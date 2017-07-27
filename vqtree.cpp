#include "onlineaverage.cpp"
#include <cstring>
#include <cmath>
#include <cfloat>
#include <cassert>
#include <stack>
#include <iostream>
#include <queue>
#include <map>
#include <cstdint>
#include <bitset>
#include "prettyprint.hpp"
//#undef DNDEBUG

#define BITSET_DIM 32

int VQ_NODE_COUNT = 0;

template <class Node> class VQTree;

class KTreeNode;

template <class T> class MinNQueue : public std::priority_queue<T> {
  public:
    size_t n;
    MinNQueue(size_t n) : n(n) {}

    inline void add(const T& value) {
      if (this->size() < n) {
        this->push(value);
      } else if (value < this->top()) {
        this->pop();
        this->push(value);
      }
    }

    inline bool isMature() {
      return this->size() >= n;
    }

    inline std::vector<T>* container() { return &this->c; }
};

template <class K, class V> class MinNMap : public std::multimap<K,V> {
  public:
    size_t n;
    MinNMap(size_t n) : n(n) {}

    inline void add(const K& key, const V& value) {
      if (this->size() < n) {
        this->emplace(key,value);
      } else if (n > 0 && key < this->rbegin()->first) {
        this->erase(std::prev(this->end()));
        this->emplace(key,value);
      }
    }

    inline bool isMature() {
      return this->size() == n;
    }
};

template <class Node_> class VQTreeNode {
  public:
    typedef Node_ Node;
    typedef VQTree<Node> Tree;

    int _id = VQ_NODE_COUNT++;
    std::deque<int>* _contents;
    OnlineAverage _avg;
    Node* parent;
    Tree* tree;

  public:
    VQTreeNode(Tree* tree) : VQTreeNode(tree, nullptr) {}

    VQTreeNode(Tree* tree, Node* parent) : _contents(new std::deque<int>()),
        _avg(tree->dim), parent(parent), tree(tree) {}

    virtual ~VQTreeNode() {
      freeContents();
    }

    inline int id() { return _id; }
    inline OnlineAverage* avg() { return &_avg; }
    inline double* position() { return _avg.position(); }
    inline bool isRoot() { return parent == nullptr; }
    //inline bool isLeaf() { return numChildren() == 0; }
    inline bool isLeaf() { return _contents != nullptr; }
    inline std::deque<int>* contents() { return _contents; }

    inline void freeContents() {
      if (_contents != nullptr) {
        delete _contents;
        _contents = nullptr;
      }
    }

    inline void freeTree() {
      for (int i = 0; i < numChildren(); i++) {
        child(i)->freeTree();
      }
      delete this;
    }

    virtual int numChildren() = 0;
    virtual Node* child(int ndx) = 0;
    virtual int add(int ndx) = 0;
    virtual void remove(int ndx) = 0;
    virtual double* childPosition(int ndx) { return child(ndx)->position(); }
};

template <class Node> class VQTree {
  public:
    using MinDistQ = MinNQueue<std::pair<double, int>>;
    using LookupSet = std::unordered_set<Node*>;
    size_t currNdx = 0;
    bool wrap = false;

    size_t dim;
    size_t maxSize;
    size_t maxLeafSize;
    size_t branchFactor;
    size_t minLeaves;
    size_t minN;
    double spill;
    int defaultSearchType;

    Node* root;
    LookupSet* nodeLookup;
    double* dat;
    double* lbl;

    bool removeDups;
    std::bitset<BITSET_DIM>* driftHist;
    size_t driftThreshold;
    uint32_t driftMask;
    long driftCount = 0;

  public:
    VQTree(size_t dim, size_t maxSize, size_t maxLeafSize=64, size_t branchFactor=16, size_t minLeaves=100, size_t minN=100, int searchType=3, double spill=-1., bool removeDups=true, size_t driftHistLen=5, size_t driftThreshold=4) :
        dim(dim), maxSize(maxSize), maxLeafSize(maxLeafSize), branchFactor(branchFactor),
        minLeaves(minLeaves), minN(minN), spill(spill), defaultSearchType(searchType),
        root(new Node(this)), nodeLookup(new LookupSet[maxSize]),
        dat(new double[dim*maxSize]), lbl(new double[dim*maxSize]),
        removeDups(removeDups), driftHist(new std::bitset<BITSET_DIM>[maxSize]),
        driftThreshold(driftThreshold) {
      driftMask = 0;
      for (size_t i = 0; i < driftHistLen; i++) {
        driftMask = (driftMask << 1) | 1;
      }
    }

    ~VQTree() {
      root->freeTree();
      delete[] nodeLookup;
      delete[] dat;
      delete[] lbl;
      delete[] driftHist;
    }

    double* getData(int ndx) { return dat+dim*ndx; }
    double getLabel(int ndx) { return lbl[ndx]; }

    void add(double* data, double label) {
      if (currNdx >= maxSize) {
        wrap = true;
        currNdx = 0;
      }
      int ndx = currNdx++;
      if (wrap) {
        clear(ndx);
      }
      if (driftThreshold > 0) {
        driftHist[ndx].reset();
        enforceDriftConsistency(data, label);
      }
      std::memcpy(getData(ndx), data, dim*sizeof(*data));
      lbl[ndx] = label;
      int oldNdx = root->add(ndx);
      if (oldNdx != -1) {
        lbl[ndx] = (.1)*lbl[oldNdx] + .9*lbl[ndx];
        clear(oldNdx);
      }
    }

    void clear(int ndx) {
      for (Node* tmpNode : nodeLookup[ndx]) {
        tmpNode->remove(ndx);
      }
      nodeLookup[ndx].clear();
    }

    Node* closestChild(Node* n, double* data) {
      assert(!n->isLeaf());
      double minDist = DBL_MAX;
      Node* minNode = nullptr;
      for (int i = 0; i < n->numChildren(); i++) {
        Node* tmpNode = n->child(i);
        double tmpDist = dist(tmpNode->position(), data);
        if (tmpDist < minDist) {
          minDist = tmpDist;
          minNode = tmpNode;
        }
      }
      return minNode;
    }

    double query(double* queryData, int searchType=-1) {
      double sum = 0, norm = 0;
      auto callback = [this, queryData, &sum, &norm](Node* node) {
        for (size_t i = 0; i < node->contents()->size(); i++) {
          double k = kernel(getData((*node->contents())[i]), queryData);
          double lbl = getLabel((*node->contents())[i]);
          sum += lbl*k;
          norm += k;
        }
      };
      search(queryData, callback, searchType);
      return sum / norm;
    }

    std::vector<int>* nearestNeighbors(double* queryData, int n, int searchType=-1) {
      MinDistQ q(n);
      auto callback = [this, queryData, &q](Node* node) {
        for (int i : *node->contents()) {
          q.add(std::make_pair(dist(getData(i), queryData), i));
        }
      };
      search(queryData, callback, searchType);
      std::sort(q.container()->begin(), q.container()->end());

      std::vector<int>* out = new std::vector<int>();
      for (auto pair : *q.container()) {
        out->push_back(pair.second);
      }
      return out;
    }

    //TODO: Not accurate for spill
    void enforceDriftConsistency(double* queryData, double target, int searchType=5) {
      driftCount = 0;
      double sum = 0, norm = 0;
      std::unordered_set<int> visited;
      std::vector<std::tuple<Node*,int,double,double>> candidates;
      //TODO: Why duplicates?
      auto callback = [this, queryData, &sum, &norm, &candidates, &visited](Node* node) {
        //for (size_t i = 0; i < node->contents()->size(); i++) {
        //  double k = kernel(getData((*node->contents())[i]), queryData);
        //  double lbl = getLabel((*node->contents())[i]);
        for (int i : *node->contents()) {
          if (visited.find(i) != visited.end()) { continue; }
          visited.insert(i);
          double k = kernel(getData(i), queryData);
          double lbl = getLabel(i);
          sum += lbl*k;
          norm += k;
          candidates.push_back(std::make_tuple(node, i, k, lbl));
        }
      };
      search(queryData, callback, searchType);

      double prediction = sum/norm;
      double predictionErr = std::abs(target - prediction);
      //std::cout << "candidates: " << candidates << std::endl;
      for (auto &candidate : candidates) {
        Node* node = std::get<0>(candidate);
        int ndx = std::get<1>(candidate);
        double k = std::get<2>(candidate);
        double lbl = std::get<3>(candidate);
        double subPrediction = (sum-lbl*k) / (norm-k);
        double subPredictionErr = std::abs(target - subPrediction);

        driftHist[ndx] <<= 1;
        driftHist[ndx] &= driftMask;
        if (subPredictionErr < predictionErr - 1e-10) {
          //printf("Marking %d: %.4f vs %.4f (%.4f, %.4f)\n", ndx, prediction, subPrediction, k, lbl);
          driftHist[ndx] |= 1;
          if (driftHist[ndx].count() >= driftThreshold) {
            //printf("Evicting %d (%.4f, %.4f)\n", ndx, k, lbl);
            driftCount++;
            node->remove(ndx);
            unregisterLeaf(node, ndx);
            //TODO: Rebalance prediction?
            //sum -= lbl*k;
            //norm -= k;
            //predictionErr = std::abs(target - sum/norm);
          }
        }
      }

      //for (Node* node : nodes) {
      //  for (int i : *node->contents()) {
      //  //for (size_t i = 0; i < node->contents()->size(); i++) {
      //    double k = kernel(getData(i), queryData);
      //    double lbl = getLabel(i);
      //    double subPredictionErr = std::abs(target - (sum-lbl*k) / (norm-k));
      //    driftHist[i] <<= 1;
      //    driftHist[i] &= driftMask;
      //    if (subPredictionErr > predictionErr) {
      //      driftHist[i] |= 1;
      //      if (driftHist[i].count() >= driftThreshold) {
      //        node->remove(i);
      //        //TODO: Rebalance prediction?
      //        //sum -= lbl*k;
      //        //norm -= k;
      //        //predictionErr = std::abs(target - sum/norm);
      //      }
      //    }
      //  }
      //}
    }

    Node* nearestLeaf(Node* tmp, double* query) {
      while (!tmp->isLeaf()) {
        tmp = closestChild(tmp, query);
      }
      return tmp;
    }

    int size() {
      return wrap ? maxSize : currNdx;
    }

    int countNodes() {
      std::deque<Node*> q;
      q.push_back(root);
      int out = 0;
      while (q.size() > 0) {
        Node* node = q.front();
        q.pop_front();
        out++;
        if (!node->isLeaf()) {
          for (int i = 0; i < node->numChildren(); i++) {
            q.push_back(node->child(i));
          }
        }
      }
      return out;
    }

    double dist(double* a, double* b) {
      double out = 0, tmp;
      for (size_t i = 0; i < dim; i++) {
        tmp = a[i]-b[i];
        out += tmp*tmp;
      }
      return out;
    }

    inline double relDistToPlane(double* p, double* wPlus, double* wMinus) {
      return relDistToPlane(dist(p,wPlus), dist(p,wMinus), dist(wPlus,wMinus));
    }
    inline double relDistToPlane(double dPlus, double dMinus, double* wPlus, double* wMinus) {
      return relDistToPlane(dPlus, dMinus, dist(wPlus, wMinus));
    }
    inline double relDistToPlane(double dPlus, double dMinus, double dPM) {
      return std::abs(dPlus-dMinus)/dPM;
    }

    inline double distToPlane(double* p, double* wPlus, double* wMinus) {
      return distToPlane(dist(p,wPlus), dist(p,wMinus), dist(wPlus,wMinus));
    }
    inline double distToPlane(double dPlus, double dMinus, double* wPlus, double* wMinus) {
      return distToPlane(dPlus, dMinus, dist(wPlus, wMinus));
    }
    inline double distToPlane(double dPlus, double dMinus, double dPM) {
      double dDiff = dPlus-dMinus;
      return dDiff*dDiff / (4 * dPM);
    }

    inline double kernel(double dist) {
      return 1/(dist+1e-5);
    }

    inline double kernel(double* a, double* b) {
      return 1/(dist(a,b)+1e-5);
    }

    void registerLeaf(Node* n, int ndx) {
      nodeLookup[ndx].insert(n);
    }

    void unregisterLeaf(Node* n, int ndx) {
      nodeLookup[ndx].erase(n);
    }

    std::vector<int>* leafStats() {
      std::vector<int>* out = new std::vector<int>();
      std::deque<Node*> q;
      Node* node;
      q.push_back(root);
      q.push_back(nullptr);
      while (q.size() > 1) {
        int leafCount = 0, innerCount = 0;
        while ((node = q.front())) {
          q.pop_front();
          if (node->isLeaf()) {
            leafCount += node->contents()->size();
          } else {
            innerCount++;
            for (int i = 0; i < node->numChildren(); i++) {
              q.push_back(node->child(i));
            }
          }
        }
        out->push_back(innerCount);
        out->push_back(leafCount);
        q.pop_front();
        q.push_back(nullptr);
      }
      q.clear();

      return out;
    }

    void printTree() { printTree(root); }
    void printTree(Node* start) {
      std::stack<Node*> stack;
      stack.push(start);

      int depth = 0;
      while (stack.size() > 0) {
        Node* tmp = stack.top(); stack.pop();
        if (tmp == nullptr) {
          depth--;
        } else {
          printNode(tmp, depth);
          if (!tmp->isLeaf()) {
            depth++;
            stack.push(nullptr);
            for (int i = tmp->numChildren()-1; i >= 0; i--) {
              stack.push(tmp->child(i));
            }
          }
        }
      }
      fflush(stdout);
    }

  private:
    template <typename Func> void search(double* query, const Func& callback, int searchType=-1) {
      using SearchMethod = void (VQTree::*)(double*, const Func&);
      if (searchType == -1) {
        searchType = defaultSearchType;
      }
      static SearchMethod searches[] = {
        /*0*/ &VQTree::searchBrute,
        /*1*/ &VQTree::searchExact,
        /*2*/ &VQTree::searchDefeatist,
        /*3*/ &VQTree::searchMultiLeafProt,
        /*4*/ &VQTree::searchMultiLeafProtRecur,
        /*5*/ &VQTree::searchMultiLeafPlane,
        /*6*/ &VQTree::searchMultiLeafPlaneRecur,
      };
      (this->*searches[searchType])(query, callback);
    }

    template <typename Func> void searchBrute(double* query, const Func& callback) {
      std::stack<int> ndxes;
      ndxes.push(0);

      Node* tmp = root;
      while (tmp != nullptr) {
        if (tmp->isLeaf()) {
          callback(tmp);
          tmp = tmp->parent;
          ndxes.pop();
        } else if (ndxes.top() == tmp->numChildren()) {
          tmp = tmp->parent;
          ndxes.pop();
        } else {
          tmp = tmp->child(ndxes.top()++);
          ndxes.push(0);
        }
      }
    }
    template <typename Func> void searchExact(double* query, const Func& callback) {
      MinNQueue<double> closest(minN);
      searchExact(root, query, &closest, callback);
    }
    template <typename Func> void searchExact(Node* node, double* query, MinNQueue<double>* closest, const Func& callback) {
      //TODO: Why is this not actually exact.. ?
      if (node->isLeaf()) {
        for (int i : *node->contents()) {
          closest->add(i);
        }
        callback(node);
        return;
      }

      int numChildren = node->numChildren();
      std::vector<std::pair<double, Node*>> dists;
      dists.reserve(numChildren);
      for (int i = 0; i < numChildren; i++) {
        Node* tmpChild = node->child(i);
        dists.emplace_back(dist(tmpChild->position(), query), tmpChild);
      }
      std::sort(dists.begin(), dists.end());

      std::pair<double, Node*> minPair = dists[0];
      double minDist = minPair.first;
      Node* minNode = minPair.second;
      double* minPos = minNode->position();

      searchExact(minNode, query, closest, callback);
      for (int i = 1; i < numChildren; i++) {
        std::pair<double, Node*> tmp = dists[i];
        
        double tmpDistToPlane = distToPlane(minDist, tmp.first, minPos, tmp.second->position());
        if (!closest->isMature() || tmpDistToPlane < closest->top()) {
          searchExact(tmp.second, query, closest, callback);
        }
      }
    }

    template <typename Func> void searchDefeatist(double* query, const Func& callback) {
      puts("This shouldn't be here.");
      Node* tmp = nearestLeaf(root, query)->parent;
      for (int i = 0; i < tmp->numChildren(); i++) {
        callback(tmp->child(i));
      }
      //callback(nearestLeaf(root, query));
    }

    template <typename Func> void searchMultiLeafProt(double* query, const Func& callback) {
      MinNQueue<std::pair<double, Node*>> searchQ(minLeaves-1);
      Node* tmp = root;

      while (!tmp->isLeaf()) {
        Node* minNode = nullptr;
        double minDist = DBL_MAX;
        for (int i = 0; i < tmp->numChildren(); i++) {
          Node* tmpNode = tmp->child(i);
          double tmpDist = dist(query, tmpNode->position());
          if (tmpDist < minDist) {
            minNode = tmpNode;
            minDist = tmpDist;
          } else {
            searchQ.add(std::make_pair(tmpDist, tmpNode));
          }
        }
        tmp = minNode;
      }
      callback(tmp);
      for (auto pair : *searchQ.container()) {
        callback(nearestLeaf(pair.second, query));
      }
    }

    template <typename Func> void searchMultiLeafProtRecur(double* query, const Func& callback) {
      using DistPair = std::pair<double, Node*>;
      MinNMap<double, Node*> searchQ(minLeaves);
      searchQ.add(0, root);

      for (int i = 0; i < minLeaves && !searchQ.empty(); i++) {
        auto tmpP = searchQ.begin();
        Node* tmp = tmpP->second;
        searchQ.erase(tmpP);
        searchQ.n--;

        while (!tmp->isLeaf()) {
          DistPair minPair(DBL_MAX, nullptr);
          for (int i = 0; i < tmp->numChildren(); i++) {
            Node* tmpNode = tmp->child(i);
            double tmpDist = dist(query, tmpNode->position());
            if (tmpDist < minPair.first) {
              minPair = std::make_pair(tmpDist, tmpNode);
            } else {
              searchQ.add(tmpDist, tmpNode);
            }
          }
          tmp = minPair.second;
        }

        callback(tmp);
      }
    }

    template <typename Func> void searchMultiLeafPlane(double* query, const Func& callback) {
      double* dists = new double[branchFactor];
      MinNMap<double, Node*> searchQ(minLeaves-1);
      searchQ.add(0, root);

      Node* tmp = root;

      while (!tmp->isLeaf()) {
        for (int i = 0; i < tmp->numChildren(); i++) {
          dists[i] = dist(query, tmp->child(i)->position());
        }
        auto minChildP = std::min_element(dists, dists+tmp->numChildren());
        int minNdx = minChildP - dists;
        Node* minChild = tmp->child(minNdx);
        double* minPos = minChild->position();

        for (int i = 0; i < tmp->numChildren(); i++) {
          if (i == minNdx) { continue; }
          Node* tmpChild = tmp->child(i);
          double* tmpPos = tmpChild->position();
          double tmpDTP = distToPlane(dists[minNdx], dists[i], minPos, tmpPos);
          searchQ.add(tmpDTP, tmpChild);
        }
        
        tmp = minChild;
      }

      callback(tmp);
      for (auto pair : searchQ) {
        callback(nearestLeaf(pair.second, query));
      }
      delete[] dists;
    }

    template <typename Func> void searchMultiLeafPlaneRecur(double* query, const Func& callback) {
      double* dists = new double[branchFactor];
      MinNMap<double, Node*> searchQ(minLeaves);
      searchQ.add(0, root);

      for (int i = 0; i < minLeaves && !searchQ.empty(); i++) {
        auto tmpP = searchQ.begin();
        Node* tmp = tmpP->second;
        searchQ.erase(tmpP);
        searchQ.n--;

        while (!tmp->isLeaf()) {
          for (int i = 0; i < tmp->numChildren(); i++) {
            dists[i] = dist(query, tmp->child(i)->position());
          }
          auto minChildP = std::min_element(dists, dists+tmp->numChildren());
          int minNdx = minChildP - dists;
          Node* minChild = tmp->child(minNdx);
          double* minPos = minChild->position();

          for (int i = 0; i < tmp->numChildren(); i++) {
            if (i == minNdx) { continue; }
            Node* tmpChild = tmp->child(i);
            double* tmpPos = tmpChild->position();
            double tmpDTP = distToPlane(dists[minNdx], dists[i], minPos, tmpPos);
            searchQ.add(tmpDTP, tmpChild);
          }
          
          tmp = minChild;
        }

        callback(tmp);
      }
      delete[] dists;
    }

    void printNode(Node* node, int depth) {
      printIndent(depth);
      printf("%d ", node->id());
      //printData(node->position());
      //printf("(%d) ", node->avg()->count);

      if (node->isLeaf()) {
        putchar('\n');
        printIndent(depth+1);
        printf("* ");
        printf("%zu",node->contents()->size());
        putchar(' ');
        printContents(node);
      } else {
        printChildren(node);
      }
      putchar('\n');
    }
    void printIndent(int depth) {
      for (int i = 0; i < depth; i++) {
        printf("  ");
      }
    }
    void printContents(Node* node) {
      printf("- ");
      auto& contents = *node->contents();
      for (int i = 0; i < contents.size(); i++) {
        printData(getData(contents[i]), getLabel(contents[i]));
      }
      printf(" [");
      for (int i = 0; i < contents.size(); i++) {
        if (i != 0) { putchar(','); }
        printf("%d", contents[i]);
      }
      putchar(']');
    }
    void printData(double* data, double target) {
      //putchar('[');
      //for (size_t i = 0; i < dim; i++) {
      //  if (i != 0) { putchar(','); }
      //  printf("%0.4f", data[i]);
      //}
      //printf("(%.2f)", target);
      //putchar(']');
      printf("[%.4f]", target);
    }
    void printChildren(Node* node) {
      putchar('[');
      for (int i = 0; i < node->numChildren(); i++) {
        if (i != 0) { putchar(','); }
        printf("%d", node->child(i)->avg()->count);
      }
      putchar(']');
    }
};

class MeanTreeNode : public VQTreeNode<MeanTreeNode> {
  public:
    using VQTreeNode::VQTreeNode;

    Node* childA = nullptr;
    Node* childB = nullptr;

    inline int numChildren() { return childA == nullptr ? 0 : 2; }

    inline Node* child(int ndx) { return ndx == 0 ? childA : childB; }

    int add(int ndx) {
      double* data = tree->getData(ndx);
      double label = tree->getLabel(ndx);
      Node* tmp = this;

      while (!tmp->isLeaf()) {
        tmp->avg()->add(data, label);
        tmp = tree->closestChild(tmp, data);
      }

      std::deque<int>* contents = tmp->contents();
      int matchNdx = -1;
      for (auto tmpNdx : *contents) {
        double* dataTmp = tree->getData(tmpNdx);
        bool flag = true;
        for (size_t i = 0; flag && i < tree->dim; i++) {
          flag = data[i] == dataTmp[i];
        }
        if (flag) {
          matchNdx = tmpNdx;
          break;
        }
      }

      if (matchNdx == -1) {
        tmp->avg()->add(data, label);
        contents->push_back(ndx);
        tree->registerLeaf(tmp, ndx);

        if (contents->size() > tree->maxLeafSize) {
          tmp->convertLeafToInner();
        }
      } else {
        while (!tmp->isRoot()) {
          tmp = tmp->parent;
          tmp->avg()->remove(data, label);
        }
      }
      return matchNdx;
    }

    void remove(int ndx) {
      if (isLeaf() && !isRoot() && contents()->size() == 1) {
        Node* sibling = this == parent->childA ? parent->childB : parent->childA;
        if (parent->isRoot()) {
          tree->root = sibling;
          sibling->parent = nullptr;
        } else {
          for (size_t i = 0; i < contents()->size(); i++) {
            parent->remove((*contents())[i]);
          }
          Node* grandpa = parent->parent;
          sibling->parent = grandpa;
          (grandpa->childA == parent ? grandpa->childA : grandpa->childB) = sibling;
        }
        delete parent;
        delete this;
      } else {
        double* data = tree->getData(ndx);
        double label = tree->getLabel(ndx);

        avg()->remove(data, label);

        if (isLeaf()) {
          contents()->pop_front();
        }
        if (!isRoot()) {
          parent->remove(ndx);
        }
      }
    }

  private:
    void convertLeafToInner() {
      childA = new Node(tree, this);
      childB = new Node(tree, this);
      OnlineAverage posA(tree->dim), posB(tree->dim), posMean(tree->dim);
      for (int tmp : *contents()) {
        posMean.add(tree->getData(tmp), 0);
      }
      posA.add(posMean.position(), 0);
      posB.add(posMean.position(), 0);

      std::vector<int> labels(contents()->size(), -1);

      bool flag = true, first = true;
      while (flag) {
        flag = false;
        for (size_t i = 0; i < contents()->size(); i++) {
          int ndx = (*contents())[i];
          double* dat = tree->getData(ndx);
          Node* c = tree->closestChild(this, dat);
          int newLabel = c == childA ? 0 : 1;

          if (labels[i] != newLabel) {
            double lbl = tree->getLabel(ndx);
            if (labels[i] == 0) {
              posA.remove(dat, lbl);
            } else if (labels[i] == 1) {
              posB.remove(dat, lbl);
            }
            c->avg()->add(dat,lbl);
            flag = true;
            labels[i] = newLabel;
          }
        }
        if (first) {
          first = false;
          posA.remove(posMean.position(), 0);
          posB.remove(posMean.position(), 0);
        }
      }
      for (size_t i = 0; i < contents()->size(); i++) {
        tree->unregisterLeaf(this, (*contents())[i]);
        (labels[i] == 0 ? childA : childB)->add((*contents())[i]);
      }
      freeContents();
    }
};

class KTreeNode : public VQTreeNode<KTreeNode> {
  public:
    using VQTreeNode::VQTreeNode;

    std::vector<Node*>* children = nullptr;
    size_t childNdx = -1;

    inline int numChildren() { return children == nullptr ? 0 : children->size(); }

    inline Node* child(int ndx) { return (*children)[ndx]; }

    int add(int ndx) {
      double* data = tree->getData(ndx);
      double label = tree->getLabel(ndx);
      int out = -1;

      if (tree->spill >= 0) {
        double* dists = new double[tree->branchFactor];
        std::queue<Node*> q;
        q.push(this);
        while (q.size() > 0) {
          Node* tmp = q.front();
          q.pop();
          if (tmp->isLeaf()) {
            out = std::max(out, tmp->addToLeaf(ndx, data, label));
            continue;
          }

          int nChildren = tmp->numChildren();
          for (int i = 0; i < nChildren; i++) {
            dists[i] = tree->dist(data, tmp->child(i)->position());
          }
          auto minChildP = std::min_element(dists, dists+nChildren);
          int minNdx = minChildP - dists;
          Node* minChild = tmp->child(minNdx);
          double* minPos = minChild->position();

          q.push(minChild);

          for (int i = 0; i < nChildren; i++) {
            if (i == minNdx) { continue; }
            Node* tmpChild = tmp->child(i);
            double* tmpChildPos = tmpChild->position();
            double tmpDTP = tree->relDistToPlane(dists[minNdx], dists[i], minPos, tmpChildPos);
            if (tmpDTP < tree->spill) {
              q.push(tmpChild);
            }
          }
        }
        delete[] dists;
      } else {
        Node* tmp = this;
        while (!tmp->isLeaf()) {
          tmp = tree->closestChild(tmp, data);
        }
        out = tmp->addToLeaf(ndx, data, label);
      }

      return out;
    }

    void remove(int ndx) {
      assert(isLeaf());

      double* data = tree->getData(ndx);
      double label = tree->getLabel(ndx);

      auto delP = std::find(contents()->begin(), contents()->end(), ndx);
      assert(delP != contents()->end());
      size_t delNdx = delP - contents()->begin();

      if (delNdx != contents()->size()-1) {
        (*contents())[delNdx] = contents()->back();
      }

      contents()->pop_back();
      for (Node* tmp = this; tmp != nullptr; tmp = tmp->parent) {
        tmp->avg()->remove(data, label);
      }
      if (contents()->size() == 0) {
        parent->freeChild(this);
      }
    }

    ~KTreeNode() {
      if (children != nullptr) {
        delete children;
      }
    }

    bool checkInvariant() {
      for (size_t i = 0; i < numChildren(); i++) {
        if (!child(i)->checkInvariant()) {
          return false;
        }
      }
      if (!checkInvariantLocal()) {
        return false;
      }
      return true;
    }
    bool checkInvariantLocal() {
      if (!std::isfinite(position()[0])) {
        printf("Fail PosFinite: %f\n", position()[0]);
        return false;
      }
      if (!isLeaf()) {
        if (numChildren() > tree->branchFactor) {
          printf("Fail BranchFactor: %d vs %zu\n", numChildren(), tree->branchFactor);
          return false;
        }
        int childPosCount = 0;
        for (size_t i = 0; i < children->size(); i++) {
          Node* child = (*children)[i];
          childPosCount += child->avg()->count;
          if (child->parent != this || child->childNdx != i) {
            puts("Fail Parent/ChildNdx");
            return false;
          }
        }
        if (avg()->count != childPosCount) {
          printf("Fail PosCount: %d vs %d\n", avg()->count, childPosCount);
          return false;
        }
      } else {
        if (contents()->size() > tree->maxLeafSize) {
          printf("Fail MaxLeafSize: %zu vs %zu\n", contents()->size(), tree->maxLeafSize);
          return false;
        }
        if (avg()->count != contents()->size()) {
          printf("Fail Leaf PosCount: %d vs %zu\n", avg()->count, contents()->size());
          return false;
        }
      }
      return true;
    }

    int distUp() {
      int cnt = 0;
      for (Node* tmp = this; !tmp->isRoot(); tmp = tmp->parent, cnt++) { }
      return cnt;
    }

    int distDown() {
      int cnt = 0;
      for (Node* tmp = this; !tmp->isLeaf(); tmp = tmp->child(0), cnt++) { }
      return cnt;
    }

  private:
    void freeChild(Node* child) {
      size_t ndx = child->childNdx;
      assert(ndx >= 0);
      assert(ndx < children->size());
      assert((*children)[ndx] == child);
      if (children->size() == 1) {
        if (!isRoot()) {
          parent->freeChild(this);
        }
      } else {
        if (ndx != children->size()-1) {
          (*children)[ndx] = children->back();
          (*children)[ndx]->childNdx = ndx;
        }
        children->pop_back();
      }
      delete child;
    }

    Node* makeInnerNode() {
      Node* out = new Node(tree, parent);
      out->freeContents();
      out->children = new std::vector<Node*>();
      out->children->reserve(tree->branchFactor+1);
      return out;
    }

    void makeRootNode(Node* sibling) {
      assert(this->parent == nullptr);
      assert(sibling->parent == nullptr);
      Node* newRoot = makeInnerNode();

      newRoot->children->push_back(this);
      newRoot->children->push_back(sibling);
      this->childNdx = 0;
      sibling->childNdx = 1;

      newRoot->avg()->add(this->avg());
      newRoot->avg()->add(sibling->avg());
      tree->root = this->parent = sibling->parent = newRoot;
    }

    template <class T> void labeledMove(std::vector<int>* labels, T* src, T* dest) {
      size_t newSize = 0;
      for (size_t i = 0; i < labels->size(); i++) {
        if ((*labels)[i] == 0) {
          (*src)[newSize++] = (*src)[i];
        } else {
          dest->push_back((*src)[i]);
        }
      }

      while (src->size() > newSize) {
        src->pop_back();
      }
    }

    void addChild(Node* newChild) {
      assert(!isLeaf());
      newChild->childNdx = children->size();
      children->push_back(newChild);
      if (children->size() > tree->branchFactor) {
#ifndef DNDEBUG
        auto s1 = children->size();
#endif
        Node* sibling = splitInner();
#ifndef DNDEBUG
        auto s2 = children->size(), s3 = sibling->children->size();
        if (s2 >= s1 || s3 >= s1) {
          printf("\n%zu -> %zu %zu\n", s1, s2, s3);
          puts("PANIC!!!");
          tree->printTree(this->parent);
        }
#endif

        if (isRoot()) {
          makeRootNode(sibling);
        } else {
          parent->addChild(sibling);
        }
      }
    }

    Node* splitLeaf() {
      Node* sibling = new Node(tree, parent);

      std::vector<int>* labels = twoMeansLeaf();
      /*for (int i = 0; i < labels->size(); i++) {
        if ((*labels)[i] == 1) {
          int tmpNdx = (*contents())[i];
          tree->unregisterLeaf(this, tmpNdx);
          tree->registerLeaf(sibling, tmpNdx);
        }
      }*/
      labeledMove(labels, this->contents(), sibling->contents());
      delete labels;

      for (size_t i = 0; i < sibling->contents()->size(); i++) {
        int tmpNdx = (*sibling->contents())[i];
        sibling->avg()->add(tree->getData(tmpNdx), tree->getLabel(tmpNdx));
        tree->unregisterLeaf(this, tmpNdx);
        tree->registerLeaf(sibling, tmpNdx);
      }
      avg()->remove(sibling->avg());

      return sibling;
    }
    std::vector<int>* twoMeansLeaf() {
      std::vector<double*> data(contents()->size());
      for (size_t i = 0; i < contents()->size(); i++) {
        data[i] = tree->getData((*contents())[i]);
      }
      return twoMeans(&data);
    }

    Node* splitInner() {
      Node* sibling = makeInnerNode();

      std::vector<int>* labels = twoMeansInner();
      labeledMove(labels, children, sibling->children);
#ifdef DNDEBUG
      if (children->size() == 0 || sibling->children->size() == 0) {
        printf("Invalid children size after move... %zu %zu\n", this->children->size(), sibling->children->size());
        std::cout << *labels << std::endl;
      }
#endif
      delete labels;

      for (size_t i = 0; i < this->children->size(); i++) {
        this->child(i)->childNdx = i;
      }
      for (size_t i = 0; i < sibling->children->size(); i++) {
        Node* tmp = sibling->child(i);
        sibling->avg()->add(tmp->avg());
        tmp->parent = sibling;
        tmp->childNdx = i;
      }
      this->avg()->remove(sibling->avg());
      return sibling;
    }
    std::vector<int>* twoMeansInner() {
      std::vector<double*> data(children->size());
      for (size_t i = 0; i < children->size(); i++) {
        data[i] = (*children)[i]->position();
      }
      return twoMeans(&data);
    }

    std::vector<int>* twoMeans(std::vector<double*>* data) {
      OnlineAverage posA(tree->dim), posB(tree->dim), posMean(tree->dim);
      for (double* d : *data) {
        posMean.add(d, 0);
      }
      posA.add(posMean.position(), 0);
      posB.add(posMean.position(), 0);

      std::vector<int>* labels = new std::vector<int>(data->size(), -1);

      bool flag = true, first = true;
      for (int count = 0; count < 50 && flag; count++) {
        flag = false;
        for (size_t i = 0; i < data->size(); i++) {
          double* dat = (*data)[i];
          double dA = tree->dist(posA.position(), dat);
          double dB = tree->dist(posB.position(), dat);
          int newLabel = dA < dB ? 0 : 1;

          if ((*labels)[i] != newLabel) {
            if ((*labels)[i] == 0) {
              posA.remove(dat, 0);
            } else if ((*labels)[i] == 1) {
              posB.remove(dat, 0);
            }
            (newLabel == 0 ? posA : posB).add(dat, 0);
            flag = true;
            (*labels)[i] = newLabel;
          }
        }
        if (first) {
          first = false;
          posA.remove(posMean.position(), 0);
          posB.remove(posMean.position(), 0);
        }
      }
      return labels;
    }

    bool doesMatch(double* a, double* b) {
      for (size_t i = 0; i < tree->dim; i++) {
        if (a[i] != b[i]) {
          return false;
        }
      }
      return true;
    }

    int addToLeaf(int ndx, double* data, double label) {
      tree->registerLeaf(this, ndx);
      if (tree->removeDups) {
        auto matchPtr = std::find_if(contents()->begin(), contents()->end(), [data, this](int i) {
          return doesMatch(data, tree->getData(i));
        });
        if (matchPtr != contents()->end()) {
          int out = *matchPtr;
          tree->unregisterLeaf(this, out);
          *matchPtr = ndx;
          return out;
        }
      }
      for (Node* tmp = this; tmp != nullptr; tmp = tmp->parent) {
        tmp->avg()->add(data, label);
      }

      contents()->push_back(ndx);
      if (contents()->size() > tree->maxLeafSize) {
        if (isRoot()) {
          makeRootNode(splitLeaf());
        } else {
          parent->addChild(splitLeaf());
        }
      }
#ifndef DNDEBUG
      for (Node* tmp = this; tmp != nullptr; tmp = tmp->parent) {
        if (!tmp->checkInvariantLocal()) {
          printf("!!!!!! %d\ndistUp:%d distDown:%d id:%d\n", ndx, distUp(), distDown(), tmp->_id);
          tree->printTree();
        }
      }
#endif
      return -1;
    }
};

using MeanTree = VQTree<MeanTreeNode>;
using KTree = VQTree<KTreeNode>;
