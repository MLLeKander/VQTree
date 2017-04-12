#include "onlineaverage.cpp"
#include <cstring>
#include <cmath>
#include <cfloat>
#include <cassert>
#include <boost/circular_buffer.hpp>
#include <stack>
#include <iostream>
#include <queue>
#include <map>
#include "prettyprint.hpp"
#undef DNDEBUG

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
  private:
    int _id = VQ_NODE_COUNT++;
    //TODO: Smart pointer?
    std::deque<int>* _contents;
    OnlineAverage _pos;

  public:
    typedef Node_ Node;
    typedef VQTree<Node> Tree;
    Node* parent;
    Tree* tree;

    VQTreeNode(Tree* tree) : VQTreeNode(tree, nullptr) {}

    VQTreeNode(Tree* tree, Node* parent) : _contents(new std::deque<int>()),
        _pos(tree->dim), parent(parent), tree(tree) {}

    virtual ~VQTreeNode() {
      freeContents();
    }

    inline int id() { return _id; }
    inline OnlineAverage* pos() { return &_pos; }
    inline bool isRoot() { return parent == nullptr; }
    inline bool isLeaf() { return numChildren() == 0; }
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
    Node* root;
    LookupSet* nodeLookup;
    double* dat;
    double* lbl;
    bool removeDups;

  public:
    VQTree(size_t dim, size_t maxSize, size_t maxLeafSize=64, size_t branchFactor=16, size_t minLeaves=100, size_t minN=100, bool removeDups=true) :
        dim(dim), maxSize(maxSize), maxLeafSize(maxLeafSize), branchFactor(branchFactor),
        minLeaves(minLeaves), minN(minN), root(new Node(this)), nodeLookup(new LookupSet[maxSize]),
        dat(new double[dim*maxSize]), lbl(new double[dim*maxSize]), removeDups(removeDups) {}

    ~VQTree() {
      root->freeTree();
      delete[] nodeLookup;
      delete[] dat;
      delete[] lbl;
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
        double tmpDist = dist(tmpNode->pos()->position(), data);
        if (tmpDist < minDist) {
          minDist = tmpDist;
          minNode = tmpNode;
        }
      }
      return minNode;
    }

    double query(double* data, int type=3) {
      double sum = 0, norm = 0;
      auto callback = [this, &sum, &norm](Node* node, double* data) {
        for (size_t i = 0; i < node->contents()->size(); i++) {
          double k = kernel(getData((*node->contents())[i]), data);
          double lbl = getLabel((*node->contents())[i]);
          sum += lbl*k;
          norm += k;
        }
      };
      search(data, type, callback);
      return sum / norm;
    }

    std::vector<int>* nearestNeighbors(double* query, int n, int type=0) {
      MinDistQ q(n);
      //buildNNQ(query, &q, type);
      auto callback = [this, &q](Node* node, double* query) {
        for (int i : *node->contents()) {
          q.add(std::make_pair(dist(getData(i), query), i));
        }
      };
      search(query, type, callback);

      std::vector<int>* out = new std::vector<int>();
      std::sort(q.container()->begin(), q.container()->end());
      for (auto pair : *q.container()) {
        out->push_back(pair.second);
      }
      return out;
    }

    Node* nearestLeaf(Node* tmp, double* query) {
      while (!tmp->isLeaf()) {
        tmp = closestChild(tmp, query);
      }
      return tmp;
    }

    int size() {
      //printf("CPP size %zu %p\n", currNdx, this);
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

    void printTree() {
      std::stack<Node*> stack;
      stack.push(root);

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
    /*void printTree() {
      std::stack<Node*> nodeStack;
      std::stack<int> ndxStack;
      Node* tmp = root;

      int depth = 0;
      while (tmp != nullptr) {
        printNode(tmp, depth);

        if (!tmp->isLeaf()) {
          stack.push(0);
        } else {
          while (stack.size() > 0 && stack.top() == tmp->numChildren()) {
            tmp = tmp->parent;
          }
        }
      }
      fflush(stdout);
    }*/

  private:
    template <typename Func> void search(double* query, int type, const Func& callback) {
      using SearchMethod = void (VQTree::*)(double*, const Func&);
      static SearchMethod searches[] = {
        /*0*/ &VQTree::searchBrute,
        /*1*/ &VQTree::searchExact,
        /*2*/ &VQTree::searchDefeatist,
        /*3*/ &VQTree::searchMultiLeafProt,
        /*4*/ &VQTree::searchMultiLeafProtRecur,
        /*5*/ &VQTree::searchMultiLeafPlane,
        /*6*/ &VQTree::searchMultiLeafPlaneRecur,
      };
      (this->*searches[type])(query, callback);
    }

    template <typename Func> void searchBrute(double* query, const Func& callback) {
      std::stack<int> ndxes;
      ndxes.push(0);

      Node* tmp = root;
      while (tmp != nullptr) {
        if (tmp->isLeaf()) {
          callback(tmp, query);
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
        callback(node, query);
        return;
      }

      int numChildren = node->numChildren();
      std::vector<std::pair<double, Node*>> dists;
      dists.reserve(numChildren);
      for (int i = 0; i < numChildren; i++) {
        Node* tmpChild = node->child(i);
        dists.emplace_back(dist(tmpChild->pos()->position(), query), tmpChild);
      }
      std::sort(dists.begin(), dists.end());

      std::pair<double, Node*> minPair = dists[0];
      double minDist = minPair.first;
      Node* minNode = minPair.second;
      double* minPos = minNode->pos()->position();

      searchExact(minNode, query, closest, callback);
      for (int i = 1; i < numChildren; i++) {
        std::pair<double, Node*> tmp = dists[i];
        
        double tmpDistToPlane = distToPlane(minDist, tmp.first, minPos, tmp.second->pos()->position());
        if (!closest->isMature() || tmpDistToPlane < closest->top()) {
          searchExact(tmp.second, query, closest, callback);
        }
      }
    }

    template <typename Func> void searchDefeatist(double* query, const Func& callback) {
      Node* tmp = nearestLeaf(root, query)->parent;
      for (int i = 0; i < tmp->numChildren(); i++) {
        callback(tmp->child(i), query);
      }
      //callback(nearestLeaf(root, query), query);
    }

    template <typename Func> void searchMultiLeafProt(double* query, const Func& callback) {
      MinNQueue<std::pair<double, Node*>> searchQ(minLeaves-1);
      Node* tmp = root;

      while (!tmp->isLeaf()) {
        Node* minNode = nullptr;
        double minDist = DBL_MAX;
        for (int i = 0; i < tmp->numChildren(); i++) {
          Node* tmpNode = tmp->child(i);
          double tmpDist = dist(query, tmpNode->pos()->position());
          if (tmpDist < minDist) {
            minNode = tmpNode;
            minDist = tmpDist;
          } else {
            searchQ.add(std::make_pair(tmpDist, tmpNode));
          }
        }
        tmp = minNode;
      }
      callback(tmp, query);
      for (auto pair : *searchQ.container()) {
        callback(nearestLeaf(pair.second, query), query);
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
            double tmpDist = dist(query, tmpNode->pos()->position());
            if (tmpDist < minPair.first) {
              minPair = std::make_pair(tmpDist, tmpNode);
            } else {
              searchQ.add(tmpDist, tmpNode);
            }
          }
          tmp = minPair.second;
        }

        callback(tmp, query);
      }
    }

    template <typename Func> void searchMultiLeafPlane(double* query, const Func& callback) {
      double* dists = new double[branchFactor];
      MinNMap<double, Node*> searchQ(minLeaves-1);
      searchQ.add(0, root);

      Node* tmp = root;

      while (!tmp->isLeaf()) {
        for (int i = 0; i < tmp->numChildren(); i++) {
          dists[i] = dist(query, tmp->child(i)->pos()->position());
        }
        auto minChildP = std::min_element(dists, dists+tmp->numChildren());
        int minNdx = minChildP - dists;
        Node* minChild = tmp->child(minNdx);
        double* minPos = minChild->pos()->position();

        for (int i = 0; i < tmp->numChildren(); i++) {
          if (i == minNdx) { continue; }
          Node* tmpChild = tmp->child(i);
          double* tmpPos = tmpChild->pos()->position();
          double tmpDTP = distToPlane(dists[minNdx], dists[i], minPos, tmpPos);
          searchQ.add(tmpDTP, tmpChild);
        }
        
        tmp = minChild;
      }

      callback(tmp, query);
      for (auto pair : searchQ) {
        callback(nearestLeaf(pair.second, query), query);
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
            dists[i] = dist(query, tmp->child(i)->pos()->position());
          }
          auto minChildP = std::min_element(dists, dists+tmp->numChildren());
          int minNdx = minChildP - dists;
          Node* minChild = tmp->child(minNdx);
          double* minPos = minChild->pos()->position();

          for (int i = 0; i < tmp->numChildren(); i++) {
            if (i == minNdx) { continue; }
            Node* tmpChild = tmp->child(i);
            double* tmpPos = tmpChild->pos()->position();
            double tmpDTP = distToPlane(dists[minNdx], dists[i], minPos, tmpPos);
            searchQ.add(tmpDTP, tmpChild);
          }
          
          tmp = minChild;
        }

        callback(tmp, query);
      }
      delete[] dists;
    }

    void printNode(Node* node, int depth) {
      printIndent(depth);
      printf("%d ", node->id());
      //printData(node->pos()->position());
      //printf(" %p(%p[%d]) ", node, node->parent, node->childNdx);

      if (node->isLeaf()) {
        putchar('\n');
        printIndent(depth+1);
        printf("* ");
        printf("%zu",node->contents()->size());
        //putchar(' ');
        //printContents(node);
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
        printData(getData(contents[i]));
      }
      printf(" [");
      for (int i = 0; i < contents.size(); i++) {
        if (i != 0) { putchar(','); }
        printf("%d", contents[i]);
      }
      putchar(']');
    }
    void printData(double* data) {
      putchar('[');
      for (size_t i = 0; i < dim; i++) {
        if (i != 0) { putchar(','); }
        printf("%0.4f", data[i]);
      }
      putchar(']');
    }
    void printChildren(Node* node) {
      putchar('[');
      for (int i = 0; i < node->numChildren(); i++) {
        if (i != 0) { putchar(','); }
        printf("%d", node->child(i)->pos()->count);
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
        tmp->pos()->add(data, label);
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
        tmp->pos()->add(data, label);
        contents->push_back(ndx);
        tree->registerLeaf(tmp, ndx);

        if (contents->size() > tree->maxLeafSize) {
          tmp->convertLeafToInner();
        }
      } else {
        while (!tmp->isRoot()) {
          tmp = tmp->parent;
          tmp->pos()->remove(data, label);
        }
      }
      return matchNdx;
    }

    void remove(int ndx) {
      //TODO: Update
#ifndef NDEBUG
      if (isLeaf()) {
        //printf("Node::remove id:%d ndx:%d front:%d\n",id, ndx, contents()->front());
        assert(contents()->front() == ndx);
      }
#endif

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

        pos()->remove(data, label);

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
            c->pos()->add(dat,lbl);
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

      Node* tmp = this;
      while (!tmp->isLeaf()) {
        tmp->pos()->add(data, label);
        tmp = tree->closestChild(tmp, data);
      }
      tree->registerLeaf(tmp, ndx);

      if (!tree->removeDups) {
        tmp->addToLeaf(ndx, data, label);
        return -1;
      }

      std::deque<int>* contents = tmp->contents();
      auto matchPtr = std::find_if(contents->begin(), contents->end(), [data, this](int i) {
        return doesMatch(data, tree->getData(i));
      });

      if (matchPtr == contents->end()) {
        tmp->addToLeaf(ndx, data, label);
        return -1;
      }

      int out = *matchPtr;
      tree->unregisterLeaf(tmp, out);
      *matchPtr = ndx;
      while (!tmp->isRoot()) {
        tmp = tmp->parent;
        tmp->pos()->remove(data, label);
      }
      return out;
      /*std::deque<int>* contents = tmp->contents();
      int matchCNdx = -1, matchNdx = -1;
      for (size_t tmpCNdx = 0; tmpCNdx < contents->size(); tmpCNdx++) {
        int tmpNdx = (*contents)[tmpCNdx];
        if (doesMatch(data, tree->getData(tmpNdx))) {
          matchCNdx = tmpCNdx;
          matchNdx = tmpNdx;
          break;
        }
      }

      tree->registerLeaf(tmp, ndx);
      if (matchNdx == -1) {
        tmp->pos()->add(data, label);
        tmp->contents()->push_back(ndx);

        if (tmp->contents()->size() > tree->maxLeafSize) {
          if (tmp->isRoot()) {
            tmp->makeRootNode(tmp->splitLeaf());
          } else {
            tmp->parent->addChild(tmp->splitLeaf());
          }
        }
      } else {
        tree->unregisterLeaf(tmp, matchNdx);
        (*contents)[matchCNdx] = ndx;
        while (!tmp->isRoot()) {
          tmp = tmp->parent;
          tmp->pos()->remove(data, label);
        }
      }
      return matchNdx;*/
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
        tmp->pos()->remove(data, label);
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
      if (!isLeaf()) {
        for (size_t i = 0; i < children->size(); i++) {
          Node* child = (*children)[i];
          if (child->parent != this || child->childNdx != i) {
            return false;
          }
          //assert(child->parent == this);
          //assert(child->childNdx == i);
          if (!child->checkInvariant()) {
            return false;
          }
        }
      }
      return true;
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

      newRoot->pos()->add(this->pos());
      newRoot->pos()->add(sibling->pos());
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
        Node* sibling = splitInner();

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
        sibling->pos()->add(tree->getData(tmpNdx), tree->getLabel(tmpNdx));
        tree->unregisterLeaf(this, tmpNdx);
        tree->registerLeaf(sibling, tmpNdx);
      }
      pos()->remove(sibling->pos());

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
      delete labels;

      for (size_t i = 0; i < this->children->size(); i++) {
        Node* tmp = (*this->children)[i];
        tmp->childNdx = i;
      }
      for (size_t i = 0; i < sibling->children->size(); i++) {
        Node* tmp = (*sibling->children)[i];
        sibling->pos()->add(tmp->pos());
        tmp->parent = sibling;
        tmp->childNdx = i;
      }
      this->pos()->remove(sibling->pos());
      return sibling;
    }
    std::vector<int>* twoMeansInner() {
      std::vector<double*> data(children->size());
      for (size_t i = 0; i < children->size(); i++) {
        data[i] = (*children)[i]->pos()->position();
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

    void addToLeaf(int ndx, double* data, double label) {
      pos()->add(data, label);
      contents()->push_back(ndx);

      if (contents()->size() > tree->maxLeafSize) {
        if (isRoot()) {
          makeRootNode(splitLeaf());
        } else {
          parent->addChild(splitLeaf());
        }
      }
    }
};

using MeanTree = VQTree<MeanTreeNode>;
using KTree = VQTree<KTreeNode>;
