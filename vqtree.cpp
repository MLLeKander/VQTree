#include "onlineaverage.cpp"
#include <cstring>
#include <cmath>
#include <cfloat>
#include <cassert>
#include <stack>
#include <iostream>
#include <cstdint>
#include <numeric>
#include <random>
#include <stdexcept>
#include "prettyprint.hpp"
#include "utils.hpp"

#define MAX_ITERS 50

int VQ_NODE_COUNT = 0;

template <class Node> class VQTree;

class KTreeNode;


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
      for (size_t i = 0; i < numChildren(); i++) {
        child(i)->freeTree();
      }
      delete this;
    }

    virtual size_t numChildren() = 0;
    virtual Node* child(int ndx) = 0;
    virtual void add(int ndx) = 0;
    virtual void remove(int ndx) = 0;
    virtual double* childPosition(int ndx) { return child(ndx)->position(); }
};

template <class Node> class VQForest {
  public:
    using MinDistQ = MinNQueue<std::pair<double, int>>;
    using Tree = VQTree<Node>;
    std::vector<Tree*> trees;
    ssize_t headNdx = -1;
    size_t tailNdx = 0;
    size_t dim;
    size_t memorySize;

    size_t minLeaves;
    size_t minN = -1;
    double exactEps;
    int defaultSearchType;

    double* dat;
    double* lbl;
    std::default_random_engine randEngine;

    bool removeDups;
    ArrMap<double, size_t>* lookupExact = nullptr;

  public:
    VQForest(size_t dim, size_t memorySize, size_t maxLeafSize, size_t branchFactor, double spill, bool removeDups, size_t numTrees, size_t minLeaves, double exactEps, int searchType, long randSeed) :
        dim(dim), memorySize(memorySize), minLeaves(minLeaves), exactEps(exactEps),
        defaultSearchType(searchType), dat(new double[dim*memorySize]),
        lbl(new double[memorySize]), randEngine(randSeed), removeDups(removeDups) {
      if (numTrees <= 0) {
        throw std::invalid_argument("numTrees must not be zero.");
      }
      for (size_t i = 0; i < numTrees; i++) {
        trees.push_back(new Tree(dim, memorySize, maxLeafSize, branchFactor, spill, removeDups, this));
      }
      for (size_t i = 0; i < dim*memorySize; i++) {
        dat[i] = i;
      }
      for (size_t i = 0; i < memorySize; i++) {
        lbl[i] = i;
      }
      if (removeDups) {
        lookupExact = new ArrMap<double, size_t>(memorySize*1.2, ArrHasher<double>(dim), ArrEqualer<double>(dim));
      }
    }

    ~VQForest() {
      delete[] dat;
      delete[] lbl;
      if (removeDups) {
        delete lookupExact;
      }
    }

    double* getData(int ndx) { return dat+dim*ndx; }
    double getLabel(int ndx) { return lbl[ndx]; }

    size_t getLookupExact(double* data) {
      if (removeDups) {
        auto findResult = lookupExact->find(data);
        if (findResult != lookupExact->end()) {
          return findResult->second;
        }
      }
      return memorySize;
    }

    size_t add(double* data, double label, bool includeClears=false) {
      if (removeDups) {
        auto findResult = lookupExact->find(data);
        if (findResult != lookupExact->end()) {
          int oldNdx = findResult->second;
          label = std::max(label, getLabel(oldNdx));
          clearAndReplace(oldNdx);
        }
      }

      if (size() == memorySize) {
        clearAndReplace(tailNdx);
      }

      headNdx++;
      if (((size_t)headNdx) >= memorySize) {
        headNdx = 0;
      }

      size_t ndx = headNdx;
      std::memcpy(getData(ndx), data, dim*sizeof(*data));
      lbl[ndx] = label;
      if (removeDups) {
        assert(lookupExact->find(getData(ndx)) == lookupExact->end() && "Key was already inserted?");
        auto insertResult = lookupExact->insert(std::make_pair(getData(ndx), ndx));
        assert(insertResult.second && "Insert failed?");
        _unused(insertResult);
      }

      for (Tree* tree : trees) {
        tree->add(ndx);
      }

      return ndx;
    }

    size_t ndxWrapUp(ssize_t ndx) {
      return ndx >= ((ssize_t)tailNdx) ? ndx : ndx + memorySize;
    }

    bool isValidNdx(size_t ndx) {
      return size() != 0 && ndx >= 0 && ndx < memorySize && ndxWrapUp(ndx) <= ndxWrapUp(headNdx);
    }

    bool checkConsistency() {
      bool out = true;
      for (size_t i = 0; i < memorySize; i++) {
        if (isValidNdx(i) != isActive(i)) {
          printf("Inconsistent: %zu\n", i);
          out = false;
        }
      }
      return out;
    }

    size_t clearAndReplace(size_t ndx) {
      if (!isValidNdx(ndx)) {
        throw std::invalid_argument("Attempt to clear invalid ndx.");
      }

      for (Tree* tree : trees) {
        tree->clear(ndx);
      }
      if (removeDups) {
        size_t eraseResult = lookupExact->erase(getData(ndx));
        assert(eraseResult == 1 && "Attempt to clear index not in lookupExact?");
        _unused(eraseResult);
      }

      size_t oldNdx = tailNdx;
      if (oldNdx != ndx) {
        std::memcpy(getData(ndx), getData(oldNdx), dim*sizeof(*getData(oldNdx)));
        lbl[ndx] = lbl[oldNdx];
        if (removeDups) {
          size_t eraseResult = lookupExact->erase(getData(oldNdx));
          assert(eraseResult == 1 && "oldNdx wasn't in lookupExact?");
          _unused(eraseResult);
          auto insertResult = lookupExact->insert(std::make_pair(getData(ndx), ndx));
          assert(insertResult.second && "Insert failed?");
          _unused(insertResult);
        }

        for (Tree* tree : trees) {
          tree->relabel(oldNdx, ndx);
        }
      }

      if (size() == 1) { // Special case if buffer will be empty
        headNdx = -1;
        tailNdx = 0;
      } else {
        tailNdx++;
        if (tailNdx >= memorySize) {
          tailNdx = 0;
        }
      }

      return oldNdx;
    }

    std::vector<int>* nearestNeighborsNdxes(double* queryData, int n, int searchType=-1) {
      MinDistQ* q = nearestNeighbors(queryData, n, searchType);
      std::vector<int>* out = new std::vector<int>();
      for (auto pair : *q->container()) {
        out->push_back(pair.second);
      }
      delete q;
      return out;
    }

    MinDistQ* nearestNeighbors(double* queryData, int n, int searchType=-1) {
      if (searchType == -1) {
        searchType = defaultSearchType;
      }

      minN = n;
      MinDistQ* q = new MinDistQ(n);
      std::unordered_set<int> visited;
      auto callback = [this, queryData, &visited, q](Node* node) {
        for (int i : *node->contents()) {
          if (!visited.insert(i).second) {
            continue;
          }
          q->add(std::make_pair(dist(getData(i), queryData), i));
        }
      };

      for (Tree* tree : trees) {
        tree->search(queryData, callback, searchType, nullptr);
      }
      std::sort(q->container()->begin(), q->container()->end());
      return q;
    }

    bool isActive(size_t ndx) {
      for (Tree* tree : trees) {
        if (!tree->isActive(ndx)) {
          return false;
        }
      }
      return true;
    }

    size_t enforceTreeConsistencyFull() {
      size_t out = 0;
      for (Tree* tree : trees) {
        out += tree->enforceTreeConsistencyFull();
      }
      return out;
    }

    size_t enforceTreeConsistencyAt(size_t ndx) {
      size_t out = 0;
      for (Tree* tree : trees) {
        if (tree->enforceTreeConsistencyAt(ndx)) {
          out++;
        }
      }
      return out;
    }

    size_t enforceTreeConsistencyRandom() {
      size_t out = 0;
      for (Tree* tree : trees) {
        if (tree->enforceTreeConsistencyRandom()) {
          out++;
        }
      }
      return out;
    }

    size_t size() {
      return headNdx == -1 ? 0 : ndxWrapUp(headNdx) - tailNdx + 1;
    }

    int countNodes() {
      int out = 0;
      for (Tree* tree : trees) {
        out += tree->countNodes();
      }
      return out;
    }

    std::vector<int>* leafStats() {
      std::vector<int>* out = new std::vector<int>();
      out->push_back(-1);
      for (Tree* tree : trees) {
        tree->leafStats(out);
        out->push_back(-1);
      }
      return out;
    }

    void printTree() {
      const char* sep = "";
      for (Tree* tree : trees) {
        printf("%s", sep);
        sep = " ----------\n";
        tree->printTree();
      }
    }

    double dist(double* a, double* b) {
      double out = 0, tmp;
      for (size_t i = 0; i < dim; i++) {
        tmp = a[i]-b[i];
        out += tmp*tmp;
      }
      return out;
    }
};

template <class Node> class VQTree {
  public:
    using LookupSet = std::unordered_set<Node*>;

    size_t dim;
    size_t maxLeafSize;
    size_t branchFactor;

    Node* root;
    double spill;
    LookupSet* leafLookup;

    bool removeDups;
    VQForest<Node>* forest;

  public:
    VQTree(size_t dim, size_t memorySize, size_t maxLeafSize, size_t branchFactor, double spill, bool removeDups, VQForest<Node>* forest) :
        dim(dim), maxLeafSize(maxLeafSize), branchFactor(branchFactor),
        root(new Node(this)), spill(spill),
        leafLookup(new LookupSet[memorySize]), removeDups(removeDups), forest(forest) {
    }

    ~VQTree() {
      root->freeTree();
      delete[] leafLookup;
    }

    double* getData(int ndx) { return forest->getData(ndx); }
    double getLabel(int ndx) { return forest->getLabel(ndx); }

    void add(int ndx) {
      root->add(ndx);
    }

    void clear(size_t ndx) {
      for (Node* tmpNode : leafLookup[ndx]) {
        tmpNode->remove(ndx);
      }
      leafLookup[ndx].clear();
    }

    void relabel(size_t oldNdx, size_t newNdx) {
      assert(!isActive(newNdx));
      for (Node* tmpNode : leafLookup[oldNdx]) {
        auto* contents = tmpNode->contents();
        auto search = std::find(contents->begin(), contents->end(), oldNdx);
        assert(search != contents->end() && "Couldn't find item in leaf node.");
        *search = newNdx;
      }
      leafLookup[newNdx] = leafLookup[oldNdx];
      leafLookup[oldNdx].clear();
    }

    Node* closestChild(Node* n, double* data) {
      assert(!n->isLeaf());
      double minDist = DBL_MAX;
      Node* minNode = nullptr;
      for (size_t i = 0; i < n->numChildren(); i++) {
        Node* tmpNode = n->child(i);
        double tmpDist = dist(tmpNode->position(), data);
        if (tmpDist < minDist) {
          minDist = tmpDist;
          minNode = tmpNode;
        }
      }
      return minNode;
    }

    bool isActive(size_t ndx) {
      return leafLookup[ndx].size() > 0;
    }

    size_t enforceTreeConsistencyFull() {
      size_t cycles = 0, errs = 1;
      while (errs > 0) {
        errs = 0;
        for (size_t i = 0; i < forest->size(); i++) {
          if (enforceTreeConsistencyAt(i)) {
            errs++;
          }
        }
        fprintf(stderr, "errs:%zu\n",errs);
        cycles++;
      }
      return cycles;
    }
    bool enforceTreeConsistencyAt(size_t ndx) {
      if (!isActive(ndx)) {
        return false;
      }
      std::deque<int>* defeatist = nearestLeaf(root, getData(ndx))->contents();
      if (!contains(defeatist, ndx)) {
        clear(ndx);
        root->add(ndx);
        return true;
      }
      return false;
    }

    bool enforceTreeConsistencyRandom() {
      std::uniform_int_distribution<int> distribution(0, forest->size()-1);
      for (int i = 0; i < MAX_ITERS; i++) {
        int ndx = distribution(forest->randEngine);
        if (isActive(ndx)) {
          return enforceTreeConsistencyAt(ndx);
        }
      }
      return false;
    }

    Node* nearestLeaf(Node* tmp, double* query) {
      while (!tmp->isLeaf()) {
        tmp = closestChild(tmp, query);
      }
      return tmp;
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
          for (size_t i = 0; i < node->numChildren(); i++) {
            q.push_back(node->child(i));
          }
        }
      }
      return out;
    }

    double dist(double* a, double* b) { return forest->dist(a,b); }
    //  double out = 0, tmp;
    //  for (size_t i = 0; i < dim; i++) {
    //    tmp = a[i]-b[i];
    //    out += tmp*tmp;
    //  }
    //  return out;
    //}

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

    void registerLeaf(Node* n, int ndx) {
      leafLookup[ndx].insert(n);
    }

    void unregisterLeaf(Node* n, int ndx) {
      leafLookup[ndx].erase(n);
    }

    std::vector<int>* leafStats() {
      std::vector<int>* out = new std::vector<int>();
      leafStats(out);
      return out;
    }

    void leafStats(std::vector<int>* out) {
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
            for (size_t i = 0; i < node->numChildren(); i++) {
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

    template <typename Func> void search(double* query, const Func& callback, int searchType, va_list* args) {
      using SearchMethod = void (VQTree::*)(double*, const Func&, va_list*);
#define VQSEARCH_BRUTE 0
#define VQSEARCH_EXACT 1
#define VQSEARCH_DEFEATIST 2
#define VQSEARCH_PROT_DIST 3
#define VQSEARCH_PLANE_DIST 4
#define VQSEARCH_LEAFGRAPH 5
      static SearchMethod searches[] = {
        /*0*/ &VQTree::searchBrute,
        /*1*/ &VQTree::searchExact,
        /*2*/ &VQTree::searchDefeatist,
        /*3*/ &VQTree::searchMultiLeafProt,
        /*4*/ &VQTree::searchMultiLeafPlane,
        /*5*/ &VQTree::searchLeafGraph,
      };
      (this->*searches[searchType])(query, callback, args);
    }

  private:
    // args: ???
    template <typename Func> void searchBrute(double* query, const Func& callback, va_list* args) {
      std::stack<size_t> ndxes;
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

    // args: minN
    template <typename Func> void searchExact(double* query, const Func& callback, va_list* args) {
      assert(forest->minN > 0);
      MinNQueue<double> closest(forest->minN);
      std::unordered_set<int> visited;
      searchExact(root, query, &closest, &visited, callback);
    }
    template <typename Func> void searchExact(Node* node, double* query, MinNQueue<double>* closest, std::unordered_set<int>* visited, const Func& callback) {
      if (node->isLeaf()) {
        for (int i : *node->contents()) {
          if (visited->insert(i).second) {
            closest->add(dist(getData(i), query));
          }
        }
        callback(node);
        return;
      }

      size_t numChildren = node->numChildren();
      std::vector<std::pair<double, Node*>> dists;
      dists.reserve(numChildren);
      for (size_t i = 0; i < numChildren; i++) {
        Node* tmpChild = node->child(i);
        dists.emplace_back(dist(tmpChild->position(), query), tmpChild);
      }
      //std::cout << "dists: " << dists << std::endl;
      std::sort(dists.begin(), dists.end());

      std::pair<double, Node*> minPair = dists[0];
      double minDist = minPair.first;
      Node* minNode = minPair.second;
      double* minPos = minNode->position();

      searchExact(minNode, query, closest, visited, callback);
      for (size_t i = 1; i < numChildren; i++) {
        std::pair<double, Node*> tmp = dists[i];
        
        double tmpDistToPlane = distToPlane(minDist, tmp.first, minPos, tmp.second->position());
        if (!closest->isMature() || tmpDistToPlane*(1.0+forest->exactEps) < closest->top()) {
          searchExact(tmp.second, query, closest, visited, callback);
        }
      }
    }

    // args: none
    template <typename Func> void searchDefeatist(double* query, const Func& callback, va_list* args) {
      callback(nearestLeaf(root, query));
      //Node* leaf = nearestLeaf(root, query);
      //Node* tmp = leaf->parent;
      //if (tmp != NULL) {
      //  for (size_t i = 0; i < tmp->numChildren(); i++) {
      //    callback(tmp->child(i));
      //  }
      //} else {
      //  callback(leaf);
      //}
    }

    // args: minLeaves
    template <typename Func> void searchMultiLeafProt(double* query, const Func& callback, va_list* args) {
      using DistPair = std::pair<double, Node*>;
      MinNList<double, Node*> searchQ(forest->minLeaves);
      searchQ.add(0, root);

      for (size_t i = 0; i < forest->minLeaves && !searchQ.empty(); i++) {
        auto tmpPair = searchQ.begin();
        Node* tmpNode = tmpPair->second;
        searchQ.erase(tmpPair);
        searchQ.n--;

        while (!tmpNode->isLeaf()) {
          DistPair minPair(DBL_MAX, nullptr);
          for (size_t i = 0; i < tmpNode->numChildren(); i++) {
            Node* tmpChild = tmpNode->child(i);
            double tmpDist = dist(query, tmpChild->position());
            if (tmpDist < minPair.first) {
              if (minPair.second != nullptr) {
                searchQ.add(minPair.first, minPair.second);
              }
              minPair = std::make_pair(tmpDist, tmpChild);
            } else {
              searchQ.add(tmpDist, tmpChild);
            }
          }
          tmpNode = minPair.second;
        }

        callback(tmpNode);
      }
    }

    // args: minLeaves
    template <typename Func> void searchMultiLeafPlane(double* query, const Func& callback, va_list* args) {
      double* dists = new double[branchFactor];
      MinNList<double, Node*> searchQ(forest->minLeaves);
      searchQ.add(0, root);

      for (size_t i = 0; i < forest->minLeaves && !searchQ.empty(); i++) {
        auto tmpPair = searchQ.begin();
        Node* tmpNode = tmpPair->second;
        searchQ.erase(tmpPair);
        searchQ.n--;

        while (!tmpNode->isLeaf()) {
          for (size_t i = 0; i < tmpNode->numChildren(); i++) {
            dists[i] = dist(query, tmpNode->child(i)->position());
          }
          auto minChildP = std::min_element(dists, dists+tmpNode->numChildren());
          size_t minNdx = minChildP - dists;
          Node* minChild = tmpNode->child(minNdx);
          double* minPos = minChild->position();

          for (size_t i = 0; i < tmpNode->numChildren(); i++) {
            if (i == minNdx) { continue; }
            Node* tmpChild = tmpNode->child(i);
            double* tmpPos = tmpChild->position();
            double tmpDTP = distToPlane(dists[minNdx], dists[i], minPos, tmpPos);
            searchQ.add(tmpDTP, tmpChild);
          }
          
          tmpNode = minChild;
        }

        callback(tmpNode);
      }
      delete[] dists;
    }

    // args: minLeaves
    template <typename Func> void searchLeafGraph(double* query, const Func& callback, va_list *args) {
      std::unordered_set<Node*> visitedNodes;
      MinNList<double, Node*> searchQ(forest->minLeaves);
      Node* defeatistNode = nearestLeaf(root, query);

      visitedNodes.insert(defeatistNode);
      searchQ.add(0, defeatistNode);

      for (size_t i = 0; i < forest->minLeaves && !searchQ.empty(); i++) {
        auto tmpPair = searchQ.begin();
        Node* tmpNode = tmpPair->second;
        searchQ.erase(tmpPair);
        searchQ.n--;

        for (int ndx : *tmpNode->contents()) {
          for (Node* tmpNeighbor : leafLookup[ndx]) {
            if (!visitedNodes.insert(tmpNeighbor).second) {
              continue;
            }
            
            double tmpDist = dist(query, tmpNeighbor->position());
            searchQ.add(tmpDist, tmpNeighbor);
          }
        }

        callback(tmpNode);
      }
    }

    void printNode(Node* node, size_t depth) {
      printIndent(depth);
      printf("%d ", node->id());
      printData(node->position(), node->avg()->label());
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
    void printIndent(size_t depth) {
      for (size_t i = 0; i < depth; i++) {
        printf("  ");
      }
    }
    void printContents(Node* node) {
      printf("- ");
      auto& contents = *node->contents();
      for (size_t i = 0; i < contents.size(); i++) {
        printData(getData(contents[i]), getLabel(contents[i]));
      }
      printf(" [");
      for (size_t i = 0; i < contents.size(); i++) {
        if (i != 0) { putchar(','); }
        printf("%d", contents[i]);
      }
      putchar(']');
    }
    void printData(double* data, double target) {
      putchar('[');
      for (size_t i = 0; i < dim; i++) {
        if (i != 0) { putchar(','); }
        printf("%0.4f", data[i]);
      }
      //printf("(%.2f)", target);
      putchar(']');
      //printf("[%.4f]", target);
    }
    void printChildren(Node* node) {
      putchar('[');
      for (size_t i = 0; i < node->numChildren(); i++) {
        if (i != 0) { putchar(','); }
        printf("%zu", node->child(i)->avg()->count);
      }
      putchar(']');
    }
};

class MeanTreeNode : public VQTreeNode<MeanTreeNode> {
  public:
    using VQTreeNode::VQTreeNode;

    Node* childA = nullptr;
    Node* childB = nullptr;

    inline size_t numChildren() { return childA == nullptr ? 0 : 2; }

    inline Node* child(int ndx) { return ndx == 0 ? childA : childB; }

    void add(int ndx) {
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
        assert(false);
        while (!tmp->isRoot()) {
          tmp = tmp->parent;
          tmp->avg()->remove(data, label);
        }
      }
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

    inline size_t numChildren() { return children == nullptr ? 0 : children->size(); }

    inline Node* child(int ndx) { return (*children)[ndx]; }

    void add(int ndx) {
      double* data = tree->getData(ndx);
      double label = tree->getLabel(ndx);

      if (tree->spill >= 0) {
        double* dists = new double[tree->branchFactor];
        std::queue<Node*> q;
        q.push(this);
        while (q.size() > 0) {
          Node* tmp = q.front();
          q.pop();
          if (tmp->isLeaf()) {
            tmp->addToLeaf(ndx, data, label);
            continue;
          }

          size_t nChildren = tmp->numChildren();
          for (size_t i = 0; i < nChildren; i++) {
            dists[i] = tree->dist(data, tmp->child(i)->position());
          }
          auto minChildP = std::min_element(dists, dists+nChildren);
          size_t minNdx = minChildP - dists;
          Node* minChild = tmp->child(minNdx);
          double* minPos = minChild->position();

          q.push(minChild);

          for (size_t i = 0; i < nChildren; i++) {
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
        tmp->addToLeaf(ndx, data, label);
      }
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
      if (contents()->size() == 0 && parent != nullptr) {
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
        fprintf(stderr, "Fail PosFinite: %f\n", position()[0]);
        return false;
      }
      if (!isLeaf()) {
        if (numChildren() > tree->branchFactor) {
          fprintf(stderr, "Fail BranchFactor: %zu vs %zu\n", numChildren(), tree->branchFactor);
          return false;
        }
        size_t childPosCount = 0;
        for (size_t i = 0; i < children->size(); i++) {
          Node* child = (*children)[i];
          childPosCount += child->avg()->count;
          if (child->parent != this || child->childNdx != i) {
            puts("Fail Parent/ChildNdx");
            return false;
          }
        }
        if (avg()->count != childPosCount) {
          fprintf(stderr, "Fail PosCount: %zu vs %zu\n", avg()->count, childPosCount);
          return false;
        }
      } else {
        if (contents()->size() > tree->maxLeafSize) {
          fprintf(stderr, "Fail MaxLeafSize: %zu vs %zu\n", contents()->size(), tree->maxLeafSize);
          return false;
        }
        if (avg()->count != contents()->size()) {
          fprintf(stderr, "Fail Leaf PosCount: %zu vs %zu\n", avg()->count, contents()->size());
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
      /*for (size_t i = 0; i < labels->size(); i++) {
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
        fprintf(stderr, "Invalid children size after move... %zu %zu\n", this->children->size(), sibling->children->size());
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

      std::vector<size_t> perm(data->size());
      std::iota(perm.begin(), perm.end(), 0);
      std::shuffle(perm.begin(), perm.end(), tree->forest->randEngine);

      std::vector<int>* labels = new std::vector<int>(data->size(), -1);

      bool flag = true, first = true;
      for (int count = 0; count < MAX_ITERS && flag; count++) {
        flag = false;
        for (size_t i : perm) {
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
      tree->registerLeaf(this, ndx);
#ifndef NDEBUG
      if (tree->removeDups) {
        auto matchPtr = std::find_if(contents()->begin(), contents()->end(), [data, this](int i) {
          return doesMatch(data, tree->getData(i));
        });
        assert(matchPtr == contents()->end() && "Duplicates should be handled elsewhere...");
      }
#endif
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
#ifndef NDEBUG
      for (Node* tmp = this; tmp != nullptr; tmp = tmp->parent) {
        if (!tmp->checkInvariantLocal()) {
          fprintf(stderr, "!!!!!! %d\ndistUp:%d distDown:%d id:%d\n", ndx, distUp(), distDown(), tmp->_id);
          tree->printTree();
        }
      }
#endif
    }
};

using MeanTree = VQTree<MeanTreeNode>;
using MeanForest = VQForest<MeanTreeNode>;
using KTree = VQTree<KTreeNode>;
using KForest = VQForest<KTreeNode>;

template <class Node> class VQForestBuilder {
  public:
    size_t dim_, memorySize_;

    VQForestBuilder(size_t dim, size_t memorySize) : dim_(dim), memorySize_(memorySize) {}
    
#define SETTER(type, var, def) type var##_ = def; VQForestBuilder<Node>& var(type var) { this->var##_ = var; return *this; }
    SETTER(size_t, maxLeafSize, 64)
    SETTER(size_t, branchFactor, 16)
    SETTER(double, spill, -1.)
    SETTER(bool, removeDups, true)
    SETTER(size_t, numTrees, 1)
    SETTER(size_t, minLeaves, 100)
    SETTER(double, exactEps, 0.1)
    SETTER(int, searchType, VQSEARCH_PLANE_DIST)
    SETTER(long, randSeed, 9)
#undef SETTER
  
    VQForest<Node>* build() {
      return new VQForest<Node>(dim_, memorySize_, maxLeafSize_, branchFactor_, spill_, removeDups_, numTrees_, minLeaves_, exactEps_, searchType_, randSeed_);
    }
};

using KForestBuilder = VQForestBuilder<KTreeNode>;
using MeanForestBuilder = VQForestBuilder<MeanTreeNode>;
