class RandTreeNode {
  public:
    typedef VQTree<RandTreeNode> Tree;

    int _id = VQ_NODE_COUNT++;
    std::vector<int>* _contents;
    OnlineAverage* _avg;
    std::vector<Node*>* _children = nullptr;
    double* childPositions = nullptr;
    Node* parent;
    Tree* tree;

  public:
    VQTreeNode(Tree* tree) : VQTreeNode(tree, nullptr) {}

    VQTreeNode(Tree* tree, Node* parent) : _contents(new std::vector<int>()),
        _avg(new OnlineAverage(tree->dim)), parent(parent), tree(tree) {}

    virtual ~VQTreeNode() {
      freeLeaf();
      freeInner();
    }

    inline void freeLeaf() {
      if (_contents != nullptr) {
        delete _contents;
        delete _avg;
        _avg = _contents = nullptr;
      }
    }

    inline void freeInner() {
      if (_children != nullptr) {
        delete _children;
        _children = nullptr;
      }
    }

    inline void freeTree() {
      for (int i = 0; i < numChildren(); i++) {
        child(i)->freeTree();
      }
      delete this;
    }

    inline int id() { return _id; }
    inline OnlineAverage* avg() { return _avg; }
    inline double* position() { return _avg->position(); }
    inline bool isRoot() { return parent == nullptr; }
    inline bool isLeaf() { return _contents != nullptr; }
    inline std::vector<int>* contents() { return _contents; }

    virtual int numChildren() { return _children == nullptr ? 0 : _children->size(); }
    virtual Node* child(int ndx) { return (*_children)[i]; }
    virtual double* childPosition(int ndx) { return childPositions+ndx*tree->dim; }

    virtual int add(int ndx) = 0;
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
        out = nearestLeaf(this, data)->addToLeaf(ndx, data, label);
      }

      return out;
    }
    virtual void remove(int ndx) = 0;
};
