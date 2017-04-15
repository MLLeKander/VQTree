#include "vqtree.cpp"
#include "prettyprint.hpp"
#include "timer.cpp"
#include <cstdio>
#include <string>
#include <vector>
#include <cassert>

#define NUM_TYPES 7
#define DATASET "sift"

template <class T> bool readBinaryVec(const std::string& fname, std::vector<std::vector<T>>* out) {
  FILE* f = fopen(fname.c_str(), "rb");
  if (f == nullptr) { return false; }

  int dim;
  fread(&dim, sizeof(dim), 1, f);
  auto vecsize = sizeof(dim) + sizeof(T)* dim;

  fseek(f, 0, SEEK_END);
  int numel = ftell(f)/vecsize;
  out->reserve(out->size()+numel);

  fseek(f, 0, SEEK_SET);
  int tmpDim;
  for (int i = 0; i < numel; i++) {
    fread(&tmpDim, sizeof(tmpDim), 1, f);
    if (tmpDim != dim) {
      fprintf(stderr, "Inconsistent dimensions? expected:%d got:%d\n", dim, tmpDim);
    }

    out->emplace_back(tmpDim);
    fread(out->back().data(), sizeof(T), tmpDim, f);
  }

  fclose(f);
  return true;
}

template <class T, class U> double dist(T a, U b, int dims=128) {
  double out = 0;
  for (int i = 0; i < dims; i++) {
    double tmp = a[i]-b[i];
    out += tmp*tmp;
  }
  return out;
}

std::vector<int>* bruteForce(const std::vector<std::vector<float>>& baseF, const std::vector<float>& queryF, int n) {
  MinNQueue<std::pair<double, int>> q(n);
  int dims = queryF.size();
  for (int i = 0; i < baseF.size(); i++) {
    q.add(std::make_pair(dist(baseF[i],queryF,dims),i));
  }
  std::vector<int>* out = new std::vector<int>();
  std::sort(q.container()->begin(), q.container()->end());
  for (auto pair : *q.container()) {
    out->push_back(pair.second);
  }
  return out;
}

template <class T> size_t intersectSize(T* a, T* b) {
  T v;
  std::sort(a->begin(), a->end());
  std::sort(b->begin(), b->end());
  std::set_intersection(a->begin(), a->end(), b->begin(), b->end(), std::back_inserter(v));
  return v.size();
}

void buildTree(KTree* tree, const std::vector<std::vector<float>>& baseF) {
  double t = progressBar(0, baseF.size(), [&baseF, tree](int i) {
    std::vector<double> row(baseF[i].begin(), baseF[i].end());
    tree->add(row.data(), 0);
  });
  printf("Build: %f %d %d\n", t, tree->size(), tree->countNodes());
}

void checkLookups(KTree* tree) {
  for (int i = 0; i < tree->size(); i++) {
    auto nodeSet = tree->nodeLookup[i];
    for (auto tmpNode : nodeSet) {
      auto contents = tmpNode->contents();
      auto tmpPtr = std::find(contents->begin(), contents->end(), i);
      if (tmpPtr == contents->end()) {
        printf("Uh oh... %d\n", i);
      }
    }
  }
}

int checkBins(KTree* tree) {
  int errs = 0;
  for (int i = 0; i < tree->size(); i++) {
    auto defeatist = tree->nearestLeaf(tree->root, tree->getData(i));
    auto actual = *tree->nodeLookup[i].begin();
    //int count = 0;
    //while (defeatist != actual) {
    //  defeatist = defeatist->parent;
    //  actual = actual->parent;
    //  count++;
    //}
    //if (count != 0) {
    if (defeatist != actual) {
      errs++;
    }
  }
  if (errs != 0) {
    printf("errs:%d\n", errs);
  }
  return errs;
}

int main() {
  std::vector<std::vector<float>> baseF, queryF;
  std::vector<std::vector<int>> groundtruth;
  StopWatch bruteTimer;
  StopWatch treeTimers[NUM_TYPES];
  bool res = true;
  res &= readBinaryVec(DATASET "/base.fvecs", &baseF);
  res &= readBinaryVec(DATASET "/query.fvecs", &queryF);
  res &= readBinaryVec(DATASET "/groundtruth.ivecs", &groundtruth);
  assert(res);
  int n = 1;
  size_t intersectCount[NUM_TYPES] = {0}; 
  //size_t bruteCount = 0;
  //int types[] = {0, 2, 3};
  /*
     0 // &VQTree::searchBrute,              
     1 // &VQTree::searchExact,              
     2 // &VQTree::searchDefeatist,            
     3 // &VQTree::searchMultiLeafProt,      
     4 // &VQTree::searchMultiLeafProtRecur, 
     5 // &VQTree::searchMultiLeafPlane,     
     6 // &VQTree::searchMultiLeafPlaneRecur,     
  */
  int types[] = {2, 5, 6};

  KTree tree(baseF[0].size(), baseF.size(), 64, 16, 100, 100, -1, false);
  buildTree(&tree, baseF);
  //buildTree(&tree, baseF);
  //checkLookups(&tree);
  //checkBins(&tree);


  //queryF.resize(100);
  progressBar(0, queryF.size(), [&](int i) {
    std::vector<double> query(queryF[i].begin(), queryF[i].end());
    std::vector<int> truth(groundtruth[i].begin(), groundtruth[i].begin()+n);

      //bruteTimer.start();
      //auto bruteResult = bruteForce(baseF, queryF[i], n);
      //bruteTimer.pause();
      //bruteCount += intersectSize(&truth, bruteResult);
      //delete bruteResult;

    for (int type : types) {
      treeTimers[type].start();
      auto result = tree.nearestNeighbors(query.data(), n, type);
      treeTimers[type].pause();
      intersectCount[type] += intersectSize(&truth, result);
      delete result;
    }
  });

  auto leafStats = tree.leafStats();
  std::cout << *leafStats << std::endl;
  delete leafStats;

  //printf("brute: %.1f%% accuracy, %f QPS\n", 100.*bruteCount/(double)(n*queryF.size()), queryF.size()/bruteTimer.elapsed());
  for (int type : types) {
    double qps = queryF.size()/treeTimers[type].elapsed();
    printf("search%d: %.1f%% accuracy, %f QPS\n", type, 100.*intersectCount[type]/(double)(n*queryF.size()), qps);
  }
}
