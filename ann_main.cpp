#include "vqtree.cpp"
#include "prettyprint.hpp"
#include "timer.cpp"
#include <cstdio>
#include <string>
#include <vector>
#include <cassert>

#define NUM_SEARCH_TYPES 8
#define DATASET "sift"
#define CONSISTINCY_ITERS 6
#define N 10

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

double buildTree(KForest* forest, const std::vector<std::vector<float>>& baseF) {
  double t = progressBar(0, baseF.size(), [&baseF, forest](int i) {
    std::vector<double> row(baseF[i].begin(), baseF[i].end());
    forest->add(row.data(), 0);
    for (int i = 0; i < std::min(forest->size(), (size_t)CONSISTINCY_ITERS); i++) {
      forest->enforceTreeConsistencyRandom();
    }
  });
  printf("Build: %f %zu %d\n", t, forest->size(), forest->countNodes());
  return t;
}

double enforceBins(KForest* forest) {
  StopWatch timer;
  timer.start();
  int cycles = forest->enforceTreeConsistencyFull();
  timer.pause();

  printf("Enforce Bins: %d %f\n", cycles, timer.elapsed());
  return timer.elapsed();
}

void checkLookups(KForest* forest) {
  for (int i = 0; i < forest->size(); i++) {
    for (auto& subTree : forest->trees) {
      auto nodeSet = subTree->leafLookup[i];
      for (auto tmpNode : nodeSet) {
        auto contents = tmpNode->contents();
        auto tmpPtr = std::find(contents->begin(), contents->end(), i);
        if (tmpPtr == contents->end()) {
          printf("Uh oh... %d\n", i);
        }
      }
    }
  }
}

int checkBins(KForest* forest) {
  int errs = 0;
  for (int i = 0; i < forest->size(); i++) {
    for (auto& subTree : forest->trees) {
      auto defeatist = subTree->nearestLeaf(subTree->root, forest->getData(i));
      auto actual = *subTree->leafLookup[i].begin();
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
  }
  if (errs != 0) {
    printf("errs:%d\n", errs);
  }
  return errs;
}

void testBrute(std::vector<std::vector<float>>& baseF, std::vector<std::vector<float>>& queryF) {
  StopWatch bruteTimer;
  bruteTimer.start();
  bruteTimer.pause();
}

void testQueries(size_t qParam, KForest* forest, double buildTime, size_t n, std::vector<std::vector<float>>& queryF, std::vector<std::vector<int>>& groundtruth) {
  size_t intersectCount[NUM_SEARCH_TYPES] = {0}; 
  //int types[] = {VQSEARCH_EXACT, VQSEARCH_DEFEATIST, VQSEARCH_PROT_DIST, VQSEARCH_PLANE_DIST, VQSEARCH_LEAFGRAPH};
  //int types[] = {VQSEARCH_DEFEATIST, VQSEARCH_PROT_DIST, VQSEARCH_PLANE_DIST, VQSEARCH_LEAFGRAPH};
  int types[] = {VQSEARCH_DEFEATIST};
  StopWatch treeTimers[NUM_SEARCH_TYPES];
  forest->minLeaves = qParam;
  forest->exactEps = 0.1*qParam;

  //queryF.resize(100);
  progressBar(0, queryF.size(), [&](int i) {
    std::vector<double> query(queryF[i].begin(), queryF[i].end());
    std::vector<int> truth(groundtruth[i].begin(), groundtruth[i].begin()+n);
    //std::cout << "A  " << truth << std::endl;

      //bruteTimer.start();
      //auto bruteResult = bruteForce(baseF, queryF[i], n);
      //bruteTimer.pause();
      //bruteCount += intersectSize(&truth, bruteResult);
      ////std::cout << "B  " << *bruteResult << " " << intersectSize(&truth, bruteResult) << std::endl;
      //delete bruteResult;

    for (int type : types) {
      treeTimers[type].start();
      auto result = forest->nearestNeighborsNdxes(query.data(), n, type);
      treeTimers[type].pause();
      intersectCount[type] += intersectSize(&truth, result);
      //std::cout << "C" << type << " " << *result << " " << intersectSize(&truth, result) << std::endl;
      delete result;
    }
  });

  //printf("brute: %.4f accuracy, %g QPS\n", bruteCount/(double)(n*queryF.size()), 1./(8.*queryF.size()/bruteTimer.elapsed()));
  for (int type : types) {
    double qps = queryF.size()/treeTimers[type].elapsed();
    double spq = 1./(8.*qps);
    double accuracy = intersectCount[type]/(double)(n*queryF.size());
    //printf("search%d: %.1f%% accuracy, %f QPS\n", type, 100.*intersectCount[type]/(double)(n*queryF.size()), qps);
    printf("ktree(%d)\t", type);
    KTree* tree = forest->trees[0];
    printf("KForest(searchType=%d,qParam=%zu,spill=%f,maxLeafSize=%zu,branchFactor=%zu,numTrees=%zu)\t", 
                  type, qParam, tree->spill, tree->maxLeafSize, tree->branchFactor, forest->trees.size());
    printf("%g\t%g\t%g\n", buildTime, spq, accuracy);
    //printf("search%d: %.4f accuracy, %g SPQ\n", type, intersectCount[type]/(double)(n*queryF.size()), 1./(8.*qps));
  }
}


int main() {
  std::vector<std::vector<float>> baseF, queryF;
  std::vector<std::vector<int>> groundtruth;
  bool res = true;
  res &= readBinaryVec(DATASET "/base.fvecs", &baseF);
  res &= readBinaryVec(DATASET "/query.fvecs", &queryF);
  res &= readBinaryVec(DATASET "/groundtruth.ivecs", &groundtruth);
  assert(res);

  //baseF.resize(100000);
  KForest* forest = KForestBuilder(baseF[0].size(), baseF.size())
      .maxLeafSize(64)
      .branchFactor(16)
      .spill(0.1)
      .removeDups(false)
      .numTrees(1)
      .randSeed(9)
      .build();

  double buildTime = buildTree(forest, baseF);
  buildTime += enforceBins(forest);
  checkLookups(forest);

  auto leafStats = forest->leafStats();
  std::cout << *leafStats << std::endl;
  delete leafStats;

  for (size_t qParam : {2,4,8,16,32,64,128,256,512,1024,2048}) {
  //for (size_t qParam : {2048,1024,512,256,128,64,32,16,8,4,2,1}) {
    testQueries(qParam, forest, buildTime, N, queryF, groundtruth);
  }
  delete forest;
}
