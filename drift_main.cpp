#include "vqtree.cpp"
#define PRINT() if (tree.maxSize <= 100) { tree.printTree(); }

void randVec(double *data, int dim) {
  for (int i = 0; i < dim; i++) {
    data[i] = (2.0*(rand()/(double)RAND_MAX-0.5));
  }
}

// https://nl.mathworks.com/help/matlab/ref/peaks.html
double f(double *d, bool partA) {
  if (partA) {
    //return -10;
    return 0.1*d[0] + 0.2*d[1] + 0.3*d[2] + 0.4*d[3];
  } else {
    //return 10;
    return 0.4*d[0] + 0.3*d[1] + 0.2*d[2] + 0.1*d[3] - 10;
  }
}

double f_noisy(double *d, bool partA) {
  //return f(d, partA);
  return f(d, partA) + 0.005*(rand()/(double)RAND_MAX-0.5);
}

void test(KTree& tree) {
  int dim = tree.dim, size = tree.maxSize;
  double* d = new double[dim];
  srand(10);

  long removed = 0;
  for (int i = 0; i < size; i++) {
    randVec(d, dim);
    double target = f_noisy(d, true);
    tree.add(d, target);
    if (tree.driftCount > 0) {
      removed += tree.driftCount;
      printf("Block1 Drift %d: %ld (%ld)\n", i, tree.driftCount, tree.size()-removed);
    }
  }

  puts("Block1 complete");
  PRINT();

  double MSE = 0, MAE = 0;
  for (int i = 0; i < size; i++) {
    randVec(d, dim);
    double target = f_noisy(d, false);
    double query = tree.query(d);
    double diff = query-target;
    printf("query:%.4f diff:%.4f\n", query, diff);
    MSE += diff*diff;
    MAE += std::abs(diff);

    tree.add(d, target);
    if (tree.driftCount > 0) {
      removed += tree.driftCount;
      printf("Block2 Drift %d: %ld (%ld)\n", i, tree.driftCount, tree.size()-removed);
      //PRINT();
    }
  }

  puts("Block2 complete");
  PRINT();

  delete[] d;
  printf("MAE: %f\n", MAE/tree.size());
  printf("MSE: %f\n", MSE/tree.size());
}

void test(size_t dim, size_t maxSize, size_t maxLeafSize=64, size_t branchFactor=16, size_t minLeaves=100, size_t minN=100, int searchType=6, double spill=-1., bool removeDups=true, size_t driftHistLen=100, size_t driftThreshold=100) {
  KTree tree(dim, maxSize, maxLeafSize, branchFactor, minLeaves, minN, searchType, spill, removeDups, driftHistLen, driftThreshold);
  //VQTree(size_t dim, size_t maxSize, size_t maxLeafSize=64, size_t branchFactor=16, size_t minLeaves=100, size_t minN=100, int searchType=6, double spill=-1., bool removeDups=true, size_t driftHistLen=5, size_t driftThreshold=4) :
  test(tree);
}

int main(int argc, char *argv[]) {
  if (argc < 2) {
    fprintf(stderr, "usage: %s size [dims=4] [leafSize=64] [branchFactor=16]\n", argv[0]);
    return -1;
  }
  int size = std::stoi(argv[1]);
  int dims = 4, leafSize = 64, branchFactor = 16;
  if (argc >= 3) {
    dims = std::stoi(argv[2]);
  }
  if (argc >= 4) {
    leafSize = std::stoi(argv[3]);
  }
  if (argc >= 5) {
    branchFactor = std::stoi(argv[4]);
  }
  printf("size:%d dims:%d leafSize:%d branchFactor:%d\n", size, dims, leafSize, branchFactor);
  test(dims, size, leafSize, branchFactor);
}
