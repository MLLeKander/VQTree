#include "vqtree.cpp"
#include "timer.cpp"

void randVec(double *data, int dim) {
  for (int i = 0; i < dim; i++) {
    data[i] = (2.0*(rand()/(double)RAND_MAX-0.5));
  }
}

// https://nl.mathworks.com/help/matlab/ref/peaks.html
double f(double *d) {
  double x = d[0], y = d[1];
  double x1 = x-1, x2 = x+1, y1 = y+1;
  return 3*x1*x1*exp(-x*x - y1*y1) 
      - 10*(x/5 - x*x*x - y*y*y*y*y)*exp(-x*x-y*y)
      - 1/3*exp(-x2*x2 - y*y);
}

double f_noisy(double *d) {
  return f(d) + 0.05*(rand()/(double)RAND_MAX-0.5);
}

template <class Node> void test(int size, int dim, int leafSize, int branchFactor, int minLeaves=50, int minN=50, double spill=-1, bool removeDups=true) {
  VQTree<Node> tree(dim, size, leafSize, branchFactor, minLeaves, minN, spill, removeDups);
  test(tree);
}
template <class Node> void test(VQTree<Node>& tree) {
  int dim = tree.dim, size = tree.maxSize;
  double* d = new double[dim];
  srand(10);

  double buildTime1 = progressBar(0, size, [&,d](int i) {
    randVec(d, dim);
    tree.add(d, f_noisy(d));
  });
  puts("build1 complete");
  //tree.printTree();
  fflush(stdout);

  double buildTime2 = progressBar(0, size, [&,d](int i) {
    randVec(d, dim);
    tree.add(d, f_noisy(d));
  });
  puts("build2 complete");
  //tree.printTree();
  fflush(stdout);

  double MSE = 0, MAE = 0;
  double queryTime = progressBar(0, size, [&,d](int i) {
    randVec(d, dim);
    double diff = tree.query(d, 2)-f(d);
    MSE += diff*diff;
    MAE += std::abs(diff);
  });
  puts("query complete");
  delete[] d;

  auto stats = tree.leafStats();
  for (auto i : *stats) {
    printf("%d ", i);
  }
  puts("");
  delete stats;

  tree.printTree();
  printf("treesize: %d\n", tree.size());
  printf("nodes: %d\n", tree.countNodes());
  printf("MAE: %f\n", MAE/tree.size());
  printf("MSE: %f\n", MSE/tree.size());
  printf("buildTime1:  %fs\n", buildTime1);
  printf("buildTime2: %fs\n", buildTime2);
  printf("queryTime:  %fs\n", queryTime);
}

int main(int argc, char *argv[]) {
  if (argc < 4) {
    fprintf(stderr, "usage: %s type size dims [leafSize=64] [branchFactor=16]\n", argv[0]);
    return -1;
  }
  char type = argv[1][0];
  int size = std::stoi(argv[2]), dim = std::stoi(argv[3]);
  int leafSize = 64, branchFactor = 16;
  if (argc >= 5) {
    leafSize = std::stoi(argv[3]);
  }
  if (argc >= 6) {
    branchFactor = std::stoi(argv[4]);
  }
  if (type == 'm') {
    test<MeanTreeNode>(size, dim, leafSize, branchFactor);
  } else if (type == 'k') {
    test<KTreeNode>(size, dim, leafSize, branchFactor);
  } else {
    fprintf(stderr, "Unknown type: %c\n", type);
  }
}
