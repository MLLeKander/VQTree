#include <algorithm>

//TODO: This requires 2x memory than it absolutely needs to...
class OnlineAverage {
  public:
    int dim;
    double* posSum;
    double* pos;
    double lblSum = 0;
    double lbl = 0;
    int count = 0;

    OnlineAverage(int dim_) : dim(dim_), posSum(new double[dim_]()), pos(new double[dim_]()) {}

    ~OnlineAverage() {
      delete[] pos;
      delete[] posSum;
    }

    void add(double* pos_, double lbl_) {
      count++;
      for (int i = 0; i < dim; i++) {
        posSum[i] += pos_[i];
      }
      lblSum += lbl_;
      updateAverage();
    }

    void add(OnlineAverage* o) {
      count += o->count;
      for (int i = 0; i < dim; i++) {
        posSum[i] += o->posSum[i];
      }
      lblSum += o->lblSum;
      updateAverage();
    }

    void remove(OnlineAverage* o) {
      count -= o->count;
      for (int i = 0; i < dim; i++) {
        posSum[i] -= o->posSum[i];
      }
      lblSum -= o->lblSum;
      updateAverage();
    }

    void remove(double* pos_, double lbl_) {
      count--;
      for (int i = 0; i < dim; i++) {
        posSum[i] -= pos_[i];
      }
      lblSum -= lbl_;
      updateAverage();
    }

    void reset() {
      count = lblSum = lbl = 0;
      std::fill_n(pos, dim, 0);
      std::fill_n(posSum, dim, 0);
    }

    double* position() {
      return pos;
    }

    double label() {
      return lbl;
    }

    void updateAverage() {
      for (int i = 0; i < dim; i++) {
        pos[i] = posSum[i]/count;
      }
      lbl = lblSum/count;
    }
};

