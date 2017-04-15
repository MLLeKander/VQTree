// http://stackoverflow.com/a/19471595
#include <ctime>
#include <cassert>
class Timer {
  public:
    Timer() { clock_gettime(CLOCK_REALTIME, &beg_); }

    double elapsed() {
      clock_gettime(CLOCK_REALTIME, &end_);
      return end_.tv_sec - beg_.tv_sec +
        (end_.tv_nsec - beg_.tv_nsec) / 1000000000.;
    }

    void reset() { clock_gettime(CLOCK_REALTIME, &beg_); }

  private:
    timespec beg_, end_;
};

// http://stackoverflow.com/a/6993279
class StopWatch {
  public:
    StopWatch() { reset(); }

    double elapsed() {
      clock_t out = _stored;
      if (isRunning) {
        out += clock() - _start;
      }
      return out/(double)CLOCKS_PER_SEC;
    }

    void start() {
      _start = clock();
      isRunning = true;
    }

    void pause() {
      if (isRunning) {
        _stored += clock() - _start;
        _start = 0;
        isRunning = false;
      }
    }

    template <class Func, class Out> Out timeF(const Func& f) {
      assert(!isRunning);
      start();
      Out out = f();
      pause();
      return out;
    }

    template <class Func> void timeF(const Func& f) {
      assert(!isRunning);
      start();
      f();
      pause();
    }

    void reset() { _start = _stored = 0; }

  private:
    bool isRunning = false;
    clock_t _start = 0, _stored = 0;
};

template <class Func> double progressBar(size_t min, size_t max, const Func& callback, double refreshRate=2.0) {
  clock_t prevClock = clock(), startClock = prevClock;
  for (size_t i = min; i < max; i++) {
    callback(i);
    clock_t currClock = clock();
    if ((currClock-prevClock)*refreshRate >= CLOCKS_PER_SEC) {
      printf("%zu\r", i);
      fflush(stdout);
      prevClock = currClock;
    }
  }
  return (clock()-startClock)/(double)CLOCKS_PER_SEC;
}
