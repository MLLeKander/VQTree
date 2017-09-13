#pragma once

#include <algorithm>
#include <map>
#include <queue>
#include <unordered_map>


template<class Container, class E> bool contains(Container* lst, E element) {
  return std::find(lst->begin(), lst->end(), element) != lst->end();
}

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

template <class K, class V> class MinNList : public std::multimap<K,V> {
  public:
    size_t n;
    MinNList(size_t n) : n(n) {}

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


template <class E> class ArrHasher {
  public:
    const size_t dim;
    ArrHasher(size_t dim) : dim(dim) {}

    const size_t operator()(E* d) const {
      size_t seed = 0;

      for (size_t i = 0; i < dim; i++) {
        seed = hash_combine(seed,std::hash<E>{}(d[i]));
      }

      return seed;
    }
  private:
    // http://www.boost.org/doc/libs/1_35_0/doc/html/boost/hash_combine_id241013.html
    size_t hash_combine(const size_t seed, const size_t val) const {
      return val + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    }
};

template <class E> class ArrEqualer {
  public:
    const size_t dim;
    ArrEqualer(size_t dim) : dim(dim) {}

    bool operator()(E* a, E* b) const {
      for (size_t i = 0; i < dim; i++) {
        if (a[i] != b[i]) {
          return false;
        }
      }
      return true;
    }
};

template <class K, class V> using ArrMap = std::unordered_map<K*, V, ArrHasher<K>, ArrEqualer<K>>;
// Intended use:
//   ArrMap<double, int> m(10, ArrHasher<double>(2), ArrEqualer<double>(2));
