#pragma once

#include <algorithm>
#include <map>
#include <queue>


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
