#include <Python.h>
#include "vqtree.cpp"
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include "numpy/arrayobject.h"

//TODO: use PyCapsule_CheckExact?
//TODO: capsule destructor?
const char* VQForest_NAME = "vqtree.VQForest";

template <class Node> static PyArrayObject* sanitizeArray(VQForest<Node>* forest, PyObject* arrayObj) {
  PyArrayObject* array = (PyArrayObject*)PyArray_ContiguousFromAny(arrayObj, NPY_DOUBLE, 1, 2);
  //TODO: What is appropriate for max_depth?
  if (array == NULL) {
    return NULL;
  }
  if (PyArray_NDIM(array) != 1 ||
      PyArray_DTYPE(array)->type != 'd' ||
      (size_t)PyArray_DIM(array,0) != forest->dim) {
    Py_DECREF(array);
    PyErr_SetString(PyExc_ValueError, "array must be same length as dim and of type d");
    return NULL;
  }
  return array;
}

static bool capsuleConvert(PyObject* obj, void* out) {
  if (!PyCapsule_IsValid(obj, VQForest_NAME)) {
    PyErr_SetString(PyExc_ValueError, "invalid VQForest");
    return false;
  }
  *((void**)out) = PyCapsule_GetPointer(obj, VQForest_NAME);
  return true;
}

template <class E> static PyObject* arrToNpy1D(const E* arr, size_t dim1, int npyType) {
  npy_intp dims[1] = {(npy_intp)dim1};
  PyObject* out = PyArray_SimpleNew(1, dims, npyType);
  std::copy(arr, arr+dim1, (E*)PyArray_DATA((PyArrayObject*)out));
  return out;
}
template <class E> static PyObject* vecToNpy1D(const std::vector<E>& vec, int npyType) {
  npy_intp dims[1] = {(npy_intp)vec.size()};
  PyObject* out = PyArray_SimpleNew(1, dims, npyType);
  std::copy(vec.begin(), vec.end(), (E*)PyArray_DATA((PyArrayObject*)out));
  return out;
}
template <class E> static PyObject* vecToNpy2D(const std::vector<E*>& vec, size_t dim2, int npyType) {
  npy_intp dims[2] = {(npy_intp)vec.size(), (npy_intp)dim2};
  PyObject* out = PyArray_SimpleNew(2, dims, npyType);
  E* dest = (E*)PyArray_DATA((PyArrayObject*)out);
  for (E* p : vec) {
    std::copy(p, p+dim2, dest);
    dest += dim2;
  }
  return out;
}



PyDoc_STRVAR(init__doc__,"init(dim, memory_size, max_leaf_size, branch_factor, spill, remove_dups, num_trees, min_leaves, exact_eps, search_type, rand_seed) -> VQForest");
template <class Node> static PyObject* py_init(PyObject* self, PyObject* args) {
  size_t dim, memorySize, maxLeafSize, branchFactor, minLeaves, numTrees;
  int removeDups, searchType;
  double spill, exactEps;
  long randSeed;
  if (!PyArg_ParseTuple(args, "nnnndinndil", &dim, &memorySize, &maxLeafSize, &branchFactor, &spill, &removeDups, &numTrees, &minLeaves, &exactEps, &searchType, &randSeed)) {
    return NULL;
  }

  VQForest<Node>* forest = new VQForest<Node>(dim, memorySize, maxLeafSize, branchFactor, spill, removeDups != 0, numTrees, minLeaves, exactEps, searchType, randSeed);
  PyObject* out = PyCapsule_New(forest, VQForest_NAME, NULL);
  return out;
}

PyDoc_STRVAR(free__doc__,"free(VQForest)");
template <class Node> static PyObject* py_free(PyObject* self, PyObject* args) {
  VQForest<Node>* forest;
  if (!PyArg_ParseTuple(args, "O&", &capsuleConvert, &forest)) {
    return NULL;
  }
  
  delete forest;
  Py_RETURN_NONE;
}

PyDoc_STRVAR(add__doc__,"add(VQForest, data, label) -> ndx");
template <class Node> static PyObject* py_add(PyObject* self, PyObject* args) {
  PyObject* arrayObj;
  VQForest<Node>* forest;
  double label;
  if (!PyArg_ParseTuple(args, "O&Od", &capsuleConvert, &forest, &arrayObj, &label)) {
    return NULL;
  }

  PyArrayObject* array = sanitizeArray(forest, arrayObj);
  if (array == NULL) {
    return NULL;
  }

  size_t ndx;
  Py_BEGIN_ALLOW_THREADS
  ndx = forest->add((double*)PyArray_DATA(array),label);
  Py_END_ALLOW_THREADS

  Py_DECREF(array);
  return PyInt_FromSize_t(ndx);
}

PyDoc_STRVAR(clear__doc__,"clear(VQForest, ndx) -> old_ndx");
template <class Node> static PyObject* py_clear(PyObject* self, PyObject* args) {
  VQForest<Node>* forest;
  size_t ndx;
  if (!PyArg_ParseTuple(args, "O&n", &capsuleConvert, &forest, &ndx)) {
    return NULL;
  }

  size_t oldNdx;
  try {
    oldNdx = forest->clearAndReplace(ndx);
  } catch (const std::invalid_argument& e) {
    PyErr_SetString(PyExc_ValueError, "Attempt to clear invalid index.");
    return NULL;
  }

  return PyInt_FromSize_t(oldNdx);
}

PyDoc_STRVAR(enforceTreeConsistencyFull__doc__,"enforce_tree_consistency_full(VQForest) -> num_cycles");
template <class Node> static PyObject* py_enforceTreeConsistencyFull(PyObject* self, PyObject* args) {
  VQForest<Node>* forest;
  if (!PyArg_ParseTuple(args, "O&", &capsuleConvert, &forest)) {
    return NULL;
  }

  size_t cycles;
  Py_BEGIN_ALLOW_THREADS
  cycles = forest->enforceTreeConsistencyFull();
  Py_END_ALLOW_THREADS

  return PyInt_FromSize_t(cycles);
}

PyDoc_STRVAR(enforceTreeConsistencyAt__doc__,"enforce_tree_consistency_random(VQForest) -> num_corrections");
template <class Node> static PyObject* py_enforceTreeConsistencyAt(PyObject* self, PyObject* args) {
  VQForest<Node>* forest;
  size_t ndx;
  if (!PyArg_ParseTuple(args, "O&n", &capsuleConvert, &forest, &ndx)) {
    return NULL;
  }

  size_t corrections;
  Py_BEGIN_ALLOW_THREADS
  corrections = forest->enforceTreeConsistencyAt(ndx);
  Py_END_ALLOW_THREADS

  return PyInt_FromSize_t(corrections);
}

PyDoc_STRVAR(enforceTreeConsistencyRandom__doc__,"enforce_tree_consistency_random(VQForest) -> num_corrections");
template <class Node> static PyObject* py_enforceTreeConsistencyRandom(PyObject* self, PyObject* args) {
  VQForest<Node>* forest;
  if (!PyArg_ParseTuple(args, "O&", &capsuleConvert, &forest)) {
    return NULL;
  }

  size_t corrections;
  Py_BEGIN_ALLOW_THREADS
  corrections = forest->enforceTreeConsistencyRandom();
  Py_END_ALLOW_THREADS

  return PyInt_FromSize_t(corrections);
}

/*PyDoc_STRVAR(query__doc__,"query(VQForest, data) -> value");
template <class Node> static PyObject* py_query(PyObject* self, PyObject* args) {
  VQForest<Node>* forest;
  PyObject* arrayObj;
  if (!PyArg_ParseTuple(args, "O&O", &capsuleConvert, &forest, &arrayObj)) {
    return NULL;
  }

  puts("Inside query...");
  printf("%p\n", arrayObj);
  puts(PyString_AsString(PyObject_Repr(arrayObj)));
  puts("???");

  PyArrayObject* array = sanitizeArray(forest, arrayObj);
  if (array == NULL) {
    return NULL;
  }

  double result;
  Py_BEGIN_ALLOW_THREADS
  printf("a[0] = %f\n", ((double*)PyArray_DATA(array))[1]);
  result = forest->query((double*)PyArray_DATA(array));
  Py_END_ALLOW_THREADS

  Py_DECREF(array);
  return PyFloat_FromDouble(result);
}*/

PyDoc_STRVAR(neighbors__doc__,"neighbors(VQForest, data, n)");
template <class Node> static PyObject* py_neighbors(PyObject* self, PyObject* args) {
  VQForest<Node>* forest;
  PyObject* arrayObj;
  int n;
  if (!PyArg_ParseTuple(args, "O&Oi", &capsuleConvert, &forest, &arrayObj, &n)) {
    return NULL;
  }

  PyArrayObject* array = sanitizeArray(forest, arrayObj);
  if (array == NULL) {
    return NULL;
  }

  typename VQForest<Node>::MinDistQ* result;
  Py_BEGIN_ALLOW_THREADS
  result = forest->nearestNeighbors((double*)PyArray_DATA(array), n);
  Py_END_ALLOW_THREADS

  Py_DECREF(array);

  std::vector<int> ndxes;
  std::vector<double> dists, labels;
  std::vector<double*> datas;
  
  for (std::pair<double,int>& pair : *result->container()) {
    dists.push_back(pair.first);

    int ndx = pair.second;
    ndxes.push_back(ndx);
    labels.push_back(forest->getLabel(ndx));
    datas.push_back(forest->getData(ndx));
  }
  delete result;

  PyObject* npy_dists = vecToNpy1D(dists, NPY_DOUBLE);
  PyObject* npy_labels = vecToNpy1D(labels, NPY_DOUBLE);
  PyObject* npy_ndxes = vecToNpy1D(ndxes, NPY_INT);
  PyObject* npy_datas = vecToNpy2D(datas, forest->dim, NPY_DOUBLE);

  return Py_BuildValue("NNNN", npy_dists, npy_labels, npy_ndxes, npy_datas);
}

PyDoc_STRVAR(getData__doc__,"get_data(VQForest, ndx) -> data");
template <class Node> static PyObject* py_getData(PyObject* self, PyObject* args) {
  VQForest<Node>* forest;
  int ndx;
  if (!PyArg_ParseTuple(args, "O&i", &capsuleConvert, &forest, &ndx)) {
    return NULL;
  }

  return arrToNpy1D(forest->getData(ndx), forest->dim, NPY_DOUBLE);
}

PyDoc_STRVAR(getLabel__doc__,"get_label(VQForest, ndx) -> label");
template <class Node> static PyObject* py_getLabel(PyObject* self, PyObject* args) {
  VQForest<Node>* forest;
  int ndx;
  if (!PyArg_ParseTuple(args, "O&i", &capsuleConvert, &forest, &ndx)) {
    return NULL;
  }
  
  return PyFloat_FromDouble(forest->getLabel(ndx));
}

PyDoc_STRVAR(size__doc__,"size(VQForest) -> size");
template <class Node> static PyObject* py_size(PyObject* self, PyObject* args) {
  VQForest<Node>* forest;
  if (!PyArg_ParseTuple(args, "O&", &capsuleConvert, &forest)) {
    return NULL;
  }
  
  return PyInt_FromSize_t(forest->size());
}

PyDoc_STRVAR(countNodes__doc__,"count_nodes(VQForest) -> num_nodes");
template <class Node> static PyObject* py_countNodes(PyObject* self, PyObject* args) {
  VQForest<Node>* forest;
  if (!PyArg_ParseTuple(args, "O&", &capsuleConvert, &forest)) {
    return NULL;
  }

  int count;
  Py_BEGIN_ALLOW_THREADS
  count = forest->countNodes();
  Py_END_ALLOW_THREADS
  
  return PyInt_FromLong(count);
}

PyDoc_STRVAR(printTree__doc__,"print_tree(VQForest)");
template <class Node> static PyObject* py_printTree(PyObject* self, PyObject* args) {
  VQForest<Node>* forest;
  if (!PyArg_ParseTuple(args, "O&", &capsuleConvert, &forest)) {
    return NULL;
  }
  
  Py_BEGIN_ALLOW_THREADS
  forest->printTree();
  Py_END_ALLOW_THREADS

  Py_RETURN_NONE;
}

PyDoc_STRVAR(leafStats__doc__,"tree_stats(VQForest)");
template <class Node> static PyObject* py_leafStats(PyObject* self, PyObject* args) {
  VQForest<Node>* forest;
  if (!PyArg_ParseTuple(args, "O&", &capsuleConvert, &forest)) {
    return NULL;
  }
  
  std::vector<int>* stats;
  Py_BEGIN_ALLOW_THREADS
  stats = forest->leafStats();
  Py_END_ALLOW_THREADS

  //PyObject* out = PyArray_SimpleNew(1, dims, NPY_INT);
  //std::copy(stats->begin(), stats->end(), (int*)PyArray_DATA((PyArrayObject*)out));

  PyObject* out = vecToNpy1D(*stats, NPY_INT);

  delete stats;
  return out;
}

PyDoc_STRVAR(isActive__doc__,"is_active(VQForest, ndx) -> bool");
template <class Node> static PyObject* py_isActive(PyObject* self, PyObject* args) {
  VQForest<Node>* forest;
  int ndx;
  if (!PyArg_ParseTuple(args, "O&i", &capsuleConvert, &forest, &ndx)) {
    return NULL;
  }
  
  return PyBool_FromLong(forest->isActive(ndx));
}



#define GETTER(cppName, pyName, accessor)\
PyDoc_STRVAR(cppName##__doc__,#pyName "(CQForest)");\
template <class Node> static PyObject* py_##cppName(PyObject* self, PyObject* args) {\
  VQForest<Node>* forest;\
  if (!PyArg_ParseTuple(args, "O&", &capsuleConvert, &forest)) {\
    return NULL;\
  }\
  return accessor;\
}

GETTER(dim, dim, PyInt_FromSize_t(forest->dim))
GETTER(memorySize, memory_size, PyInt_FromSize_t(forest->memorySize))
GETTER(maxLeafSize, max_leaf_size, PyInt_FromSize_t(forest->trees[0]->maxLeafSize))
GETTER(branchFactor, branch_factor, PyInt_FromSize_t(forest->trees[0]->branchFactor))
GETTER(spill, spill, PyFloat_FromDouble(forest->trees[0]->spill))
GETTER(removeDups, remove_dups, PyBool_FromLong(forest->trees[0]->removeDups))
GETTER(numTrees, num_trees, PyInt_FromSize_t(forest->trees.size()))
GETTER(minLeaves, min_leaves, PyInt_FromSize_t(forest->minLeaves))
GETTER(exactEps, exact_eps, PyFloat_FromDouble(forest->exactEps))
GETTER(defaultSearchType, search_type, PyInt_FromLong(forest->defaultSearchType))
#undef GETTER

#define SETTER(cppName, pyName, type, format)\
PyDoc_STRVAR(set_##cppName##__doc__,"set_" #pyName "(CQForest, new_" #pyName ")");\
template <class Node> static PyObject* py_set_##cppName(PyObject* self, PyObject* args) {\
  VQForest<Node>* forest;\
  type cppName;\
  if (!PyArg_ParseTuple(args, "O&" format, &capsuleConvert, &forest, &cppName)) {\
    return NULL;\
  }\
  forest->cppName = cppName;\
  Py_RETURN_NONE;\
}

SETTER(minLeaves, min_leaves, size_t, "n")
SETTER(exactEps, exact_eps, double, "d")
SETTER(defaultSearchType, search_type, int, "i")
#undef SETTER
