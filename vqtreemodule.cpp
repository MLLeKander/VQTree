#include <Python.h>
#include "vqtree.cpp"
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include "numpy/arrayobject.h"

//TODO: use PyCapsule_CheckExact?
//TODO: capsule destructor?
const char* VQTree_NAME = "vqtree.VQTree";

template <class Node> static PyArrayObject* sanitizeArray(VQTree<Node>* tree, PyObject* arrayObj) {
  PyArrayObject* array = (PyArrayObject*)PyArray_ContiguousFromAny(arrayObj, NPY_DOUBLE, 1, 2);
  //TODO: What is appropriate for max_depth?
  if (PyArray_NDIM(array) != 1 ||
      PyArray_DTYPE(array)->type != 'd' ||
      (size_t)PyArray_DIM(array,0) != tree->dim) {
    Py_DECREF(array);
    PyErr_SetString(PyExc_ValueError, "array must be same length as dim and of type d");
    return NULL;
  }
  return array;
}

static bool capsuleConvert(PyObject* obj, void* out) {
  if (!PyCapsule_IsValid(obj, VQTree_NAME)) {
    PyErr_SetString(PyExc_ValueError, "invalid VQTree");
    return false;
  }
  *((void**)out) = PyCapsule_GetPointer(obj, VQTree_NAME);
  return true;
}



PyDoc_STRVAR(init__doc__,"init(dims, maxSize, maxLeafSize) -> TestObj");
template <class Node> static PyObject* py_init(PyObject* self, PyObject* args) {
  size_t dim, maxSize, maxLeafSize, branchFactor;
  //TODO: Is this the right format specifier?
  if (!PyArg_ParseTuple(args, "nnnn", &dim, &maxSize, &maxLeafSize, &branchFactor)) {
    return NULL;
  }
  VQTree<Node>* tree = new VQTree<Node>(dim, maxSize, maxLeafSize, branchFactor);
  PyObject* out = PyCapsule_New(tree, VQTree_NAME, NULL);
  return out;
}

PyDoc_STRVAR(free__doc__,"free(VQTree)");
template <class Node> static PyObject* py_free(PyObject* self, PyObject* args) {
  VQTree<Node>* tree;
  if (!PyArg_ParseTuple(args, "O&", &capsuleConvert, &tree)) {
    return NULL;
  }
  
  delete tree;
  Py_RETURN_NONE;
}

PyDoc_STRVAR(add__doc__,"add(VQTree, data, label)");
template <class Node> static PyObject* py_add(PyObject* self, PyObject* args) {
  PyObject* arrayObj;
  VQTree<Node>* tree;
  double label;
  if (!PyArg_ParseTuple(args, "O&Od", &capsuleConvert, &tree, &arrayObj, &label)) {
    return NULL;
  }

  PyArrayObject* array = sanitizeArray(tree, arrayObj);
  if (array == NULL) {
    return NULL;
  }

  tree->add((double*)PyArray_DATA(array),label);
  Py_DECREF(array);
  Py_RETURN_NONE;
}

PyDoc_STRVAR(query__doc__,"query(VQTree, data) -> value");
template <class Node> static PyObject* py_query(PyObject* self, PyObject* args) {
  PyObject* arrayObj;
  VQTree<Node>* tree;
  if (!PyArg_ParseTuple(args, "O&O", &capsuleConvert, &tree, &arrayObj)) {
    return NULL;
  }

  PyArrayObject* array = sanitizeArray(tree, arrayObj);
  if (array == NULL) {
    return NULL;
  }

  double result = tree->query((double*)PyArray_DATA(array));
  Py_DECREF(array);
  return PyFloat_FromDouble(result);
}

PyDoc_STRVAR(size__doc__,"size(VQTree) -> size");
template <class Node> static PyObject* py_size(PyObject* self, PyObject* args) {
  VQTree<Node>* tree;
  if (!PyArg_ParseTuple(args, "O&", &capsuleConvert, &tree)) {
    return NULL;
  }
  
  return PyInt_FromLong(tree->size());
}

PyDoc_STRVAR(countNodes__doc__,"count_nodes(VQTree) -> num_nodes");
template <class Node> static PyObject* py_countNodes(PyObject* self, PyObject* args) {
  VQTree<Node>* tree;
  if (!PyArg_ParseTuple(args, "O&", &capsuleConvert, &tree)) {
    return NULL;
  }
  
  return PyInt_FromLong(tree->countNodes());
}

PyDoc_STRVAR(printTree__doc__,"print_tree(VQTree)");
template <class Node> static PyObject* py_printTree(PyObject* self, PyObject* args) {
  VQTree<Node>* tree;
  if (!PyArg_ParseTuple(args, "O&", &capsuleConvert, &tree)) {
    return NULL;
  }
  
  tree->printTree();
  Py_RETURN_NONE;
}

PyDoc_STRVAR(leafStats__doc__,"tree_stats(VQTree)");
template <class Node> static PyObject* py_leafStats(PyObject* self, PyObject* args) {
  VQTree<Node>* tree;
  if (!PyArg_ParseTuple(args, "O&", &capsuleConvert, &tree)) {
    return NULL;
  }
  
  std::vector<int>* stats = tree->leafStats();
  npy_intp dims[1];
  dims[0] = stats->size();

  PyObject* out =  PyArray_SimpleNew(1, dims, NPY_INT);
  std::copy(stats->begin(), stats->end(), (int*)PyArray_DATA((PyArrayObject*)out));

  delete stats;
  return out;
}

static PyMethodDef ktreeMethods[] = {
  {"init",        py_init<KTreeNode>,       METH_VARARGS, init__doc__      },
  {"free",        py_free<KTreeNode>,       METH_VARARGS, free__doc__      },
  {"add",         py_add<KTreeNode>,        METH_VARARGS, add__doc__       },
  {"query",       py_query<KTreeNode>,      METH_VARARGS, query__doc__     },
  {"size",        py_size<KTreeNode>,       METH_VARARGS, size__doc__      },
  {"count_nodes", py_countNodes<KTreeNode>, METH_VARARGS, countNodes__doc__},
  {"print_tree",  py_printTree<KTreeNode>,  METH_VARARGS, printTree__doc__ },
  {"leaf_stats",  py_leafStats<KTreeNode>,  METH_VARARGS, leafStats__doc__ },
  {NULL, NULL, 0, NULL}
};

PyDoc_STRVAR(ktree__doc__,"KTree data structure for online kernel regression");
PyMODINIT_FUNC init_ktree(void) {
  import_array();
  Py_InitModule3("_ktree", vqtreeMethods, vqtree__doc__);
}
