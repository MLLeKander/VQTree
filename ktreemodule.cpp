#include "module_base.cpp"

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
  Py_InitModule3("_ktree", ktreeMethods, ktree__doc__);
}
