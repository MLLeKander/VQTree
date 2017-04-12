#include "module_base.cpp"

static PyMethodDef meantreeMethods[] = {
  {"init",        py_init<MeanTreeNode>,       METH_VARARGS, init__doc__      },
  {"free",        py_free<MeanTreeNode>,       METH_VARARGS, free__doc__      },
  {"add",         py_add<MeanTreeNode>,        METH_VARARGS, add__doc__       },
  {"query",       py_query<MeanTreeNode>,      METH_VARARGS, query__doc__     },
  {"size",        py_size<MeanTreeNode>,       METH_VARARGS, size__doc__      },
  {"count_nodes", py_countNodes<MeanTreeNode>, METH_VARARGS, countNodes__doc__},
  {"print_tree",  py_printTree<MeanTreeNode>,  METH_VARARGS, printTree__doc__ },
  {"leaf_stats",  py_leafStats<MeanTreeNode>,  METH_VARARGS, leafStats__doc__ },
  {NULL, NULL, 0, NULL}
};

PyDoc_STRVAR(meantree__doc__,"MeanTree data structure for online kernel regression");
PyMODINIT_FUNC init_meantree(void) {
  import_array();
  Py_InitModule3("_meantree", meantreeMethods, meantree__doc__);
}
