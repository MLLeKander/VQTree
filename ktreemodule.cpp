#include "module_base.cpp"

#define METHDEF(cppName, pyName) {#pyName, py_##cppName<KTreeNode>, METH_VARARGS, cppName##__doc__ }
static PyMethodDef ktreeMethods[] = {
  METHDEF(init, init),
  METHDEF(free, free),
  METHDEF(add, add),
  METHDEF(clear, clear),
  METHDEF(neighbors, neighbors),
  METHDEF(getData, get_data),
  METHDEF(getLabel, get_label),
  METHDEF(size, size),
  METHDEF(countNodes, count_nodes),
  METHDEF(printTree, print_tree),
  METHDEF(leafStats, leaf_stats),
  METHDEF(isActive, is_active),
  METHDEF(enforceTreeConsistencyFull, enforce_tree_consistency_full),
  METHDEF(enforceTreeConsistencyAt, enforce_tree_consistency_at),
  METHDEF(enforceTreeConsistencyRandom, enforce_tree_consistency_random),
  METHDEF(dim, get_dim),
  METHDEF(memorySize, get_memory_size),
  METHDEF(maxLeafSize, get_max_leaf_size),
  METHDEF(branchFactor, get_branch_factor),
  METHDEF(spill, get_spill),
  METHDEF(removeDups, get_remove_dups),
  METHDEF(numTrees, get_num_trees),
  METHDEF(minLeaves, get_min_leaves),
  METHDEF(exactEps, get_exact_eps),
  METHDEF(defaultSearchType, get_search_type),
  METHDEF(set_minLeaves, set_min_leaves),
  METHDEF(set_exactEps, set_exact_eps),
  METHDEF(set_defaultSearchType, set_search_type),
  {NULL, NULL, 0, NULL}
};
#undef METHDEF

PyDoc_STRVAR(ktree__doc__,"KTree data structure for online kernel regression");
PyMODINIT_FUNC init_ktree(void) {
  import_array();
  Py_InitModule3("_ktree", ktreeMethods, ktree__doc__);
}
