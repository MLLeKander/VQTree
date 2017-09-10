import _ktree #, _meantree

SEARCH_BRUTE = 0
SEARCH_EXACT = 1
SEARCH_DEFEATIST = 2
SEARCH_PROT_DIST = 3
SEARCH_PLANE_DIST = 4
SEARCH_LEAFGRAPH = 5

class _ForestBase(object):
    def __init__(self, module, dim, memory_size, max_leaf_size=64, branch_factor=16, spill=-1., remove_dups=True, num_trees=1, min_leaves=100, exact_eps=0.1, search_type=SEARCH_PLANE_DIST, rand_seed=9):
        self.module = module
        self._forest = module.init(dim, memory_size, max_leaf_size, branch_factor, spill, remove_dups, num_trees, min_leaves, exact_eps, search_type, rand_seed)

    def free(self):
        self.module.free(self._forest)

    def add(self, data, label):
        return self.module.add(self._forest, data, label)

    def clear(self, ndx):
        return self.module.clear(self._forest, ndx)

    def get_data(self, ndx):
        return self.module.get_data(self._forest, ndx)

    def get_label(self, ndx):
        return self.module.get_label(self._forest, ndx)

    def size(self):
        return self.module.size(self._forest)

    def count_nodes(self):
        return self.module.count_nodes(self._forest)

    def print_tree(self):
        self.module.print_tree(self._forest)

    def leaf_stats(self):
        return self.module.leaf_stats(self._forest)

    def is_active(self, ndx):
        return self.module.is_active(self._forest, ndx)



    def neighbors(self, data, n):
        return self.module.neighbors(self._forest, data, n)

    def enforce_tree_consistency_full(self):
        return self.module.enforce_tree_consistency_full(self._forest)
    def enforce_tree_consistency_at(self, ndx):
        return self.module.enforce_tree_consistency_at(self._forest, ndx)
    def enforce_tree_consistency_random(self):
        return self.module.enforce_tree_consistency_random(self._forest)



    def get_dim(self):
        return self.module.get_dim(self._forest)
    def get_memory_size(self):
        return self.module.get_memory_size(self._forest)
    def get_max_leaf_size(self):
        return self.module.get_max_leaf_size(self._forest)
    def get_branch_factor(self):
        return self.module.get_branch_factor(self._forest)
    def get_spill(self):
        return self.module.get_spill(self._forest)
    def get_remove_dups(self):
        return self.module.get_remove_dups(self._forest)
    def get_num_trees(self):
        return self.module.get_num_trees(self._forest)

    def get_min_leaves(self):
        return self.module.get_min_leaves(self._forest)
    def get_exact_eps(self):
        return self.module.get_exact_eps(self._forest)
    def get_search_type(self):
        return self.module.get_search_type(self._forest)

    def set_min_leaves(self, new_min_leaves):
        return self.module.set_min_leaves(self._forest, new_min_leaves)
    def set_exact_eps(self, new_exact_eps):
        return self.module.set_exact_eps(self._forest, new_exact_eps)
    def set_search_type(self, new_search_type):
        return self.module.set_search_type(self._forest, new_search_type)

class KForest(_ForestBase):
    #def __init__(self, dim, memory_size, max_leaf_size=64, branch_factor=16, spill=-1., remove_dups=True, num_trees=1, min_leaves=100, exact_eps=0.1, search_type=SEARCH_PLANE_DIST, rand_seed=1):
    def __init__(self, *args, **kwargs):
        super(KForest, self).__init__(_ktree, *args, **kwargs)

#class MeanForest(_ForestBase):
#  def __init__(self, *args, **kwargs):
#    kwargs['branch_factor'] = -1
#    super(MeanForest, self).__init__(_meantree, *args, **kwargs)
