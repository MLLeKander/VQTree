import _ktree #, _meantree

class _TreeBase(object):
    def __init__(self, module, dim, max_size, max_leaf_size, branch_factor, min_count=100):
        self.module = module
        self._tree = module.init(dim, max_size, max_leaf_size, branch_factor, min_count)

    def free(self):
        self.module.free(self._tree)

    def add(self, data, label):
        self.module.add(self._tree, data, label)

    def query(self, data):
        return self.module.query(self._tree, data)

    def size(self):
        return self.module.size(self._tree)

    def count_nodes(self):
        return self.module.count_nodes(self._tree)

    def print_tree(self):
        self.module.print_tree(self._tree)

    def leaf_stats(self):
      return self.module.leaf_stats(self._tree)

class KTree(_TreeBase):
    def __init__(self, dim, max_size, max_leaf_size, branch_factor, min_count=100):
        super(KTree, self).__init__(_ktree, dim, max_size, max_leaf_size, branch_factor, min_count)

#class MeanTree(_TreeBase):
#  def __init__(self, *args, **kwargs):
#    kwargs['branch_factor'] = -1
#    super(MeanTree, self).__init__(_meantree, *args, **kwargs)
