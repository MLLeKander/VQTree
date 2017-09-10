import vqtree

dims = 1000
mem_size = 10000

forest = vqtree.KForest(dim=dims, memory_size=mem_size, max_leaf_size=3, branch_factor=3, spill=-2., remove_dups=False, num_trees=2, min_leaves=10, exact_eps=0.2, search_type=vqtree.SEARCH_LEAFGRAPH, rand_seed=2)

print 'get_dim', forest.get_dim()
print 'get_memory_size', forest.get_memory_size()
print 'get_max_leaf_size', forest.get_max_leaf_size()
print 'get_branch_factor', forest.get_branch_factor()
print 'get_spill', forest.get_spill()
print 'get_remove_dups', forest.get_remove_dups()
print 'get_num_trees', forest.get_num_trees()
print 'get_min_leaves', forest.get_min_leaves()
print 'get_exact_eps', forest.get_exact_eps()
print 'get_search_type', forest.get_search_type()

print '...'
print 'set_min_leaves(2)', forest.set_min_leaves(2)
print 'get_min_leaves', forest.get_min_leaves()
print 'set_exact_eps(2.0)', forest.set_exact_eps(2.)
print 'get_exact_eps', forest.get_exact_eps()
print 'set_search_type(1)', forest.set_search_type(1)
print 'get_search_type', forest.get_search_type()

for i in range(mem_size):
#    print 'add', i, forest.add([i]*dims,1)
    forest.add([i]*dims,1)
#    print 'get_data(i)', forest.get_data(i)
#    print 'get_label(i)', forest.get_label(i)
#forest.print_tree()

print 'enforce_tree_consistency_at(0)', forest.enforce_tree_consistency_at(0)
print 'enforce_tree_consistency_random', forest.enforce_tree_consistency_random()
#print 'enforce_tree_consistency_full', forest.enforce_tree_consistency_full()
#forest.print_tree()

print 'clear(0)', forest.clear(0)
#forest.print_tree()

#for i in range(10):
#    print 'neighbor %d %s' %(i, forest.neighbors([i]*dims,3))

forest.set_search_type(vqtree.SEARCH_EXACT)
forest.set_exact_eps(0)

while True:
    print len(forest.neighbors([i]*dims, mem_size)[0])
