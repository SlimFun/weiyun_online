from graph_utils import *
from matrix_utils import print_matrix
from weiyun.mic_cloud.cloud import *
from matrix_utils import copy_matrix

# adj_matrix, w_vex = generate_DAG(5)
# print_matrix(adj_matrix)
# print w_vex
#
# print randomly_choice_a_path(adj_matrix, len(adj_matrix) - 2, '')
# paths = all_path(adj_matrix)
# for p in paths:
#     print p
# line =  DFS_DAG(adj_matrix)
# print line

# pruned_matrix = prune(adj_matrix, line)
# print_matrix(pruned_matrix)

# adj_matrix =  [([0] * 5) for i in range(5)]
# print adj_matrix[:][1]
# adj_matrix[4][3] = 1
# print_matrix(adj_matrix)

# strA = "abcdegfgijlk"
# print strA[::-1]

# print normal_random(0.7, 1)

# adj_matrix, w_vex, topo_order = generate_DAG(5)
# print_matrix(adj_matrix)
# print topo_order

adj_matrix, w_vex, topo_order = generate_DAG(5)
print_matrix(adj_matrix)
print w_vex

cp_m = copy_matrix(adj_matrix)
print 'cp_m'
print_matrix(cp_m)
adj_matrix[1][2] = 100
print_matrix(adj_matrix)
print 'after change'
print_matrix(cp_m)
#
n = len(adj_matrix)
# traced = [0 for i in range(len(adj_matrix))]
# dp = [0 for i in range(len(adj_matrix))]
# last = [0 for i in range(len(adj_matrix))]
# print critical_path(adj_matrix, n - 1, traced, dp, last)
# print last
# new_graph = reshape_graph(adj_matrix, w_vex, schedule_queue([2, 3, 4, 0, 1], topo_order))
# print 'new graph : '
# print_matrix(new_graph)

# dst = {}
# print dag_sp(adj_matrix, 0, n-1, dst)
# print dst
# anc_matrix = generate_ancestor_matrix(adj_matrix)
# print 'anc_matrix : '
# print_matrix(anc_matrix)
#
#
# a = [[7, 2, 5, 1, 3], [4, 0, 6, -1, -1]]
# n = 3
# v = [x[n] for x in a]
# print v
#
#
# for i in range(len(a)):
#     print i