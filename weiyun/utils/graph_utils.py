# -*- coding: UTF-8 -*-

from random import shuffle as sl
from random import randint as rd
import random
from weiyun.utils.log_utils import logger
import weiyun.mic_cloud.cloud as cloud
import weiyun.mic_cloud.local as local
import weiyun.mic_cloud.trans
# import weiyun.mic_cloud.trans as trans
# import weiyun.mic_cloud.trans as trans

EDGE_WEIGHT_MEAN = 0.6
EDGE_WEIGHT_VAR = 0.6

VEX_WEIGHT_MEAN = 0.6
VEX_WEIGHT_VAR = 0.6


# 返回DAG的邻接矩阵
# 行为节点的出度
# 列为节点的入度
# 为方便构造，头节点和结束节点为邻接矩阵的最后两维
def generate_DAG(n):
    node = list(range(1, n + 1))
    sl(node)
    sl(node)
    m = rd(1, (n - 1) * n / 2)
    adj_matrix = [([0] * (n+2)) for i in range(n+2)]
    # lines = ''
    ln = 0
    # for i in range(0, m):
    for i in range(0, min(10, m)):
        p1 = rd(1, n - 1)
        p2 = rd(p1 + 1, n)
        x = node[p1 - 1]
        y = node[p2 - 1]
        # print '{},{}'.format(x-1, y - 1)
        if adj_matrix[x-1][y-1] == 0:
            adj_matrix[x-1][y - 1] = gauss_random(EDGE_WEIGHT_MEAN, EDGE_WEIGHT_VAR)
            ln += 1
        # if lines.find('{0},{1}'.format(x, y)) == - 1:
        #     ln = ln + 1
        #     lines = lines + '{0},{1};'.format(x, y)
    for i in range(0, n):
        # 出度为0的节点，需要连接到结束节点
        if outdere(adj_matrix, i) == 0:
            adj_matrix[i][n+1] = gauss_random(EDGE_WEIGHT_MEAN, EDGE_WEIGHT_VAR)
    w_vex = []
    for i in range(0, n):
        # 头节点连向入度为0的节点
        if indere(adj_matrix, i) == 0:
            adj_matrix[n][i] = gauss_random(EDGE_WEIGHT_MEAN, EDGE_WEIGHT_VAR)
        # w_vex.append(gauss_random(VEX_WEIGHT_MEAN, VEX_WEIGHT_VAR))
    logger.debug('{0} nodes, {1} edges'.format(n, ln))
    # logger.debug('lines : ' + lines)
    for i in range(n+2):
        w_vex.append(gauss_random(VEX_WEIGHT_MEAN, VEX_WEIGHT_VAR))
    # topo_order = generate_topo_order(node)
    return adj_matrix, w_vex, generate_topo_order(node)

def generate_topo_order(node):
    r = []
    n = len(node)
    r.append(n)
    for i in range(len(node)):
        r.append(node[i] - 1)
    r.append(n+1)
    return r

def outdere(adj_matrix, n):
    return sum(adj_matrix[n][:])

def indere(adj_matrix, n):
    v = [x[n] for x in adj_matrix]
    return sum(v)

def indere_vex(adj_matrix, n):
    v = [x[n] for x in adj_matrix]
    r = []
    for i in range(len(v)):
        if v[i] != 0:
            r.append(i)
    return r

def outdere_vex(adj_matrix, n):
    v = adj_matrix[n][:]
    r = []
    for i in range(len(v)):
        if v[i] != 0:
            r.append(i)
    return r

# def DFS(matrix, i, line):
#     line = line + str(i)
#     logger.debug(line)
#     if outdere(matrix, i) == 0:
#         return line
#     v = matrix[i][:]
#     l = []
#     for j in range(len(v)):
#         logger.debug(v[j])
#         if v[j] == 1:
#             l.append(j)
#     n = random.choice(l)
#     return DFS(matrix, n, line)

def DFS(matrix, i, path, paths):
    path += str(i)
    if outdere(matrix, i) == 0:
        paths.append(path)
        return path
    v = matrix[i][:]
    for j in range(len(v)):
        if v[j] == 1:
            DFS(matrix, j, path, paths)
    return path

def all_path(adj_matrix):
    n = len(adj_matrix)
    paths = []
    DFS(adj_matrix, n - 2, '', paths)
    for p in paths:
        logger.debug(p)
    return paths

def randomly_choice_a_path(adj_matrix, i, path):
    # path = path + str(i)
    path.append(i)
    logger.debug(path)
    if outdere(adj_matrix, i) == 0:
        return path
    v = adj_matrix[i][:]
    l = []
    for j in range(len(v)):
        logger.debug(v[j])
        if v[j] != 0:
            l.append(j)
    n = random.choice(l)
    return randomly_choice_a_path(adj_matrix, n, path)

# 剪去一条路径
# 1. 剪去line上途径的路径
# 2. 剪去line上所有节点的所有入度
def prune(matrix, line):

    pro_vex = 5
    for i in range(1, len(line)):
        cur_vex = int(line[i])
        # 1. 剪去line上途径的路径
        matrix[pro_vex][cur_vex] = 0
        # 2. 剪去line上所有节点的所有入度
        for j in range(len(matrix)):
            matrix[j][cur_vex] = 0
    return matrix

def gauss_random(mu, sigma):
    # return random.randint(1, 20)
    rd = random.gauss(0, sigma)
    while abs(rd) > mu:
        rd = random.gauss(0, sigma)
    return mu + rd

def reshape_graph(adj_matrix, w_vex, schedule_queue, v_tag, cloud, trans):
    n = len(adj_matrix)
    for i in range(len(schedule_queue)):
        sp_cols = []
        start_index = 0
        if len(schedule_queue[i]) == 0:
            continue
        if schedule_queue[i][0] == -1:
            sp_cols.append(0)
        for j in range(1, len(schedule_queue[i])):
            if schedule_queue[i][j] == -1:
                sp_cols.append(j)
            elif adj_matrix[schedule_queue[i][start_index]][schedule_queue[i][j]] == 0:
                adj_matrix[schedule_queue[i][start_index]][schedule_queue[i][j]] = -1
                start_index = j
            else:
                start_index = j
        if len(sp_cols) == 0:
            continue
        # r = max_col_compute(w_vex, schedule_queue, 0, sp_cols)
        r = 0
        for in_i in range(1, len(sp_cols)):
            # 如果前后sp_cols是连续的，累加列最大的计算量
            if sp_cols[in_i] == sp_cols[in_i-1] + 1:
                # col = adj_matrix[:][sp_cols[i]]
                # r += max(col)
                r += max_col_compute(w_vex, schedule_queue, in_i-1, sp_cols)
            else:
                r += max_col_compute(w_vex, schedule_queue, in_i-1, sp_cols)
                w_vex[schedule_queue[i][sp_cols[in_i-1] + 1]] += r
                r = 0
        index = len(sp_cols) - 1
        if sp_cols[index] != len(schedule_queue[i]) - 1:
            r += max_col_compute(w_vex, schedule_queue, index, sp_cols)
            w_vex[schedule_queue[i][sp_cols[index] + 1]] += r

    new_graph = [([0] * (n * 2)) for i in range(n * 2)]
    # 把节点的权值拉到边上，原节点产生出相连节点
    for i in range(n):
        if v_tag[i] == 1:
            new_graph[i][i + n] = w_vex[i] / cloud.compute_ability
        if v_tag[i] == -1:
            new_graph[i][i + n] = w_vex[i] / local.COMPUTE_ABILITY
        out_vex = outdere_vex(adj_matrix, i)
        for v in out_vex:
            if v_tag[i] + v_tag[v] != 0:
                new_graph[i + n][v] = -1
            else:
                new_graph[i + n][v] = adj_matrix[i][v] / trans.band_width
    return new_graph

def max_col_compute(w_vex, schedule_queue, c, sp_cols):
    col = [x[sp_cols[c]] for x in schedule_queue]
    # col = schedule_queue[:][sp_cols[c]]
    mc = 0
    for vc in col:
        if w_vex[vc] > mc and vc != -1:
            mc = w_vex[vc]
    return mc

        # for sc in sp_cols:
        #     col = adj_matrix[:][sc]
        #     mc = max(col)

# int solve(int i)
# {
#     if (vis[i]) return dp[i];
#     vis[i] = true;
#     for (int j = 1; j <= n; ++j) {
#         if (G[j][i] != INF) {
#             if (dp[i] < G[j][i] + solve(j)) {
#                 dp[i] = G[j][i] + solve(j);
#                 last[i] = j;
#             }
#         }
#     }
#     return dp[i];
# }

# 生成图的祖先矩阵 anc_matrix
# anc_matrix[i][j] == 1，说明i为j的祖先
def generate_ancestor_matrix(adj_matrix):
    n = len(adj_matrix)
    anc_matrix = [([0] * (n)) for i in range(n)]
    for i in range(n-2):
        dst = {}
        dag_sp(adj_matrix, i, n-1, dst)
        for j in dst:
            if i != j:
                anc_matrix[i][j] = 1
    return anc_matrix


def dag_sp(W, s, t, d):
    if s == t:
        return 0
    if s not in d:
        d[s] = max(W[s][v] + dag_sp(W, v, t, d) for v in outdere_vex(W, s))
    return d[s]

def critical_path(adj_matrix, i, traced, dp, last):
    if traced[i]:
        return dp[i]
    traced[i] = True
    in_v = indere_vex(adj_matrix, i)
    for v in in_v:
        l = adj_matrix[v][i]
        if l == -1:
            l = 0
        len_vi = l + critical_path(adj_matrix, v, traced, dp, last)
        if dp[i] < len_vi:
            dp[i] = len_vi
            last[i] = v
    return dp[i]

