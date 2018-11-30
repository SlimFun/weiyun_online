# -*- coding: UTF-8 -*-

from weiyun.mic_cloud.local import T as local_T
from weiyun.utils.log_utils import logger
from weiyun.utils.graph_utils import *

class Trans:
    def __init__(self, cloud, band_width=3.0):
        self.band_width = band_width
        self.cloud = cloud

    def select_trans_seq(self, adj_matrix, path, w_vex, v_tag):

        entry_task = 1
        exit_task = 1
        n = len(path) - 1
        node_num = len(adj_matrix)
        start = 1
        # path 的首尾节点只有两种情况：
        # 1. path 是图的完整路径，故首尾都为 untagged
        # 2. path 的首尾节点都是 tagged 节点
        # if v_tag[int(path[0])] == 0 and v_tag[int(path[-1])] == 0:
        # if int(path[0]) == node_num-2 and int(path[-1]) == node_num-1:
        #     # path的首尾节点都未tag，即第一次选择的完整的头尾路径
        #     # 完整路径中，首尾节点是图的头尾节点，图的头尾节点规定在本地
        #     start = 1
        #     entry_task = 1
        #     exit_task = 1
        #     n -= 1
        T_min = self.T(adj_matrix, path, entry_task, exit_task, w_vex, v_tag)

        for i in range(start, n):
            for j in range(i, n):
                t = self.T(adj_matrix, path, i, j, w_vex, v_tag)
                if t < T_min:
                    entry_task = i
                    exit_task = j
                    T_min = t
        # else:
        #     # path 的首尾节点都是 tagged 节点
        t = self.T(adj_matrix, path, -1, -1, w_vex, v_tag)
        if t < T_min:
            return -1, -1
        return entry_task, exit_task

    def edge_weight(self, adj_matrix, path, i, j):
        return adj_matrix[int(path[i])][int(path[j])]

    def trans_time(self, adj_matrix, path, entry_task, exit_task, v_tag):
        total_trans = 0
        if entry_task == len(path) - 1 and exit_task == len(path) - 1:
            entry_node = path[1]
            exit_node = path[len(path) - 2]
            pre_entry_node = path[0]
            af_exit_node = path[len(path)-1]
            if v_tag[pre_entry_node] == 1:
                total_trans += adj_matrix[pre_entry_node][entry_node]
            if v_tag[af_exit_node] == 1:
                total_trans += adj_matrix[exit_node][af_exit_node]
        else:
            entry_node = path[entry_task]
            exit_node = path[exit_task]
            pre_entry_node = path[entry_task - 1]
            af_exit_node = path[exit_task + 1]

            if v_tag[pre_entry_node] == -1 or v_tag[pre_entry_node] == 0:
                total_trans += adj_matrix[pre_entry_node][entry_node]
            if v_tag[af_exit_node] == -1 or v_tag[af_exit_node] == 0:
                total_trans += adj_matrix[exit_node][af_exit_node]
        return total_trans / self.band_width

    def T(self, adj_matrix, path, entry_task, exit_task, w_vex, v_tag):
        if entry_task == -1 and exit_task == -1:
            entry_task = len(path) - 1
            exit_task = len(path) - 1
        T_local = 0
        T_local += local_T(path, 1, entry_task, w_vex)
        T_local += local_T(path, exit_task+1, len(path) - 1, w_vex)
        T_cloud = 0
        if entry_task == len(path) - 1 and exit_task == len(path) - 1:
            T_cloud += 0
        else:
            T_cloud += self.cloud.T(path, entry_task, exit_task, w_vex)
        T_trans = self.trans_time(adj_matrix, path, entry_task, exit_task, v_tag)
        if v_tag[path[0]] == -1:
            T_local += w_vex[path[0]] / local.COMPUTE_ABILITY
        else:
            T_cloud += w_vex[path[0]] / self.cloud.compute_ability
        if v_tag[path[-1]] == -1:
            T_local += w_vex[path[-1]] / local.COMPUTE_ABILITY
        else:
            T_cloud += w_vex[path[-1]] / self.cloud.compute_ability
        logger.debug('{0},{1} : {2}'.format(entry_task, exit_task, T_local + T_cloud + T_trans))
        return T_local + T_cloud + T_trans

    def tag_trans_vex(self, path, entry_task, exit_task, v_tag):
        if entry_task == -1 and exit_task == -1:
            for i in range(1, len(path)-1):
                v_tag[path[i]] = -1
        else:
            for i in range(entry_task, exit_task+1):
                v_tag[int(path[i])] = 1
            for i in range(1, entry_task):
                v_tag[int(path[i])] = -1
            for i in range(exit_task+1, len(path) - 1):
                v_tag[int(path[i])] = -1

    def choose_trans_vex_update_tag(self, adj_matrix, path, w_vex, v_tag):
        # logger.debug('path : {0};'.format(path))
        # logger.debug('v_tag : {0};'.format(v_tag))
        # logger.debug('v_trans : {0};'.format(v_trans))

        entry_task, exit_task = self.select_trans_seq(adj_matrix, path, w_vex, v_tag)
        self.tag_trans_vex(path, entry_task, exit_task, v_tag)
        logger.debug('{0}->{1}'.format(entry_task, exit_task))
        logger.debug('path : {0};'.format(path))
        logger.debug('v_tag : {0};'.format(v_tag))

    def all_tagged(self, v_tag):
        for t in v_tag:
            if t == 0:
                return False
        return True

    def backtrace_trans_graph(self, adj_matrix, v_tag, w_vex):
        n = len(adj_matrix)
        cur_v = n-1
        # path = ''
        path = []

        # sum(v_tag) < n : 图中还有未标记的节点
        while not self.all_tagged(v_tag):
            in_v = indere_vex(adj_matrix, cur_v)
            untag_v = []
            for i in in_v:
                if v_tag[i] == 0:
                    untag_v.append(i)
            if v_tag[cur_v] == 0:
                if len(untag_v) == 0:
                    # 入度节点都是tag节点
                    # cur_v : untagged; in_vex : tagged  (1->0)
                    # 结束path的构建，执行迁移方案选择，并清空path
                    # path += str(cur_v)
                    path.append(cur_v)
                    # path += str(random.choice(in_v))
                    path.append(random.choice(in_v))
                    path = path[::-1]
                    self.choose_trans_vex_update_tag(adj_matrix, path, w_vex, v_tag)
                    # path = ''
                    path = []

                    cur_v = random.choice(in_v)
                else:
                    # 入度节点还有未tag的节点
                    # cur_v : untagged; in_vex : untagged  (0->0)
                    # 将 cur_v 添加到path，随机选择一个 untagged 入度节点，继续回溯
                    # path += str(cur_v)
                    path.append(cur_v)

                    cur_v = random.choice(untag_v)
            else:
                if len(untag_v) == 0:
                    # 入度节点都是tag节点
                    # cur_v : tagged; in_vex : tagged  (1->1)
                    # 随机选择一个入度节点，继续回溯
                    cur_v = random.choice(in_v)
                else:
                    # 入度节点还有未tag的节点
                    # cur_v : tagged; in_vex : untagged  (0->1)
                    # 开始一个新的path构建
                    # path += str(cur_v)
                    path.append(cur_v)

                    cur_v = random.choice(untag_v)
            n = len(adj_matrix)
            if cur_v == n - 2:
                cur_v = n - 1

    def trans_modules_graph(self, adj_matrix, w_vex):
        n = len(adj_matrix)
        # 节点迁移方案的tag
        # tag为1，节点迁移方案确定；为0，节点迁移方案不确定
        v_tag = [0 for i in range(len(adj_matrix))]
        # 图的头尾节点规定在本地
        v_tag[-1] = -1
        v_tag[-2] = -1
        # 节点的迁移方案
        # v_trans值为1，节点需要迁移到云端。
        # v_trans = [0 for i in range(len(adj_matrix))]
        path = []
        randomly_choice_a_path(adj_matrix, n-2, path)
        self.choose_trans_vex_update_tag(adj_matrix, path, w_vex, v_tag)

        self.backtrace_trans_graph(adj_matrix, v_tag, w_vex)
        return v_tag

    def getRealSubSet(self, fromList,toList):
        if(len(fromList) <= 1):
            return
        for id in range(len(fromList)):
            arr = [i for i in fromList if i != fromList[id]]
            self.getRealSubSet(arr,toList)
            #print arr
            if(toList.count(arr) == 0):
                toList.append(arr)
# li = []
# getRealSubSet([1,2,3,4],li)
# li.sort()
# print li


