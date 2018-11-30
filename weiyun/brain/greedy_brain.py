import numpy as np
# import weiyun.mic_cloud.trans as trans
# import weiyun.mic_cloud.cloud as cloud
from weiyun.mic_cloud.cloud import Cloud
from weiyun.mic_cloud.trans import Trans
import weiyun.mic_cloud.rp_time as rp_time
from weiyun.utils.log_utils import logger
from weiyun.utils.matrix_utils import *

MAX_CORE = 10
MAX_BANDWIDTH = 10

DISCOUNT = 0.9

class GreedyPolicy:
    def __init__(self, env):
        self.env = env

        # user_threshold 格式：[[core_threshold, bandwidth_threshold], [...], ...]
        self.graph_threshold = []

        self._init_graph_threshold()

    def _init_graph_threshold(self):
        for i in range(self.env.total_graphs):
            graph = self.env.graphs[i]
            time_matrix = self._time_matrix(graph)
            core_threshold = -1
            bandwidth_threshold = -1
            for i in range(MAX_CORE-1):
                if time_matrix[i][0] == time_matrix[i+1][0]:
                    core_threshold = i+1
            core_threshold = core_threshold if core_threshold != -1 else MAX_CORE - 1
            for j in range(MAX_BANDWIDTH-1):
                if time_matrix[core_threshold][j] == time_matrix[core_threshold][j+1]:
                    bandwidth_threshold = j
            bandwidth_threshold = bandwidth_threshold if bandwidth_threshold != -1 else MAX_BANDWIDTH - 1
            # self.graph_threshold.append([4, 2])
            core_threshold = int((core_threshold+1) * DISCOUNT) if int((core_threshold+1) * DISCOUNT) > 0 else 1
            bandwidth_threshold = int((bandwidth_threshold+1) * DISCOUNT) if int((bandwidth_threshold+1) * DISCOUNT) > 0 else 1
            self.graph_threshold.append([core_threshold, bandwidth_threshold])
        logger.info('graph_threshold : %s' % self.graph_threshold)

    def _time_matrix(self, graph):
        time_matrix = []
        for i in range(1, MAX_CORE+1):
            tm = []
            # cloud.CORE_NUM = i
            cloud = Cloud(core_num=i)
            for j in range(1, MAX_BANDWIDTH + 1):
                # trans.BAND_WIDTH = j
                trans = Trans(cloud, band_width=j)
                time, last = rp_time.ours(graph[0], graph[1], graph[2], cloud, trans)
                tm.append(time)
            time_matrix.append(tm)
        return time_matrix

    def take_a_step(self, observation):
        n_core_left = observation[1]
        n_bandwidth_left = observation[2]
        # user = self.env.users[observation[0]] if observation[0] != -1 else None

        # graph_id = np.argmax(graph_vector)
        graph_id = observation[0]
        # 一直循环，直到环境剩余足够的资源
        while n_core_left < self.graph_threshold[graph_id][0] or n_bandwidth_left < self.graph_threshold[graph_id][1]:
            n_core_left = self.env.n_core_left
            n_bandwidth_left = self.env.n_bandwidth_left

        # 环境剩余足够的资源
        return self.env.step([self.graph_threshold[graph_id][0], self.graph_threshold[graph_id][1]])