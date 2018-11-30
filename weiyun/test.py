from sklearn import preprocessing
import numpy as np
import matplotlib.pyplot as plt
import random
import pickle
from weiyun.mic_cloud.cloud import Cloud
from weiyun.mic_cloud.trans import Trans
import weiyun.utils.graph_utils as graph_utils
import weiyun.utils.matrix_utils as matrix_utils
import weiyun.mic_cloud.rp_time as rp_time
#
# arr = [0, 0, 1, 2, 0, 3]
# print(arr.count(0))
# index = 0
# for i in range(arr.count(0)):
#     index = arr.index(0, index)
#     print(index)
#     index += 1
#
# print(np.random.poisson(lam=5))
#
# import math
# import random
#
# def nextTime(rateParameter):
#     return -math.log(1.0 - random.random()) / rateParameter
#
# print(nextTime(1/40.0))
# print(sum([nextTime(3/1) for i in range(1000000)]) / 1000000)

# arr  = np.hstack([[1, 2, 3], [4, 5]])
# dic = {
#     'arr' : arr
# }
# print(dic)
# arr = np.array([1, 2, 3])
# print(arr)
# print(arr.shape)
#
# arr = arr[np.newaxis, :]
# print(arr)
# print(arr.shape)
# experience_pool = np.zeros(2)
# print(experience_pool.shape)
# arr = np.array([{'a' : 1}, {'b' : 2}])
# print(arr.shape)

# action_space = np.arange(10 * 10)
#
# n = np.random.randint(0, action_space)
# print(type(n))
# print(n)
#
# for i in range(n):
#     print('a')
# print(int(72 / 10))
#
# # action = np.argmax(action_value)
#
# fig = plt.figure()
# ax = fig.add_subplot(1, 1, 1)
# plt.ion()
# plt.show()
# record = []
# for i in range(1000):
#     # try:
#     #     ax.lines.remove(lines[0])
#     # except Exception:
#     #     pass
#     record.append(random.randint(0, 100))
#     print('-----------plt : ')
#     print(record)
#     lines = ax.plot([_ for _ in range(len(record))], record, 'r-', lw=5)
#     plt.pause(0.001)

# a = np.random.choice(a=5, size=3, p=[1, 2, 3, 4, 5])
# print(a)
# p=[1, 2, 3, 4, 5]
# print(p[:2])
#
# for i in range(10):
#     print(np.random.randint(0, 100))

# graphs = []
# with open('graph_file', 'rb') as fp:
#     graphs = pickle.load(fp)
# cloud = Cloud(core_num=7)
# trans = Trans(cloud, band_width=1)
# s = 0
# for g in graphs:
#     cp_len, last = rp_time.ours(g[0], g[1], g[2], cloud, trans)
#     print(cp_len)
#     s += cp_len
# print(s / len(graphs))

#
# b = False
#
# a = 10
# b = a
# a = 20
# print(b)

def _computation_time(assign_n_core, assign_n_bandwidth, graph):
    # cloud.CORE_NUM = user.assign_n_core
    # trans.BAND_WIDTH = user.assign_n_bandwidth
    cloud = Cloud(core_num=assign_n_core)
    trans = Trans(cloud, band_width=assign_n_bandwidth)
    print('user n_core : %d, n_bandwidth : %d' % (assign_n_core, assign_n_bandwidth))
    v_tag = trans.trans_modules_graph(graph[0], graph[1])
    schedule_queue = cloud.schedule_queue(v_tag, graph[2], graph_utils.generate_ancestor_matrix(graph[0]))

    # queue : [[9, 3, 0, 1], [7, 4, 2, -1]]
    max_computation = 0
    for i in range(len(schedule_queue)):
        sq = schedule_queue[i]
        sum_computation = 0
        for j in range(len(sq)):
            if sq[j] != -1 and sq[j] != 0:
                sum_computation += graph[1][sq[j]]
        if 0 in sq:
            index = 0
            zero_is = []
            for j in range(sq.count(0)):
                index = sq.index(0, index)
                zero_is.append(index)
                index += 1
            for zi in zero_is:
                max_vex_w = 0
                for k in range(len(schedule_queue)):
                    v_index = schedule_queue[k][zi]
                    v_w = graph[1][v_index]
                    if max_vex_w < v_w:
                        max_vex_w = v_w
                sum_computation += max_vex_w
        if max_computation < sum_computation:
            max_computation = sum_computation
    return max_computation / cloud.compute_ability

# graphs = []
# with open('graph_file', 'rb') as fp:
#     graphs = pickle.load(fp)
# # print(_computation_time(5, 5, graphs[0]))
# s = 0
# for g in graphs:
#     t = _computation_time(5, 2, g)
#     print(t)
#     s += t
# print('average')
# print(s / len(graphs))

# graphs = []
# for i in range(50):
#     n_nodes = random.randint(5, 20)
#     adj_matrix, w_vex, topo_order = graph_utils.generate_DAG(n_nodes)
#     graphs.append([adj_matrix, w_vex, topo_order])
#
# with open('graph_file', 'wb') as fp:
#     pickle.dump(graphs, fp)
# graph = self.graphs[user.graph_id]
#
# cp_time = self._computation_time(user, graph)

acd = {20.0: [78.7773044713079, 18, 4.376516915072661], 21.0: [67.9176408638688, 13, 5.224433912605292], 22.0: [22.595321148156494, 5, 4.519064229631299], 23.0: [13.189524998528928, 3, 4.396508332842976], 26.0: [5.327440849311444, 1, 5.327440849311444], 27.0: [7.635128355088308, 1, 7.635128355088308], 29.0: [10.483794856520703, 2, 5.241897428260351], 31.0: [12.636805446162718, 2, 6.318402723081359], 33.0: [43.28274649012665, 6, 7.213791081687774], 35.0: [13.27371834575575, 2, 6.636859172877875], 37.0: [8.240603504421232, 1, 8.240603504421232], 38.0: [11.942145953777572, 2, 5.971072976888786], 40.0: [4.334301240697123, 1, 4.334301240697123], 41.0: [35.29049288253271, 8, 4.411311610316589], 43.0: [7.400662509831592, 1, 7.400662509831592], 44.0: [4.694572512209439, 1, 4.694572512209439], 45.0: [6.524347336800208, 1, 6.524347336800208], 46.0: [22.74421207781385, 3, 7.58140402593795], 47.0: [56.671477028966706, 10, 5.667147702896671], 48.0: [23.144756895517183, 4, 5.786189223879296], 49.0: [4.502399290910162, 1, 4.502399290910162], 50.0: [38.105946816686185, 7, 5.443706688098026], 51.0: [150.84443057620174, 28, 5.387301092007205], 52.0: [13.240541750153298, 2, 6.620270875076649], 53.0: [106.60653852954275, 18, 5.922585473863486], 54.0: [45.24122120136815, 8, 5.6551526501710185], 56.0: [58.21710803669111, 10, 5.821710803669111], 59.0: [10.846069018283455, 2, 5.423034509141727], 60.0: [88.50420477982753, 15, 5.900280318655168], 62.0: [4.192284186937684, 1, 4.192284186937684], 64.0: [35.46721406106391, 7, 5.066744865866274], 65.0: [11.719495786444634, 2, 5.859747893222317], 66.0: [63.35548178265211, 12, 5.279623481887676], 68.0: [7.807246660758295, 1, 7.807246660758295], 70.0: [4.798337623008487, 1, 4.798337623008487], 71.0: [61.710442092269595, 12, 5.1425368410224666], 72.0: [181.29993801286602, 31, 5.848385097189227], 73.0: [24.231587239459138, 5, 4.846317447891828], 74.0: [63.98934138080296, 14, 4.570667241485926], 76.0: [4.341153694518154, 1, 4.341153694518154], 77.0: [194.3553042918181, 31, 6.269525944897358], 79.0: [11.049283957625672, 2, 5.524641978812836], 80.0: [162.88519147079512, 33, 4.935914893054398], 82.0: [18.85486647528455, 3, 6.284955491761516], 87.0: [34.54399303763141, 6, 5.757332172938568], 88.0: [25.82716754343339, 4, 6.4567918858583475], 89.0: [3.469072177601852, 1, 3.469072177601852], 90.0: [6.31994467856134, 1, 6.31994467856134], 91.0: [143.00267145593767, 23, 6.217507454605986], 92.0: [351.9131167079876, 61, 5.769067487016191], 93.0: [338.2893655751092, 52, 6.505564722598253], 95.0: [5.510941654243265, 1, 5.510941654243265], 96.0: [98.97578020913018, 18, 5.498654456062788], 97.0: [5.533886305440468, 1, 5.533886305440468]}

for a in acd.keys():
    print('%s : %s' % (a, acd[a]))
# for a in acd.keys():
#     acd[a].append(acd[a][0] / acd[a][1])
# print(acd)
#
# print(10.0 - 1.041660 + 0.227654)