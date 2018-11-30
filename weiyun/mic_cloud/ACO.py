# -*- coding: UTF-8 -*-
import random
import math

import random
from trans import *
import cloud
import local
from weiyun.utils.matrix_utils import *
import time

# from weiyun.utils.graph_utils import *
# import weiyun.mic_cloud.cloud as cloud
# import weiyun.mic_cloud.local as local
# import weiyun.mic_cloud.trans as trans
import matplotlib.pyplot as plt
import rp_time as rpTime

# def init_resources(core_num, bandwidth):
#     resources = []
#     index = 0
#     for i in range(0, core_num):
#         for j in range(0, bandwidth):
#             resources.append(index)
#             index += 1
#     return resources

def init_resources(n):
    resources = []
    for i in range(n):
        resources.append(i)
    return resources

USER_NUM = 20

# 蚁群算法一共需要迭代的次数
iterator_num = 50

# 每次迭代中蚂蚁的数量。每只蚂蚁都是一个任务调度者，
# 每次迭代中的每一只蚂蚁都需要完成所有任务的分配，这也就是一个可行解。
ant_num = 20

# 任务处理时间矩阵
# timeMatrix[i][j]=task[i]/ads[j]
time_matrix = []

# 信息素矩阵
# 记录任务i分配给节点j这条路径上的信息素浓度
pheromone_matrix = []

# 每完成一次迭代后，信息素衰减的比例
p = 0.5
# 蚂蚁每次经过一条路径，信息素增加的比例
q = 2

MAX_PHEROMONE = 1000.0
MIN_PHEROMONE = 0.0

# def init_time_matrix(resources, ads, total_core, total_bandwidth):
#     time_matrix = []
#     for j in range(len(ads[0])):
#         time_matrix_i = []
#         for i in range(len(resources)):
#             cloud.CORE_NUM = resources[i] / total_bandwidth + 1
#             trans.BAND_WIDTH = resources[i] % total_bandwidth + 1
#             rp_time, last = rpTime.ours(ads[0][j], ads[1][j], ads[2][j])
#             time_matrix_i.append(rp_time)
#         time_matrix.append(time_matrix_i)
#     # for ad in ads:
#     #     d = {}
#     #     for t in tuples:
#     #         cloud.CORE_NUM = t[0]
#     #         trans.BAND_WIDTH = t[1]
#     #         rp_time, last = rpTime.ours(ad[0], ad[1], ad[2])
#     #         d[index_of_tuple(t)] = rp_time
#     #     time_matrix.append(d)
#     return time_matrix

def init_time_matrix(resources, ads):
    time_matrix = []
    for i in range(len(ads[0])):
        time_matrix_i = []
        for j in range(len(resources)):
            cloud.CORE_NUM = j + 1
            rp_time, last = rpTime.ours(ads[0][i], ads[1][i], ads[2][i])
            time_matrix_i.append(rp_time)
        time_matrix.append(time_matrix_i)
    return time_matrix

def init_pheromone_matrix(resources, ads, time_len):
    pheromone_matrix = []
    for i in range(len(resources)):
        pheromone_matrix_i = []
        for j in range(len(ads[0])):
            pheromone_matrix_i.append(MAX_PHEROMONE)
        pheromone_matrix.append(pheromone_matrix_i)
    # for ad in ads:
    #     d = {}
    #     for t in tuples:
    #         d[index_of_tuple(t)] = 1
    #     pheromone_matrix.append(d)
    return pheromone_matrix

def init_matrix(resources, ads, num):
    matrix = []
    for i in range(len(ads[0])):
        matrix_i = []
        for j in range(len(resources)):
        # for j in range(10):
            matrix_i.append(num)
        matrix.append(matrix_i)
    # for ad in ads:
    #     d = {}
    #     for t in tuples:
    #         d[index_of_tuple(t)] = num
    #     matrix.append(d)
    return matrix

def init_array(length, num):
    arr = []
    for i in range(length):
        arr.append(num)
    return arr

def compute_p(pheromone_row, h_factor, alpha, beta):
    ps = []
    s = 0
    for i in range(len(pheromone_row)):
        x = math.pow(pheromone_row[i], alpha) * math.pow(h_factor[i], beta)
        ps.append(x)
        s += x
    prob_s = map(lambda x:x/s, ps)
    return prob_s

# def get_range_of_resources(userCount, used_core, used_bandwidth, total_core, total_bandwidth, resources, ads):
#     rag = []
#     reserved_resources = len(ads[0]) - userCount
#     if total_core - used_core <= reserved_resources and total_bandwidth - used_bandwidth <= reserved_resources:
#         return [0]
#     if total_core - used_core <= reserved_resources:
#         for i in range(0, 1 * MAX_BANDWIDTH):
#             rag.append(i)
#
#     if total_bandwidth - used_bandwidth <= reserved_resources:
#         for i in range(len(resources) / MAX_BANDWIDTH):
#             rag.append(i * MAX_BANDWIDTH)
#
#     if len(rag) == 0:
#         for i in range(min(total_core - reserved_resources - used_core - 1, MAX_CORE)):
#             for j in range(min(total_bandwidth - reserved_resources - used_bandwidth - 1, MAX_BANDWIDTH)):
#                 rag.append(i * MAX_BANDWIDTH + j)
#     return rag

def assign_one_resource(resource_index, rp_time, ads, ad_core, pheromone_matrix, time_matrix):
    reduce = []
    users = []
    for i in range(len(ads[0])):
        users.append(i)
        reduce.append(1.0)
    for j in range(len(ads[0])):
        # cloud.CORE_NUM = ad_core[j] + 1
        # cp_len, last = rpTime.ours(ads[0][j], ads[1][j], ads[2][j])
        if ad_core[j] + 1 >= 9:
            ad_core[j] = 8
        # print 'ad_core[{0}] : {1}'.format(j, ad_core[j])
        cp_len = time_matrix[j][ad_core[j] + 1]
        if rp_time[j] - cp_len > 0:
            # reduce.append(rp_time[j] - cp_len)
            reduce[j] = rp_time[j] - cp_len

    alpha = 2
    beta = 3

    h_factor = map(lambda x:x / 10, reduce)

    pheromone_row = pheromone_matrix[resource_index][:]
    ps = compute_p(pheromone_row, h_factor, alpha, beta)
    ran = random.random()
    s = 0
    j = 0
    for j, p in zip(users, ps):
        s += p
        if ran < s:
            break
    m_rp_time = time_matrix[j][ad_core[j] + 1]
    # m_rp_time, last = rpTime.ours(ads[0][j], ads[1][j], ads[2][j])
    return j, m_rp_time

def cal_time_oneIt(ant_num, rp_time_all, user_num):
    time_allAnt = []
    for i in range(ant_num):
        total_time = 0
        for j in range(len(rp_time_all[i])):
            total_time += rp_time_all[i][j]
        time_allAnt.append(total_time / user_num)
    return time_allAnt


def update_pheromone_matrix(path_matrix_all, pheromone_matrix, time_arr_oneIt, resources, ads, p, q, ant_num, time_matrix):
    # print pheromone_matrix
    # 所有信息素衰减p
    # for dis_tuples in tuples:
    for resource_index in range(len(resources)):
        for user_index in range(len(ads[0])):
            pheromone_matrix[resource_index][user_index] *= p
            if pheromone_matrix[resource_index][user_index] < MIN_PHEROMONE:
                pheromone_matrix[resource_index][user_index] = MIN_PHEROMONE
    # for i in range(len(tuples)):
    #     for j in range(len(ads)):
    #         pheromone_matrix[i][j] *= p

    # 找出任务处理时间最短的蚂蚁编号
    min_time = float('INF')
    min_index = -1
    # print time_arr_oneIt
    for i in range(len(time_arr_oneIt)):
        if time_arr_oneIt[i] < min_time:
            min_time = time_arr_oneIt[i]
            min_index = i

    # 将本次迭代中最优路径的信息素增加q
    # for dis_tuples in tuples:
    for resource_index in range(len(resources)):
        for user_index in range(len(ads[0])):
            # time_row = time_matrix[user_index][:]
            # total_time = sum(time_row)
            if path_matrix_all[min_index][user_index][resource_index] == 1:
                time_row = time_matrix[user_index][:]
                total_time = sum(time_row)
                core_num = sum(path_matrix_all[min_index][user_index][:])
                # print 'core_num : {0}'.format(core_num)
                pheromone_matrix[resource_index][user_index] += (total_time / time_matrix[user_index][core_num]) / 100
                # print total_time / time_matrix[user_index][resource_index]
                # pheromone_matrix[user_index][resource_index] *= q
                if pheromone_matrix[resource_index][user_index] > MAX_PHEROMONE:
                    pheromone_matrix[resource_index][user_index] = MAX_PHEROMONE

    # max_pheromone_matrix = []
    #
    # for user_index in range(len(ads[0])):
    #     max_pheromone = pheromone_matrix[user_index][0]
    #     max_index = 0
    #
    #     for resource_index in range(len(resources)):
    #         if pheromone_matrix[user_index][resource_index] > max_pheromone:
    #             max_pheromone = pheromone_matrix[user_index][resource_index]
    #             max_index = resource_index
    #
    #     max_pheromone_matrix.append(max_index)


    # for i in range(len(tasks)):
    #     max_pheromone = pheromone_matrix[i][0]
    #     max_index = 0
    #     sum_pheromone = pheromone_matrix[i][0]
    #     is_all_same = True
    #
    #     for j in range(len(ads)):
    #         if pheromone_matrix[i][j] > max_pheromone:
    #             max_pheromone = pheromone_matrix[i][j]
    #             max_index = j
    #
    #         if pheromone_matrix[i][j] != pheromone_matrix[i][j-1]:
    #             is_all_same = False
    #
    #         sum_pheromone += pheromone_matrix[i][j]
    #
    #     # 若本行信息素全都相等，则随机选择一个作为最大信息素
    #     if is_all_same:
    #         max_index = random.randint(0, len(ads) - 1)
    #         max_pheromone = pheromone_matrix[i][max_index]
    #
    #     # 将本行最大信息素的下标加入max_pheromone_matrix
    #     max_pheromone_matrix.append(max_index)

        # 将本次迭代的蚂蚁临界编号加入critical_point_matrix
        # (该临界点之前的蚂蚁的任务分配根据最大信息素原则，而该临界点之后的蚂蚁采用随机分配策略)
        # critical_point_matrix.append(round(ant_num * (max_pheromone / sum_pheromone)))

    # print critical_point_matrix

def aca_search(iterator_num, ant_num, resources, ads, pheromone_matrix, time_matrix, p, q, total_core, total_bandwidth):
    result_data = []
    result_path = []
    for itCount in range(iterator_num):
        # 本次迭代中，所有蚂蚁的路径
        path_matrix_all = []

        rp_time_all = []
        for antCount in range(ant_num):

            # 第antCount只蚂蚁的分配策略(pathMatrix[i][j]
            # 表示第antCount只蚂蚁将i任务分配给j节点处理)
            path_matrix_oneAnt = init_matrix(resources, ads, 0)
            # for disCount in range(len(tuples)):
            # for userCount in range(len(ads[0])):
            #     # 将第taskCount个任务分配给第nodeCount个节点处理
            #     resourceCount = assign_one_task(userCount, resources, ads, pheromone_matrix, time_matrix, total_core, total_bandwidth, used_core, used_bandwidth)
            #     # path_matrix_oneAnt[taskCount][nodeCount] = 1
            #     # path_matrix_oneAnt[userCount]['{0},{1}'.format(dis_tuple[0], dis_tuple[1])] = 1
            #     path_matrix_oneAnt[userCount][resourceCount] = 1
            ad_core = []
            rp_time = []
            for i in range(len(ads[0])):
                # cloud.CORE_NUM = 1
                ad_core.append(0)
                cp_len = time_matrix[i][0]
                # cp_len, last = rpTime.ours(ads[0][i], ads[1][i], ads[2][i])
                rp_time.append(cp_len)
            for i in range(len(resources)):
                m_index, m_rp_time = assign_one_resource(i, rp_time, ads, ad_core, pheromone_matrix, time_matrix)
                rp_time[m_index] = m_rp_time
                ad_core[m_index] += 1
                path_matrix_oneAnt[m_index][i] = 1

            # 将当前蚂蚁的路径加入pathMatrix_allAnt
            path_matrix_all.append(path_matrix_oneAnt)
            rp_time_all.append(rp_time)

        # 计算 本次迭代中 所有蚂蚁 的任务处理时间
        time_arr_oneIt = cal_time_oneIt(len(path_matrix_all), rp_time_all, len(ads[0]))
        # 将本地迭代中 所有蚂蚁的 任务处理时间加入总结果集，
        # 它是一个二维矩阵，result_data[x][y]=10代表第x次迭代中第y只蚂蚁的任务处理时间是10
        result_data.append(time_arr_oneIt)

        if itCount == iterator_num - 1:
            result_path = init_matrix(resources, ads, 0)

            min_time = float('INF')
            min_index = -1
            for i in range(len(time_arr_oneIt)):
                if time_arr_oneIt[i] < min_time:
                    min_time = time_arr_oneIt[i]
                    min_index = i

            for resource_index in range(len(resources)):
                for user_index in range(len(ads[0])):
                    if path_matrix_all[min_index][user_index][resource_index] == 1:
                        result_path[user_index][resource_index] = 1

        # 更新信息素
        update_pheromone_matrix(path_matrix_all, pheromone_matrix, time_arr_oneIt, resources, ads, p, q, ant_num, time_matrix)

    return result_data, result_path

def aca(iterator_num, ant_num, resources, ads, p, q, total_core, total_bandwidth, time_matrix):


    trans.BAND_WIDTH = 3.0

    time_len = 10

    pheromone_matrix = init_pheromone_matrix(resources, ads, time_len)

    return aca_search(iterator_num, ant_num, resources, ads, pheromone_matrix, time_matrix, p, q, total_core, total_bandwidth)

def aco_alg(ads, total_core, total_bandwidth, time_matrix):

    resources = init_resources(total_core - len(ads[0]))

    result_data, result_path = aca(iterator_num, ant_num, resources, ads, p, q, total_core, total_bandwidth, time_matrix)

    x_it = []
    y_time = []
    d = []
    # print result_data
    for i in range(iterator_num):
        x_it.append(i)
        for j in range(ant_num):
            d.append([x_it, result_data[i][j]])
    # print d
    # plt.plot(d)
    # plt.show()
    r = []
    # x = []
    for j in range(ant_num):
        r = []
        # x = []
        for i in range(iterator_num):
            # for j in range(ant_num):
            # x.append(i)
            r.append(result_data[i][j])
        # plt.scatter(x, r)
        plt.plot(r)
    plt.show()
    min_r = float('INF')
    for r in result_data[len(result_data) - 1]:
        if r < min_r:
            min_r = r

    return min_r

# print aco_alg()