# -*- coding: UTF-8 -*-

import random
# from weiyun.mic_cloud.trans import *
# import weiyun.mic_cloud.trans as trans
# import weiyun.mic_cloud.cloud as cloud
import weiyun.mic_cloud.local
from weiyun.mic_cloud.cloud import Cloud
from weiyun.mic_cloud.trans import Trans
from weiyun.utils.matrix_utils import *
from weiyun.utils.graph_utils import *
import pickle
import weiyun.mic_cloud.rp_time as rp_time
import time
import weiyun.mic_cloud.ACO as ACO
import csv

# path = '012345'
# w_vex = []
# for i in range(len(path)):
#     w_vex.append(random.randint(1,20))
#
# for i in range(len(w_vex)):
#     print w_vex[i]
#
# entry_task, exit_task = select_trans_seq(path, w_vex)
# print entry_task, exit_task

# print cloud.T(path, 0, 0, w_vex)
# print local.T(path, 0, 0, w_vex)

# print len(path)

# adj_matrix = generate_DAG(5)
# print_matrix(adj_matrix)
#
#
# trans_modules_graph(adj_matrix, [])

# print random.normalvariate(0.7, 1)

# adj_matrix, w_vex = generate_DAG(5)
# print_matrix(adj_matrix)
# print w_vex

# v_tag = [0 for i in range(len(adj_matrix))]
# v_tag[-1] = -1
# v_tag[-2] = -1

# path = randomly_choice_a_path(adj_matrix, len(adj_matrix)-2, '')
# choose_trans_vex_update_tag(adj_matrix, path, w_vex, v_tag)

# trans_modules_graph(adj_matrix, w_vex)

# modules = [3,5,7,4,2]
# topo_order = [0, 1, 2, 3, 4, 5, 6, 7, 8]
# queue = cloud.schedule_queue(modules, topo_order)
# print queue





# def best_trans(adj_matrix, w_vex):
# def select(i, vex):
#     list = []
#     if i == len(vex):
#         list.append([])
#     sub_list = select(i+1, vex, list)
#     for l  in sub_list:
#         if len(l) == 0:

    # if i > n:
    #     return
    # print '{0}, 0'.format(i)
    # select(i+1, n)
    # print '{0}, 1'.format(i)
    # select(i+1, n)



# adj_matrix, w_vex, topo_order = generate_DAG(5)
# print_matrix(adj_matrix)
# print w_vex
# #
# # v_tag = trans_modules_graph(adj_matrix, w_vex)
# v_tag = [-1, -1, -1, 1, 1, -1, -1]
# new_graph = reshape_graph(adj_matrix, w_vex, cloud.schedule_queue(v_tag, topo_order, generate_ancestor_matrix(adj_matrix)), v_tag)
# print w_vex
# print 'new graph : '
# print_matrix(new_graph)
#
# n = len(new_graph)
# traced = [0 for i in range(len(new_graph))]
# dp = [0 for i in range(len(new_graph))]
# last = [0 for i in range(len(new_graph))]
# #
# print critical_path(new_graph, n - 1, traced, dp, last)
# print last

# adj_matrix, w_vex, topo_order = generate_DAG(5)
# print_matrix(adj_matrix)
# print w_vex

def ours(matrix, w_v, topo_order):
    adj_matrix = copy_matrix(matrix)
    w_vex = copy_arr(w_v)
    v_tag = trans_modules_graph(adj_matrix, w_vex)
    new_graph = reshape_graph(adj_matrix, w_vex,
                              cloud.schedule_queue(v_tag, topo_order, generate_ancestor_matrix(adj_matrix)), v_tag)
    # print w_vex
    # print 'new graph : '
    logger.debug('w_vex : %s'.format(w_vex))
    logger.debug('new graph : ')
    print_matrix(new_graph)

    n = len(new_graph)
    traced = [0 for i in range(len(new_graph))]
    dp = [0 for i in range(len(new_graph))]
    last = [0 for i in range(len(new_graph))]
    #
    cp_len = critical_path(new_graph, n - 1, traced, dp, last)
    # print 'ours v_tag : {0}'.format(v_tag)
    logger.debug('ours v_tag : {0}'.format(v_tag))
    return cp_len, last

def best(adj_matrix, w_vex, topo_order):
    vs = []
    for i in range(len(w_vex) - 2):
        vs.append(i)
    li = []
    getRealSubSet(vs, li)
    ps = []
    for modules in li:
        v_tag = []
        for i in range(len(vs)):
            if vs[i] in modules:
                v_tag.append(1)
            else:
                v_tag.append(-1)
        ps.append(v_tag)
    l = []
    ln = []
    for v in vs:
        l.append(1)
        ln.append(-1)
    ps.append(l)
    ps.append(ln)
    # print len(ps)
    # print ps

    min_cp = float("inf")
    total_cp = 0
    for p in ps:
        p.append(-1)
        p.append(-1)
        if sum(p) == len(p) * (-1):
            continue
        # print 'p : {0}'.format(p)
        logger.debug('p : {0}'.format(p))
        cp_m = copy_matrix(adj_matrix)
        cp_a = copy_arr(w_vex)
        new_graph = reshape_graph(cp_m, cp_a,
                                  cloud.schedule_queue(p, topo_order, generate_ancestor_matrix(cp_m)), p)
        # print w_vex
        # print 'new graph : '
        # print_matrix(new_graph)

        n = len(new_graph)
        traced = [0 for i in range(len(new_graph))]
        dp = [0 for i in range(len(new_graph))]
        last = [0 for i in range(len(new_graph))]
        #
        cp_len = critical_path(new_graph, n - 1, traced, dp, last)
        # print cp_len
        logger.debug(cp_len)
        total_cp += cp_len
        if cp_len < min_cp:
            min_cp = cp_len
    # print 'mean_cp : {0}'.format(total_cp / len(ps))
    logger.debug('mean_cp : {0}'.format(total_cp / len(ps)))
    return min_cp, last

def distribution(matrix, w_v, topo_order, core_num):
    adj_matrix = copy_matrix(matrix)
    w_vex = copy_arr(w_v)
    v_tag = trans_modules_graph(adj_matrix, w_vex)
    cloud.CORE_NUM = core_num
    queue = cloud.schedule_queue(v_tag, topo_order, generate_ancestor_matrix(adj_matrix))
    new_graph = reshape_graph(adj_matrix, w_vex, queue, v_tag)
    print_matrix(new_graph)

    n = len(new_graph)
    traced = [0 for i in range(len(new_graph))]
    dp = [0 for i in range(len(new_graph))]
    last = [0 for i in range(len(new_graph))]
    #
    cp_len = critical_path(new_graph, n - 1, traced, dp, last)
    return cp_len, last


# ours_cp, ours_last = ours(adj_matrix, w_vex, topo_order)
# cloud.CORE_NUM = 3
# best_cp, best_last = best(cp_m, cp_a, topo_order)
#
# print 'ours_cp : {0}'.format(ours_cp)
# print 'best_cp : {0}'.format(best_cp)

def generate_multiple_users(user_num):
    ads = []
    wvs = []
    tp_orders = []
    for i in range(user_num):
        adj_matrix, w_vex, topo_order = generate_DAG(random.randint(10, 50))
        ads.append(adj_matrix)
        wvs.append(w_vex)
        tp_orders.append(topo_order)
    return ads, wvs, tp_orders

def greed(ads, wvs, tp_orders, total_core, time_matrix):
    # 20个用户，每个用户初始两个核
    t = 1 * len(tp_orders)
    ad_core = []
    rp_time = []
    for i in range(len(tp_orders)):
        # cloud.CORE_NUM = 1
        ad_core.append(0)
        # cp_len, last = ours(ads[i], wvs[i], tp_orders[i])
        cp_len = time_matrix[i][0]
        rp_time.append(cp_len)
    for i in range(total_core - t):
        max_reduce = 0
        m_index = len(rp_time) + 1
        m_rp_time = 0
        for j in range(len(rp_time)):
            # cloud.CORE_NUM = ad_core[j] + 1
            # cp_len, last = ours(ads[j], wvs[j], tp_orders[j])
            if ad_core[j] + 1 >= 9:
                ad_core[j] = 8
            cp_len = time_matrix[j][ad_core[j] + 1]
            if rp_time[j] - cp_len < 0:
                # print 'rp_time : {0}'.format(rp_time[j])
                # print 'cp_len : {0}'.format(cp_len)
                logger.debug('rp_time : {0}'.format(rp_time[j]))
                logger.debug('cp_len : {0}'.format(cp_len))
            if rp_time[j] - cp_len  > max_reduce:
                m_index = j
                max_reduce = rp_time[j] - cp_len
                m_rp_time = cp_len
        if m_index == len(rp_time) + 1:
            m_index = random.randint(0, len(rp_time)-1)
            # cloud.CORE_NUM = ad_core[m_index] + 1
            # m_rp_time, last = ours(ads[m_index], wvs[m_index], tp_orders[m_index])
            m_rp_time = time_matrix[m_index][ad_core[m_index] + 1]
        rp_time[m_index] = m_rp_time
        ad_core[m_index] += 1
    return rp_time, ad_core

def greed_with_bandwidth(ads, wvs, tp_orders, total_core, total_bandwidth):
    # 20个用户，每个用户初始两个核
    t = 1 * len(tp_orders)
    ad_core = []
    ad_bandwidth = []
    rp_time = []
    for i in range(len(tp_orders)):
        cloud.CORE_NUM = 1
        trans.BAND_WIDTH = 1
        ad_core.append(1)
        ad_bandwidth.append(1)
        cp_len, last = ours(ads[i], wvs[i], tp_orders[i])
        rp_time.append(cp_len)

    for i in range(total_bandwidth - t):
        max_reduce = 0
        m_index = len(rp_time) + 1
        m_rp_time = 0
        for j in range(len(rp_time)):
            trans.BAND_WIDTH = ad_bandwidth[j] + 1
            cloud.CORE_NUM = ad_core[j]
            cp_len, last = ours(ads[j], wvs[j], tp_orders[j])
            if rp_time[j] - cp_len  > max_reduce:
                m_index = j
                max_reduce = rp_time[j] - cp_len
                m_rp_time = cp_len
        if m_index == len(rp_time) + 1:
            m_index = random.randint(0, len(rp_time)-1)
            cloud.CORE_NUM = ad_bandwidth[m_index] + 1
            m_rp_time, last = ours(ads[j], wvs[j], tp_orders[j])
        rp_time[m_index] = m_rp_time
        ad_bandwidth[m_index] += 1

    for i in range(total_core - t):
        max_reduce = 0
        m_index = len(rp_time) + 1
        m_rp_time = 0
        for j in range(len(rp_time)):
            cloud.CORE_NUM = ad_core[j] + 1
            trans.BAND_WIDTH = ad_bandwidth[j]
            cp_len, last = ours(ads[j], wvs[j], tp_orders[j])
            if rp_time[j] - cp_len < 0:
                # print 'rp_time : {0}'.format(rp_time[j])
                # print 'cp_len : {0}'.format(cp_len)
                logger.debug('rp_time : {0}'.format(rp_time[j]))
                logger.debug('cp_len : {0}'.format(cp_len))
            if rp_time[j] - cp_len  > max_reduce:
                m_index = j
                max_reduce = rp_time[j] - cp_len
                m_rp_time = cp_len
        if m_index == len(rp_time) + 1:
            m_index = random.randint(0, len(rp_time)-1)
            cloud.CORE_NUM = ad_core[m_index] + 1
            m_rp_time, last = ours(ads[j], wvs[j], tp_orders[j])
        rp_time[m_index] = m_rp_time
        ad_core[m_index] += 1

    return rp_time,ad_core,ad_bandwidth

def mean_alg(ads, wvs, tp_orders, total_core):
    cloud.CORE_NUM = total_core / len(tp_orders)
    left = total_core - cloud.CORE_NUM * len(tp_orders)
    rp_time = []
    index = []
    while len(index) - left != 0:
        ind = random.randint(0, len(tp_orders) - 1)
        if ind in index:
            continue
        index.append(ind)
    for i in range(len(tp_orders)):
        cloud.CORE_NUM = total_core / len(tp_orders)
        if i in index:
            cloud.CORE_NUM += 1
        cp_len, last = ours(ads[i], wvs[i], tp_orders[i])
        rp_time.append(cp_len)
    return rp_time

# for i in range(100):
#     adj_matrix, w_vex, topo_order = generate_DAG(5)
#     # print_matrix(adj_matrix)
#     # print w_vex
#
#     TOTAL_CORE = 40
#     TOTAL_BAND_WIDTH = 60
#     cloud.CORE_NUM = 2
#     cp_len_two, last_two = ours(adj_matrix, w_vex, topo_order)
#
#     cloud.CORE_NUM = 3
#     cp_len_three, last_three = ours(adj_matrix, w_vex, topo_order)
#     if cp_len_three > cp_len_two:
#         print 'two : {0}; three : {1}'.format(cp_len_two, cp_len_three)
#         print '{0} - {1} = {2}'.format(cp_len_two, cp_len_three, cp_len_two - cp_len_three)
#         print 'i={0}'.format(i)
#         break


# start_time = time.time()

# end_time = time.time()
# print end_time - start_time
# print '2 core : {0}'.format(min_cp)

# cloud.CORE_NUM = 2
# min_cp, last = best(adj_matrix, w_vex, topo_order)
# print '3 core : {0}'.format(min_cp)
# #
# cp_len, last = ours(adj_matrix, w_vex, topo_order)
# print cp_len
# print last

# # print 'mean - greed = {0}'.format(sum(mean_rp_time) / USER_NUM - sum(greed_rp_time) / USER_NUM)
#
# print 'greed average rp_time : {0}'.format(sum(greed_rp_time) / USER_NUM)
# print 'mean average rp_time : {0}'.format(sum(mean_rp_time) / USER_NUM)
# print 'greed_with_bw average rp_time : {0}'.format(sum(greed_with_bw_rp_time) / USER_NUM)

# adj_matrix, w_vex, topo_order = generate_DAG(5)
# print_matrix(adj_matrix)
# print w_vex
# #
# # v_tag = trans_modules_graph(adj_matrix, w_vex)
# v_tag = [-1, -1, -1, 1, 1, -1, -1]
# new_graph = reshape_graph(adj_matrix, w_vex, cloud.schedule_queue(v_tag, topo_order, generate_ancestor_matrix(adj_matrix)), v_tag)
# print w_vex
# print 'new graph : '
# print_matrix(new_graph)
#
# n = len(new_graph)
# traced = [0 for i in range(len(new_graph))]
# dp = [0 for i in range(len(new_graph))]
# last = [0 for i in range(len(new_graph))]
# #
# print 'from {0}'.format(n-1)
# print critical_path(new_graph, n - 1, traced, dp, last)
# print last

# TOTAL_CORE = 40
# TOTAL_BANDWIDTH = 60
# USER_NUM = 20
# ads, wvs, tp_orders = generate_multiple_users(USER_NUM)
#
# def init_resources(n):
#     resources = []
#     for i in range(n):
#         resources.append(i)
#     return resources
#
# def init_time_matrix(resources, ads):
#     time_matrix = []
#     for i in range(len(ads[0])):
#         time_matrix_i = []
#         for j in range(30):
#             cloud.CORE_NUM = j + 1
#             rp_time, last = ours(ads[0][i], ads[1][i], ads[2][i])
#             time_matrix_i.append(rp_time)
#         time_matrix.append(time_matrix_i)
#     return time_matrix
#
# resources = init_resources(TOTAL_CORE - USER_NUM)
#
# time_matrix = init_time_matrix(resources, [ads, wvs, tp_orders])
#
# greed_rp_time, ad_core = greed(ads, wvs, tp_orders, TOTAL_CORE, time_matrix)
#
# mean_rp_time = mean_alg(ads, wvs, tp_orders, TOTAL_CORE)
# #
# #
# # greed_with_bw_rp_time, greed_with_bw_ad_core, greed_with_bw_ad_bw = greed_with_bandwidth(ads, wvs, tp_orders, TOTAL_CORE, TOTAL_BANDWIDTH)
# #
# aco_rp_time = ACO.aco_alg([ads, wvs, tp_orders], TOTAL_CORE, TOTAL_BANDWIDTH, time_matrix)
# #
# print('mean rp_time : {0}'.format(mean_rp_time))
# print('mean average rp_time : {0}'.format(sum(mean_rp_time) / USER_NUM))
#
# # print 'greed rp_time : {0}'.format(greed_rp_time)
# print('greed ad : {0}'.format(ad_core))
# print('greed average rp_time : {0}'.format(sum(greed_rp_time) / USER_NUM))
#
# # print 'greed_with_bw rp_time : {0}'.format(greed_with_bw_rp_time)
# # print 'greed_with_bw average rp_time : {0}'.format(sum(greed_with_bw_rp_time) / USER_NUM)
#
# print('aco rp_time : {0}'.format(aco_rp_time))


# adj_matrix, w_vex, topo_order = generate_DAG(10)
# cp_len, last = ours(adj_matrix, w_vex, topo_order)
# print(cp_len)

# def compute_time_matrix(adj_matrix, w_vex, topo_order):
#     time_matrix = []
#     for i in range(10):
#         time_matrix_i = []
#         for j in range(10):
#             cloud.CORE_NUM = j + 1
#             trans.BAND_WIDTH = i + 1
#             rp_time, last = ours(adj_matrix, w_vex, topo_order)
#             time_matrix_i.append(rp_time)
#         time_matrix.append
#     return time_matrix
#
# adj_matrix, w_vex, topo_order = generate_DAG(15)
# print_matrix(compute_time_matrix(adj_matrix, w_vex, topo_order))

# graphs = []
# for i in range(50):
#     adj_matrix, w_vex, topo_order = generate_DAG(15)
#     graphs.append([adj_matrix, w_vex, topo_order])
#
# cloud.CORE_NUM = 10
# trans.BAND_WIDTH = 10
# s = 0
# for g in graphs:
#     cp_len, last = rp_time.ours(g[0], g[1], g[2])
#     s += cp_len
#     print(cp_len)
# print(s / len(graphs))



# graphs = []
# s = 0
# for i in range(50):
#     n_nodes = random.randint(10, 20)
#     adj_matrix, w_vex, topo_order = generate_DAG(n_nodes)
#     graphs.append([adj_matrix, w_vex, topo_order])
#     cloud = Cloud(core_num=2)
#     trans = Trans(cloud)
#     cp_len, last = rp_time.ours(adj_matrix, w_vex, topo_order, cloud, trans)
#     s += cp_len
# print(s)

# with open('graph_file', 'wb') as fp:
#     pickle.dump(graphs, fp)

# graphs = []
# with open('graph_file', 'rb') as fp:
#     graphs = pickle.load(fp)
# cloud = Cloud(core_num=7)
# trans = Trans(cloud, band_width=4)
# s = 0
# for g in graphs:
#     cp_len, last = rp_time.ours(g[0], g[1], g[2])
#     s += cp_len
# print(s / len(graphs))

# def confirm_result():
#     with open('resources_num_test_ours', 'rb') as fp:
#         results_ours = pickle.load(fp)
#     with open('resources_num_test_best', 'rb') as fp:
#         results_best = pickle.load(fp)
#     results = []
#     for i in range(len(results_ours)):
#         ros = results_ours[i]
#         rbs = results_best[i]
#         results.append([sum(ros)/len(ros), sum(rbs)/len(rbs)])
#     print(results)
# confirm_result()

# graphs = []
# for i in range(100):
#     adj_matrix, w_vex, topo_order = generate_DAG(15)
#     graphs.append([adj_matrix, w_vex, topo_order])
# result = []
# with open('graph_type_topo', 'wb') as fp:
#     pickle.dump(graphs, fp)
#
# for g in graphs:
#     cloud = Cloud(core_num=2)
#     trans = Trans(cloud, band_width=2)
#
#     cp_len, last = rp_time.ours(g[0], g[1], g[2], cloud, trans)
#
#     cloud = Cloud(core_num=3)
#     trans = Trans(cloud, band_width=3)
#
#     cp_len_prime, last_prime = rp_time.ours(g[0], g[1], g[2], cloud, trans)
#     result.append((cp_len - cp_len_prime) / cp_len)
#
# with open('graph_type_topo.csv', 'w', newline='') as f:
#     writer = csv.writer(f)
#     writer.writerow(result)

# below_graphs, above_graphs = [], []
# below_reduce, above_reduce = [], []
# while len(below_graphs) < 100  or len(above_graphs) < 100:
#     print('%d : %d' % (len(below_graphs), len(above_graphs)))
#     adj_matrix, w_vex, topo_order = generate_DAG(15)
#     cloud = Cloud(core_num=2)
#     trans = Trans(cloud, band_width=2)
#     cp_len, last = rp_time.ours(adj_matrix, w_vex, topo_order, cloud, trans)
#
#     cloud = Cloud(core_num=3)
#     trans = Trans(cloud, band_width=3)
#     cp_len_prime, last_prime = rp_time.ours(adj_matrix, w_vex, topo_order, cloud, trans)
#     reduce = (cp_len - cp_len_prime) / cp_len
#     if reduce <= 0.15 and len(below_graphs) < 100:
#         below_graphs.append([adj_matrix, w_vex, topo_order])
#         below_reduce.append(reduce)
#     if reduce > 0.15 and len(above_graphs) < 100:
#         above_graphs.append([adj_matrix, w_vex, topo_order])
#         above_reduce.append(reduce)
#
# with open('graph_type_topo_below', 'wb') as fp:
#     pickle.dump(below_graphs, fp)
# with open('graph_type_topo_above', 'wb') as fp:
#     pickle.dump(above_graphs, fp)
# with open('graph_type_topo_below.csv', 'w', newline='') as f:
#     writer = csv.writer(f)
#     writer.writerow(below_reduce)
# with open('graph_type_topo_above.csv', 'w', newline='') as f:
#     writer = csv.writer(f)
#     writer.writerow(above_reduce)

# r = []
# reduces = []
# for j in range(100):
#     adj_matrix, w_vex, topo_order = generate_DAG(15)
#     result = []
#     for i in range(10):
#         cloud = Cloud(core_num=i+1)
#         trans = Trans(cloud, band_width=i+1)
#         cp_len, last = rp_time.ours(adj_matrix, w_vex, topo_order, cloud, trans)
#         if i == 0:
#             result.append(cp_len)
#         else:
#             result.append((result[0] - cp_len) / result[0])
#     reduces.append(sum(result[1:]) / (len(result) - 1))
#     r.append(result)
# print(sum(reduces) / len(reduces))



# with open('graph_type_topo.csv', 'w', newline='') as f:
#     writer = csv.writer(f)
#     for result in r:
#         writer.writerow(result)
# below_graphs, above_graphs = [], []
# below_reduce, above_reduce = [], []
# while len(below_graphs) < 100 or len(above_graphs) < 100:
#     print('%d : %d' % (len(below_graphs), len(above_graphs)))
#     adj_matrix, w_vex, topo_order = generate_DAG(15)
#     cloud = Cloud(core_num=1)
#     trans = Trans(cloud, band_width=1)
#     cp_len, last = rp_time.ours(adj_matrix, w_vex, topo_order, cloud, trans)
#     reduce = []
#     for i in range(9):
#         cloud = Cloud(core_num=i + 2)
#         trans = Trans(cloud, band_width=i + 2)
#         cp_len_prime, last_prime = rp_time.ours(adj_matrix, w_vex, topo_order, cloud, trans)
#
#         reduce.append((cp_len - cp_len_prime) / cp_len)
#     r = sum(reduce) / len(reduce)
#     if r > 0.37 and len(above_reduce) < 100:
#         above_graphs.append([adj_matrix, w_vex, topo_order])
#         above_reduce.append(reduce)
#     if r <= 0.37 and len(below_graphs) < 100:
#         below_graphs.append([adj_matrix, w_vex, topo_order])
#         below_reduce.append(reduce)

# with open('graph_type_topo_below', 'wb') as fp:
#     pickle.dump(below_graphs, fp)
# with open('graph_type_topo_above', 'wb') as fp:
#     pickle.dump(above_graphs, fp)
# with open('graph_type_topo_below.csv', 'w', newline='') as f:
#     writer = csv.writer(f)
#     writer.writerow(below_reduce)
# with open('graph_type_topo_above.csv', 'w', newline='') as f:
#     writer = csv.writer(f)
#     writer.writerow(above_reduce)

def generate_two_type_graphs(above_num, below_num, above_threshold, below_threshold):
    below_graphs, above_graphs = [], []
    while len(below_graphs) < below_num or len(above_graphs) < above_num:
        print('%d : %d' % (len(below_graphs), len(above_graphs)))
        adj_matrix, w_vex, topo_order = generate_DAG(15)
        cloud = Cloud(core_num=1)
        trans = Trans(cloud, band_width=1)
        cp_len, last = rp_time.ours(adj_matrix, w_vex, topo_order, cloud, trans)

        cloud = Cloud(core_num=10)
        trans = Trans(cloud, band_width=10)
        cp_len_prime, last_prime = rp_time.ours(adj_matrix, w_vex, topo_order, cloud, trans)
        if cp_len - cp_len_prime > 2 and len(above_graphs) < above_num:
            above_graphs.append([adj_matrix, w_vex, topo_order])
        if cp_len - cp_len_prime < 1.5 and len(below_graphs) < below_num:
            below_graphs.append([adj_matrix, w_vex, topo_order])
    return below_graphs, above_graphs
        # reduce = []
        # pre_cp_len = 0
        # for i in range(9):
        #     cloud = Cloud(core_num=i + 2)
        #     trans = Trans(cloud, band_width=i + 2)
        #     cp_len_prime, last_prime = rp_time.ours(adj_matrix, w_vex, topo_order, cloud, trans)
        #
        #     if i != 0:
        #         reduce.append((cp_len - cp_len_prime) / cp_len)
        # r = sum(reduce) / len(reduce)
        # if r > threshold and len(above_graphs) < above_num:
        #     above_graphs.append([adj_matrix, w_vex, topo_order])
        # if r <= threshold and len(below_graphs) < below_num:
        #     below_graphs.append([adj_matrix, w_vex, topo_order])
    # below_graphs, above_graphs = [], []
    # while len(below_graphs) < below_num or len(above_graphs) < above_num:
    #     print('%d : %d' % (len(below_graphs), len(above_graphs)))
    #     adj_matrix, w_vex, topo_order = generate_DAG(15)
    #     cloud = Cloud(core_num=1)
    #     trans = Trans(cloud, band_width=1)
    #     cp_len, last = rp_time.ours(adj_matrix, w_vex, topo_order, cloud, trans)
    #     reduce = []
    #     pre_cp_len = 0
    #     for i in range(9):
    #         cloud = Cloud(core_num=i + 2)
    #         trans = Trans(cloud, band_width=i + 2)
    #         cp_len_prime, last_prime = rp_time.ours(adj_matrix, w_vex, topo_order, cloud, trans)
    #
    #         if i != 0:
    #             reduce.append((cp_len - cp_len_prime) / cp_len)
    #     r = sum(reduce) / len(reduce)
    #     if r > threshold and len(above_graphs) < above_num:
    #         above_graphs.append([adj_matrix, w_vex, topo_order])
    #     if r <= threshold and len(below_graphs) < below_num:
    #         below_graphs.append([adj_matrix, w_vex, topo_order])

below_graphs, above_graphs = generate_two_type_graphs(100, 100, 2, 1.5)

with open('graph_type_topo_below', 'rb') as fp:
    below_graphs = pickle.load(fp)
with open('graph_type_topo_above', 'rb') as fp:
    above_graphs = pickle.load(fp)
bg = random.sample(below_graphs)
ag = random.sample(above_graphs)

results = []


# results = []
# for j in range(len(bg)):
#     results.append([])
#     g = bg[j]
#     for i in range(10):
#         cloud = Cloud(core_num=i+1)
#         trans = Trans(cloud, band_width=i+1)
#         cp_len, last = rp_time.ours(g[0], g[1], g[2], cloud, trans)
#         results[j].append(cp_len)
# with open('graph_type_topo_below.csv', 'w', newline='') as f:
#     writer = csv.writer(f)
#     for r in results:
#         writer.writerow(r)
#
# results = []
# for j in range(len(ag)):
#     results.append([])
#     g = ag[j]
#     for i in range(10):
#         cloud = Cloud(core_num=i + 1)
#         trans = Trans(cloud, band_width=i + 1)
#         cp_len, last = rp_time.ours(g[0], g[1], g[2], cloud, trans)
#         results[j].append(cp_len)
# with open('graph_type_topo_above.csv', 'w', newline='') as f:
#     writer = csv.writer(f)
#     for r in results:
#         writer.writerow(r)

# for graphs in [bg, ag]:
#     results.append([])
#     for g in graphs:
#         for i in range(10):
#             cloud = Cloud(core_num=i + 1)
#             trans = Trans(cloud, band_width=i + 1)
#             cp_len, last = rp_time.ours(g[0], g[1], g[2], cloud, trans)
#