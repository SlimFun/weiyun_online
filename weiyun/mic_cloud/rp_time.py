# -*- coding: UTF-8 -*-

from weiyun.mic_cloud.cloud import Cloud
from weiyun.mic_cloud.trans import Trans
from weiyun.utils.matrix_utils import *
from weiyun.utils.graph_utils import *

def ours(matrix, w_v, topo_order, cloud, trans):
    adj_matrix = copy_matrix(matrix)
    w_vex = copy_arr(w_v)
    v_tag = trans.trans_modules_graph(adj_matrix, w_vex)
    new_graph = reshape_graph(adj_matrix, w_vex,
                              cloud.schedule_queue(v_tag, topo_order, generate_ancestor_matrix(adj_matrix)), v_tag, cloud, trans)
    # print w_vex
    logger.debug('w_vex : %s'.format(w_vex))
    # print 'new graph : '
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

def best(adj_matrix, w_vex, topo_order, cloud, trans):
    vs = []
    for i in range(len(w_vex) - 2):
        vs.append(i)
    li = []
    trans.getRealSubSet(vs, li)
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
        logger.debug('cp_len : %d'.format(cp_len))
        total_cp += cp_len
        if cp_len < min_cp:
            min_cp = cp_len
    # print 'mean_cp : {0}'.format(total_cp / len(ps))
    logger.debug('mean_cp : {0}'.format(total_cp / len(ps)))
    return min_cp, last

def distribution(matrix, w_v, topo_order, cloud, trans):
    adj_matrix = copy_matrix(matrix)
    w_vex = copy_arr(w_v)
    v_tag = trans.trans_modules_graph(adj_matrix, w_vex)
    # cloud.CORE_NUM = core_num
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
        adj_matrix, w_vex, topo_order = generate_DAG(5)
        ads.append(adj_matrix)
        wvs.append(w_vex)
        tp_orders.append(topo_order)
    return ads, wvs, tp_orders

def greed(ads, wvs, tp_orders, total_core):
    # 20个用户，每个用户初始两个核
    t = 1 * len(tp_orders)
    ad_core = []
    rp_time = []
    for i in range(len(tp_orders)):
        cloud.CORE_NUM = 1
        ad_core.append(1)
        cp_len, last = ours(ads[i], wvs[i], tp_orders[i])
        rp_time.append(cp_len)
    for i in range(total_core - t):
        max_reduce = 0
        m_index = len(rp_time) + 1
        m_rp_time = 0
        for j in range(len(rp_time)):
            cloud.CORE_NUM = ad_core[j] + 1
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
    return rp_time,ad_core

def greed_with_bandwidth(ads, wvs, tp_orders, total_core, total_bandwidth):
    # 20个用户，每个用户初始两个核
    t = 1 * len(tp_orders)
    ad_core = []
    ad_bandwidth = []
    rp_time = []
    for i in range(len(tp_orders)):
        # cloud.CORE_NUM = 1
        # trans.BAND_WIDTH = 1
        cloud = Cloud(core_num=1)
        trans = Trans(cloud, band_width=1)
        ad_core.append(1)
        ad_bandwidth.append(1)
        cp_len, last = ours(ads[i], wvs[i], tp_orders[i], cloud, trans)
        rp_time.append(cp_len)

    for i in range(total_bandwidth - t):
        max_reduce = 0
        m_index = len(rp_time) + 1
        m_rp_time = 0
        for j in range(len(rp_time)):
            cloud = Cloud(core_num=ad_core[j])
            trans = Trans(cloud, band_width=ad_bandwidth[j] + 1)
            # trans.BAND_WIDTH = ad_bandwidth[j] + 1
            # cloud.CORE_NUM = ad_core[j]
            cp_len, last = ours(ads[j], wvs[j], tp_orders[j], cloud, trans)
            if rp_time[j] - cp_len  > max_reduce:
                m_index = j
                max_reduce = rp_time[j] - cp_len
                m_rp_time = cp_len
        if m_index == len(rp_time) + 1:
            m_index = random.randint(0, len(rp_time)-1)
            # cloud.CORE_NUM = ad_bandwidth[m_index] + 1
            cloud = Cloud(core_num=ad_bandwidth[m_index] + 1)
            m_rp_time, last = ours(ads[j], wvs[j], tp_orders[j], cloud, trans)
        rp_time[m_index] = m_rp_time
        ad_bandwidth[m_index] += 1

    for i in range(total_core - t):
        max_reduce = 0
        m_index = len(rp_time) + 1
        m_rp_time = 0
        for j in range(len(rp_time)):
            # cloud.CORE_NUM = ad_core[j] + 1
            # trans.BAND_WIDTH = ad_bandwidth[j]
            cloud = Cloud(core_num=ad_core[j] + 1)
            trans = Trans(cloud, band_width=ad_bandwidth[j])
            cp_len, last = ours(ads[j], wvs[j], tp_orders[j], cloud, trans)
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
            cloud = Cloud(core_num=ad_core[m_index] + 1)
            m_rp_time, last = ours(ads[j], wvs[j], tp_orders[j], cloud, trans)
        rp_time[m_index] = m_rp_time
        ad_core[m_index] += 1

    return rp_time,ad_core,ad_bandwidth

def mean_alg(ads, wvs, tp_orders, total_core):
    # cloud.CORE_NUM = total_core / len(tp_orders)
    cloud = Cloud(total_core / len(tp_orders))
    trans = Trans(cloud)
    rp_time = []
    for i in range(len(tp_orders)):
        cp_len, last = ours(ads[i], wvs[i], tp_orders[i], cloud, trans)
        rp_time.append(cp_len)
    return rp_time