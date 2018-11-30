# -*- coding: UTF-8 -*-
import random
import math
import matplotlib.pyplot as plt

def init_random_array(num):
    arr = []
    for i in range(num):
        arr.append(random.uniform(10.0, 100.0))
    return arr

# 任务数组，数组的下标表示任务的编号，数组的值表示任务的长度
tasks = init_random_array(500)

# 处理节点的数组。数组的下标表示处理节点的编号，数组值表示节点的处理速度
nodes = init_random_array(50)

# 蚁群算法一共需要迭代的次数
iterator_num = 50

# 每次迭代中蚂蚁的数量。每只蚂蚁都是一个任务调度者，
# 每次迭代中的每一只蚂蚁都需要完成所有任务的分配，这也就是一个可行解。
ant_num = 10

# 任务处理时间矩阵
# timeMatrix[i][j]=task[i]/nodes[j]
time_matrix = []

# 信息素矩阵
# 记录任务i分配给节点j这条路径上的信息素浓度
pheromone_matrix = []
# pheromoneMatrix矩阵的每一行中最大信息素的下标
max_pheromone_matrix = []

# 在一次迭代中，采用随机分配策略的蚂蚁的临界编号:
# criticalPointMatrix[0]=5，那么也就意味着，在分配第0个任务的时候，
# 编号是0～5的蚂蚁根据信息素浓度进行任务分配;
# 6～9号蚂蚁则采用随机分配的方式
critical_point_matrix = []

# 每完成一次迭代后，信息素衰减的比例
p = 0.5
# 蚂蚁每次经过一条路径，信息素增加的比例
q = 2

def init_time_matrix(tasks, nodes):
    time_matrix = []
    for i in range(len(tasks)):
        time_matrix_i = []
        for j in range(len(nodes)):
            time_matrix_i.append(tasks[i] / nodes[j])
        time_matrix.append(time_matrix_i)
    return time_matrix

def init_pheromone_matrix(tasks, nodes):
    pheromone_matrix = []
    for i in range(len(tasks)):
        pheromone_matrix_i = []
        for j in range(len(nodes)):
            pheromone_matrix_i.append(1)
        pheromone_matrix.append(pheromone_matrix_i)
    return pheromone_matrix

def init_matrix(tasks, nodes, num):
    matrix = []
    for i in range(len(tasks)):
        matrix_i = []
        for j in range(len(nodes)):
            matrix_i.append(num)
        matrix.append(matrix_i)
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


def assign_one_task(antCount, taskCount, nodes, critical_point_matrix, max_pheromone_matrix, pheromone_matrix, time_matrix):
    # 若当前蚂蚁编号在临界点之前，则采用最大信息素的分配方式
    # if antCount <= critical_point_matrix[taskCount]:
    # # if True:
    #     return max_pheromone_matrix[taskCount]
    # # 若当前蚂蚁编号在临界点之后，则采用随机分配方式
    # return random.randint(0, len(nodes) - 1)

    alpha = 1
    beta = 1

    time_row = time_matrix[taskCount][:]
    h_factor = map(lambda x:1/(1-x), time_row)


    pheromone_row = pheromone_matrix[taskCount][:]
    sum_pheromone = sum(pheromone_row)
    # ps = map(lambda x:x/sum_pheromone,pheromone_row)
    ps = compute_p(pheromone_row, h_factor, alpha, beta)
    index = []
    for i in range(len(nodes)):
        index.append(i)
    ran = random.random()
    s = 0
    j = 0
    for j, p in zip(index, ps):
        s += p
        if ran < s:
            break
    return j

def cal_time_oneIt(path_matrix_all, tasks, nodes, time_matrix):
    time_allAnt = []
    for i in range(len(path_matrix_all)):
        # 获取第i只蚂蚁的行走路径
        path_matrix = path_matrix_all[i]

        # 获取处理时间最长的节点 对应的处理时间
        max_time = -1
        for node_index in range(len(nodes)):
            # 计算节点taskIndex的任务处理时间
            time = 0
            for task_index in range(len(tasks)):
                if path_matrix[task_index][node_index] == 1:
                    time += time_matrix[task_index][node_index]

            # 更新maxTime
            if time > max_time:
                max_time = time

        time_allAnt.append(max_time)
    return time_allAnt


def update_pheromone_matrix(path_matrix_all, pheromone_matrix, time_arr_oneIt, tasks, nodes, p, q, ant_num):
    # 所有信息素衰减p
    for i in range(len(tasks)):
        for j in range(len(nodes)):
            pheromone_matrix[i][j] *= p

    # 找出任务处理时间最短的蚂蚁编号
    min_time = float('INF')
    min_index = -1
    # print time_arr_oneIt
    for i in range(len(time_arr_oneIt)):
        if time_arr_oneIt[i] < min_time:
            min_time = time_arr_oneIt[i]
            min_index = i

    # 将本次迭代中最优路径的信息素增加q
    for i in range(len(tasks)):
        for j in range(len(nodes)):
            if path_matrix_all[min_index][i][j] == 1:
                pheromone_matrix[i][j] *= q

    max_pheromone_matrix = []
    critical_point_matrix = []

    for i in range(len(tasks)):
        max_pheromone = pheromone_matrix[i][0]
        max_index = 0
        sum_pheromone = pheromone_matrix[i][0]
        is_all_same = True

        for j in range(len(nodes)):
            if pheromone_matrix[i][j] > max_pheromone:
                max_pheromone = pheromone_matrix[i][j]
                max_index = j

            if pheromone_matrix[i][j] != pheromone_matrix[i][j-1]:
                is_all_same = False

            sum_pheromone += pheromone_matrix[i][j]

        # 若本行信息素全都相等，则随机选择一个作为最大信息素
        if is_all_same:
            max_index = random.randint(0, len(nodes) - 1)
            max_pheromone = pheromone_matrix[i][max_index]

        # 将本行最大信息素的下标加入max_pheromone_matrix
        max_pheromone_matrix.append(max_index)

        # 将本次迭代的蚂蚁临界编号加入critical_point_matrix
        # (该临界点之前的蚂蚁的任务分配根据最大信息素原则，而该临界点之后的蚂蚁采用随机分配策略)
        critical_point_matrix.append(round(ant_num * (max_pheromone / sum_pheromone)))

    # print critical_point_matrix

def aca_search(iterator_num, ant_num, tasks, nodes, pheromone_matrix, time_matrix, p, q):
    result_data = []
    for itCount in range(iterator_num):
        # 本次迭代中，所有蚂蚁的路径
        path_matrix_all = []

        for antCount in range(ant_num):
            # 第antCount只蚂蚁的分配策略(pathMatrix[i][j]
            # 表示第antCount只蚂蚁将i任务分配给j节点处理)
            path_matrix_oneAnt = init_matrix(tasks, nodes, 0)
            for taskCount in range(len(tasks)):
                # 将第taskCount个任务分配给第nodeCount个节点处理
                nodeCount = assign_one_task(antCount, taskCount, nodes, critical_point_matrix, max_pheromone_matrix, pheromone_matrix, time_matrix)
                path_matrix_oneAnt[taskCount][nodeCount] = 1
            # 将当前蚂蚁的路径加入pathMatrix_allAnt
            path_matrix_all.append(path_matrix_oneAnt)

        # 计算 本次迭代中 所有蚂蚁 的任务处理时间
        time_arr_oneIt = cal_time_oneIt(path_matrix_all, tasks, nodes, time_matrix)
        # 将本地迭代中 所有蚂蚁的 任务处理时间加入总结果集，
        # 它是一个二维矩阵，result_data[x][y]=10代表第x次迭代中第y只蚂蚁的任务处理时间是10
        result_data.append(time_arr_oneIt)

        # 更新信息素
        update_pheromone_matrix(path_matrix_all, pheromone_matrix, time_arr_oneIt, tasks, nodes, p, q, ant_num)

    return result_data

def aca(iterator_num, ant_num, tasks, nodes, pheromone_matrix, time_matrix, p, q):
    time_matrix = init_time_matrix(tasks, nodes)

    pheromone_matrix = init_pheromone_matrix(tasks, nodes)

    return aca_search(iterator_num, ant_num, tasks, nodes, pheromone_matrix, time_matrix, p, q)

# critical_point_matrix = init_array(len(tasks), len(nodes) / 2)
# max_pheromone_matrix = init_array(len(tasks), 0)
for i in range(len(tasks)):
    critical_point_matrix.append(random.randint(0, len(nodes) - 1))
for i in range(len(tasks)):
    max_pheromone_matrix.append(random.randint(0, len(nodes) - 1))
result_data = aca(iterator_num, ant_num, tasks, nodes, pheromone_matrix, time_matrix, p, q)
# print result_data
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