import numpy as np
# import weiyun.mic_cloud.trans as trans
# import weiyun.mic_cloud.cloud as cloud
from weiyun.mic_cloud.cloud import Cloud
from weiyun.mic_cloud.trans import Trans
import weiyun.utils.graph_utils as graph_utils
import weiyun.mic_cloud.rp_time as rp_time
import time, threading
import math, random
from weiyun.utils.log_utils import logger
from weiyun.brain.prio_DQN_brain import Memory
import matplotlib.pyplot as plt

MAX_CORE = 10
MAX_BANDWIDTH = 10

TOTAL_CORE = 40
TOTAL_BANDWIDTH = 60

TOTAL_GRAPHS = 50
TOTAL_NODE = 15

POOL_SIZE = 2000

POISSON_RATE = 11.0 / 1
MAX_QUEUE_SIZE = 50

class User:
    # 对于目前阶段需要传入 graph_id，用 graph_id 的 one-hot 编码来产生 graph_vex
    # 下一阶段需要直接通过 graph 产生 graph_vex，届时则可不传入 graph_id
    # graph : [ad_matrix, wvs, topo_order]
    def __init__(self, user_id, graph, graph_id=None):
        self.graph = graph
        self.generate_time = time.time()
        self.start_excu_time = None
        self.graph_id = graph_id
        self.user_id = user_id
        # graph_id 为空，应该用 graph2vec 处理
        self.graph_vex = self.graph_id2vec(graph_id) if graph_id is not None else None
        self.assign_n_core = 0
        self.assign_n_bandwidth = 0
        self.queue_time = None
        self.state = None

    def graph_id2vec(self, graph_id):
        pass

class OnlineWyEnv:
    def __init__(self, graphs=None, prioritized = False, memory_size = POOL_SIZE):
        self.action_space = [_ for _ in range(MAX_CORE * MAX_BANDWIDTH)]
        self.n_actions = len(self.action_space)
        self.n_features = None

        # queue = [user1_id, user2_id, ...]
        self.queue = []

        self.n_core_left = TOTAL_CORE
        self.n_bandwidth_left = TOTAL_BANDWIDTH
        self.observation = None

        # 应用库
        # graphs 格式： [[ad_matrix, wvs, topo_order], [...], ...]
        self.graphs = self._generate_graphs() if graphs is None else graphs

        self.total_graphs = TOTAL_GRAPHS

        self.prioritized = prioritized

        # 环境中存在的用户
        # users : {
        #   id : user
        # }
        # user : {
        #   graph : [ad_matrix, wvs, topo_order],
        #   generate_time,
        #   start_excu_time,
        # }
        if self.prioritized:
            self.memory = Memory(capacity=memory_size)

        self.users = {}
        self.is_stop_generate_user = True

        self.n_actions = MAX_CORE * MAX_BANDWIDTH
        # n_feature = len(state)
        self.n_features = 3
        self.experience_pool_size = memory_size
        self.experience_pool = np.zeros((self.experience_pool_size, self.n_features * 2 + 2))
        self.transitions = [0 for _ in range(self.experience_pool_size)]
        self.generate_users_thread = None

        self.plt_record = []
        self.thread_id = 0

        self.out = True

        # self.shutdown_flag = 0
        self.running_users = []

    def start_generate_user(self):
        # 如果 thread 没有开始，才开启 thread
        # 需要等待前一个 generate 线程彻底结束
        while not self.out:
            pass
        if self.out:
            self.thread_id += 1
            self.generate_users_thread = threading.Thread(target=self._generate_user, args=(self.thread_id, ))
            self.is_stop_generate_user = False
            self.generate_users_thread.start()

    def stop_generate_user(self):
        # self.generate_users_thread.join()
        self.is_stop_generate_user = True

    def shutdown_generate_user(self):
        self.queue = [self._choose_user_from_queue()]
        logger.info('shutdown_env : %s' % self.queue)
        self.stop_generate_user()

    def _generate_user(self, thread_id):
        id = 0
        self.out = False
        while not self.is_stop_generate_user:
            nt = self._next_time(POISSON_RATE)
            time.sleep(nt)
            if self.is_stop_generate_user:
                break
            # print('thread id : %d, shutdown flag : %d' % (thread_id, self.shutdown_flag))
            # if thread_id == self.shutdown_flag:
            #     break
            self._assert_graphs()
            graph_id = random.randint(0, len(self.graphs)-1)
            user = User(user_id=id, graph=self.graphs[graph_id], graph_id=graph_id)
            self.users[user.user_id] = user
            self.queue.append(user.user_id)
            logger.info('thread id : %d , generate user : %d; queue : %s' % (thread_id, id, self.queue))
            id += 1
        self.out = True

    def _assert_graphs(self):
        self.graphs = self._generate_graphs() if self.graphs is None else self.graphs

    def _next_time(self, rateParameter):
        return -math.log(1.0 - random.random()) / rateParameter

    def _generate_graphs(self):
        graphs = []
        for i in range(TOTAL_GRAPHS):
            adj_matrix, w_vex, topo_order = rp_time.generate_DAG(TOTAL_NODE)
            graphs.append([adj_matrix, w_vex, topo_order])
        return graphs

    def _choose_user_from_queue(self):
        # 若队列中没有用户，挂起程序
        while len(self.queue) == 0:
            pass
        return self.queue[0]

    def _choose_next_user_from_queue(self):
        return self.queue[1] if len(self.queue) >= 2 else None

    # 返回 observation
    # observation 格式：[user, n_core_left, n_bandwidth_left]
    def reset(self):
        while len(self.running_users) != 0:
            pass
        self.running_users = []
        self.n_core_left = TOTAL_CORE
        self.n_bandwidth_left = TOTAL_BANDWIDTH
        self.queue = []
        # self.users = {}

        self.start_generate_user()

        user_id = self._choose_user_from_queue()
        # user = self.users[user_id]

        return [self.users[user_id].graph_id, self.n_core_left, self.n_bandwidth_left]

    def _computation_time(self, user, graph):
        # cloud.CORE_NUM = user.assign_n_core
        # trans.BAND_WIDTH = user.assign_n_bandwidth
        cloud = Cloud(core_num=user.assign_n_core)
        trans = Trans(cloud, band_width=user.assign_n_bandwidth)
        print('user n_core : %d, n_bandwidth : %d' % (user.assign_n_core, user.assign_n_bandwidth))
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

    # 首先确定分配给用户 graph_id 的资源后，计算该用户所需的计算时间，
    # 然后开启一个线程计时，计时结束更新环境资源数量，并 pop 等待队列
    def _start_process_user(self, user, s, action, next_user_id, queue_len, done):
        user.queue_time = time.time() - user.generate_time
        graph = self.graphs[user.graph_id]

        cp_time = self._computation_time(user, graph)

        self.queue.pop(0)
        logger.info('start process user : %d, cp_time : %f, queue : %s ; action : <%d, %d>' % (user.user_id, cp_time, self.queue, action[0], action[1]))

        t = threading.Thread(target=self._process_user, args=(user, cp_time, s, action, next_user_id, queue_len, done))
        self.running_users.append(user.user_id)
        t.start()

    def _process_user(self, user, cp_time, s, action, next_user_id, queue_len, done):

        # 从 sleep 中醒来后发现 env 已经 reset 了，直接返回，不再执行该 user_thread
        # if user.user_id not in self.running_users:
        #     return
        # self.running_users.remove(user.user_id)
        cloud = Cloud(core_num=user.assign_n_core)
        trans = Trans(cloud, band_width=user.assign_n_bandwidth)
        cp_len, last = rp_time.ours(user.graph[0], user.graph[1], user.graph[2], cloud, trans)

        # is_done = len(self.queue) == 0 and self.is_stop_generate_user

        if not done:
            # 这里貌似还会有问题？？？？当进入循环等待后，shutdown env 会使得此处产生死循环
            # while (next_user_id is None) and (not self.is_stop_generate_user):
            if (next_user_id is None) and (not self.is_stop_generate_user):
                next_user_id = self._choose_user_from_queue()
            # 更新当前 experience 的时机：当前 user 执行完毕（得到执行时间）并且下一个 user 开始执行（得到下一个 user 的排队时间）
            # 等待下一个 user 开始执行
            while next_user_id != -1 and (self.users[next_user_id].queue_time is None):
                pass
            print("next_user_id : %d" % next_user_id)
            if next_user_id == -1:
                self.store_transition(done, user, s, cp_len, 0,
                                  user.state,
                                  queue_len, a=action)
            else:
                next_user_queue_time = self.users[next_user_id].queue_time
                self.store_transition(done, user, s, cp_len, next_user_queue_time, self.users[next_user_id].state, queue_len, a=action)

            time.sleep(cp_time)
            self._release_user_resources(user)
        else:
            self.store_transition(done, user, s, cp_len, 0,
                                  user.state,
                                  queue_len, a=action)
            # self.store_transition(done, user, s, cp_len, 0, [user.state[0] - user.assign_n_core, user.state[1] - user.assign_n_bandwidth], queue_len, a=action)
        logger.info('end process user : %d, n_core_left : %f, n_bandwidth_left : %f, queue : %s' % (user.user_id, self.n_core_left, self.n_bandwidth_left, self.queue))
        self.running_users.remove(user.user_id)

    def _store_transition_without_prioritized(self, experience):
        # 总 memory 大小是固定的，如果超出总大小，旧 memory 被新 memory 替换
        index = self.memory_counter % self.experience_pool_size

        self.experience_pool[index, :] = experience

    def _store_transition_with_prioritized(self, experience):
        self.memory.store(experience)  # have high priority for newly arrived transition

    def store_transition(self, done, user, s, run_time, queue_time, s_, queue_len, a):
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0

        punish = 300 - user.user_id if done else 0
        if done:
            print('punish : %d' % punish)
        # experience = np.hstack([s, (a[0] - 1) * 10.0 + (a[1] - 1), 300.0 - (20 * run_time) - 10 * punish, s_])
        experience = np.hstack([s, (a[0] - 1) * 10.0 + (a[1] - 1), 300.0 - (20 * run_time), s_])
        if np.shape(experience)[0] == 6:
            raise TypeError('experience : %s, user.id : %d, state : %s' % (experience, user.user_id, user.state))
        if self.prioritized:
            self._store_transition_with_prioritized(experience)
        else:
            self._store_transition_without_prioritized(experience)
        # logger.info('user : %d, action : <%d, %d>, time : %f + %f = %f' % (user.user_id, user.assign_n_core, user.assign_n_bandwidth, run_time, queue_time, run_time + queue_time))

        self.memory_counter += 1

        index = self.memory_counter % self.experience_pool_size
        self.transitions[index] = {
            'user': user,
            'user_id': user.user_id,
            'run_time': run_time,
            # 'next_user_queue_time': queue_time
        }

        if self.memory_counter % 100 == 0:
            total_time = 0
            total_users = 0
            total_queue_time = 0
            for e in self.transitions:
                if e != 0:
                    total_time += e['run_time'] + e['user'].queue_time
                    total_queue_time += e['user'].queue_time
                    total_users += 1
            logger.info('epoch %d : average time : %f , queue time occupy %f' % (
            int(self.memory_counter / 300), total_time / total_users * 1.0, total_queue_time / total_time))
            self.plt_record.append([total_time / total_users * 1.0, total_queue_time / total_users * 1.0])

    def _release_user_resources(self, user):
        self.n_core_left += user.assign_n_core
        self.n_bandwidth_left += user.assign_n_bandwidth

    # action 格式： [assign_n_core, assign_n_bandwidth]
    # return [observation, reward, done]
    def step(self, action):
        assign_n_core, assign_n_bandwidth = action
        user_id = self._choose_user_from_queue()
        next_user_id = self._choose_next_user_from_queue()

        # graph_vec = self._choose_user_from_queue()
        logger.info('choose user %d from queue' % user_id)
        user = self.users[user_id]
        s = [float(user.graph_id), float(self.n_core_left), float(self.n_bandwidth_left)]
        self.n_core_left -= assign_n_core
        self.n_bandwidth_left -= assign_n_bandwidth
        user.state = s
        # graph_id = np.argmax(graph_vec)
        user.assign_n_core = assign_n_core
        user.assign_n_bandwidth = assign_n_bandwidth

        logger.info('assign user n_core : %f, n_bandwidth : %f; n_core_left : %f, n_bandwidth_left : %f' % (assign_n_core, assign_n_bandwidth, self.n_core_left, self.n_bandwidth_left))
        done = self.is_stop_generate_user
        self._start_process_user(user, s, action, next_user_id, len(self.queue), done)

        # is_done = (len(self.queue) == 0) and self.is_stop_generate_user

        next_user_id = -1
        if not done:
            next_user_id = self._choose_user_from_queue()
        # if not is_done or not self.is_stop_generate_user:
        #     next_user_id = self._choose_user_from_queue()
        # if len(self.queue) > MAX_QUEUE_SIZE:
        #     is_done = True
        return [self.users[next_user_id].graph_id if next_user_id != -1 else None, self.n_core_left, self.n_bandwidth_left], None, done