import tensorflow as tf
import numpy as np
from weiyun.utils.log_utils import logger

class DeepQNetwork:

    def __init__(
            self,
            n_actions,
            n_features,
            env,
            learning_rate = 0.01,
            reward_decay = 0.9,
            e_greedy = 0.9,
            replace_target_iter = 300,
            memory_size = 500,
            batch_size = 32,
            e_greedy_increment = None,
            output_graph = False,
            training = True
    ):
        self.env = env
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0.6 if e_greedy_increment is not None else self.epsilon_max
        self.training = training



        self.learn_step_counter = 0

        self.memory = np.zeros((self.memory_size, n_features * 2 + 2))

        self._build_net()

        # 替换 target net 的参数
        t_params = tf.get_collection('target_net_params')   # 提取 target net 的参数
        e_params = tf.get_collection('eval_net_params')  # 提取 eval net 的参数
        self.replace_target_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]  # 更新 target net 的参数

        self.sess = tf.Session()

        # 输出 tensorboard 文件
        if output_graph:
                # $tensorboard --logdir=logs
            tf.summary.FileWriter('logs/', self.sess.graph)

        self.sess.run(tf.global_variables_initializer())
        self.cost_his = []

        self.saver = tf.train.Saver()
        if not self.training:
            self.restore_model()

    def _build_net(self):
        # observation
        self.s = tf.placeholder(tf.float32, [None, self.n_features], name='s')
        self.q_target = tf.placeholder(tf.float32, [None, self.n_actions], name='Q_target')

        with tf.variable_scope('eval_net'):
            c_names, n_l1, n_l2, w_initializer, b_initializer = \
            ['eval_net_params', tf.GraphKeys.GLOBAL_VARIABLES], 30, 50, \
            tf.random_normal_initializer(0, 0.3), tf.constant_initializer(0.1)

            # eval_net 的第一层，collections 在更新 target_net 参数时会用到
            with tf.variable_scope('l1'):
                w1 = tf.get_variable('w1', [self.n_features, n_l1], initializer=w_initializer, collections=c_names)
                b1 = tf.get_variable('b1', [1, n_l1], initializer=b_initializer, collections=c_names)
                l1 = tf.nn.relu(tf.matmul(self.s, w1) + b1)

            with tf.variable_scope('l2'):
                w2 = tf.get_variable('w2', [n_l1, n_l2], initializer=w_initializer, collections=c_names)
                b2 = tf.get_variable('b2', [1, n_l2], initializer=b_initializer, collections=c_names)
                l2 = tf.nn.relu(tf.matmul(l1, w2) + b2)

            with tf.variable_scope('l3'):
                w3 = tf.get_variable('w3', [n_l2, self.n_actions], initializer=w_initializer, collections=c_names)
                b3 = tf.get_variable('b3', [1, self.n_actions], initializer=b_initializer, collections=c_names)
                self.q_eval = tf.matmul(l2, w3) + b3

        with tf.variable_scope('loss'):
            self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval))
        with tf.variable_scope('train'):
            self._train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)

        # 接收下一个 observation
        self.s_ = tf.placeholder(tf.float32, [None, self.n_features], name='s_')
        with tf.variable_scope('target_net'):
            c_names = ['target_net_params', tf.GraphKeys.GLOBAL_VARIABLES]

            with tf.variable_scope('l1'):
                w1 = tf.get_variable('w1', [self.n_features, n_l1], initializer=w_initializer, collections=c_names)
                b1 = tf.get_variable('b1', [1, n_l1], initializer=b_initializer, collections=c_names)
                l1 = tf.nn.relu(tf.matmul(self.s_, w1) + b1)

            with tf.variable_scope('l2'):
                w2 = tf.get_variable('w2', [n_l1, n_l2], initializer=w_initializer, collections=c_names)
                b2 = tf.get_variable('b2', [1, n_l2], initializer=b_initializer, collections=c_names)
                l2 = tf.nn.relu(tf.matmul(l1, w2) + b2)

            with tf.variable_scope('l3'):
                w3 = tf.get_variable('w3', [n_l2, self.n_actions], initializer=w_initializer, collections=c_names)
                b3 = tf.get_variable('b3', [1, self.n_actions], initializer=b_initializer, collections=c_names)
                self.q_next = tf.matmul(l2, w3) + b3

    def store_transition(self, s, a, r, s_):
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0

        # 记录一条[s, a, r, s_]记录
        transition = np.hstack([s, [a, r], s_])

        # 总 memory 大小是固定的，如果超出总大小，旧 memory 被新 memory 替换
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition

        self.memory_counter += 1

    def choose_action(self, observation):
        # observation 的 shape (1, size_of_observation)
        observation = np.array(observation)
        observation = observation[np.newaxis, :]
        # print(observation)

        if self.training:
            if np.random.uniform() < self.epsilon:
                # 让 eval_net 神经网络生成所有 action 的值，并选择值最大的 action
                action_value = self.sess.run(self.q_eval, feed_dict={self.s : observation})
                # print('observation : {%s}, action_value : %s' % (observation, action_value))
                action = np.argmax(action_value)
            else:
                action = np.random.randint(0, self.n_actions)
        else:
            # test 模型时不探索
            action_value = self.sess.run(self.q_eval, feed_dict={self.s: observation})
            # print('observation : {%s}, action_value : %s' % (observation, action_value))
            action = np.argmax(action_value)
        return action

    def plot_cost(self):
        import matplotlib.pyplot as plt
        plt.plot(np.arange(len(self.cost_his)), self.cost_his)
        plt.ylabel('Cost')
        plt.xlabel('training steps')
        plt.show()

    def learn(self):
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.sess.run(self.replace_target_op)
            print('\ntarget_params_replaced\n')

        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
        batch_memory = self.memory[sample_index, :]
        # if self.env.memory_counter > self.env.experience_pool_size:
        #     # rewards = [_[8] for _ in self.env.experience_pool]
        #     # s = sum(rewards)
        #     # sample_index = np.random.choice(self.env.experience_pool_size, size=self.batch_size, p=[(_ / s) for _ in rewards])
        #     sample_index = np.random.choice(self.env.experience_pool_size, size=self.batch_size)
        # else:
        #     # rewards = [_[8] for _ in self.env.experience_pool[:self.env.memory_counter]]
        #     # s = sum(rewards)
        #     # sample_index = np.random.choice(self.env.memory_counter, size=self.batch_size, p=[(_ / s) for _ in rewards])
        #     sample_index = np.random.choice(self.env.memory_counter, size=self.batch_size)
        # experiences = [np.hstack(e['experience']) for e in self.env.experience_pool[sample_index]]
        # for e in self.env.experience_pool[sample_index]:
        #     experiences.append(np.hstack(e['experience']))
        # batch_memory = self.env.experience_pool[sample_index, :]

        q_next, q_eval = self.sess.run(
            [self.q_next, self.q_eval],
            feed_dict={
                self.s_: batch_memory[:, -self.n_features:],
                self.s: batch_memory[:, :self.n_features]
            }
        )

        q_target = q_eval.copy()
        batch_index = np.arange(self.batch_size, dtype=np.int32)
        eval_act_index = batch_memory[:, self.n_features].astype(int)
        reward = batch_memory[:, self.n_features + 1]
        # print(reward)

        q_target[batch_index, eval_act_index] = reward + self.gamma * np.max(q_next, axis=1)

        _, self.cost = self.sess.run(
            [self._train_op, self.loss],
            feed_dict={
                self.s: batch_memory[:, :self.n_features],
                self.q_target: q_target
            }
        )
        self.cost_his.append(self.cost)

        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1

    def store_model(self):
        self.saver.save(self.sess, 'my_net/save_net.ckpt')
        logger.info('store model to my_net/save_net.ckpt')
        # s = np.array([[12.0, 31.0, 15.0, 5.0]])
        # action_value = self.sess.run(self.q_eval, feed_dict={self.s: s})
        # print(action_value)

    def restore_model(self):
        self.saver.restore(self.sess, 'my_net/save_net.ckpt')
        logger.info('restore model from my_net/save_net.ckpt')
        # s = np.array([[12.0, 31.0, 15.0, 5.0]])
        # action_value = self.sess.run(self.q_eval, feed_dict={self.s: s})
        # print(action_value)

    def take_a_step(self, observation, action):
        n_core_left = observation[1]
        n_bandwidth_left = observation[2]

        while n_core_left < action[0] or n_bandwidth_left < action[1]:
            n_core_left = self.env.n_core_left
            n_bandwidth_left = self.env.n_bandwidth_left

        return self.env.step(action)

