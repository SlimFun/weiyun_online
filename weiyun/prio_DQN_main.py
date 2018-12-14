"""
The DQN improvement: Prioritized Experience Replay (based on https://arxiv.org/abs/1511.05952)

View more on my tutorial page: https://morvanzhou.github.io/tutorials/

Using:
Tensorflow: 1.0
gym: 0.8.0
"""


from weiyun.mic_cloud.online_env import OnlineWyEnv
from weiyun.brain.prio_DQN_brain import DQNPrioritizedReplay
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import pickle
from weiyun.utils.log_utils import logger

# env = gym.make('MountainCar-v0')
# env = env.unwrapped
# env.seed(21)


plt_flag = False
TRAINING = True
MEMORY_SIZE = 200

def trans_action(action):
    print('return action : %d' % int(action))
    return int(action / 10) + 1, int(action % 10) + 1

def summarize_experience(experience_pool):
    actions = [_[3] for _ in experience_pool]
    rewards = [_[4] for _ in experience_pool]
    acd = {}
    for i in range(len(actions)):
        a = actions[i]
        if a not in acd.keys():
            acd[a] = [0, 0]
        acd[a][0] += rewards[i]
        acd[a][1] += 1
    for a in acd.keys():
        acd[a].append(acd[a][0] / acd[a][1])
    for a in acd.keys():
        print('%s : %s' % (a, acd[a]))

def run_env(env, agent):
    step = 0
    observation = env.reset()
    done = False
    if plt_flag:
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        plt.ion()
        plt.show()

    for episode in range(5000):
        action = trans_action(agent.choose_action(observation))

        observation_, reward, done = agent.take_a_step(observation, action)

        if TRAINING and step > MEMORY_SIZE and step % 50 == 0:
            agent.learn()

        observation = observation_

        if step % 150 == 0 and plt_flag:
            try:
                ax.lines.remove(lines[0])
            except Exception:
                pass
            lines = ax.plot([_ for _ in range(len(env.plt_record))], [10 - _[0] for _ in env.plt_record], 'r-', lw=5)
            plt.pause(0.0001)

        step += 1

        if len(env.queue) > 20 and TRAINING:
            env.shutdown_generate_user()

            while not done:
                action = trans_action(agent.choose_action(observation))
                observation_, reward, done = agent.take_a_step(observation, action)
                observation = observation_

            observation = env.reset()
            print('env reset, thread id : %d' % env.thread_id)

    # env.stop_generate_user()
    env.shutdown_generate_user()

    # if plt_flag:
    #     agent.plot_cost()

    cp_time_record = [_[0] for _ in env.plt_record]
    queue_time_record = [_[1] for _ in env.plt_record]
    logger.info('average time : %f, queue_time : %f' % (sum(cp_time_record) / len(cp_time_record), sum(queue_time_record) / len(queue_time_record)))

    if TRAINING:
        agent.store_model()
        summarize_experience(env.experience_pool)

    while not done:

        action = trans_action(agent.choose_action(observation))

        observation_, reward, done = agent.take_a_step(observation, action)

        observation = observation_

        step += 1

    print('game over')


if __name__ == '__main__':
    graphs = []
    with open('graph_file', 'rb') as fp:
        graphs = pickle.load(fp)

    env = OnlineWyEnv(graphs=graphs, prioritized=True, memory_size=MEMORY_SIZE)

    # sess = tf.Session()
    with tf.variable_scope('DQN_with_prioritized_replay'):
        agent = DQNPrioritizedReplay(env.n_actions, env.n_features, env,
                                     memory_size=MEMORY_SIZE,
                                     learning_rate=0.1,
                                     reward_decay=0.9,
                                     e_greedy=0.9,
                                     replace_target_iter=20,
                                     e_greedy_increment=0.3 / 100,
                                     # sess=sess,
                                     prioritized=True,
                                     output_graph=True,
                                     training=TRAINING
        )
    # sess.run(tf.global_variables_initializer())

    run_env(env, agent)
