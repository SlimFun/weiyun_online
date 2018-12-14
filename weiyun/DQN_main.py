from weiyun.mic_cloud.online_env import OnlineWyEnv
from weiyun.brain.DQN_brain import DeepQNetwork
import matplotlib.pyplot as plt
import threading
import pickle
from weiyun.utils.log_utils import logger
import time

plt_flag = False

TRAINING = True

# return assign_n_cores, assign_n_bandwidths
def trans_action(action):
    # print('action')
    # print(type(action))
    print('return action : %d' % int(action))
    return int(action / 10) + 1, int(action % 10) + 1

# def plot_record(env):
#     fig = plt.figure()
#     ax = fig.add_subplot(1, 1, 1)
#     plt.ion()
#     plt.show()
#     while plt_flag:
#         # if len(env.plt_record) == 0:
#         #     continue
#         try:
#             ax.lines.remove(lines[0])
#         except Exception:
#             pass
#         # print('-----------plt : ')
#         # print(env.plt_record)
#         lines = ax.plot([_ for _ in range(len(env.plt_record))], env.plt_record, 'r-', lw=5)
#         plt.pause(1)

def run_env(env, agent):
    step = 0
    observation = env.reset()
    done = False
    # plt_thread = threading.Thread(target=plot_ record, args=(env, ))
    # plt_thread.start()
    if plt_flag:
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        plt.ion()
        plt.show()

    for episode in range(1000):
        action = trans_action(agent.choose_action(observation))

        observation_, reward, done = agent.take_a_step(observation, action)

        # if done:
        #     print('env reset')
        #     observation = env.reset()
        #     continue

        # agent.store_transition(observation, action, reward, observation_)

        if TRAINING and step > 50 and step % 50 == 0:
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

        if (len(env.queue) > 50 or env.processed_user >= 500) and TRAINING:
            # env.stop_generate_user()
            env.shutdown_generate_user()

            while not done:
                action = trans_action(agent.choose_action(observation))
                observation_, reward, done = agent.take_a_step(observation, action)
                observation = observation_

            observation = env.reset()
            print('env reset, thread id : %d' % env.thread_id)

    env.stop_generate_user()

    agent.plot_cost()

    cp_time_record = [_[0] for _ in env.plt_record]
    queue_time_record = [_[1] for _ in env.plt_record]
    logger.info('average time : %f, queue_time : %f' % (sum(cp_time_record) / len(cp_time_record), sum(queue_time_record) / len(queue_time_record)))

    if TRAINING:
        agent.store_model()
        summarize_experience(env.experience_pool)

    while not done:

        action = trans_action(agent.choose_action(observation))

        observation_, reward, done = agent.take_a_step(observation, action)

        # agent.store_transition(observation, action, reward, observation_)

        # if step > 200 and step % 5 == 0:
        #     agent.learn()

        observation = observation_

        step += 1

    print('game over')

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
    # print(acd)
    for a in acd.keys():
        print('%s : %s' % (a, acd[a]))

if __name__ == '__main__':
    graphs = []
    with open('graph_file', 'rb') as fp:
        graphs = pickle.load(fp)

    env = OnlineWyEnv(graphs=graphs)

    # print('env.n_actions : %d' % env.n_actions)
    agent = DeepQNetwork(env.n_actions, env.n_features,
                         env,
                      learning_rate=0.01,
                      reward_decay=0.9,
                      e_greedy=0.9,
                      replace_target_iter=200,
                      memory_size=2000,
                      output_graph=True,
                         # e_greedy_increment=0.35 / 1000,
                       training=TRAINING)
    run_env(env, agent)
    # agent.plot_cost()
