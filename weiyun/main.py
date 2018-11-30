from weiyun.brain.greedy_brain import GreedyPolicy
from weiyun.mic_cloud.online_env import OnlineWyEnv
from weiyun.utils.log_utils import logger
import csv
import pickle
import matplotlib.pyplot as plt

def run_env(env, agent):
    step = 0

    observation = env.reset()

    done = False

    # fig = plt.figure()
    # ax = fig.add_subplot(1, 1, 1)
    # plt.ion()
    # plt.show()

    for episode in range(1000):

        # observation = env.reset()WWWWWWWW

        # action = agent.choose_action(observation)
        #
        # observation_, reward, done = env.step(action)

        # RL.store_transition(observation, action, reward, observation_)
        #
        # if step > 200 and step % 5 == 0:
        #     RL.learn()

        observation_, reward, done = agent.take_a_step(observation)

        observation = observation_

        # if step % 150 == 0:
        #     try:
        #         ax.lines.remove(lines[0])
        #     except Exception:
        #         pass
        #     lines = ax.plot([_ for _ in range(len(env.plt_record))], [_[0] for _ in env.plt_record], 'r-', lw=5)
        #     plt.pause(0.0001)

        step += 1

    env.stop_generate_user()

    cp_time_record = [_[0] for _ in env.plt_record]
    queue_time_record = [_[1] for _ in env.plt_record]
    logger.info('average time : %f, queue_time : %f' % (sum(cp_time_record) / len(cp_time_record), sum(queue_time_record) / len(queue_time_record)))

    while not done:
        observation_, reward, done = agent.take_a_step(observation)

        observation = observation_

    # print(env.get_experiences())
    # print(len(env.get_experiences()))

    # experience_pool = env.experience_pool
    # wr = csv.writer(open('experience_pool', 'wb'), quoting=csv.QUOTE_ALL)
    # for ep in experience_pool:
    #     wr.writerow([ep['run_time'], ep['next_user_queue_time'], ep['reward']])

    # total_time = 0
    # for e in env.experience_pool:
    #     total_time += e['reward']
    # logger.info('total generate %d users, average time : %f' % (len(env.experience_pool), total_time / len(env.experience_pool) * 1.0))

if __name__ == '__main__':
    graphs = []
    with open('graph_file', 'rb') as fp:
        graphs = pickle.load(fp)

    env = OnlineWyEnv(graphs=graphs)
    agent = GreedyPolicy(env)
    run_env(env, agent)