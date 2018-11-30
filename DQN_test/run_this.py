from DQN_test.maze_env import Maze
from DQN_test.DQN_brain import DeepQNetwork
import matplotlib.pyplot as plt

def run_maze():
    step = 0

    # fig = plt.figure()
    # ax = fig.add_subplot(1, 1, 1)
    # plt.ion()
    # plt.show()

    trs = []
    for episode in range(300):
        observation = env.reset()

        tr = 0
        while True:
            env.render()

            action = RL.choose_action(observation)

            observation_, reward, done = env.step(action)
            tr += reward

            RL.store_transition(observation, action, reward, observation_)

            if step > 200 and step % 5 == 0:
                RL.learn()

            observation = observation_

            # if done:
            #     trs.append(tr)
            #     try:
            #         ax.lines.remove(lines[0])
            #     except Exception:
            #         pass
            #     lines = ax.plot([_ for _ in range(len(trs))], [_ for _ in trs], 'r-',
            #                     lw=5)
            #     plt.pause(0.0001)
            #     break
            if done:
                break

            step += 1

    RL.plot_cost()

    RL.store_model()
    print('game over')
    env.destroy()

if __name__ == '__main__':
    env = Maze()

    RL = DeepQNetwork(env.n_actions, env.n_features, env=env,
                      learning_rate=0.01,
                      reward_decay=0.9,
                      e_greedy=0.9,
                      replace_target_iter=200,
                      memory_size=2000,
                      output_graph=True,
                      training=True)
    env.after(100, run_maze)
    env.mainloop()
    RL.plot_cost()
