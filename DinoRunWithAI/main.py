from Game import *
from Agent import *
from GameState import *
from Agents.dueling_dqn import DuelingDQNAgent
from Agents.dqqn import  DDQNAgent
from copy import deepcopy

FILE_NAME = 'stat.txt'
#game parameters
ACTIONS = 3  # possible actions: jump, do nothing
GAMMA = 0.99  # decay rate of past observations original 0.99
OBSERVATION = 100.  # timesteps to observe before training
EXPLORE = 100000  # frames over which to anneal epsilon
FINAL_EPSILON = 0.0001  # final value of epsilon
INITIAL_EPSILON = 0.1  # starting value of epsilon
REPLAY_MEMORY = 50000  # number of previous transitions to remember
BATCH = 16  # size of minibatch
FRAME_PER_ACTION = 1
LEARNING_RATE = 1e-4
img_rows, img_cols = 80, 80
img_channels = 4  #We stack 4 frames
GAMES = 10000
EPISODES = 10000
INPUT_SHAPE = (80, 80, 4)
SAVE_MODEL = True
LOAD_MODEL = True


#main function
def playGame(observe=False):
    game = Game()
    dino = DinoAgent(game)
    game_state = Game_state(dino, game)
    agent = DDQNAgent(ACTIONS, INPUT_SHAPE)
    if LOAD_MODEL:
        agent.load()

    for cur_game in range(GAMES):
        state, reward, _ = game_state.get_state([0, 0, 0])
        big_state = np.append(state, state[:, :, :], axis=2)
        big_state = np.append(big_state, state[:, :, :], axis=2)
        big_state = np.append(big_state, state[:, :, :], axis=2)
        sum_reward = 0
        total_reward = 0
        for episode in range(EPISODES):
            action = agent.act(np.array(big_state))
            next_state, reward, done = game_state.get_state(action)
            next_big_state = np.append(next_state, big_state[:, :, :3], axis=2)
            if action[0] != 1:
                reward -= 0.05
            if done:
                total_reward = reward
            else:
                total_reward += reward
            agent.remember(np.array(big_state), action, reward, np.array(next_big_state), done)
            big_state = next_big_state
            state = next_state
            sum_reward += reward
            if episode % 80 == 0 and not(episode == 0):
                agent.update_target_model()
            if len(agent.memory) > BATCH:
                game.pause()
                agent.replay(BATCH)
                game.resume()
            if done:
                break

        game.pause()
        print("game - %s, s_r - %s, episodes - %s, eps - %s" % (str(cur_game), str(sum_reward), str(episode), str(agent.epsilon)))
        for i in range(10):
            agent.replay(BATCH)
        if SAVE_MODEL:
            agent.save()
        game.restart()

    pass
    game.end()


def save_game(game, sum_reward, episodes):
    pass


def init_cache():
    """initial variable caching, done only once"""
    save_obj(INITIAL_EPSILON, "epsilon")
    t = 0
    save_obj(t, "time")
    D = deque()
    save_obj(D, "D")


if __name__ == "__main__":
    # init_cache()
    playGame()

