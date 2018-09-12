import Agent
from util import *


loss_df = pd.read_csv(loss_file_path) if os.path.isfile(loss_file_path) else pd.DataFrame(columns=['loss'])
scores_df = pd.read_csv(scores_file_path) if os.path.isfile(loss_file_path) else pd.DataFrame(columns=['scores'])
actions_df = pd.read_csv(actions_file_path) if os.path.isfile(actions_file_path) else pd.DataFrame(columns=['actions'])
q_values_df = pd.read_csv(actions_file_path) if os.path.isfile(q_value_file_path) else pd.DataFrame(columns=['qvalues'])


class Game_state:
    def __init__(self, agent: Agent, game):
        self._agent = agent
        self._game = game
        self._display = show_img()  #display the processed image on screen using openCV, implemented using python coroutine
        self._display.__next__()  # initiliaze the display coroutine

    def get_state(self, actions):
        actions_df.loc[len(actions_df)] = actions[1]  # storing actions in a dataframe
        score = self._game.get_score()
        reward = 0.1
        is_over = False  #game over
        if actions[1] == 1:
            self._agent.jump()
        elif actions[2] == 1:
            self._agent.duck()
        image = grab_screen(self._game._driver)
        image = image.reshape((80, 80, 1))
        self._display.send(image)  #display the image on screen
        if self._agent.is_crashed():
            scores_df.loc[len(loss_df)] = score  # log the score when game is over
            self._game.restart()
            reward = -10
            is_over = True
        return image, reward, is_over  #return the Experience tuple

    def pause(self):
        self._agent.pause()

    def resume(self):
        self._agent.resume()

    def restart(self):
        self._game.restart()