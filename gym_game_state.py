# -*- coding: utf-8 -*-
import sys
import numpy as np
#import cv2
#from ale_python_interface import ALEInterface

import gym

#from constants import ROM

#from constants import ACTION_SIZE

class GymGameState(object):
  def __init__(self, rand_seed, env, display=False, no_op_max=7 ):

# result of calling gym.make('envname')
    self.env = env

    self._no_op_max = no_op_max

    self.display = display

    # TODO: handle continous action spaces
    # discrete ok for now

    self.reset()

  def reset(self):


    self.env.reset()

    # TODO: maybe revisit? random noop initialization
    # # randomize initial state
    # if self._no_op_max > 0:
    #   no_op = np.random.randint(0, self._no_op_max + 1)
    #   for _ in range(no_op):
    #     self.ale.act(0)
    #

    action = self.env.action_space.sample()
    observation, reward, done, info = self.env.step(action)

    x_t = observation

    # _, _, x_t = self._process_frame(0, False)

    self.reward = 0
    self.terminal = False
    # self.s_t = np.stack((x_t, x_t, x_t, x_t), axis=1)
    self.s_t = np.stack((x_t, ), axis=1)

  def process(self, action):

    if self.display:
        self.env.render()

    observation, reward, done, info = self.env.step(action)
    observation = np.reshape(observation, (len(observation), 1))    # reshape to make compatible with s_t for .append

    self.reward = reward
    self.terminal = done
    self.s_t1 = np.append(self.s_t[:,1:], observation, axis=1)

  def update(self):
    self.s_t = self.s_t1

