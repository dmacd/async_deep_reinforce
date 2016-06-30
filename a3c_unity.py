# import stuff
# -*- coding: utf-8 -*-
import tensorflow as tf
import threading
import time
import numpy as np

import signal
import random
import math
import os
from datetime import datetime

from game_ac_network import GameACNetwork
from lowdim_ac_network import LowDimACNetwork
from a3c_training_thread import A3CTrainingThread
from rmsprop_applier import RMSPropApplier



from utils import write_spinner

import constants
import gym
import gym_game_state



##########################################################
## command line args

import argparse


def parse_args():

    parser = argparse.ArgumentParser(description='Run a3c.')
    # parser.add_argument('--render', dest='visualize_global', action='store_true',
    #                     help='render the global network applied to a test environment')
    #
    # parser.add_argument('--tag', dest='tag', action='store',
    #                     default="",
    #                     help='tag to label the summary data')
    #
    # parser.add_argument('--hidden', dest='hidden_sizes', action='append',
    #                     default=[200],
    #                     help='number of units in 1st fc hidden layer')
    #
    # parser.add_argument('--lstm', dest='lstm_sizes', action='append',
    #                     default=[128],
    #                     help='number of units in lstm layer')
    #
    # parser.add_argument('--lr', dest='initial_learning_rate', action='store',
    #                     default=initial_learning_rate, type=float,
    #                     help='initial learning rate')
    #
    # parser.add_argument('--max_time_step', dest='max_time_step', action='store',
    #                     default=constants.MAX_TIME_STEP, type=int,
    #                     help='max time step to run for')
    #
    # parser.add_argument('--gym_env', dest='gym_env', action='store',
    #                     default=constants.GYM_ENV,
    #                     help='gym environment to use')
    #
    # parser.add_argument('--unity_env', dest='unity_env', action='store_true',
    #                     default=False,
    #                     help='listen for connections from unity env')
    #
    #
    # parser.add_argument('--threads', dest='threads', action='store',
    #                     default=constants.PARALLEL_SIZE, type=int,
    #                     help='number of agents/threads to use in training')

    parser.add_argument('--listen-address', dest='listen_address', action='store',
                        default="tcp://*:17100", type=int,
                        help='listen address for command server')

    return parser.parse_args()







def init_network():


    global_network = LowDimACNetwork(continuous_mode=constants.CONTINUOUS_MODE,
                                     action_size=action_size,
                                     input_size=input_size,
                                     hidden_sizes=args.hidden_sizes, #[200],
                                     lstm_sizes=args.lstm_sizes, #[128],
                                     network_name="global-net",
                                     device=device)


def start_training_threads(args, global_network, device="/cpu:0"):
    ###########################################################
    training_threads = []

    learning_rate_input = tf.placeholder("float")

    grad_applier = RMSPropApplier(learning_rate = learning_rate_input,
                                  decay = constants.RMSP_ALPHA,
                                  momentum = 0.0,
                                  epsilon = constants.RMSP_EPSILON,
                                  clip_norm = constants.GRAD_NORM_CLIP,
                                  device = device)

    for i in range(args.threads):
      training_thread = A3CTrainingThread(i, global_network, args.initial_learning_rate,
                                          learning_rate_input,
                                          grad_applier, args.max_time_step,
                                          device = device,
                                          environment=make_env(i+1)) #gym.make(args.gym_env))
      training_threads.append(training_thread)










#############################################################
## start master bridge server


master_env = { 'args': args,

               'init_network': init_network,
               'start_agent_thread': start_agent_thread,
               'stop_agent_thread': stop_agent_thread,

               }

master_server = BridgeServer(args.listen_address, master_env)