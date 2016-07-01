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

#from constants import ACTION_SIZE
from constants import INITIAL_ALPHA_LOW
from constants import INITIAL_ALPHA_HIGH
from constants import INITIAL_ALPHA_LOG_RATE
from constants import MAX_TIME_STEP
from constants import CHECKPOINT_DIR
from constants import LOG_FILE
from constants import RMSP_EPSILON
from constants import RMSP_ALPHA
from constants import GRAD_NORM_CLIP
from constants import USE_GPU

from utils import write_spinner

import constants
import gym
import gym_game_state

## TODO: verify this actually does the loguniform thing...

def log_uniform(lo, hi, rate):
  log_lo = math.log(lo)
  log_hi = math.log(hi)
  v = log_lo * (1-rate) + log_hi * rate
  return math.exp(v)

device = "/cpu:0"
if USE_GPU:
  device = "/gpu:0"

initial_learning_rate = log_uniform(INITIAL_ALPHA_LOW,
                                    INITIAL_ALPHA_HIGH,
                                    INITIAL_ALPHA_LOG_RATE)

print("initial_learning_rate = %f" % initial_learning_rate)

global_t = 0

stop_requested = False


##########################################################
## command line args

import argparse

parser = argparse.ArgumentParser(description='Run a3c.')
parser.add_argument('--render', dest='visualize_global', action='store_true',
                    help='render the global network applied to a test environment')

parser.add_argument('--tag', dest='tag', action='store',
                    default="",
                    help='tag to label the summary data')

parser.add_argument('--hidden', dest='hidden_sizes', action='append',
                    default=[200],
                    help='number of units in 1st fc hidden layer')

parser.add_argument('--lstm', dest='lstm_sizes', action='append',
                    default=[128],
                    help='number of units in lstm layer')

parser.add_argument('--lr', dest='initial_learning_rate', action='store',
                    default=initial_learning_rate, type=float,
                    help='initial learning rate')

parser.add_argument('--max_time_step', dest='max_time_step', action='store',
                    default=constants.MAX_TIME_STEP, type=int,
                    help='max time step to run for')

parser.add_argument('--gym_env', dest='gym_env', action='store',
                    default=constants.GYM_ENV,
                    help='gym environment to use')

parser.add_argument('--unity_env', dest='unity_env', action='store_true',
                    default=False,
                    help='listen for connections from unity env')


parser.add_argument('--threads', dest='threads', action='store',
                    default=constants.PARALLEL_SIZE, type=int,
                    help='number of agents/threads to use in training')



args = parser.parse_args()
print "*********************************************************"
print "Run configuration:", args
print "*********************************************************"

visualize_global = args.visualize_global


print "If it hangs, kill it: "
print "pkill -f a3c.py"

##########################################################
# env setup

from a3c_unity import make_unity_env

def make_env(index=0):

    if args.unity_env:
        make_unity_env(index)
    else:
        return gym.make(args.gym_env)




##########################################################
## network setup
# global_network = GameACNetwork(ACTION_SIZE, device)

# env = gym.make(args.gym_env)
env = make_env(index=0)

print "Env observation space:", env.observation_space


input_size = len(env.observation_space.sample().flatten())

# HACK: only works on discrete action spaces
if constants.CONTINUOUS_MODE:
    action_size = env.action_space.shape[0]
else:
    action_size = env.action_space.n

env.close() # needed so we can reuse connection 0

global_network = LowDimACNetwork(continuous_mode=constants.CONTINUOUS_MODE,
                                 action_size=action_size,
                                 input_size=input_size,
                                 hidden_sizes=args.hidden_sizes, #[200],
                                 lstm_sizes=args.lstm_sizes, #[128],
                                 network_name="global-net",
                                 device=device)


###########################################################
training_threads = []

learning_rate_input = tf.placeholder("float")

grad_applier = RMSPropApplier(learning_rate = learning_rate_input,
                              decay = RMSP_ALPHA,
                              momentum = 0.0,
                              epsilon = RMSP_EPSILON,
                              clip_norm = GRAD_NORM_CLIP,
                              device = device)

for i in range(args.threads):
  training_thread = A3CTrainingThread(i, global_network, args.initial_learning_rate,
                                      learning_rate_input,
                                      grad_applier, args.max_time_step,
                                      device = device,
                                      environment=make_env(i+1)) #gym.make(args.gym_env))
  training_threads.append(training_thread)

#os._exit(0)

############################################################################
# summary for tensorboard

from summaries import setup_summaries



############################################################################
# prepare session
sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))

summary_writer, record_score_fn = setup_summaries(sess, env.spec.id, args)

init = tf.initialize_all_variables()
sess.run(init)


## print all variable names in the graph...jjuuuuust to make sure

vars = tf.get_collection(tf.GraphKeys.VARIABLES)
for v in vars:
    print v.name



###################################################################
## checkpoints

import checkpoints

checkpoint_name = checkpoints.get_checkpoint_name(env.spec.id, args)

def train_function(parallel_index):
    global global_t
    global stop_requested

    training_thread = training_threads[parallel_index]

    checkpoint_count = 0

    while True:
        if stop_requested:
            break
        if global_t > args.max_time_step:
            break

        diff_global_t = training_thread.process(sess, global_t, summary_writer,
                                                record_score_fn)
        # score_summary_op, score_input)
        global_t += diff_global_t


        # global checkpoint saving
        if parallel_index == 0 and checkpoints.should_save_checkpoint(global_t, 1000000, checkpoint_count):
            checkpoint_count += 1
            print "Saving checkpoint %d at t=%d" % (checkpoint_count, global_t)
            checkpoints.save_checkpoint(checkpoint_name=checkpoint_name)

    
def signal_handler(signal, frame):
    global stop_requested
    print ("***************************************************************************************************")
    print('You pressed Ctrl+C!')
    stop_requested = True

    # TODO: in order for this to be interruptible with unity shut down, need to somehow broadcast interrupt to
    # all the environments in the training threads. plumbing.
    # WONTFIX for now
    # shut down python side first if want to cleanly exit
  
train_threads = []
for i in range(args.threads):
  train_threads.append(threading.Thread(target=train_function, args=(i,)))
  


########################################################################################
## visualization

def vis_network_thread(network):

    import pyglet

    global stop_requested
    global global_t

    # init env & states
    # env = gym.make(args.gym_env)
    env = make_env(args.threads + 1)
    game_state = gym_game_state.GymGameState(0, display=True, no_op_max=0, env=env)  # resets env already
    lstm_state = network.lstm_initial_state_value

    print "vis thread started"

    episode_reward = 0
    while True:


        #print "vis thread step"
        if stop_requested:
            break
        if global_t > args.max_time_step:
            break

        # print "lstm state:", lstm_state
        # pi_values = network.run_policy(sess, game_state.s_t)
        pi_values, v, lstm_state = network.run(sess, game_state.s_t, lstm_state)

        # what the learner does
        # action = choose_action(pi_values)
        action = network.sample_action(pi_values)

        # audition mode: always make the best choice
        #action = pi_values.tolist().index(max(pi_values))


        game_state.process(action)

        game_state.update()
        #env.render()

        episode_reward += game_state.reward
        if game_state.terminal:


            print "EPISODE REWARD: ", episode_reward
            episode_reward = 0

            #  TODO: try trapping exceptions on this thread...see if can avoid bad state that way


            # env.render(close=True) # does close window but doesnt help reset after exceptions...
            # env.close() # TODO: validate that this is stable/efficient

            # reinit env & states
            # env = gym.make(args.gym_env)
            game_state = gym_game_state.GymGameState(0, display=True, no_op_max=0, env=env)
            lstm_state = network.lstm_initial_state_value




        #time.sleep(0.001)


        # # except render() already does all this crap...sooo....wtf
        # ## massively hack-y event loop for pyglet event loop windows so we arent stuck with them
        # # TODO: break out in to separate utility fn or use EventLoop or....??
        # pyglet.clock.tick()
        #
        # for window in pyglet.app.windows:
        #     #window.switch_to()
        #     window.dispatch_events()
        #     window.dispatch_event('on_draw')
        #     #window.flip()




    print "vis thread exiting"
    env.close()


# if visualize_global:
#     vis_thread = threading.Thread(target=vis_network_thread, args=(global_network,))
#     vis_thread.daemon = False
#     vis_thread.start()
# else:

for t in train_threads:
    t.start()

########################################################################################

signal.signal(signal.SIGINT, signal_handler)

print('Press Ctrl+C to stop')


try:
    # run vis on main thread always...yes, much better
    if not args.unity_env: # no need for vis thread if using unity
        vis_network_thread(global_network)
except Exception as e:
    print "exception in vis thread, exiting"
    stop_requested = True
    print e




# signal.pause() # vis thread effectively pauses!




# if visualize_global:
#     vis_thread.join()
#
# else:

for t in train_threads:
    t.join()



print("*****************************************************************************************")
print('Now saving data. Please wait')

checkpoints.save_checkpoint(checkpoint_name)
