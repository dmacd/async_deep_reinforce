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
from constants import PARALLEL_SIZE
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

args = parser.parse_args()

visualize_global = args.visualize_global

##########################################################
## network setup
# global_network = GameACNetwork(ACTION_SIZE, device)

env = gym.make(constants.GYM_ENV)

print("Env observation space:", env.observation_space)


input_size = len(env.observation_space.sample().flatten())

# HACK: only works on discrete action spaces
if constants.CONTINUOUS_MODE:
    action_size = env.action_space.shape[0]
else:
    action_size = env.action_space.n

global_network = LowDimACNetwork(continuous_mode=constants.CONTINUOUS_MODE,
                                 action_size=action_size,
                                 input_size=input_size,
                                 hidden_sizes=[64],
                                 lstm_sizes=[64],
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

for i in range(PARALLEL_SIZE):
  training_thread = A3CTrainingThread(i, global_network, initial_learning_rate,
                                      learning_rate_input,
                                      grad_applier, MAX_TIME_STEP,
                                      device = device,
                                      environment=gym.make(constants.GYM_ENV))
  training_threads.append(training_thread)


#os._exit(0)



############################################################################
# summary for tensorboard


def setup_summaries(sess):
    ROOT_LOG_DIR = constants.LOG_FILE #os.getcwd() + "/tf-log/"
    TODAY_LOG_DIR = ROOT_LOG_DIR + "/" + datetime.now().date().isoformat()

    LOG_DIR = TODAY_LOG_DIR + "/" + datetime.now().isoformat('_').replace(':', '.')

    score_input = tf.placeholder(tf.float32,name="score_input")
    score_input_avg = tf.placeholder(tf.float32,name="score_input_avg")
    score_smooth = tf.Variable(dtype=tf.float32, initial_value=0, name="score_avg")
    score_smooth_assign_op = tf.assign(score_smooth, score_input * 0.01 + score_smooth*0.99)

    score_summary_op = [tf.merge_summary([
            tf.scalar_summary("score", score_input),
            tf.scalar_summary("score_avg", score_input_avg),
            tf.scalar_summary("score_smooth", score_smooth),
        ]),
        score_smooth_assign_op]

    from collections import deque

    moving_avg_scores = deque(maxlen=100)


    # summary_op = tf.merge_all_summaries()
    summary_writer = tf.train.SummaryWriter(LOG_DIR, sess.graph_def)

    print("logs written to: %s " % LOG_DIR)
    print("tensorboard --logdir=%s" % LOG_DIR)

    # v1
    def _record_score_fn(sess, summary_writer, score, global_t):

        moving_avg_scores.append(score)
        score_avg = np.mean(moving_avg_scores)

        summary_str, _ = sess.run(score_summary_op, feed_dict={
            score_input: score,
            score_input_avg: score_avg
        })

        moving_avg_scores.append(score)


        # print "record_score_fn:", summary_str
        summary_writer.add_summary(summary_str, global_t)





    return summary_writer, _record_score_fn


############################################################################
# prepare session
sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))

summary_writer, record_score_fn = setup_summaries(sess)

init = tf.initialize_all_variables()
sess.run(init)


## print all variable names in the graph...jjuuuuust to make sure

vars = tf.get_collection(tf.GraphKeys.VARIABLES)
for v in vars:
    print v.name


###############################################################################
# init or load checkpoint with saver
saver = tf.train.Saver() # TODO: just save/restore global net params
checkpoint = tf.train.get_checkpoint_state(CHECKPOINT_DIR)
if checkpoint and checkpoint.model_checkpoint_path:
  saver.restore(sess, checkpoint.model_checkpoint_path)
  print "checkpoint loaded:", checkpoint.model_checkpoint_path
  tokens = checkpoint.model_checkpoint_path.split("-")
  # set global step
  global_t = int(tokens[1])
  print ">>> global step set: ", global_t
else:
  print "Could not find old checkpoint"


def train_function(parallel_index):
  global global_t
  global stop_requested
  
  training_thread = training_threads[parallel_index]

  while True:
    if stop_requested:
      break
    if global_t > MAX_TIME_STEP:
      break

    diff_global_t = training_thread.process(sess, global_t, summary_writer,
                                            record_score_fn)
                                            #score_summary_op, score_input)
    global_t += diff_global_t


    
    
def signal_handler(signal, frame):
  global stop_requested
  print ("***************************************************************************************************")
  print('You pressed Ctrl+C!')
  stop_requested = True
  
train_threads = []
for i in range(PARALLEL_SIZE):
  train_threads.append(threading.Thread(target=train_function, args=(i,)))
  


########################################################################################
## visualization

def vis_network_thread(network):
    import pyglet

    global stop_requested
    global global_t
    #from a3c_training_thread import choose_action
    env = gym.make(constants.GYM_ENV)

    game_state = gym_game_state.GymGameState(0, display=True, no_op_max=0, env=env)  # resets env already
    #env.render()
    print "vis thread started"
    lstm_state = network.lstm_initial_state_value

    episode_reward = 0
    while True:


        #print "vis thread step"
        if stop_requested:
            break
        if global_t > MAX_TIME_STEP:
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

            # todo: avg episode rewards and display those
            print "EPISODE REWARD: ", episode_reward
            episode_reward = 0
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


# run vis on main thread always...yes, much better
vis_network_thread(global_network)

# signal.pause() # vis thread effectively pauses!




# if visualize_global:
#     vis_thread.join()
#
# else:

for t in train_threads:
    t.join()



print("*****************************************************************************************")
print('Now saving data. Please wait')


if not os.path.exists(CHECKPOINT_DIR):
  os.mkdir(CHECKPOINT_DIR)  

saver.save(sess, CHECKPOINT_DIR + '/' + 'checkpoint', global_step = global_t)



