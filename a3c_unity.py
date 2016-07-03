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



from utils import write_spinner, MutableCounter

import constants
import gym
import gym_game_state

import checkpoints

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
                        default="tcp://*:17100", type=str,
                        help='listen address for command server')

    parser.add_argument('--checkpoint-dir', dest='checkpoint_dir', action='store',
                        default=constants.CHECKPOINT_DIR, type=str,
                        help='directory from which checkpoints will be read and written')

    parser.add_argument('--device', dest='device', action='store',
                        default="/cpu:0", type=str,
                        help='default tensorflow device')


    return parser.parse_args()




def make_unity_env(index=0, unity_baseport=4440):

        # unity_baseport = 4440
        port = (unity_baseport+index)

        print "Using unity environment, base port %d, target port %d " % (unity_baseport, port)
        # instantiante unityenv
        import envs
        env = envs.UnityEnv(listen_address="tcp://*:%d" % port)  # NEXT STEP: establish discovery scheme...

        print "Waiting for unity env... "
        while not env.ready:
            write_spinner(0.1)

        return env



def init_network(args, input_size, action_size, device="/cpu:0"):

    global_network = LowDimACNetwork(continuous_mode=constants.CONTINUOUS_MODE,
                                     action_size=action_size,
                                     input_size=input_size,
                                     hidden_sizes=args.hidden_sizes, #[200],
                                     lstm_sizes=args.lstm_sizes, #[128],
                                     network_name="global-net",
                                     device=device)

    return global_network


def create_training_networks(args, num_threads, unity_baseport, global_network, device="/cpu:0"):
    ###########################################################
    training_nets = []

    learning_rate_input = tf.placeholder("float")

    grad_applier = RMSPropApplier(learning_rate = learning_rate_input,
                                  decay = constants.RMSP_ALPHA,
                                  momentum = 0.0,
                                  epsilon = constants.RMSP_EPSILON,
                                  clip_norm = constants.GRAD_NORM_CLIP,
                                  device = device)

    for i in range(num_threads):
        training_net = A3CTrainingThread(i, global_network, args.initial_learning_rate,
                                            learning_rate_input,
                                            grad_applier, args.max_time_step,
                                            device=device,
                                            environment=make_unity_env(i, unity_baseport=unity_baseport))  # gym.make(args.gym_env))
        # TODO: include unity baseport
        training_nets.append(training_net)

    return training_nets


def prod_thread(env, sess, network, shutdown_signal_callback):

    # global stop_requested
    # global global_t

    # init env & states
    # env = gym.make(args.gym_env)
    # env = make_unity_env(index=0, unity_baseport=port)
    game_state = gym_game_state.GymGameState(0, display=True, no_op_max=0, env=env)  # resets env already
    lstm_state = network.lstm_initial_state_value

    print "prod thread started"

    episode_reward = 0
    while True:


        #print "prod thread step"
        if shutdown_signal_callback():
            break
        # if global_t > args.max_time_step:
        #     break

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


    print "prod thread exiting"
    env.close()


from summaries import setup_summaries

def prepare_session(args):
    ############################################################################
    # prepare session
    sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))

    env_id = "Unity-online-please-take-from-script"
    summary_writer, record_score_fn = setup_summaries(sess, env_id, args)

    init = tf.initialize_all_variables()
    sess.run(init)

    # debug
    # vars = tf.get_collection(tf.GraphKeys.VARIABLES)
    # for v in vars:
    #     print v.name

    return sess, summary_writer, record_score_fn




def train_function(training_net, sess, summary_writer, record_score_fn, shutdown_signal_callback, global_timestep):

    while True:
        if shutdown_signal_callback():
            break

        diff_global_t = training_net.process(sess, global_timestep.value, summary_writer,
                                                record_score_fn)
        # score_summary_op, score_input)
        global_timestep.inc(diff_global_t)


def start_training_threads(sess, training_nets, summary_writer, record_score_fn, shutdown_signal_callback):

    training_timestep = MutableCounter()

    #  when does session prep need to happen?
    # should_shutdown = False
    # def shutdown_signal_callback():
    #     return should_shutdown

    def train_wrapper(training_net):

        train_function(training_net=training_net,
                       sess=sess,
                       summary_writer=summary_writer,
                       record_score_fn=record_score_fn,
                       shutdown_signal_callback=shutdown_signal_callback,
                       global_timestep=training_timestep)

    training_threads = []

    for tn in training_nets:
        t = {'shutdown_requested': False} # neatly group the shutdown signal with the thread object
        t['thread'] =threading.Thread(target=train_wrapper, args=(tn, lambda: t['shutdown_requested']))
        training_threads.append(t)

    # start them

    for tt in training_threads:
        tt['thread'].daemon = True
        tt['thread'].start()


    return training_threads, training_timestep

def stop_training_threads(training_threads):

    print "Requested shutdown of training threads"

    for tt in training_threads:
        tt['shutdown_requested'] = True

    for tt in training_threads:
        tt['thread'].join()
        print "Shutdown training thread."

    print "Shutdown all training threads."



class AgentAlreadyRunningException(Exception):
    pass

class AgentThreadGroup(object):

    def __init__(self, args, sess, network, device):

        self._args = args
        self._sess = sess
        self._network = network
        self._device = device

        self._threads = {} # port -> thread mapping
        self._should_shutdown = {} # port -> bool mapping for shutdown signals
        self._envs = {}

    def start(self, port):

        if port in self._threads:
            raise AgentAlreadyRunningException("Agent already running on port %d" % port)

        env = make_unity_env(0, port)
        self._envs[port] = env

        def agent_fn():
            # TODO: this isnt quite good enough....need a way to interrupt the env
            # better if i create it here so i have a ref to it so i can call close() on it
            # when time to shutdown
            prod_thread(env=env,
                        sess=self._sess,
                        network=self._network,
                        shutdown_signal_callback=lambda: self._should_shutdown[port])


        self._should_shutdown[port] = False
        thread = threading.Thread(target=agent_fn, args=())
        thread.daemon = True
        thread.start()
        self._threads[port] = thread


    def stop(self, port):
        if not port in self._threads:
            print("Warning: no agent on port %d to stop " % port)

        # TODO: will need to send queue interrupts
        print("Stopping agent on port %d" % port)
        self._envs[port].close()
        self._should_shutdown[port] = True
        self._threads[port].join()

        print("Stopped agent on port %d" % port)


def do_exit():
    # requested to exit immediately
    import os
    os._exit(0)

# def start_prod_thread(port):
#
#     pass
#
# def stop_prod_thread(port):
#
#     pass


class ManagedAPIWrapper(object):

    def __init__(self, args):
        self._args = args
        pass

    def init(self, input_size, action_size):

        global_network = init_network(args, input_size=input_size, action_size=action_size, device=args.device)

        create_training_networks(args,) # should this be separate?

        # next step: finish assembling the wrapper
        # such that host doesnt have to worry about tracking any state between api calls

    def start_training(self, num_threads):

        # inits training networks and session if needed
        pass

    def save_checkpoint(self):

        pass

    def restore_checkpoint(self):

        pass

    def start_agent(self, port):

        # inits session if not already
        # starts an agent on port

        pass



    def shutdown(self):

        pass


if __name__ == "__main__":

    import BridgeServer

    #############################################################
    ## start master bridge server

    args = parse_args()

    master_server = None
    master_env = {
        'args': args,
        'init_network': init_network,
        'create_training_networks': create_training_networks,       # create training threads:    training_nets = create_training_networks(args, num_threads, baseport, network, device)
        'prepare_session': prepare_session,                         # must be called after all networks created
        'start_training_threads': start_training_threads,           # training_threads, training_timestep =
        'stop_training_threads': stop_training_threads,
        'AgentThreadGroup': AgentThreadGroup,                       # agents = AgentThreadGroup(args, sess, network, device)
        'checkpoints': checkpoints,                                 # save_checkpoints(sess, global_step, name='checkpoint', path=args.checkpoint_dir)
        'shutdown': lambda: master_server.shutdown(),

        # 'start_prod_thread': start_prod_thread,
        # 'stop_prod_thread': stop_prod_thread,

        # next step: refine unity client api...figure out how we're protecting/exposing internals like network, device, sess, et
        # - opaque references to be passed around by the client code
        # x and how agentthreadgroups get created
        # NO maybe default one just gets created by init_network?
       }


    simple_api = {
        'init': simple_init,
        'start_training',



    }

    master_server = BridgeServer.BridgeServer(args.listen_address, master_env)

    master_server.join()