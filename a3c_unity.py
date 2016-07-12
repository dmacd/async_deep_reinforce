# import stuff
# -*- coding: utf-8 -*-
import tensorflow as tf
import threading
import time
import numpy as np

import bunch

import zmq
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

    # here process args are mainly for global process setup, independent of model details, however the client api is free to
    # override/inject

    parser = argparse.ArgumentParser(description='Run a3c.')
    # parser.add_argument('--render', dest='visualize_global', action='store_true',
    #                     help='render the global network applied to a test environment')
    #
    parser.add_argument('--tag', dest='tag', action='store',
                        default="",
                        help='tag to label the summary data')
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




def make_unity_env_specific_port(index=0, unity_baseport=4440, wait_for_ready=False):

        # unity_baseport = 4440
        port = (unity_baseport+index)

        print "Using unity environment, base port %d, target port %d " % (unity_baseport, port)
        # instantiante unityenv
        import envs
        env = envs.UnityEnv(listen_address="tcp://*:%d" % port)  # NEXT STEP: establish discovery scheme...


        # next step: figure out whether i actually need to wait here or not...not sure that i do..*[]:

        if wait_for_ready:
            print "Waiting for unity env..."
            while not env.ready:
                write_spinner(0.1)
        else:
            print "Env created, not waiting for unity ready ack..."

        env.port = unity_baseport
        env.index = index

        return env


class NoPortsAvailableError(Exception):
    pass

def make_unity_env_random_port(min_port=50000, max_port=60000, max_tries=100):

    tries = 0;
    while tries < max_tries:
        # print "make unity env try ", tries
        try:
            tries += 1
            port = random.randrange(min_port, max_port)
            env = make_unity_env_specific_port(index=0, unity_baseport=port)
            # print "made env...returning..."
            return env
        except zmq.ZMQError as e:
            print e
            print "Failed to bind on random port %d, retrying... (%d/%d) " % (port, tries, max_tries)

    raise NoPortsAvailableError


def make_unity_env(index=0, unity_baseport=4440):
    """
    :param index:
    :param unity_baseport: base of port range, unless zero, in which case try random ports
    :return:
    """

    if (unity_baseport == 0):
        env = make_unity_env_random_port()
    else:
        env = make_unity_env_specific_port(index=index, unity_baseport=unity_baseport)

    return env



def init_network(args, input_size, action_size, hidden_sizes, lstm_sizes, device="/cpu:0"):

    global_network = LowDimACNetwork(continuous_mode=constants.CONTINUOUS_MODE,
                                     action_size=action_size,
                                     input_size=input_size,
                                     hidden_sizes=hidden_sizes, #[200],
                                     lstm_sizes=lstm_sizes, #[128],
                                     network_name="global-net",
                                     device=device)

    return global_network


def create_training_thread_objects(args, num_threads, unity_baseport, global_network, device="/cpu:0"):
    ###########################################################
    training_thread_objs = [None]*num_threads

    learning_rate_input = tf.placeholder("float")

    grad_applier = RMSPropApplier(learning_rate = learning_rate_input,
                                  decay = constants.RMSP_ALPHA,
                                  momentum = 0.0,
                                  epsilon = constants.RMSP_EPSILON,
                                  clip_norm = constants.GRAD_NORM_CLIP,
                                  device = device)




    def make_thread_obj(index):

        env = make_unity_env(i, unity_baseport=unity_baseport)

        # horrible naming here do to lack of full encapsulation of thread operations in the so-named thread class
        # TODO: when refactoring, move thread management and loop state in to the class
        obj = A3CTrainingThread(i, global_network, args.initial_learning_rate,
                                            learning_rate_input,
                                            grad_applier, args.max_time_step,
                                            device=device,
                                            environment=env)  # gym.make(args.gym_env))

        obj.port = env.port        # slight hack: cache port value with the thread
        # TODO: include unity baseport
        # print "created training net %d" % i
        # training_thread_objs.append(obj)
        training_thread_objs[index] = obj


    # run all thread inits in parallel
    for i in range(num_threads):

        # tf.Graph is NOT thread safe. cant do this without much more care

        # TODO: revisit later to make boot up time MUCH faster

        make_thread_obj(i)
        # t = threading.Thread(target=make_thread_obj, args=(i,))
        # t.start()


    while not all(training_thread_objs):
        time.sleep(.5)

    print "Done creating training thread objects"


    return training_thread_objs


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




def train_function(training_thread_obj, sess, summary_writer, record_score_fn, shutdown_signal_callback, global_timestep):

    # need to issue reset to the unity env here to force it to wait for the unity side to connect
    training_thread_obj.reset()

    while True:
        if shutdown_signal_callback():
            break

        diff_global_t = training_thread_obj.process(sess, global_timestep.value, summary_writer,
                                                    record_score_fn)
        # score_summary_op, score_input)
        global_timestep.inc(diff_global_t)


def start_training_threads(sess, training_thread_objs, summary_writer, record_score_fn):
    # todo?: maybe include shutdown_signal_callback param here?? unclear why i would need it

    training_timestep = MutableCounter()

    #  when does session prep need to happen?
    # should_shutdown = False
    # def shutdown_signal_callback():
    #     return should_shutdown

    def train_wrapper(training_thread_obj, shutdown_cb):

        train_function(training_thread_obj=training_thread_obj,
                       sess=sess,
                       summary_writer=summary_writer,
                       record_score_fn=record_score_fn,
                       shutdown_signal_callback=shutdown_cb,
                       global_timestep=training_timestep)

    training_threads = []

    for tn in training_thread_objs:
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
        self._envs = {} # port -> env mapping

    @property
    def ports(self):
        return self._threads.keys()

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
        # this could be triggered by monitoring a proprietary thread-local property perhaps? or a signal?
        # or just bite bullet, do it right, plumb the signal through the queue constructors :L

        print("Stopping agent on port %d" % port)
        self._envs[port].close()
        self._should_shutdown[port] = True
        self._threads[port].join()

        # cleanup
        del self._threads[port]
        del self._envs[port]
        del self._should_shutdown[port]

        print("Stopped agent on port %d" % port)

    def stop_all(self):

        # signal shutdown to all threads at once
        print("Stopping all agents...")

        for p in self._threads:
            self._should_shutdown[p] = True

        # then wait
        for p in self._threads:
            self.stop(p)



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

    """
    TODO: error handling
    - NoPortsAvailable

    """

    def __init__(self, args): #, input_size, action_size, hidden_sizes, lstm_sizes):
        self._args = args
        self._session_prepared = False
        self._device = args.device

        print "init called..."
        self._global_network = init_network(args,
                                            input_size=args.input_size,
                                            action_size=args.action_size,
                                            hidden_sizes=args.hidden_sizes,
                                            lstm_sizes=args.lstm_sizes,
                                            device=self._device)

        self._training_timestep = None
        self._training_thread_objs = None
        self._training_threads = None
        self._session = None
        self._summary_writer = None
        self._record_score_fn = None

        self._should_shutdown_training = False

        self._agents = None


    @property
    def ready(self):
        """
        :return: is api ready to instantiate prod agents
        """

        print "ready:", self._session, self._agents
        return (self._session is not None) and \
               (self.agents is not None)
               # (self._agents is not None)

        # and ...whatever else is needed before agents can be started and training can commence

    @property
    def training_ready(self):
        """
        :return: is api ready to instantiate training agents
        """
        return len(self._training_thread_objs) > 0 and self.ready

    @property
    def training_ports(self):
        """
        :return: list of ports which have training threads running on them. allow easy reconnection without having to wrangle all the port thread ranges
        """
        ports =  map(lambda e: e.port, self._training_thread_objs)
        print "training_ports", ports
        return ports

    # @property
    # def agent_ports(self):
    #     if self._agents is not None:
    #         return self._agents

    def init_training(self, num_threads):
        """
        Prepares the host for training by starting a fixed number of training threads and preparing the TF session.
        :param num_threads:
        :return:
        """

        self._training_thread_objs = create_training_thread_objects(self._args,
                                                                    num_threads=num_threads,
                                                                    unity_baseport=0,
                                                                    global_network=self._global_network,
                                                                    device=self._device)


        # x assign thread ports randomnly (or semi-randomnly at least)
        # x store thread ports so clients can use them to reconnect

        # inits training networks and session if needed


        # if self._session :

        # we will have to reinit the session if the training networks have been recreated
        # unknown what will happen to the existing one though...may result in collisions?
        # todo: investigate how to reset TF
        self._prepare_session()

        # start training threads

        self._training_threads, self._training_timestep = \
            start_training_threads(self._session,
                                   training_thread_objs=self._training_thread_objs,
                                   summary_writer=self._summary_writer,
                                   record_score_fn=self._record_score_fn)
                               #,shutdown_signal_callback=lambda: self._should_shutdown_training)


        return self.training_ports

    def _prepare_session(self):
                self._session, self._summary_writer, self._record_score_fn = prepare_session(self._args)



    def shutdown_training(self):
        stop_training_threads(self._training_threads)

    def save_checkpoint(self, name, path):
        checkpoints.save_checkpoint(self._session, global_step=self._training_timestep,checkpoint_name=name, path=path)

    def restore_checkpoint(self, name, path):
        checkpoints.restore_checkpoint(self._session, checkpoint_name=name, path=path)


    # expose the threadgroup api directly rather than repeat wrappers here
    # clients can call start, stop, and threads
    @property
    def agents(self):
        if self._session is None:
            self._prepare_session()

        if self._agents is None:
            self._agents = AgentThreadGroup(self._args, self._session, self._global_network, self._device)

        return self._agents


    # def start_agent(self, port):
    #
    #
    #
    #     self._agents.start(port)
    #
    #     # do i need to wait for readiness in some fashion??
    #
    # def


    def shutdown(self):

        # self._should_shutdown_training = True

        # todo: wait for training threads to join
        stop_training_threads(self._training_threads)

        # wait for all agent threads to join
        self._agents.stop_all()


def merge_dicts(x, y):
    '''Given two dicts, merge them into a new dict as a shallow copy.'''
    print 'merge_dicts:', x, y

    z = x.copy()
    z.update(y)
    print "merged:", z
    return bunch.bunchify(z)

if __name__ == "__main__":

    import BridgeServer

    #############################################################
    ## start master bridge server

    args = bunch.bunchify(vars(parse_args()))

    master_server = None

    # expose more complex direct api, requires client to manage lots of state
    # master_env = {
    #     'args': args,
    #     'init_network': init_network,
    #     'create_training_networks': create_training_networks,       # create training threads:    training_nets = create_training_networks(args, num_threads, baseport, network, device)
    #     'prepare_session': prepare_session,                         # must be called after all networks created
    #     'start_training_threads': start_training_threads,           # training_threads, training_timestep =
    #     'stop_training_threads': stop_training_threads,
    #     'AgentThreadGroup': AgentThreadGroup,                       # agents = AgentThreadGroup(args, sess, network, device)
    #     'checkpoints': checkpoints,                                 # save_checkpoints(sess, global_step, name='checkpoint', path=args.checkpoint_dir)
    #     'shutdown': lambda: master_server.shutdown(),
    #
    #     # 'start_prod_thread': start_prod_thread,
    #     # 'stop_prod_thread': stop_prod_thread,
    #
    #
    #    }

    # next step: refine unity client api...figure out how we're protecting/exposing internals like network, device, sess, et
    # - opaque references to be passed around by the client code..ick
    # x and how agentthreadgroups get created
    # NO maybe default one just gets created by init_network?

    # NEXT STEP: figure out why args arent merging

    # simplified managed api
    master_env = bunch.bunchify({
        'args': args,
        'merge_dicts': merge_dicts,         # cuz python is amazingly bad
        'api': { 'managed': {'factory' : ManagedAPIWrapper,
                             'instance' : None } },
        'shutdown': lambda: master_server.shutdown(),
    })

    # NEXT STEP: wrap initialization state up in a factor helper so we have singleton api wrapper class
    # or just look up python singleton pattern
    # then write C# wrapper side

    # todo: utils and server management stuff too

    master_server = BridgeServer.BridgeServer(args.listen_address, master_env)

    master_server.join()