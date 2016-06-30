from Queue import Queue
import Queue as Q               # to get the Queue.Full & Empties. god python == fail

import threading

# ugly hacks because python blows

import sys

from bridge.server import run_server
from gym.core import Env

class UnityEnv(Env):


    def __init__(self, listen_address):

        self.metadata['render.modes'] = ['human'] # since thats what unity is doing. video cap later possible too

        self._listen_address = listen_address
        self._action_queue = InterruptibleQueue()
        self._observation_queue = InterruptibleQueue()

        self._shutdown_requested = False

        self._server_thread = threading.Thread(target=self._server_thread_fn)

        # pybridge client will communicated by interacting with these internal states directly
        self._server_env = {  'action_queue': self._action_queue,
                              'observation_queue': self._observation_queue,
                              'action_space': None,
                              'observation_space': None,
                              'ready': False                # set when unity side considers itself ready
                            }

        # doenst appear to be necessary
        # super(Env).__init__()
        self._server_thread.daemon = True                   # dont wait for exit...otherwise exceptions just hang process
        self._server_thread.start()

    def _server_thread_fn(self):
        print "starting server thread for UnityEnv on %s" % self._listen_address
        run_server(self._listen_address, lambda: self._shutdown_requested,
                   calling_env=self._server_env)


    @property
    def ready(self):
        return not self._shutdown_requested and self._server_env['ready']

    # TODO: may need to rework these...unclear how can work with the static? members set by the base class
    @property
    def observation_space(self):
        assert self._server_env['observation_space'] is not None
        return self._server_env['observation_space']

    @property
    def action_space(self):
        assert self._server_env['action_space'] is not None
        return self._server_env['action_space']

    @property
    def spec(self):
        """
        :return: faked EnvSpec
        """
        assert self._server_env['spec'] is not None
        return self._server_env['spec']

    def _get_obs(self):
        return self._observation_queue.interruptible_get(block=True, timeout=None) # block indefinitely


    def _step(self, action):
        """
        enqueues an action

        waits for an observation

        :param action:
        :return:
        """

        # TODO: add correlation ids for action, obs pairs to ensure synchronization9
        self._action_queue.interruptible_put(action, block=True, timeout=None)

        ob, reward, done, info = self._get_obs()                # obs entries are tuples including reward, done, and info

        # print info['scope'] + ":reward=", reward
        return ob, reward, done, info

    def _close(self):

        print "closing unity env"

        # todo: debug this...looks like we're stuck
        # bet its the pyb client using non-interruptible methods
        self._shutdown_requested = True

        self._action_queue.interrupt_all()
        self._observation_queue.interrupt_all()

        print "interrupted queues"
        print "waiting for join"
        self._server_thread.join()
        self._server_env['ready'] = False

        print "closed"
        # TODO: could wait for shutdown to confirm

    def _reset(self):
        print "RESET CALLED"
        # print "observation queue has ", self._observation_queue


        self._action_queue.interruptible_put("reset", block=True, timeout=None)  # action=="reset" signals unity to reset episode

        # then try here
        # try:
        #     # while self._observation_queue.full():
        #     self._observation_queue.get(block=True, timeout=1)
        #     print "obs queue had stuff in it already!"
        # except Q.Empty:
        #     print "timed out trying to empty queue"


        obs = self._get_obs()

        if obs[2]: # i.e. still reporting terminal state. wait for another observation in that case
            obs = self._get_obs()

        return obs


    def _render(self, mode='human', close=False):
        pass



#  OOOORRRRR i didnt need to do this
#  envspec has the info i need and is standard. bleh.

#
# class ReacherEnvDJM(reacher.ReacherEnv):
#     def __init__(self):
#         reacher.ReacherEnv.__init__(self)
#
#     def _step(self, a):
#
#         ob, reward, done, info = reacher.ReacherEnv._step(self, a)
#
#
#         # vec = self.get_body_com("fingertip")-self.get_body_com("target")
#         # reward_dist = - np.linalg.norm(vec)
#
#         reward_dist = info['reward_dist']
#         print "reward_dist", reward_dist
#         if (reward_dist > -1): # since its negative of distance to target
#
#             print "DONE"
#             done = True
#
#
#
#         return ob, reward, done, info
#
#         # vec = self.get_body_com("fingertip")-self.get_body_com("target")
#
#         # reward_ctrl = - np.square(a).sum()
#         # reward = reward_dist + reward_ctrl
#         # self.do_simulation(a, self.frame_skip)
#         # ob = self._get_obs()
#         # done = False
#
#
#
# from gym.envs.registration import registry, register, make, spec
#
# register(
#     id='Reacher-v1.djm',
#     entry_point='envs:ReacherEnvDJM',
#     timestep_limit=50,
#     reward_threshold=-3.75,
# )

class QueueInterrupt(Exception):
    pass

class InterruptibleQueue(Queue):
    """ my slightly hacky interruptible queue class.
    doenst interrupt anything immediately, but waits for timeouts to expire

    if this proves inadequate, try basing on:
    http://code.activestate.com/recipes/576461-interruptible-queue/

    (still needs surfacing loops though, unclear how to integrate)

    """
    def __init__(self):

        # super(InterruptibleQueue, self).__init__() # the F? doesnt work??? Queue not a new-style class i guess. fucking python
        Queue.__init__(self)

        self._interrupt_consumers = False
        self._interrupt_producers = False


    def interrupt_all(self):
        self._interrupt_consumers = True
        self._interrupt_producers = True


    def interruptible_get(self, block=True, timeout=None):
        # print "INTERRUPTIBLE GET: "
        while True:
            try:
                value = self.get(block=block, timeout=timeout if timeout is not None else 1)
                # print "GET COMPLETE", value
                return value
            except Q.Empty:
                pass

            # print "interrupt_consumers:", self._interrupt_consumers
            if self._interrupt_consumers:
                self._interrupt_consumers = False
                raise QueueInterrupt()


    def interruptible_put(self, value, block=True, timeout=None):
        # print "INTERRUPTIBLE PUT: ", value
        while True:
            try:
                self.put(value, block=block, timeout=1) #timeout=timeout if timeout is not None else 1000)
                break
            except Q.Full:
                pass

            if self._interrupt_producers:
                self._interrupt_producers = False
                raise QueueInterrupt()
        # print "PUT COMPLETE"
