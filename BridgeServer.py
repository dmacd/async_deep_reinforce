from Queue import Queue
import Queue as Q               # to get the Queue.Full & Empties. god python == fail

import threading

# ugly hacks because python blows
import sys

from bridge.server import run_server

"""
command server for a3c

use cases

- process starts, starts up command server
    - and does nothing else?

- enable checkpoint save/restore to be remotely invoked
- ditto with experience

- some global "expert input" markers could come over this channel





later:
- support running multiple models in the same process
    - idk about that. tf may choke

- subclass? or reuse if simple enough to form the basis of each train thread's interaction:
    - unity directly invokes "action = step(obs, r, done)" which runs the process loop


"""
class BridgeServer(object):

    def __init__(self, listen_address, command_env):
        """
        :param listen_address:
        :param command_env:
        :return:
        """

        self._listen_address = listen_address

        self._shutdown_requested = False

        self._server_thread = threading.Thread(target=self._server_thread_fn)

        # pybridge client will communicated by interacting with these internal states directly
        self._server_env = command_env

        # doenst appear to be necessary
        # super(Env).__init__()
        self._server_thread.daemon = True                   # dont wait for exit...otherwise exceptions just hang process
        self._server_thread.start()

    def _server_thread_fn(self):
        print "Starting server thread for BridgeServer on %s" % self._listen_address
        run_server(self._listen_address, lambda: self._shutdown_requested,
                   calling_env=self._server_env)

    def shutdown(self):
        self._shutdown_requested = True

        print "BridgeServer: signaled shutdown; waiting for server thread to join..."

        if threading.current_thread() != self._server_thread: # allow shutdown to be called from the server thread itself by skipping self-join
            self._server_thread.join()

    def join(self):
        """ allow caller thread to just wait for exit if the want
        """
        self._server_thread.join()