# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import random

from accum_trainer import AccumTrainer
from game_state import GameState
from gym_game_state import GymGameState
# from game_state import ACTION_SIZE
from game_ac_network import GameACNetwork

from constants import GAMMA
from constants import LOCAL_T_MAX
from constants import ENTROPY_BETA


class A3CTrainingThread(object):
    def __init__(self,
                 thread_index,
                 global_network,
                 initial_learning_rate,
                 learning_rate_input,
                 grad_applier,
                 max_global_time_step,
                 device,
                 environment):

        self.thread_index = thread_index
        self.learning_rate_input = learning_rate_input
        self.max_global_time_step = max_global_time_step

        # self.local_network = GameACNetwork(ACTION_SIZE, device)

        self.local_network = global_network.structural_clone(network_name="thread-net-%s" % self.thread_index)

        self.local_network.prepare_loss(ENTROPY_BETA)

        self.trainer = AccumTrainer(device)
        self.trainer.prepare_minimize(self.local_network.total_loss,
                                      self.local_network.get_vars())

        self.accum_gradients = self.trainer.accumulate_gradients()
        self.reset_gradients = self.trainer.reset_gradients()

        self.apply_gradients, self.grad_summary_op = grad_applier.apply_gradients(
            global_network.get_vars(),
            self.trainer.get_accum_grad_list())

        self.sync = self.local_network.sync_from(global_network)

        # self.game_state = GameState(113 * thread_index)
        self.game_state = GymGameState(113 * thread_index, env=environment)

        self.local_t = 0

        self.initial_learning_rate = initial_learning_rate

        self.episode_reward = 0

        self.lstm_last_output_state = None          # cache last lstm hidden states here

    def _anneal_learning_rate(self, global_time_step):
        learning_rate = self.initial_learning_rate * (
        self.max_global_time_step - global_time_step) / self.max_global_time_step
        if learning_rate < 0.0:
            learning_rate = 0.0
        return learning_rate

    def choose_action(self, pi_values):
        values = []
        sum = 0.0
        for rate in pi_values:
            sum = sum + rate
            value = sum
            values.append(value)

        r = random.random() * sum
        for i in range(len(values)):
            if values[i] >= r:
                return i;
        # fail safe
        return len(values) - 1

    # def _record_score(self, sess, summary_writer, summary_op, score_input, score, global_t):
    #     summary_str = sess.run(summary_op, feed_dict={
    #         score_input: score,
    #     })
    #     summary_writer.add_summary(summary_str, global_t)

    """next steps
      x init the lstm state before process is called, somewhere

      x reinit lstm state after terminal episodes

      ?!? allow lstm state to persist even after global weights are copied (i guess)

      x feed state in to lstm during policy evals
      how does state work in gradient backups?



        Tests:

            - inspect lstm state inputs, outputs, and episode stored values
            -

    """

    def process(self, sess, global_t, summary_writer, record_score_fn):  #summary_op, score_input):
        states = []
        actions = []
        rewards = []
        values = []
        lstm_states = []

        terminal_end = False

        # reset accumulated gradients
        sess.run(self.reset_gradients)

        # copy weights from shared to local
        sess.run(self.sync)

        # write weight summaries...only need them from one thread really
        if (self.thread_index == 0):
            param_summary = sess.run(self.local_network.param_summary_op)
            summary_writer.add_summary(param_summary, global_step=global_t)

        start_local_t = self.local_t

        # resume with wherever we left off on last time through the action loop
        # TODO: no reason the network itself current should care about this

        if (self.lstm_last_output_state is None):
            self.lstm_last_output_state = self.local_network.lstm_initial_state_value

        lstm_state = self.lstm_last_output_state

        # lstm_state = self.local_network.lstm_last_output_state_value

        # t_max times loop
        for i in range(LOCAL_T_MAX):

            states.append(self.game_state.s_t)
            lstm_states.append(lstm_state)

            pi_, value_, lstm_state = self.local_network.run(sess, self.game_state.s_t,
                                                             lstm_state)

            action = self.local_network.sample_action(pi_)

            # print "a3c train: pi_: ", pi_
            # print "a3c train: action: ", action
            # pi_ = self.local_network.run_policy(sess, self.game_state.s_t)
            # action = choose_action(pi_)  # self.choose_action(pi_)

            actions.append(action)
            # value_ = self.local_network.run_value(sess, self.game_state.s_t)
            values.append(value_)

            if (self.thread_index == 0) and (self.local_t % 100) == 0:
                print "pi=", pi_
                print " V=", value_

            # process game
            self.game_state.process(action)

            # receive game result
            reward = self.game_state.reward
            terminal = self.game_state.terminal

            self.episode_reward += reward

            # clip reward
            rewards.append(np.clip(reward, -1, 1000))

            self.local_t += 1

            # s_t1 -> s_t
            self.game_state.update()

            if terminal:
                terminal_end = True
                print "terminal score =", self.episode_reward

                # self._record_score(sess, summary_writer, summary_op, score_input,
                #                    self.episode_reward, global_t)

                record_score_fn(sess, summary_writer, self.episode_reward, global_t)

                self.episode_reward = 0
                self.game_state.reset()

                #  ugh. reset lstm state!
                lstm_state = self.local_network.lstm_initial_state_value

                break

        R = 0.0
        if not terminal_end:
            # R = self.local_network.run_value(sess, self.game_state.s_t)

            _, R, _ = self.local_network.run(sess, self.game_state.s_t, lstm_state)

        # self.local_network.lstm_last_output_state_value = lstm_state # preserve for next time through the loop
        self.lstm_last_output_state = lstm_state

        #  TODO: cant store the lists i pass directly since they'll be destructively reversed by
        # this call....hmmmmm
        # maybe just reverse them here and leave it?
        #  start with copying the lists
        self.backup_and_accum_gradients(sess, global_t, summary_writer,
                                        states=states,
                                        lstm_states=lstm_states,
                                        actions=actions,
                                        values=values,
                                        rewards=rewards,
                                        final_reward_estimate=R)


        if (self.thread_index == 0) and (self.local_t % 100) == 0:
            print("TIMESTEP %d GLOBAL %d" % (self.local_t, global_t))

        # 進んだlocal step数を返す
        diff_local_t = self.local_t - start_local_t
        return diff_local_t

    def backup_and_accum_gradients(self, sess, global_t, summary_writer,
                                   states, lstm_states, actions, values, rewards,
                                   final_reward_estimate):
        """ inputs are lists reflecting a recorded episode fragment in the order they occured

            a = sample{ pi(a | s, lstm_s ) }
            v = V(s, lstms)
            r = env.step(a)

        :param states: states
        :param actions:
        :param rewards:
        :param lstm_states:
        :return:
        """


        # TODO: copy these and leave the originals alone...
        actions.reverse()
        states.reverse()
        rewards.reverse()
        values.reverse()
        lstm_states.reverse()

        R = final_reward_estimate

        # compute and accmulate gradients
        for (ai, ri, si, Vi, lstm_si) in zip(actions, rewards, states, values, lstm_states):
            R = ri + GAMMA * R
            td = R - Vi

            a = self.local_network.feedback_action(ai)
            # a = np.zeros([self.local_network.action_size])
            # a[ai] = 1

            _, loss_summary = sess.run([self.accum_gradients, self.local_network.loss_summary_op],
                                       feed_dict=self.local_network.loss_feed_dictionary(si, a, td, R, lstm_si)
                                       # feed_dict={
                                       #     self.local_network.s: [si],
                                       #     self.local_network.a: [a],
                                       #     self.local_network.td: [td],
                                       #     self.local_network.r: [R],
                                       #     self.local_network.lstm_current_state_tensor: lstm_si
                                       # }
                                       )

            if (self.thread_index == 0):
                summary_writer.add_summary(loss_summary, global_step=global_t)


        """ idea: maybe possible to do n-step TBPTT after having retroactively computed R for each state
        feed in batches of size up to n_max to a set of parallel networks with 
        
        
        idea: set up the lstm with say 5 recursive calls. then the initial inputs would need to be padded...maybe?
        would work if made the inputs in batches and altered iteration logic to cycle inputs through the history...
        

        """

        cur_learning_rate = self._anneal_learning_rate(global_t)

        _, grad_summary = sess.run([self.apply_gradients, self.grad_summary_op],
                                   feed_dict={self.learning_rate_input: cur_learning_rate})

        if (self.thread_index == 0):
            summary_writer.add_summary(grad_summary, global_step=global_t)


    #  TODO: rename 'states' variable as 'observations' in next version just to be fucking crystal clear


    def process_memory(self, sess, global_t, summary_writer,
                       states, initial_lstm_state, actions, rewards, final_state):
        """
        :param sess:
        :param global_t:
        :param summary_writer:
        :param states:
        :param initial_lstm_state:
        :param actions:
        :param rewards:
        :param final_state:         observation after the last game step...use None to signal terminal, otherwise used
         to compute the final boostrap Value
        :return:
        """

        # TODO: gotcha initial_lstm_state must be set carefully
        # if the episode reflects t=0, the state is always known
        # otherwise how can we know what the lstm state output of the *current* policy might plausibly have been
        # unless the same policy was executed from the very beginning of the historical episode and propagated
        # we could just record the lstm_state prior to the beginning of the history episode as an approximation
        # we might expect it to converge reasonably after a number of steps to something from the plausible distribution
        # for the current policy...however, over time, the policy will drift away further and further from what
        # created the original lstm_state
        # this suggests the solution that we update the stored initial lstm state in the replay memory after every refresh
        # ...almost like a real memory trace in a human brain might...
        # but how can we update it ????
        # maybe keep one state in reserve just to prime...but then we can only update the lstm state after it, not the one
        # that initial state needs....HMMM. maybe just
        #
        # for certain environments we could just apply the network to s_t+0 repeatedly until the lstm state converges
        # this works if the problem and/or env dont depend on any direct measure of time..perhaps
        #
        # easiest solution might just be to always reference the episodes to t=0
        # or just ignore first k states when backing up and computing gradients...since presumably we'll have converged
        # to something reasonable by that point



        values = []
        lstm_states = []

        terminal_end = False

        # reset accumulated gradients
        sess.run(self.reset_gradients)

        # copy weights from shared to local
        sess.run(self.sync)
        lstm_state = initial_lstm_state

        for (s_t, a_t, r_t) in zip(states, actions, rewards):

            # accum lstm states
            lstm_states.append(lstm_state)

            pi_, value_, lstm_state = self.local_network.run(sess, s_t, lstm_state)

            # get values
            values.append(values)



        # what to do if terminal?
        # set R = 0
        # if not, recompute final_reward_estimate using value net and final


# overkill for now but may be useful to tag with other info later
class ReplayEpisode(object):

    def __init__(self,
                 observations,
                 initial_lstm_state,
                 actions,
                 rewards,
                 final_state
                 ):
        self.observations = observations
        self.initial_lstm_state = initial_lstm_state
        self.action = actions
        self.rewards = rewards
        self.final_state = final_state

        self._total_reward = np.sum(rewards)

    @property
    def total_reward(self):
        return self._total_reward

import sortedcontainers

class ReplayMemory(object):

    def __init__(self, max_episodes):

        self._max_episodes = max_episodes
        # need a sorted map...by reward

        self.episodes = sortedcontainers.SortedListWithKey(key=lambda x: x.total_reward)




    def submit_episode(self, episode):

        # check ep total reward and insert it in the map, discarding out of the middle on overflow


        if len(self.episodes) > self._max_episodes:
            # remove from random position in the middle 25% of the list
            self.episodes.remove