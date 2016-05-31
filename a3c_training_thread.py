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
      init the lstm state before process is called, somewhere

      reinit lstm state after terminal episodes

      allow lstm state to persist even after global weights are copied (i guess)

      feed state in to lstm during policy evals
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
        lstm_state = self.local_network.lstm_last_output_state_value

        # t_max times loop
        for i in range(LOCAL_T_MAX):

            pi_, value_, lstm_state = self.local_network.run(sess, self.game_state.s_t,
                                                             lstm_state)

            # pi_ = self.local_network.run_policy(sess, self.game_state.s_t)
            action = choose_action(pi_)  # self.choose_action(pi_)

            states.append(self.game_state.s_t)
            actions.append(action)
            # value_ = self.local_network.run_value(sess, self.game_state.s_t)
            values.append(value_)
            lstm_states.append(lstm_state)

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
                break

        R = 0.0
        if not terminal_end:
            # R = self.local_network.run_value(sess, self.game_state.s_t)

            _, R, _ = self.local_network.run(sess, self.game_state.s_t, lstm_state)

        self.local_network.lstm_last_output_state_value = lstm_state # preserve for next time through the loop

        actions.reverse()
        states.reverse()
        rewards.reverse()
        values.reverse()
        lstm_states.reverse()

        # compute and accmulate gradients


        for (ai, ri, si, Vi, lstm_si) in zip(actions, rewards, states, values, lstm_states):
            R = ri + GAMMA * R
            td = R - Vi
            a = np.zeros([self.local_network.action_size])
            a[ai] = 1

            _, loss_summary = sess.run([self.accum_gradients, self.local_network.loss_summary_op],
                                       feed_dict={
                                           self.local_network.s: [si],
                                           self.local_network.a: [a],
                                           self.local_network.td: [td],
                                           self.local_network.r: [R],
                                           self.local_network.lstm_current_state_tensor: lstm_si
                                       })

            if (self.thread_index == 0):
                summary_writer.add_summary(loss_summary, global_step=global_t)

        cur_learning_rate = self._anneal_learning_rate(global_t)

        _, grad_summary = sess.run([self.apply_gradients, self.grad_summary_op],
                                   feed_dict={self.learning_rate_input: cur_learning_rate})

        if (self.thread_index == 0):
            summary_writer.add_summary(grad_summary, global_step=global_t)

        if (self.thread_index == 0) and (self.local_t % 100) == 0:
            print("TIMESTEP %d GLOBAL %d" % (self.local_t, global_t))

        # 進んだlocal step数を返す
        diff_local_t = self.local_t - start_local_t
        return diff_local_t


# todo: surely there is a builtin np function for this
def choose_action(pi_values):
    values = []
    sum = 0.0
    for rate in pi_values:
        sum = sum + rate
        value = sum
        values.append(value)

    r = random.random() * sum
    for i in range(len(values)):
        if values[i] >= r:
            return i
    # fail safe
    return len(values) - 1
