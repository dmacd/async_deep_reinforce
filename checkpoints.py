
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


import constants

###############################################################################
# init or load checkpoint with saver

def get_checkpoint_name(env_id, args):

    checkpoint_name = "checkpoint"
    checkpoint_name += " %s" % env_id #env.spec.id #args.gym_env
    checkpoint_name += " hs=%s" % args.hidden_sizes
    checkpoint_name += " lstms=%s " % args.lstm_sizes

    if len(args.tag) > 0:
        checkpoint_name += " -- %s" % args.tag


def restore_checkpoint(sess, checkpoint_name='checkpoint', path=constants.CHECKPOINT_DIR):
    """

    :param sess:
    :param checkpoint_name: friendly name of checkpoint
    :param path: directory containing all checkpoints for a single network arch and/or run
    :return:
    """

    # TODO: restore named checkpoint in a safe way:
    # - terminate all training threads first
    # then load
    # then resume them

    # for now...just load the main model and call it good

    saver = tf.train.Saver() # TODO: just save/restore global net params


    checkpoint = tf.train.get_checkpoint_state(path)
    if checkpoint and checkpoint.model_checkpoint_path:
      saver.restore(sess, checkpoint.model_checkpoint_path)
      print "checkpoint loaded:", checkpoint.model_checkpoint_path
      tokens = checkpoint.model_checkpoint_path.split("-")
      # set global step
      global_t = int(tokens[1])
      print ">>> global step set: ", global_t
    else:
      print "Could not find old checkpoint"



def save_checkpoint(sess, global_step, checkpoint_name='checkpoint', path=constants.CHECKPOINT_DIR):


    if not os.path.exists(path):
      os.mkdir(path)

    saver = tf.train.Saver()

    # TODO: just save/restore global net params
    saver.save(sess, path + '/' + checkpoint_name, global_step=global_step)




def should_save_checkpoint(global_t, checkpoint_every, checkpoint_count):
    return (math.floor(global_t/checkpoint_every) > checkpoint_count)