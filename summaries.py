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


def setup_summaries(sess, env_id, args):
    ROOT_LOG_DIR = constants.LOG_FILE #os.getcwd() + "/tf-log/"
    TODAY_LOG_DIR = ROOT_LOG_DIR + "/" + datetime.now().date().isoformat()

    LOG_DIR = TODAY_LOG_DIR + "/" + datetime.now().time().replace(second=0, microsecond=0).isoformat()[0:-3].replace(':', '.')

    LOG_DIR += " %s" % env_id #env.spec.id # args.gym_env
    LOG_DIR += " lr=%f" % args.initial_learning_rate
    LOG_DIR += " hs=%s" % args.hidden_sizes
    LOG_DIR += " lstms=%s " % args.lstm_sizes

    if len(args.tag) > 0:
        LOG_DIR += " -- %s" % args.tag


    score_input = tf.placeholder(tf.float32,name="score_input")
    score_input_avg = tf.placeholder(tf.float32,name="score_input_avg")
    score_smooth = tf.Variable(dtype=tf.float32, initial_value=0, name="score_avg")
    score_smooth_assign_op = tf.assign(score_smooth, score_input * 0.01 + score_smooth * 0.99)

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
