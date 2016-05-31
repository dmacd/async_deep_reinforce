# -*- coding: utf-8 -*-
import tensorflow as tf

from tensorflow.python.training import training_ops
from tensorflow.python.training import slot_creator

class RMSPropApplier(object):

  def __init__(self,
               learning_rate,
               decay=0.9,
               momentum=0.0,
               epsilon=1e-10,
               clip_norm=40.0,
               device="/cpu:0",
               name="RMSPropApplier"):

    self._name = name
    self._learning_rate = learning_rate
    self._decay = decay
    self._momentum = momentum
    self._epsilon = epsilon
    self._clip_norm = clip_norm
    self._device = device

    # Tensors for learning rate and momentum.  Created in _prepare.
    self._learning_rate_tensor = None
    self._decay_tensor = None
    self._momentum_tensor = None
    self._epsilon_tensor = None

    self._slots = {}

  def _create_slots(self, var_list):
    for v in var_list:
      # 'val' is Variable's intial value tensor.
      val = tf.constant(1.0, dtype=v.dtype, shape=v.get_shape())
      self._get_or_make_slot(v, val, "rms", self._name)
      self._zeros_slot(v, "momentum", self._name)

  def _prepare(self):
      self._learning_rate_tensor = tf.convert_to_tensor(self._learning_rate,
                                                      name="learning_rate")
      self._decay_tensor = tf.convert_to_tensor(self._decay, name="decay")
      self._momentum_tensor = tf.convert_to_tensor(self._momentum,
                                                 name="momentum")
      self._epsilon_tensor = tf.convert_to_tensor(self._epsilon,
                                                name="epsilon")

  def _slot_dict(self, slot_name):
    named_slots = self._slots.get(slot_name, None)
    if named_slots is None:
      named_slots = {}
      self._slots[slot_name] = named_slots
    return named_slots

  def _get_or_make_slot(self, var, val, slot_name, op_name):
    named_slots = self._slot_dict(slot_name)
    if var not in named_slots:
      named_slots[var] = slot_creator.create_slot(var, val, op_name)
    return named_slots[var]

  def get_slot(self, var, name):
    named_slots = self._slots.get(name, None)
    if not named_slots:
      return None
    return named_slots.get(var, None)

  def _zeros_slot(self, var, slot_name, op_name):
    named_slots = self._slot_dict(slot_name)
    if var not in named_slots:
      named_slots[var] = slot_creator.create_zeros_slot(var, op_name)
    return named_slots[var]

  def _apply_dense(self, grad, var):
    rms = self.get_slot(var, "rms")
    mom = self.get_slot(var, "momentum")
    return training_ops.apply_rms_prop(
      var, rms, mom,
      self._learning_rate_tensor,
      self._decay_tensor,
      self._momentum_tensor,
      self._epsilon_tensor,
      grad,
      use_locking=False).op

  # Apply accumulated gradients to var.
  #
  # TODO: in RMSProp native code, memcpy() (for CPU) and
  # cudaMemcpyAsync() (for GPU) are used when updating values,
  # and values might tend to be overwritten with results from other threads.
  # (Need to check the learning performance with replacing it)
  def apply_gradients(self, var_list, accum_grad_list, name=None):
    """
    :param var_list:
    :param accum_grad_list:
    :param name:
    :return: update_op, summary_op
    """
    update_ops = []
    summary_ops = []

    with tf.device(self._device):
      with tf.control_dependencies(None):
        self._create_slots(var_list)

      with tf.op_scope([], name, self._name) as name:
        self._prepare()

        # new: clip all gradients by their global norm

        # NB: slightly slower since global norm needs to be computed over all tensors at once;

        clipped_accum_grad_list, global_norm = tf.clip_by_global_norm(accum_grad_list,
                                                         self._clip_norm,
                                                         name="clip_by_global_norm")
        # print "**************************************************************************"
        # print "accum_grad_list: ", accum_grad_list
        # print "clipped_accum_grad_list: ", clipped_accum_grad_list
        # print "**************************************************************************"

        for var, accum_grad, clipped_accum_grad in zip(var_list, accum_grad_list, clipped_accum_grad_list):

          with tf.name_scope("update_" + var.op.name), tf.device(var.device):
            # clipped_accum_grad = tf.clip_by_norm(accum_grad, self._clip_norm) # taken care of by global scaling now
            update_ops.append(self._apply_dense(clipped_accum_grad, var))

            # add in summaries. slight hack
            # print("apply_gradients adding histogram summary: %s, %s" % (var.name, var.op.name))
            summary_ops += [tf.histogram_summary(name + "clipped_accum_grad/" + var.op.name + "<-" + accum_grad.name, clipped_accum_grad)]
            summary_ops += [tf.histogram_summary(name + "accum_grad/"         + var.op.name + "<-" + accum_grad.name, accum_grad)]

        return tf.group(*update_ops, name=name), tf.merge_summary(summary_ops)



""" questions:

how do i know the right gradients get applied to the right variables?

how do i know momentum is updated right?

- can print out value of momentum slot before and after an apply i guess


why were there so many apply ops in the first place?

rms apply should only apply to global thread...
"""