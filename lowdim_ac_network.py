

# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import math
import random

# utils

import uuid


def _is_iterable(x):
    return hasattr(x, "__iter__") \
           and not isinstance(x, basestring) \
           and not isinstance(x, tf.Variable)

def flatten(x):
    """ flattens a nested structure.
        mind-bogglingly omitted from python.
        what a joke
    """
    result = []

    for el in x:
        # if hasattr(el, "__iter__") and not isinstance(el, basestring):
        if _is_iterable(el):
            result.extend(flatten(el))
        else:
            result.append(el)
    return result

# global unique_string_counter
# unique_string_counter = 0
def unique_string():
    """Helper for generating random variable names/scopes"""
    # global unique_string_counter
    # unique_string_counter += 1
    # return "[UQ" + str(unique_string_counter) + "]"
    ## UGH would have to lock to make threadsafe. fuckit

    return str(uuid.uuid4()) # good enough

def safe_log(x, arg_min=1e-10, arg_max=1):
    return tf.log(tf.clip_by_value(x, arg_min, arg_max))



def get_basic_lstm_vars_from_scope(scope):
    # print scope
    # print scope.name
    # vars = tf.get_collection(tf.GraphKeys.VARIABLES)
    # print "all graph vars:", map(lambda x: x.name, vars)

    lstm_variables = tf.get_collection(tf.GraphKeys.VARIABLES, scope=scope.name)

    print "lstm_variables:"
    print map(lambda x: x.name, lstm_variables)
    Ws = filter(lambda x: "Matrix" in x.name, lstm_variables)
    Bs = filter(lambda x: "Bias" in x.name, lstm_variables)
    return Ws, Bs

def get_lstm_vars_from_scope(scope):
    # print scope
    # print scope.name
    # vars = tf.get_collection(tf.GraphKeys.VARIABLES)
    # print "all graph vars:", map(lambda x: x.name, vars)

    lstm_variables = tf.get_collection(tf.GraphKeys.VARIABLES, scope=scope.name)

    # print "lstm_variables:"
    # print map(lambda x: x.name, lstm_variables)
    Ws = filter(lambda x: "LSTMCell/W" in x.name, lstm_variables)
    Bs = filter(lambda x: "LSTMCell/B" in x.name, lstm_variables)
    return Ws, Bs



# Actor-Critic Network (Policy network and Value network)

class LowDimACNetwork(object):
    def __init__(self,
                 action_size, # either number of discrete actions or number of dimensions, depending on mode
                 input_size,
                 hidden_sizes=[200],
                 lstm_sizes=[128],
                 continuous_mode=False,
                 network_name="default",
                 device="/cpu:0"):

        self._action_size = action_size
        self._input_size = input_size
        self._hidden_sizes = hidden_sizes
        self._lstm_sizes = lstm_sizes

        # more lstm bookkeeping
        self.lstm_current_state_tensors = []
        self.lstm_output_state_tensors = []

        # for now these are shared
        self._lstm_initial_state_value = []
        # self.lstm_last_output_state_value = None # caches last lstm output state for convenience of training thread which is managing it for now

        self._network_name = network_name # outer name scope for all graph variables. *should* be unique for all networks

        self._device = device

        self._param_summaries = []
        self._param_summary_op = None
        self.loss_summary_op = None

        self._continuous_mode = continuous_mode
        # TODO: setup sampling functions
        self._sample_action_fn = sample_continuous_action if self._continuous_mode else sample_discrete_action

        self.setup_graph()



    @property
    def action_size(self):
        return self._action_size

    def structural_clone(self, network_name=None):
        """
        :return: a network (with initialized TF graph components) structurally compatible with
        this graph but with separate weights
        """

        return LowDimACNetwork(action_size=self.action_size,
                               input_size=self._input_size,
                               hidden_sizes=self._hidden_sizes,
                               lstm_sizes=self._lstm_sizes,
                               network_name=network_name if network_name else self.network_name(),
                               # fc0_size=self._fc0_size,
                               # fc1_size=self._fc1_size,
                               continuous_mode=self._continuous_mode,
                               device=self._device)

    def variable_summaries(self, var, name):
        l = []

        with tf.name_scope("summaries"):
            # mean = tf.reduce_mean(var)
            # l.append(tf.scalar_summary('mean/' + name, mean))
            # with tf.name_scope('stddev'):
            #     stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
            # l.append(tf.scalar_summary('sttdev/' + name, stddev))
            # l.append(tf.scalar_summary('max/' + name, tf.reduce_max(var)))
            # l.append(tf.scalar_summary('min/' + name, tf.reduce_min(var)))
            l.append(tf.histogram_summary(name, var))

        self._param_summaries.extend(l)


    @property
    def network_name(self):
        return self._network_name

    @property
    def param_summary_op(self):
        return self._param_summary_op
        # print "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
        # print self._param_summaries
        #
        # print "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
        # merged_summaries = tf.merge_summary(self._param_summaries)
        # print("param_summary_op: merged_summaries", merged_summaries)
        # print "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
        # return merged_summaries

    # todo: make a property param_summary_op that returns a merged op so it can be recorded externally
    # def record_param_summaries(self, sess, summary_writer, global_t):
    #
    #     summary = sess.run(tf.merge_summary(self._param_summaries))
    #     summary_writer.add_summary(summary, global_step=global_t)

    def _fc_layer(self, x, input_size, output_size, activation=tf.nn.relu, name="anon_layer"):
        """
        :param x: input tensor
        :param input_size: width
        :param output_size:
        :param activation:
        :return: W, b, y = weight tensor, bias tensor, output tensor
        """


        W = self._fc_weight_variable([input_size, output_size],name="W_%s" % name)
        b = self._fc_bias_variable([output_size], input_size, name="b_%s" % name)

        # print("layer %s - created W:%s b:%s" % (name, W.get_shape(), b.get_shape()))
        y = activation(tf.matmul(x, W) + b)

        self.variable_summaries(W, "W_%s" % name)
        self.variable_summaries(b, "b_%s" % name)

        return W, b, y

    # NEXT STEP:
    # - add histogram summaries to weights
    # - add summaries to loss elements
    # - LSTM!!
    # - longer training window tmax


    def setup_graph(self):


        batch_size = 1
        with tf.device(self._device), tf.name_scope(self.network_name):

            input_history = 4

            # input
            self.s = tf.placeholder("float", [1, self._input_size, input_history]) #[1, 84, 84, 4])

            # flattened input, no convolution here
            input_state_flat = tf.reshape(self.s, [1, self._input_size*input_history])

            layers = []
            # x = state_flat
            input_size = self._input_size*input_history

            def hidden_layers(x, input_size, name):
                for idx, hidden_size in enumerate(self._hidden_sizes):
                    layer_name = name+"/"+"fc%d" % idx
                    W, b, y = self._fc_layer(x, input_size, hidden_size, activation=tf.nn.relu, name=layer_name)
                    layers.append((W,b,y,layer_name))

                    # feed last output in to next layer
                    x = y
                    input_size = hidden_size


                # now set up lstm layer(s)

                if (len(self._lstm_sizes) > 0):
                    assert len(self._lstm_sizes) == 1
                    layer_name = self.network_name + "/" + name + "/lstm0"
                    with tf.variable_scope(layer_name) as vs:
                        # just 1 for now
                        # lstm = tf.nn.rnn_cell.BasicLSTMCell(self._lstm_sizes[0])
                        lstm = tf.nn.rnn_cell.LSTMCell(self._lstm_sizes[0], use_peepholes=True) # try fancier lstm .... grr returns more weight matrices fuck
                        # todo: append stats somehow

                        current_state_tensor = tf.placeholder("float", [batch_size, lstm.state_size])
                        self.lstm_current_state_tensors.append(current_state_tensor)
                                                                #tf.zeros([batch_size, lstm.state_size])

                        # todo: use # of steps in second index on input
                        #  reshape the input tensor to fit better in the computations
                        #             tf.get_variable_scope().reuse_variables()


                        y, output_state_tensor = lstm(x, current_state_tensor)
                        self.lstm_output_state_tensors.append(output_state_tensor)

                        # setup initial state
                        self._lstm_initial_state_value.append(np.zeros(dtype="float32",shape=[batch_size, lstm.state_size ]))

                        # retrieve variables
                        Ws, bs = get_lstm_vars_from_scope(vs)
                        #  we arent equipped to deal with more complex setups yet
                        # print "lstm Ws:", Ws
                        # assert len(Ws) == 1
                        # assert len(bs) == 1
                        # manually add variable summaries
                        print "Ws:", Ws
                        print "bs:", bs

                        for W in Ws:
                            print "Adding summaries for layer %s : %s=%s" % (layer_name, W.name, W)
                            self.variable_summaries(W, "W_%s_%s" % (layer_name, W.name))

                        for b in bs:
                            print "Adding summaries for layer %s : %s=%s" % (layer_name, b.name, b)
                            self.variable_summaries(b, "b_%s_%s" % (layer_name, b.name))


                        layers.append((Ws,bs,y,layer_name))     # can now add tuples of weights or biases to this list

                        x = y
                        input_size = self._lstm_sizes[0]

                return x, input_size

            if self._continuous_mode:
                # continuous output config

                hidden_policy_output, hidden_policy_output_size = hidden_layers(input_state_flat, input_size, name="mu_net")
                W_mu, b_mu, mu = self._fc_layer(hidden_policy_output, hidden_policy_output_size, self._action_size, activation=tf.identity, name="mu")
                layers.append((W_mu, b_mu, mu, "mu"))

                # hidden_sigma2_output, hidden_sigma2_output_size = hidden_layers(input_state_flat, input_size, name="sigma2_net")
                W_sigma2, b_sigma2, sigma2 = self._fc_layer(hidden_policy_output, hidden_policy_output_size, 1, activation=tf.nn.softplus, name="sigma2")
                sigma2 = tf.maximum(sigma2, 1e-3)  # MINOR HACK: bound variance so that our log probs dont get fucked
                layers.append((W_sigma2, b_sigma2, sigma2, "sigma2"))

                self.mu = mu
                self.sigma2 = sigma2

            else:
                # discrete output confif
                x, input_size = hidden_layers(input_state_flat, input_size, name="discrete")

                W_pi, b_pi, pi = self._fc_layer(x, input_size, self._action_size, activation=tf.nn.softmax, name="pi")
                layers.append((W_pi, b_pi, pi, "pi"))

                self.pi = pi


            ### UMMM...THIS OUTPUT SHOULD BE SCALAR, NO?

            hidden_v_output, hidden_v_output_size = hidden_layers(input_state_flat, input_size, name="V_net")
            W_v, b_v, v = self._fc_layer(hidden_v_output, hidden_v_output_size, 1, activation=tf.identity, name="v")
            layers.append((W_v, b_v, v, "v"))

            self.v = v
            self._layers = layers


            # initialize set up recurrent states
            # slight hax caching these here..:
            # probably want better encapsulation of network state

            # TODO: HACK: not general for multple lstm layers or sizes
            #  todo: cleanup
            # lstm_state_size = self._lstm_sizes[0]*2 # TOTAL HACK: this really needs to come from the lstm iteslf
            # self.lstm_last_output_state_value = [self.lstm_initial_state_value]*len(self.lstm_current_state_tensors) # TODO: cleanup...is this used anywhere?

            # print "initialized last_output_state_value: ", self.lstm_last_output_state_value


        self._param_summary_op = tf.merge_summary(self._param_summaries)


    @property
    def lstm_initial_state_value(self):
        return self._lstm_initial_state_value

    def prepare_loss(self, entropy_beta):

        with tf.device(self._device), tf.name_scope(self.network_name):
            if self._continuous_mode:
                policy_loss, entropy, summaries = self._prepare_policy_loss_continuous(entropy_beta)
            else:
                policy_loss, entropy, summaries = self._prepare_policy_loss_discrete(entropy_beta)


            # R (input for value)
            self.r = tf.placeholder("float", [1],name="reward")
            # value loss (output)
            # (Learning rate for Critic is half of Actor's, so multiply by 0.5)
            value_loss = 0.5 * tf.nn.l2_loss(self.r - self.v)

            # gradienet of policy and value are summed up
            self.total_loss = policy_loss + value_loss


            # todo: unclear if i really need these
            l = []
            l.extend(summaries)
            l += [tf.scalar_summary(["R"], self.r)]
            l += [tf.scalar_summary(["(R-V)"], self.td)]
            l += [tf.scalar_summary(["V (loss eval)"], tf.reshape(self.v, (1,)))]
            l += [tf.scalar_summary(["V (r-td)"], self.r - self.td)]
            l += [tf.scalar_summary(["entropy"], tf.reshape(entropy, (1,)))]
            l += [tf.scalar_summary(["policy_loss"], tf.reshape(policy_loss, (1,)))]    # TODO: HACK: when we do batch mode, will want a histogram and ditch the reshape, most likely?
            l += [tf.scalar_summary("value_loss", value_loss)]

            self.loss_summary_op = tf.merge_summary(l)


    def _multivariate_normal_pdf(self, x, mean, var):
        n = tf.cast(tf.size(mean), tf.float32)
        var_to_the_n = tf.pow(var, n)

        C = 1/(tf.sqrt(tf.pow(2*math.pi, n)*var_to_the_n))
        E = tf.exp((-0.5 * (1/var_to_the_n) \
                    * tf.reduce_sum(tf.squared_difference(x, mean), reduction_indices=(1,)))) # leave batch size alone...i hope?



        # print "//////////////////////////////////////"
        # print "mean     ", mean
        # print "n        ", n
        # print "C        ", C
        # print "var      ", var
        # print "E        ", E
        return C*E

    def _multivariate_normal_log_pdf(self, x, mean, var):

        ## TODO: validate & test
        n = tf.cast(tf.size(mean), tf.float32)
        var_to_the_n = tf.pow(var, n)

        return \
            -0.5*n*safe_log(var,arg_min=1e-6, arg_max=float("inf")) \
            + (-0.5 * (1/tf.maximum(var_to_the_n, 1e-24)) * tf.reduce_sum(tf.squared_difference(x, mean), reduction_indices=(1,)))


    def _tf_inspired_mvn_log_pdf(self, x, mean, var):

        k = tf.cast(tf.size(mean), tf.float32)

        log_two_pi = tf.constant(math.log(2 * math.pi), dtype=tf.float32)
        x_whitened = (x - mean) / tf.sqrt(var)
        x_whitened_norm = tf.reduce_sum(x_whitened * x_whitened)

        sigma_det = tf.pow(var, k)

        log_pdf_value = (
          -tf.log(sigma_det) - k * log_two_pi - x_whitened_norm) / 2

        return log_pdf_value


    def _prepare_policy_loss_continuous(self, entropy_beta):
        ## TODO: double check the loss form...from original sources. not sure what the scope of the derivative operator is...
        summaries = []
        # taken action (input for policy)
        self.a = tf.placeholder("float", [1, self.action_size], name="sparse_action")

        # temporary difference (R-V) (input for policy)
        self.td = tf.placeholder("float", [1], name="temporal_difference")

        # policy entropy
        entropy = -0.5*(tf.log(tf.maximum(1e-10, (2*math.pi*self.sigma2)))+1) #  dont clip maximum value of argument


        # policy probablity pi(a | s_t, theta)

        cov = self.sigma2 * tf.constant(np.identity(self._action_size), dtype=tf.float32)
        cov = tf.reshape(cov, shape=(1,self._action_size, self._action_size)) # mvn needs batch index to stay
        print "Sigma2 shape,", self.sigma2.get_shape()
        print "cov shape,", cov.get_shape()
        print "mu shape,", self.mu.get_shape()
        mvn = tf.contrib.distributions.MultivariateNormal(self.mu, sigma_chol=tf.sqrt(cov))

        sampled_action_probability = self._multivariate_normal_pdf(self.a, self.mu, self.sigma2)
        # sampled_action_log_probability_homebrew = self._multivariate_normal_log_pdf(self.a, self.mu, self.sigma2)
        sampled_action_log_probability_homebrew = self._tf_inspired_mvn_log_pdf(self.a, self.mu, self.sigma2)
        sampled_action_log_probability = tf.maximum(mvn.log_pdf(self.a), -1e+10)

        # summaries += [tf.scalar_summary(["log P(a) log of pdf"],
        #                                 tf.reshape(safe_log(sampled_action_probability, arg_min=1e-30), (1,)))]
        summaries += [tf.scalar_summary(["log P(a) tf"],
                                        tf.reshape(sampled_action_log_probability, (1,)))]
        summaries += [tf.scalar_summary(["log P(a) homebrew"],
                                        tf.reshape(sampled_action_log_probability_homebrew, (1,)))]
        # summaries += [tf.scalar_summary(["log P(a) (log prob - lof of pdf)"],
        #                                 tf.reshape(sampled_action_log_probability - safe_log(sampled_action_probability, arg_min=1e-30), (1,)))]

        # summaries += [tf.scalar_summary(["log P(a) (log prob - lof of pdf)"],
        #                                 tf.reshape(sampled_action_log_probability - safe_log(sampled_action_probability, arg_min=1e-30), (1,)))]

        summaries += [tf.scalar_summary(["logprob_homebrew(a) - logprob_tf(a)"],
                                        tf.reshape(sampled_action_log_probability_homebrew - sampled_action_log_probability, (1,)))]





        summaries += [tf.scalar_summary(["sigma2"], tf.reshape(self.sigma2, (1,)))]

        summaries += [tf.histogram_summary("mu", self.mu )]

        """ NEXT STEP:
        - determined that tf's mvn logprob gives right answers whereas mine doesnt for dimns > 1, but agrees with dim == 1
        - deterimend that tf's mvn logprob FUCKS the gradient, whereas mine doesnt
        - find the error in my logprob
        - may be purely numerical...could help to whiten x before reducing
         """



        # NB: revisit what correct shape should be for this
        # TODO: HACK: reduce_sum here even though

        # policy_loss = safe_log(sampled_action_probability) * (self.td + entropy * entropy_beta)
        policy_loss = sampled_action_log_probability_homebrew * (self.td + entropy * entropy_beta)

        #  TODO: cleanup non-log policy

        policy_loss = -policy_loss    # maybe this will be slightly better if we dont have the wrong gradients being computed!!!

        # print "-----------------------------------------"
        # print "entropy              ", entropy
        # print "sampled action prob  ", sampled_action_probability
        # print "td                   ", self.td
        # print "policy_loss          ", policy_loss
        # print "-----------------------------------------"


        # TODO: improve efficiency by applying the log probability
        # https://www.cs.cmu.edu/~epxing/Class/10701-08s/recitation/gaussian.pdf

        # reduces to scalar even though it was just a 1-element vector...
        # TODO: revisit when we try batch mode!
        return policy_loss, entropy, summaries


    def _prepare_policy_loss_discrete(self, entropy_beta):
        summaries = []
        # taken action (input for policy)
        self.a = tf.placeholder("float", [1, self.action_size],name="sparse_action")

        # temporary difference (R-V) (input for policy)
        self.td = tf.placeholder("float", [1],name="temporal_difference" )
        # policy entropy
        entropy = -tf.reduce_sum(self.pi * safe_log(self.pi), name="entropy") # DJM: avoid NaN when policy becomes overly deterministic!

        # more accurate expression: entropy is treated as a reward and should have same sign as R in td = R-V
        policy_loss = ( -tf.reduce_sum( tf.mul( safe_log(self.pi), self.a ) ) * (self.td +
                             entropy * entropy_beta ))

        return policy_loss, entropy, summaries


    def feedback_action(self, action):
        """Prepares the stored (sampled) action output to be fed back as input ot the loss operation"""

        if (self._continuous_mode):
            # action is already in the correct form
            return action
        else:
            # action space discrete, action is a single index
            a = np.zeros([self._action_size])
            a[action] = 1
            return a


    def run(self, sess, s_t, lstm_state):
        if self._continuous_mode:
            return self._run_continuous(sess, s_t, lstm_state)
        else:
            return self._run_discrete(sess, s_t, lstm_state)

    #  feed dictionary construction encapsulation
    #  needed due to annoying multiplicity of lstm states

    def loss_feed_dictionary(self, si, a, td, R, lstm_states):
        d = {
            self.s: [si],
            self.a: [a],
            self.td: [td],
            self.r: [R],
            # self.lstm_current_state_tensor: lstm_si
        }

        # add multiple lstm state vectors as necessary
        for i, state in enumerate(lstm_states):
            d[self.lstm_current_state_tensors[i]] = state

        # print "loss_feed_dict", d
        return d

    def _run_feed_dict(self, s_t, lstm_states):
        d = {self.s: [s_t]}
        for i, state in enumerate(lstm_states):

            d[self.lstm_current_state_tensors[i]] = state

        # print "_run_feed_dict", d
        return d

    def _run_continuous(self, sess, s_t, lstm_state):



        """
        next step:
        multiplex lstm state tensors for
        - eval
        - returning in combined form
        - feeding back in

        -- can i get sess.run to give me a map or something better as a result set??
        """
        tensors_to_eval = [self.mu, self.sigma2, self.v]
        tensors_to_eval.extend(self.lstm_output_state_tensors)

        # mu_out, sigma2_out, v_out, lstm_state \
        outputs = sess.run(tensors_to_eval,
                           feed_dict=self._run_feed_dict(s_t, lstm_state)
                                             # feed_dict={
                                             #     self.s: [s_t],
                                             #     self.lstm_current_state_tensor: lstm_state
                                             # }
                           )

        mu_out = outputs[0]
        sigma2_out = outputs[1]
        v_out = outputs[2]
        lstm_state = outputs[3:]

        # minor hacks...return single policy vector...structure is opaque to callers

        # print "run_continuous: mu_out=", mu_out, " sigma2_out=", sigma2_out
        pi_out = np.concatenate([sigma2_out[0], mu_out[0]], axis=0)
        # print "pi_out=", pi_out
        return pi_out, v_out[0][0], lstm_state

    def _run_discrete(self, sess, s_t, lstm_state):
        pi_out, v_out, lstm_state = sess.run([self.pi, self.v, self.lstm_output_state_tensor],
                                             feed_dict={
                                                 self.s: [s_t],
                                                 self.lstm_current_state_tensor: lstm_state
                                             })
        return pi_out[0], v_out[0][0], lstm_state

    # deprecated
    # def run_policy(self, sess, s_t):
    #     pi_out = sess.run( self.pi, feed_dict = {self.s : [s_t] } )
    #     return pi_out[0]
    #
    # def run_value(self, sess, s_t):
    #     v_out = sess.run( self.v, feed_dict = {self.s : [s_t]} )
    #     return v_out[0][0] # output is scalar

    def get_vars(self):

        vars = []
        for (W,b,y,name) in self._layers:
            # vars.append(W)
            vars.extend(flatten([W]))
            vars.extend(flatten([b]))

        return vars
        # return [
        #         # self.W_conv1, self.b_conv1,
        #         # self.W_conv2, self.b_conv2,
        #         self.W_fc0, self.b_fc0,
        #         self.W_fc1, self.b_fc1,
        #         self.W_fc2, self.b_fc2,
        #         self.W_fc3, self.b_fc3]

    def sync_from(self, src_network, name=None):
        src_vars = src_network.get_vars()
        dst_vars = self.get_vars()

        sync_ops = []

        with tf.device(self._device):
            with tf.op_scope([], name, "LowDimACNetwork") as name:
                for(src_var, dst_var) in zip(src_vars, dst_vars):
                    sync_op = tf.assign(dst_var, src_var)
                    sync_ops.append(sync_op)

                return tf.group(*sync_ops, name=name)


    ###############################################################
    ## actions

    def sample_action(self, pi_):
        return self._sample_action_fn(pi_)

    # weight initialization based on muupan's code
    # https://github.com/muupan/async-rl/blob/master/a3c_ale.py

    def _fc_weight_variable(self, shape, name="anon_weight"):
        input_channels = shape[0]
        d = 1.0 / np.sqrt(input_channels)
        # d = 1.0 / input_channels**2
        initial = tf.random_uniform(shape, minval=-d, maxval=d)
        return tf.Variable(initial, name=name)


    def _fc_bias_variable(self, shape, input_channels, name="anon_bias"):
        d = 1.0 / np.sqrt(input_channels)
        # d = 1.0 / input_channels**2
        initial = tf.random_uniform(shape, minval=-d, maxval=d)
        return tf.Variable(initial, name=name)


    def _conv_weight_variable(self, shape):
        w = shape[0]
        h = shape[1]
        input_channels = shape[2]
        d = 1.0 / np.sqrt(input_channels * w * h)
        initial = tf.random_uniform(shape, minval=-d, maxval=d)
        return tf.Variable(initial)


    def _conv_bias_variable(self, shape, w, h, input_channels):
        d = 1.0 / np.sqrt(input_channels * w * h)
        initial = tf.random_uniform(shape, minval=-d, maxval=d)
        return tf.Variable(initial)


    def _conv2d(self, x, W, stride):
        return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding="VALID")


    def _debug_save_sub(self, sess, prefix, var, name):
        var_val = var.eval(sess)
        var_val = np.reshape(var_val, (1, np.product(var_val.shape)))
        np.savetxt('./' + prefix + '_' + name + '.csv', var_val, delimiter=',')


    def debug_save(self, sess, prefix):
        # self._save_sub(sess, prefix, self.W_conv1, "W_conv1")
        # self._save_sub(sess, prefix, self.b_conv1, "b_conv1")
        # self._save_sub(sess, prefix, self.W_conv2, "W_conv2")
        # self._save_sub(sess, prefix, self.b_conv2, "b_conv2")



        for (W,b,y,name) in self._layers:
            # TODO: not implemented yet for multi-weight layers
            assert(len(W) == 1)
            self._save_sub(sess, prefix, W, W.name)
            self._save_sub(sess, prefix, b, b.name)


        # self._save_sub(sess, prefix, self.W_fc0, "W_fc0")
        # self._save_sub(sess, prefix, self.b_fc0, "b_fc0")
        # self._save_sub(sess, prefix, self.W_fc1, "W_fc1")
        # self._save_sub(sess, prefix, self.b_fc1, "b_fc1")
        # self._save_sub(sess, prefix, self.W_fc2, "W_fc2")
        # self._save_sub(sess, prefix, self.b_fc2, "b_fc2")
        # self._save_sub(sess, prefix, self.W_fc3, "W_fc3")
        # self._save_sub(sess, prefix, self.b_fc3, "b_fc3")


    def old_setup_graph(self):

        with tf.device(self._device):

            # self.W_conv1 = self._conv_weight_variable([8, 8, 4, 16])  # stride=4
            # self.b_conv1 = self._conv_bias_variable([16], 8, 8, 4)
            #
            # self.W_conv2 = self._conv_weight_variable([4, 4, 16, 32]) # stride=2
            # self.b_conv2 = self._conv_bias_variable([32], 4, 4, 16)

            self.W_fc0 = self._fc_weight_variable([self._input_size*4, self._fc0_size])
            self.b_fc0 = self._fc_bias_variable([self._fc0_size], self._input_size*4)  # unsure if i understand rationale here

            self.W_fc1 = self._fc_weight_variable([self._fc0_size, self._fc1_size])
            self.b_fc1 = self._fc_bias_variable([self._fc1_size], self._fc0_size)    # ditto ^^

            # weight for policy output layer
            self.W_fc2 = self._fc_weight_variable([self._fc1_size, self.action_size])
            self.b_fc2 = self._fc_bias_variable([self.action_size], self._fc1_size)

            # weight for value output layer
            self.W_fc3 = self._fc_weight_variable([self._fc1_size, 1])
            self.b_fc3 = self._fc_bias_variable([1], self._fc1_size)

            # state (input)
            self.s = tf.placeholder("float", [1, self._input_size, 4]) #[1, 84, 84, 4])

            # h_conv1 = tf.nn.relu(self._conv2d(self.s, self.W_conv1, 4) + self.b_conv1)
            # h_conv2 = tf.nn.relu(self._conv2d(h_conv1, self.W_conv2, 2) + self.b_conv2)
            #
            # h_conv2_flat = tf.reshape(h_conv2, [1, 2592])
            # h_fc1 = tf.nn.relu(tf.matmul(h_conv2_flat, self.W_fc1) + self.b_fc1)

            # TODO: may make sense to allow a tiny bit of convolution on even low-d inputs
            # that may have time symmetries
            state_flat = tf.reshape(self.s, [1, self._input_size*4])

            h_fc0 = tf.nn.relu(tf.matmul(state_flat, self.W_fc0) + self.b_fc0)
            #h_fc0 = tf.nn.relu(tf.matmul(self.W_fc0, self.s) + self.b_fc0)
            h_fc1 = tf.nn.relu(tf.matmul(h_fc0, self.W_fc1) + self.b_fc1)


            # policy (output)
            self.pi = tf.nn.softmax(tf.matmul(h_fc1, self.W_fc2) + self.b_fc2)
            # value (output)
            self.v = tf.matmul(h_fc1, self.W_fc3) + self.b_fc3


def sample_discrete_action(pi_):
    return choose_action(pi_)

# todo: surely there is a builtin np function for this

# TODO: use builtin np fn
# TODO: encapsulate this in the network class so sampling methods and action representation can be decoupled from training process
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


def sample_continuous_action(pi_values):
    """
    :param pi_values: concatenated vector [sigma^2, mu_1, mu_2, ... m_n]
    :return: vector of length n sampled from the multidimensional guassian with spherical covariance matrix sigma^2 * I
    """
    variance = pi_values[0]
    mean = pi_values[1:]
    cov = variance*np.identity(len(mean))

    return np.random.multivariate_normal(mean, cov)
