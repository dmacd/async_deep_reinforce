# -*- coding: utf-8 -*-

LOCAL_T_MAX = 100                    # repeat step size
RMSP_ALPHA = 0.99                   # decay parameter for RMSProp
RMSP_EPSILON = 0.1                  # epsilon parameter for RMSProp
CHECKPOINT_DIR = 'checkpoints'
LOG_FILE = 'tmp/a3c_log'
INITIAL_ALPHA_LOW = 1e-4            # log_uniform low limit for learning rate
INITIAL_ALPHA_HIGH = 1e-2           # log_uniform high limit for learning rate

PARALLEL_SIZE = 16                  # parallel thread size
ROM = "pong.bin"                    # action size = 3
ACTION_SIZE = 3                     # action size

INITIAL_ALPHA_LOG_RATE = 0.4226     # log_uniform interpolate rate for learning rate (around 7 * 10^-4)
# INITIAL_ALPHA_LOG_RATE = 0          # set lower for HalfCheetah
GAMMA = 0.99                        # discount factor for rewards
# ENTROPY_BETA = 0.05               # entropy regurarlization constant
ENTROPY_BETA = 1e-4                 # entropy regurarlization constant
MAX_TIME_STEP = 4 * 10**6
GRAD_NORM_CLIP = 10.0               # gradient norm clipping
USE_GPU = False                     # To use GPU, set True




CONTINUOUS_MODE = True

if CONTINUOUS_MODE:
    # continuous envs
    # GYM_ENV="InvertedPendulum-v1"
    # GYM_ENV="HalfCheetah-v1"
    GYM_ENV="Reacher-v1"

else:
    GYM_ENV="CartPole-v0"
    # GYM_ENV="MountainCar-v0"
    # GYM_ENV="Acrobot-v0"
    # GYM_ENV="Pendulum-v0"

    # GYM_ENV="LunarLander-v1" # threading problem? non-ui versions still create gl problems i guess
    # GYM_ENV="BipedalWalker-v1" # Box actionspace...not there yet anyway

