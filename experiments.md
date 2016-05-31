# a3c experiments



## decent results with cartpole

think we had


    LOCAL_T_MAX = 10 # or was it 5??
    RMSP_ALPHA = 0.99 # decay parameter for RMSProp
    RMSP_EPSILON = 0.1 # epsilon parameter for RMSProp
    CHECKPOINT_DIR = 'checkpoints'
    LOG_FILE = 'tmp/a3c_log'
    INITIAL_ALPHA_LOW = 1e-4    # log_uniform low limit for learning rate
    INITIAL_ALPHA_HIGH = 1e-2   # log_uniform high limit for learning rate

    PARALLEL_SIZE = 16 # parallel thread size
    ROM = "pong.bin"     # action size = 3
    ACTION_SIZE = 3 # action size

    INITIAL_ALPHA_LOG_RATE = 0.4226 # log_uniform interpolate rate for learning rate (around 7 * 10^-4)
    # INITIAL_ALPHA_LOG_RATE = 0 # set low for cartpole
    GAMMA = 0.99 # discount factor for rewards
    ENTROPY_BETA = 0.01 # entropy regurarlization constant
    MAX_TIME_STEP = 6 * 10**7
    GRAD_NORM_CLIP = 10.0 # gradient norm clipping



## applying to mountaincar

oohh and total fucking failure. WTF.


so, it never sees a terminal state so the constant -1 reward is what it learns to predict...and since it predicts that perfectly after a short time...loss functions drop to zero




## acrobot

also failed...though did reach terminal states a few times


### trying with small network

16 input layer, 16 lstm units


failed

### with small network, tmax=20

longer tmax seems to help it learn more from initially rare rewards




# SCREW YOU GUYS


> For physical control tasks we used reward functions which provide feedback at every step. In all
tasks, the reward contained a small action cost. For all tasks that have a static goal state (e.g.
pendulum swingup and reaching) we provide a smoothly varying reward based on distance to a goal
state, and in some cases an additional positive reward when within a small radius of the target state.
For grasping and manipulation tasks we used a reward with a term which encourages movement
towards the payload and a second component which encourages moving the payload to the target. In
locomotion tasks we reward forward action and penalize hard impacts to encourage smooth rather
than hopping gaits (Schulman et al., 2015b). In addition, we used a negative reward and early
termination for falls which were determined by simple threshholds on the height and torso angle (in
the case of walker2d).


