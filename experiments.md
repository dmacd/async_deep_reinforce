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




# continuous domain control tasks


## InvertedPendulum-v1

seems to begin to learn but diverge shortly thereafter, h=16, lstm=16 and h=32, lstm=32

symptoms: score -> 1, actions grow >> box bounds

turned down learning rate to 10^-4


basic questions:

x am i sampling the actions correctly? mu and sigma^2 mixed up?
  x checked that
x action feedback appears correct also
- learning rate still too high?
    - trying .5e-4 w/ 128h, 128lstm network
        - still failed. score never increased after 300k iterations
- fixed entropy term...sort of?
    - was prevented from being negative before, now is initially...and could easily remain so depending on the variance...think this needs more thought

- check policy pdf computed properly...check by hand

- check if train.minimize is minimizing abs val or magnitude


- initialize weight values much smaller so we are less likely to start in a divergent setting?


- initial sampled action space applied to env may make some tasks impossible out of the game?

## HalfCheetah-v1

x clip actions to space so simulation doesnt get fucked...completely
    - nope, still flips over and gets fucked right off the bat

- rethink intialization...how can i possible have a general algorithm that avoids jumping in to horrible regions of solution space right off the bat?
    - understand what the actions individually represent
    - select safe values within that space

- need to reset after get in to a bad state

## Reacher-v1


noticed intial behavior is wild and fast...maybe weights too large?
does slow down though

- decrease initial weight magnitudes by a bunch
  `[-1,+1]/sqrt(inputs)` --> `[-1,-1]/inputs**2`

- make reacher-v1.djm - reset location on success. unclear why this doesnt happen now

    - may need to update gym to get reward thresholds pulled in


- maaaybe? cost function has wrong sign/ "minimize" will go too far when rewards are all negative


*almost* converged after 3 hours...but diverged again after a period of good perf


Retrying with

- higher learning rate
- Tmax=100 (> episode timestep limit)

nope.

Retrying with:

- slightly fixed loss function (variance normalization term exponent was missing)
- 32h,32lstm network again
- higher learning rate (7e-4)
- Tmax=100

- marginally better perf, was sorta learning kinda



Retrying with: split V, sigma, and mu networks



verification checks:

- printlns
- inspect graph structure
-


Added (!) lstm state reset
