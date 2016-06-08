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

## milestones

- validate on multiple tasks, ensure robust implementation

    - half-cheetah should actually train much faster and better...try with larger network
    x try readding state history
    - look at LSTM training again...may be holding us back
        - more complex lstm model
      			- finish extending with peepholes, etc
        - try doing lstm in a `tf.while_loop` for easier graph setup for BPTT?
        - look up TF BPTT impls
        - do TF BPTT the hard way if necessary

        - or elim in favor of deeper fc network
    - try adding another fc layer
    - figure out if lstm hidden state training is working at all


- validate that i can bootstrap with experience replay and elite samples
    (necessary precondition for unity being a viable simulation environment)

    -

- connect to unity and validate that basic tasks can work there

## InvertedPendulum-v1

seems to begin to learn but diverge shortly thereafter, h=16, lstm=16 and h=32, lstm=32

symptoms: score -> 1, actions grow >> box bounds

turned down learning rate to 10^-4


basic questions:

x am i sampling the actions correctly? mu and sigma^2 mixed up?
  x checked that
x action feedback appears correct also
x learning rate still too high?
    - trying .5e-4 w/ 128h, 128lstm network
        - still failed. score never increased after 300k iterations
    - NOPE: works at 7e-4 w/ grad norm clipped to 10

x initialize weight values much smaller so we are less likely to start in a divergent setting?

x FIXED WRONG SIGN OF `policy_loss`

WORKS. COMMITTED.


## Reacher-v1


noticed intial behavior is wild and fast...maybe weights too large?
does slow down though

x decrease initial weight magnitudes by a bunch
  `[-1,+1]/sqrt(inputs)` --> `[-1,-1]/inputs**2`

x make reacher-v1.djm - reset location on success. unclear why this doesnt happen now
    x oops gym has terminal timestep count, implemented auto-reset
    - may need to update gym to get reward thresholds pulled in


x maaaybe? cost function has wrong sign/ "minimize" will go too far when rewards are all negative
  - YES THIS


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

x printlns
x inspect graph structure


x FIXED BUG: lstm state was not reset with env

### with 32h,32lstm x mu,sigma2,V nets

converged to local minima and stayed there for 30+minutes. gradients mostly zero.

- try with larger network?


### with 200hx128lstm

and fixed lstm state bug

at least doesnt diverge quickly, but learning still seems to get stuck.

CHECK: perhaps adding state history to input will let it learn velocities more easily? lstm should be able to learn them in principle but in practice...hard to say

## HalfCheetah-v1

x clip actions to space so simulation doesnt get fucked...completely
    - nope, still flips over and gets fucked right off the bat

x rethink intialization...how can i possible have a general algorithm that avoids jumping in to horrible regions of solution space right off the bat?
    x understand what the actions individually represent
    x select safe values within that space
    x restarting the task and detecting failures is how people do it in the literature. dont rathole on this

x need to reset after get in to a bad state

- P(a) too small !! need to use log prob

- logprob starts to explode...seems fine when > -1e6 or so, but
    - score peaks after a while at ~- 1e7
    - then grows after that...which makes little sense variance didnt decrease....so...wtf???

- achieved decent perf (100-200) briefly then went nuts...u diverged massively, logprob became massively negative.
    - what happened to gradients?



### retry with lower learning rate 1e-4

7e-4 may be too high according to a3c paper


### with 32h 32lstm x 3 nets

Tmax = 100

- may work better with Tmax smaller Tmax
- may work better with shorter timestep-limit
- may work better with larger networks...lots of joints to control for only 32 encoder neurons





# Possible experimental general improvements

- readd initial random action to help with exploration

- potentially boost initial weight magnitudes again since that wasnt the root of the problem



# lingering questions


- fixed entropy term...sort of?
    - was prevented from being negative before, now is initially...and could easily remain so depending on the variance...think this needs more thought


- check policy pdf computed properly...check by hand


x check if train.minimize is minimizing abs val or magnitude
    !!!!!! or if policy loss should be able to be positive for that matter??? !!!!!!!

- see if adding back in input history kills perf

- is lstm hidden state carrying useful information?

    - try comparing FF and LSTM versions...
        - difficult to compare directly

    - try setting hidden state to constant after each step and see if learning suffers
        - need to look at lstm paper to pick a good constant. 0 or 1 may not be right

