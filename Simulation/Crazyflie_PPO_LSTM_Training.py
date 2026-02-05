import gymnasium as gym
import argparse
from stable_baselines3 import DQN
from stable_baselines3 import PPO
import keyboard
import numpy as np
from gym_pybullet_drones.utils.enums import ObservationType, ActionType
from stable_baselines3.common.env_util import make_vec_env
from gym_pybullet_drones.envs.HoverAviary import HoverAviary
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
import os
from datetime import datetime
from gym_pybullet_drones.utils.enums import DroneModel, Physics, ActionType, ObservationType

from sb3_contrib import RecurrentPPO


DEFAULT_OBS = ObservationType('kin') # 'kin' or 'rgb'
DEFAULT_ACT = ActionType('rpm') # 'rpm' or 'pid' or 'vel' or 'one_d_rpm' or 'one_d_pid'


'''
UPDATES: Nicholas Navarrete    

YawVelocityReward_5 has the first implementation of reset random position gen
With learning batch_size= 2000, verbose=1,learning_rate=0.00001,n_epochs=10,n_steps=100000,ent_coef =0.4, clip_range=0.2

Random Position generation_1 has the best learning with 
batch_size= 2000, verbose=1,learning_rate=0.0001,n_epochs=10,n_steps=100000,ent_coef =0.4, clip_range=0.2,

Random Position generation_2 has implemented the random x an y target to be zero mean, the means were accidentally 2
Random Position generation_5 Learned to fall and stay near 0 x 0 y

Error_Observations_2 actually Flies!!!

Error_Observations_5 also flies, but is trained for longr with no rewards (see tensor board)

0.0001 LR batch size 2000 nsteps 100000 nepochs 10 clip range 0.2 ent_coef = 0.8

Inv Pen_11 Changes the clipping range from 0.2 to 0.1
Inv Pen 13 normalizes the advantage
Inv Pen 14 reduces number of epochs
Inv Pen 17 changes the learning rate
Inv Pen 18 increases pendulum reward coefficient from 0.3 to 0.4 and makes the action buffer size 0.
Inv Pen 19 changes entropy coefficient to 0.01 from 0.8  PEN  19 WORKS WOOOOOOOOOOOOOO BUT IT DOESNT FLY TO THE X Y COORDINATE
Inv Pen 20 readds the reward for the x y position PEN 20 FLIES!!!!!
Inv Pen 21 increases lr to 0.001 from 0.0006 This also flies 
Inv Pen 26 may have been trained with RPM2s
Inv Pen 28 is run with gui on to see the randomness
Added trained model rewards
Inv Pen 30 now has updated joint behavior
Inv Pen 31 has reduced pen ang variance from 25 to 10, maybe less noisy rewards?
Inv Pen 32 has updated pysics from pyb to PYB_GND_DRAG_DW
InvPen 33 flies with the updated physics

AttitudeRate implements the attitude rate controller as the input for the sytem using default physics
AttitudeRate 7 changes rewards 
rpos = np.exp(-(2*np.linalg.norm(self.TARGET_POS-Positions))**2)
            rsigma = np.exp(-(0.2*np.linalg.norm(Yaw))**2)
            rvel = np.exp(-(2*np.linalg.norm(Velocities))**2)

            to
rpos = np.exp(-(2*np.linalg.norm(self.TARGET_POS-Positions))**4)
            rsigma = np.exp(-(0.2*np.linalg.norm(Yaw))**2)
            rvel = np.exp(-(2*np.linalg.norm(Velocities))**2)

AttitudeRate 8 changes the reward back and changes the preprocess action to include 
def _preprocessAction(self, action):
    if action.ndim == 1:
        action = action.reshape(1, -1)
at the beginning
AttitudeRate 9 moves back to rpm control
AttitudeRate 10 moves back to inv pen and rpm control
InvPen 5 tries to make the inverted pen fly using lr 0.003 clip range 0.01 and ent coef 0.1 Physics PBY action type RPM
InvPen6 Fies with the above learning InvPen6_1 is the best model save of Inv Pen 6

DroneRPM uses the same learning settings and inputs as InvPen5
DroneRPM2 changes total timesteps to 0.4e7 from 1e7 This flies and is saved as PPO_DroneRPM

AttitudeRate 12 uses the same settings as DroneRPM2 but now uses the attituder rate controller
Attitude rate 13 sets the learning rate to be 0.01 from 0.003 in an effort to encourage exploration
AttitudeRate 17 lr =0.01 clip range = 0.5
AttitudeRate 25 changes pos reward to x^6
AttitudeRate 29 fixes the simulated attitude controller algorithm
AttitudeRate 30 adds 5 envs


SimToReal_1 Incorperates the REAL observation type, uses direct motor control at 200Hz, and the same learning settings as DroneRPM2
SimToReal_2 changes the action type to RPM

SimToReal_3 changes the policy learning to 
model = PPO('MlpPolicy',
             env,
             batch_size= 2000,
             verbose=1,
             learning_rate=0.0003,
             n_epochs=3,
             n_steps=5000,
             ent_coef =0.01,
             clip_range=0.1,
             device="cpu",tensorboard_log="./PPO_tensorboard/", normalize_advantage=True)

SimToReal_8 Flies well with the bellow learning settings
model = PPO('MlpPolicy',
             env,
             batch_size= 512,
             verbose=1,
             learning_rate = lambda frac: 3e-4 * frac,
             n_epochs=3,
             n_steps=4096,
             ent_coef = 0.01,
             clip_range=0.2,
             device="cpu",tensorboard_log="./PPO_tensorboard/", normalize_advantage=True)

             
SimToReal_9 Adds the velocity estimations
SimToReal_9 Also Flies
SimToReal_10 increases the reward of low velocity from 0.1 to 0.4
SimToReal_11 brings it to 0.2
SimToReal_12 brings it to 0.15, This yielded favorable results

SimToReal_13 reduced the xy limit from 10 to 3.7

SimToReal_14 added small angle perturbations to the initial rpy and randomizes it everytime the env is reset
    SimToReal_18 adds inefficient motors to the simulation
    SimToReal_19 updates the velocity buffer from 5 to 10
SimToReal_20 decreases the control freq from 200 to 120, and adds self.rpy_offset to represent the vicon offset, changed sym frequency from 10000 to 12000
SimToReal_21 changes the yaw reward coefficient from 0.2 to 0.0
SimToReal_22 changes the random rpy offset to be gaussian with 0 mean and 0.01 std instead of 0.1  std
SimToReal_23 changes the random rpy offset to be 0 and the yaw reward back to 0.2 with control freq 120
SimToReal_24 changes the control freq back to 200 to see if we can get the same learning as SimToReal_19 This works well
SimToReal_27 changes the control freq to be 90 and the sim freq to be 9000, learning rate was changed to 1e-4 *lambda from 3e-4 *lambda
STR25 works
SimToReal_28 increases the timesteps to 2e7 from 1e7    
SimToReal_30 brings back the control freq to 120 and causes the action to be +/-10% of hover rpm instead of 5%, but this looks like it shakes in place
SimToReal_31 Adds a reward for low angular velocity
SimToReal_32 changes the control freq to 200 to see how this affects learning
SimToReal_33 changes the reward for low angular velocity from 0.1 to 0.001 This is good, but hovers a little bit away from the target
SimToReal_34 changes the control freq to 120 Some shaking but goes to the target
SimToReal_35 changes the ctrl freq to 121 to match the actual frequency  
SimToReal_36 changes the Observation type to the newly created REAL2 type which flys very wokey (tm)
SimToRReal_37 changes the reward to omit the rotational velocity reward and increases the obs size from the last 10 obs to the last 100 obs This learns too slow
SimToReal_38 changes the obs size down to the last 10 obs and doubles the learning time to 2e7
SimToReal_39 Fixes some errors
SimToReal_40 changes the learning rate to be 5e-5 + (5e-5 * frac**0.5) from 1e-4 * frac
SimToReal_41 updates the noise for the 120Hz case and goes back to the real observation type. Learns well but did not save due to an error of the 
SimToReal_42 reduces the learning to to 1e7 and updates the lr to include the np.real() function. simulation freq is also updated to 120000 from 12000
SimToReal_45 changes the sim freq back to 12000 This works well and flies nicely
SimToReal_46 updates control freq from 120 to 240 Weird behavior but flies, just kind of hovers towards targe

SimToReal_47 changes the reward for position from 0.6 to 0.8
SimToReal_48 Adds obs Real3
SimToReal_49 changes the control frequency from 240 to 120 This flies excelent!
SimToReal_50 changes ineffienct motors to True This flies excellently   
SimToReal_51 changes the control freq to 121 fIRES GREAT!
SimToReal_52 Fails
SimToReal_53 Realizes that the inefficient motors were not enabled so we reenabel them This flies well
SimToReal_54 is for debugging and testing what kf is calculated to be 3.16e-10 The current offset kf model works well
SimToReal_55 increases the variance of KF offset to 1e-11 from 1.63e-17 This failed to train from computer shutdown.

We move onto PPO LSTM training
PPO_LSTM_2 changes the number of environments to 10 This learned something... not the best though
PPO_LSTM_3 decreases the variance of kf offset to 1e-12 This learns to hover but not fly to target well
PPO_LSTM_4 turns off Inefficient motors 
PPO_LSTM_6 decreases the total timesteps to 5e6 from 1e7. Changes the pos reward to 
rpos = np.exp(-(1.0*np.linalg.norm(self.TARGET_POS-Positions))**2) from rpos = np.exp(-(2*np.linalg.norm(self.TARGET_POS-Positions))**2)
Adds the progression reward. This flies towards the target, but hovers a bit away from it.

PPO_LSTM_7 Reintroduces inefficient motors and reduces the progression reward weight from 2.0 to 1.0 and increases the total timesteps to 1e7 from 5e6 and decreases control freq to 120
PPO_LSTM_8 Reruns PPO_LSTM_7 This flies well!

PPO_LSTM_9 changes the learning rate to start at 1e-3 instead of 1e-4 and changes the number of epochs to 5 from 3 Flies aopop
PPO_LSTM_12 changes initial position to 0,0,0  This does not work well
PPO_LSTM_13 changes initial pos to 0,0,0.1
PPO_LSTM_14 increases the rpos reward to 1.5 from 0.8
PPO_LSTM_15 changes the learning rate to 1e-3 + (1e-3 * np.real(frac**0.5)), from 5e-4 + (5e-4 * np.real(frac**0.5)),
PPO_LSTM_16 Tries the same thing
PPO_LSTM_17 changes the lr back and the eval freq to 1000 from 10000 intial pos to 0 0 0
PPO LSTM 19 changes ent coef to 0.02 from  0.01 clip range to 0.3 from 0.1
'''

training_initial_pos = np.array([[0,0,0]])
training_initial_pos.reshape(1,3)

Sim_Freq=12000
control_freq = 120 #in Hz

env = make_vec_env(HoverAviary,env_kwargs=dict(obs=ObservationType.REAL3,
                                                act=ActionType.RPM2,
                                                ctrl_freq=control_freq,
                                                pyb_freq=Sim_Freq,
                                                initial_xyzs = training_initial_pos,
                                                drone_model = DroneModel.CF2X,
                                                physics=Physics.PYB,
                                                Inefficient_Motors = True,
                                                Inflight_Motor_Variance = False
                                                ),n_envs=10)

target_reward = 10000
filename = os.path.join('SimToReal_PPO_LSTM', 'save-'+datetime.now().strftime("%m.%d.%Y_%H.%M.%S")) ##was InvPen

if not os.path.exists(filename):
    os.makedirs(filename+'/')

callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=target_reward, verbose=0)
eval_callback = EvalCallback(env,callback_on_new_best=callback_on_best,verbose=1,
best_model_save_path=filename+'/',deterministic=True,render=False, eval_freq=1000)    
  



#### Train the model ####################################### 4465
model = RecurrentPPO('MlpLstmPolicy',
             env,
             batch_size= 512,
             verbose=1,
             learning_rate=lambda frac: 5e-4 + (5e-4 * np.real(frac**0.5)),
             n_epochs=5,
             n_steps=4096,
             ent_coef = 0.02,
             clip_range=0.3,
             device="cpu",tensorboard_log="./PPO_LSTM_tensorboard/", normalize_advantage=True)

model.learn(total_timesteps= 5e6,
            callback=eval_callback,
            log_interval=10, progress_bar=True,tb_log_name="SimToReal_PPO_LSTM")  ##was InvPen

model.save("PPO_LSTM_SimToReal")#was PPO_Latest_InvPen

env = HoverAviary(gui=True, ctrl_freq=control_freq, pyb_freq=Sim_Freq,obs=ObservationType.REAL3, act=ActionType.RPM2    ,drone_model = DroneModel.CF2X, initial_xyzs=training_initial_pos, Inefficient_Motors=True )
obs,info = env.reset()

input("Training is Done! Press enter to continue")
rewardBuffer=0
rewardBufferAverage=[]

# cell and hidden state of the LSTM
lstm_states = None
num_envs = 1
# Episode start signals are used to reset the lstm states
episode_starts = np.ones((num_envs,), dtype=bool)

while not keyboard.is_pressed('esc'):
    action, lstm_states = model.predict(obs, state=lstm_states, episode_start=episode_starts, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    rewardBuffer=rewardBuffer+reward
    episode_starts = np.array([terminated or truncated])
    #print(str(action[:,3]) +'\r')
    if terminated or truncated:
        rewardBufferAverage.append(rewardBuffer)
        print("average reward over simulations:   " +str(sum(rewardBufferAverage)/len(rewardBufferAverage)))
        rewardBuffer=0
        obs, info = env.reset()
        print("Target Position: " + str(env.TARGET_POS)+ "\n")