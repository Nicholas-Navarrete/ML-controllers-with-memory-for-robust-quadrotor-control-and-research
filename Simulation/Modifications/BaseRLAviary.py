import os
import numpy as np
import pybullet as p
from gymnasium import spaces
from collections import deque

from gym_pybullet_drones.envs.BaseAviary import BaseAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics, ActionType, ObservationType, ImageType
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl

"""
UPDATES BY NICHOLAS NAVARRETE:
Updated init to initialize with self.TARGET_POS which is set by the child class
Updated _computeObs() to have the first 3 values of the observation be the errors, instead of the position

added the observation space of 16 for the inverted pendulum case same for compute obs
Added RPM2 which is just a straight 0 to 1 of 0 rpm to max rpm
added RATE_PID which is the actions space when using the attitude rate controller

Added observation REAL which is the real drone kinematic information (pose, linear and angular velocities) with emperical noise

Added rpy offset to represent the rotational offset of the vicon system

Added Real 2 which is like REAL but with a new observations structure. [x,y,z, r,p,y, last m1, last m2, last m3, last m4] stacked for the last 10 control steps
"""

class BaseRLAviary(BaseAviary):
    """Base single and multi-agent environment class for reinforcement learning."""
    
    ################################################################################

    def __init__(self,
                 drone_model: DroneModel=DroneModel.CF2X,
                 num_drones: int=1,
                 neighbourhood_radius: float=np.inf,
                 initial_xyzs=None,
                 initial_rpys=None,
                 physics: Physics=Physics.PYB,
                 pyb_freq: int = 240,
                 ctrl_freq: int = 240,
                 gui=False,
                 record=False,
                 obs: ObservationType=ObservationType.KIN,
                 act: ActionType=ActionType.RPM,
                 Inefficient_Motors = False,
                 randomize_init_rpy = False,
                 Inflight_Motor_Variance = False,
                 kf_Variance = None,
                 GoalPos = False
                 ):
        """Initialization of a generic single and multi-agent RL environment.

        Attributes `vision_attributes` and `dynamics_attributes` are selected
        based on the choice of `obs` and `act`; `obstacles` is set to True 
        and overridden with landmarks for vision applications; 
        `user_debug_gui` is set to False for performance.
        
        Parameters
        ----------
        drone_model : DroneModel, optional
            The desired drone type (detailed in an .urdf file in folder `assets`).
        num_drones : int, optional
            The desired number of drones in the aviary.
        neighbourhood_radius : float, optional
            Radius used to compute the drones' adjacency matrix, in meters.
        initial_xyzs: ndarray | None, optional
            (NUM_DRONES, 3)-shaped array containing the initial XYZ position of the drones.
        initial_rpys: ndarray | None, optional
            (NUM_DRONES, 3)-shaped array containing the initial orientations of the drones (in radians).
        physics : Physics, optional
            The desired implementation of PyBullet physics/custom dynamics.
        pyb_freq : int, optional
            The frequency at which PyBullet steps (a multiple of ctrl_freq).
        ctrl_freq : int, optional
            The frequency at which the environment steps.
        gui : bool, optional
            Whether to use PyBullet's GUI.
        record : bool, optional
            Whether to save a video of the simulation.
        obs : ObservationType, optional
            The type of observation space (kinematic information or vision)
        act : ActionType, optional
            The type of action space (1 or 3D; RPMS, thurst and torques, waypoint or velocity with PID control; etc.)

        """
        #### Create a buffer for the last .5 sec of actions ########
        self.ACTION_BUFFER_SIZE = 0 # int(ctrl_freq//2)  ## We can maybe set this to 0
        self.action_buffer = deque(maxlen=self.ACTION_BUFFER_SIZE)

        self.Real2_Size = 10

        ####
        vision_attributes = True if obs == ObservationType.RGB else False
        self.OBS_TYPE = obs
        self.ACT_TYPE = act
        self.TARGET_POS = []
        self.GoalPos = GoalPos
        
        self.velBufSize=10
        self.xVel=np.zeros(self.velBufSize)
        self.yVel=np.zeros(self.velBufSize)
        self.zVel=np.zeros(self.velBufSize)
        self.yawRate=np.zeros(self.velBufSize)
        self.rollRate=np.zeros(self.velBufSize) 
        self.pitchRate=np.zeros(self.velBufSize)

        self.Inflight_Motor_Variance = Inflight_Motor_Variance

        #### Create integrated controllers #########################
        if act in [ActionType.PID, ActionType.VEL, ActionType.ONE_D_PID]:
            os.environ['KMP_DUPLICATE_LIB_OK']='True'
            if drone_model in [DroneModel.CF2X, DroneModel.CF2P]:
                self.ctrl = [DSLPIDControl(drone_model=DroneModel.CF2X) for i in range(num_drones)]
            else:
                print("[ERROR] in BaseRLAviary.__init()__, no controller is available for the specified drone_model")
        super().__init__(drone_model=drone_model,
                         num_drones=num_drones,
                         neighbourhood_radius=neighbourhood_radius,
                         initial_xyzs=initial_xyzs,
                         initial_rpys=initial_rpys,
                         physics=physics,
                         pyb_freq=pyb_freq,
                         ctrl_freq=ctrl_freq,
                         gui=gui,
                         record=record, 
                         obstacles=True, # Add obstacles for RGB observations and/or FlyThruGat
                         user_debug_gui=True, # Remove of RPM sliders from all single agent learning aviaries False for learning
                         vision_attributes=vision_attributes,
                         Inefficient_Motors=Inefficient_Motors,
                         randomize_init_rpy=randomize_init_rpy,
                         Inflight_Motor_Variance=Inflight_Motor_Variance,
                         kf_Variance=kf_Variance
                         )
        #### Set a limit on the maximum target speed ###############
        if act == ActionType.VEL:
            self.SPEED_LIMIT = 0.03 * self.MAX_SPEED_KMH * (1000/3600)
        #The following is for the real2 observation space
        # For actions (shape: NUM_DRONES × 4)
        
        self.LastAction = np.zeros((self.NUM_DRONES, 4))
        self.LastActions = np.zeros((self.NUM_DRONES, 5, 4))
        # For obs_1000 (shape: NUM_DRONES × 1000)
        self.LastObservation = np.zeros((self.NUM_DRONES, 10*self.Real2_Size))
    ################################################################################

    def _addObstacles(self):
        """Add obstacles to the environment.

        Only if the observation is of type RGB, 4 landmarks are added.
        Overrides BaseAviary's method.

        """
        if self.OBS_TYPE == ObservationType.RGB:
            p.loadURDF("block.urdf",
                       [1, 0, .1],
                       p.getQuaternionFromEuler([0, 0, 0]),
                       physicsClientId=self.CLIENT
                       )
            p.loadURDF("cube_small.urdf",
                       [0, 1, .1],
                       p.getQuaternionFromEuler([0, 0, 0]),
                       physicsClientId=self.CLIENT
                       )
            p.loadURDF("duck_vhacd.urdf",
                       [-1, 0, .1],
                       p.getQuaternionFromEuler([0, 0, 0]),
                       physicsClientId=self.CLIENT
                       )
            p.loadURDF("teddy_vhacd.urdf",
                       [0, -1, .1],
                       p.getQuaternionFromEuler([0, 0, 0]),
                       physicsClientId=self.CLIENT
                       )
        else:
            pass

    ################################################################################

    def _actionSpace(self):
        """Returns the action space of the environment.

        Returns
        -------
        spaces.Box
            A Box of size NUM_DRONES x 4, 3, or 1, depending on the action type.

        """
        if self.ACT_TYPE in [ActionType.RPM, ActionType.VEL, ActionType.RPM2, ActionType.RATE_PID]:
            size = 4
        elif self.ACT_TYPE==ActionType.PID:
            size = 3
        elif self.ACT_TYPE in [ActionType.ONE_D_RPM, ActionType.ONE_D_PID]:
            size = 1
        else:
            print("[ERROR] in BaseRLAviary._actionSpace()")
            exit()
        act_lower_bound = np.array([-1*np.ones(size) for i in range(self.NUM_DRONES)])
        act_upper_bound = np.array([+1*np.ones(size) for i in range(self.NUM_DRONES)])
        #
        for i in range(self.ACTION_BUFFER_SIZE):
            self.action_buffer.append(np.zeros((self.NUM_DRONES,size)))
        #

        ## Uncomment if you want the thrust to be from 0 to 1
        # if self.ACT_TYPE == ActionType.RATE_PID:
        #     act_lower_bound = np.array([-1*np.ones(size) for i in range(self.NUM_DRONES)])
        #     act_upper_bound = np.array([+1*np.ones(size) for i in range(self.NUM_DRONES)])

        #     act_lower_bound[:,3]=0
        #     act_upper_bound[:,3]=1

        if self.ACT_TYPE == ActionType.RPM2:  #This is different becasue we want the lower bound to be 0
            return spaces.Box(low=np.array([0*np.ones(size) for i in range(self.NUM_DRONES)]),
                              high=act_upper_bound, dtype=np.float32)

            
        return spaces.Box(low=act_lower_bound, high=act_upper_bound, dtype=np.float32)

    ################################################################################

    def _preprocessAction(self,
                          action
                          ):
        """Pre-processes the action passed to `.step()` into motors' RPMs.

        Parameter `action` is processed differenly for each of the different
        action types: the input to n-th drone, `action[n]` can be of length
        1, 3, or 4, and represent RPMs, desired thrust and torques, or the next
        target position to reach using PID control.

        Parameter `action` is processed differenly for each of the different
        action types: `action` can be of length 1, 3, or 4 and represent 
        RPMs, desired thrust and torques, the next target position to reach 
        using PID control, a desired velocity vector, etc.

        Parameters
        ----------
        action : ndarray
            The input action for each drone, to be translated into RPMs.

        Returns
        -------
        ndarray
            (NUM_DRONES, 4)-shaped array of ints containing to clipped RPMs
            commanded to the 4 motors of each drone.

        """

        if self.Inflight_Motor_Variance == True:
            self.KF_Offset = np.array([random.gauss(0,1e-12) for _ in range(4)])

        if action.ndim == 1:
            action = action.reshape(1, -1)

        self.action_buffer.append(action)
        rpm = np.zeros((self.NUM_DRONES,4))
        for k in range(action.shape[0]):
            target = action[k, :]
            if self.ACT_TYPE == ActionType.RPM:

                self.LastAction[k] = target
                # Shift all old actions one step toward the end
                self.LastActions[k, 1:] = self.LastActions[k, :-1]
                # Store newest action at index 0
                self.LastActions[k, 0] = target

                rpm[k,:] = np.array(self.HOVER_RPM * (1+0.1*target)) ##### Changed from 5% to 10%
            elif self.ACT_TYPE == ActionType.RPM2:
                rpm[k,:] = np.array(self.MAX_RPM*target)
            elif self.ACT_TYPE == ActionType.PID:
                state = self._getDroneStateVector(k)
                next_pos = self._calculateNextStep(
                    current_position=state[0:3],
                    destination=target,
                    step_size=1,
                    )
                rpm_k, _, _ = self.ctrl[k].computeControl(control_timestep=self.CTRL_TIMESTEP,
                                                        cur_pos=state[0:3],
                                                        cur_quat=state[3:7],
                                                        cur_vel=state[10:13],
                                                        cur_ang_vel=state[13:16],
                                                        target_pos=next_pos
                                                        )
                rpm[k,:] = rpm_k

            elif self.ACT_TYPE == ActionType.RATE_PID:
                state = self._getDroneStateVector(k)
                
                rpm[k,:] = self._Attitude_Rate_PID_Simulation(state,target)

            elif self.ACT_TYPE == ActionType.VEL:
                state = self._getDroneStateVector(k)
                if np.linalg.norm(target[0:3]) != 0:
                    v_unit_vector = target[0:3] / np.linalg.norm(target[0:3])
                else:
                    v_unit_vector = np.zeros(3)
                temp, _, _ = self.ctrl[k].computeControl(control_timestep=self.CTRL_TIMESTEP,
                                                        cur_pos=state[0:3],
                                                        cur_quat=state[3:7],
                                                        cur_vel=state[10:13],
                                                        cur_ang_vel=state[13:16],
                                                        target_pos=state[0:3], # same as the current position
                                                        target_rpy=np.array([0,0,state[9]]), # keep current yaw
                                                        target_vel=self.SPEED_LIMIT * np.abs(target[3]) * v_unit_vector # target the desired velocity vector
                                                        )
                rpm[k,:] = temp
            elif self.ACT_TYPE == ActionType.ONE_D_RPM:
                rpm[k,:] = np.repeat(self.HOVER_RPM * (1+0.05*target), 4)
            elif self.ACT_TYPE == ActionType.ONE_D_PID:
                state = self._getDroneStateVector(k)
                res, _, _ = self.ctrl[k].computeControl(control_timestep=self.CTRL_TIMESTEP,
                                                        cur_pos=state[0:3],
                                                        cur_quat=state[3:7],
                                                        cur_vel=state[10:13],
                                                        cur_ang_vel=state[13:16],
                                                        target_pos=state[0:3]+0.1*np.array([0,0,target[0]])
                                                        )
                rpm[k,:] = res
            else:
                print("[ERROR] in BaseRLAviary._preprocessAction()")
                exit()
        return rpm

    ################################################################################

    def _observationSpace(self):
        """Returns the observation space of the environment.

        Returns
        -------
        ndarray
            A Box() of shape (NUM_DRONES,H,W,4) or (NUM_DRONES,12) depending on the observation type.

            Added 16 space version
        """
        if self.OBS_TYPE == ObservationType.RGB:
            return spaces.Box(low=0,
                              high=255,
                              shape=(self.NUM_DRONES, self.IMG_RES[1], self.IMG_RES[0], 4), dtype=np.uint8)
        elif self.OBS_TYPE == ObservationType.KIN or self.OBS_TYPE == ObservationType.REAL :
            if self.DRONE_MODEL == DroneModel.CF2XINVPEN:
                ############################################################
                #### OBS SPACE OF SIZE 16
                #### Observation vector ### X        Y        Z       Q1   Q2   Q3   Q4   R       P       Y       VX       VY       VZ       WX       WY       WZ  penxrot penyrot penxrotVel penyrotVel
                lo = -np.inf
                hi = np.inf
                obs_lower_bound = np.array([[lo,lo,0, lo,lo,lo,lo,lo,lo,lo,lo,lo,-np.pi,-np.pi,lo,lo] for i in range(self.NUM_DRONES)])
                obs_upper_bound = np.array([[hi,hi,hi,hi,hi,hi,hi,hi,hi,hi,hi,hi,np.pi,np.pi,hi,hi] for i in range(self.NUM_DRONES)])
                #### Add action buffer to observation space ################
                act_lo = -1
                act_hi = +1
                for i in range(self.ACTION_BUFFER_SIZE):
                    if self.ACT_TYPE in [ActionType.RPM, ActionType.VEL]:
                        obs_lower_bound = np.hstack([obs_lower_bound, np.array([[act_lo,act_lo,act_lo,act_lo] for i in range(self.NUM_DRONES)])])
                        obs_upper_bound = np.hstack([obs_upper_bound, np.array([[act_hi,act_hi,act_hi,act_hi] for i in range(self.NUM_DRONES)])])
                    elif self.ACT_TYPE==ActionType.PID:
                        obs_lower_bound = np.hstack([obs_lower_bound, np.array([[act_lo,act_lo,act_lo] for i in range(self.NUM_DRONES)])])
                        obs_upper_bound = np.hstack([obs_upper_bound, np.array([[act_hi,act_hi,act_hi] for i in range(self.NUM_DRONES)])])
                    elif self.ACT_TYPE in [ActionType.ONE_D_RPM, ActionType.ONE_D_PID]:
                        obs_lower_bound = np.hstack([obs_lower_bound, np.array([[act_lo] for i in range(self.NUM_DRONES)])])
                        obs_upper_bound = np.hstack([obs_upper_bound, np.array([[act_hi] for i in range(self.NUM_DRONES)])])
                return spaces.Box(low=obs_lower_bound, high=obs_upper_bound, dtype=np.float32)
                ############################################################
            
            else:
                ############################################################
                #### OBS SPACE OF SIZE 12
                #### Observation vector ### X        Y        Z       Q1   Q2   Q3   Q4   R       P       Y       VX       VY       VZ       WX       WY       WZ
                lo = -np.inf
                hi = np.inf
                obs_lower_bound = np.array([[lo,lo,0, lo,lo,lo,lo,lo,lo,lo,lo,lo] for i in range(self.NUM_DRONES)])
                obs_upper_bound = np.array([[hi,hi,hi,hi,hi,hi,hi,hi,hi,hi,hi,hi] for i in range(self.NUM_DRONES)])
                #### Add action buffer to observation space ################
                act_lo = -1
                act_hi = +1
                for i in range(self.ACTION_BUFFER_SIZE):
                    if self.ACT_TYPE in [ActionType.RPM, ActionType.VEL]:
                        obs_lower_bound = np.hstack([obs_lower_bound, np.array([[act_lo,act_lo,act_lo,act_lo] for i in range(self.NUM_DRONES)])])
                        obs_upper_bound = np.hstack([obs_upper_bound, np.array([[act_hi,act_hi,act_hi,act_hi] for i in range(self.NUM_DRONES)])])
                    elif self.ACT_TYPE==ActionType.PID:
                        obs_lower_bound = np.hstack([obs_lower_bound, np.array([[act_lo,act_lo,act_lo] for i in range(self.NUM_DRONES)])])
                        obs_upper_bound = np.hstack([obs_upper_bound, np.array([[act_hi,act_hi,act_hi] for i in range(self.NUM_DRONES)])])
                    elif self.ACT_TYPE in [ActionType.ONE_D_RPM, ActionType.ONE_D_PID]:
                        obs_lower_bound = np.hstack([obs_lower_bound, np.array([[act_lo] for i in range(self.NUM_DRONES)])])
                        obs_upper_bound = np.hstack([obs_upper_bound, np.array([[act_hi] for i in range(self.NUM_DRONES)])])
                return spaces.Box(low=obs_lower_bound, high=obs_upper_bound, dtype=np.float32)
                ############################################################
        elif self.OBS_TYPE == ObservationType.REAL2:
                ############################################################
                #### OBS SPACE OF SIZE 40
                #### Observation vector ### [X    Y     Z     R    P    Y   last m1, last m2, last m3, last m4] stacked for the last 10 control steps
                lo = -np.inf
                hi = np.inf
                obs_lower_bound = np.array([[lo,lo,lo, lo,lo,lo, -1,-1,-1,-1]*self.Real2_Size for i in range(self.NUM_DRONES)])
                obs_upper_bound = np.array([[hi,hi,hi,hi,hi,hi, 1,1,1,1]*self.Real2_Size for i in range(self.NUM_DRONES)])
                #### Add action buffer to observation space ################
                act_lo = -1
                act_hi = +1
                for i in range(self.ACTION_BUFFER_SIZE):
                    if self.ACT_TYPE in [ActionType.RPM,ActionType.RPM2,ActionType.VEL,ActionType.RATE_PID]:
                        obs_lower_bound = np.hstack([obs_lower_bound, np.array([[act_lo,act_lo,act_lo,act_lo] for i in range(self.NUM_DRONES)])])
                        obs_upper_bound = np.hstack([obs_upper_bound, np.array([[act_hi,act_hi,act_hi,act_hi] for i in range(self.NUM_DRONES)])])
                    elif self.ACT_TYPE==ActionType.PID:
                        obs_lower_bound = np.hstack([obs_lower_bound, np.array([[act_lo,act_lo,act_lo] for i in range(self.NUM_DRONES)])])
                        obs_upper_bound = np.hstack([obs_upper_bound, np.array([[act_hi,act_hi,act_hi] for i in range(self.NUM_DRONES)])])
                    elif self.ACT_TYPE in [ActionType.ONE_D_RPM, ActionType.ONE_D_PID]:
                        obs_lower_bound = np.hstack([obs_lower_bound, np.array([[act_lo] for i in range(self.NUM_DRONES)])])
                        obs_upper_bound = np.hstack([obs_upper_bound, np.array([[act_hi] for i in range(self.NUM_DRONES)])])
                return spaces.Box(low=obs_lower_bound, high=obs_upper_bound, dtype=np.float32)
                ############################################################

        elif self.OBS_TYPE == ObservationType.REAL3:
            ############################################################
            #### OBS SPACE OF SIZE 20
            #### Observation vector ### [X, Y,Z, VX, VY, VZ, R, P, Y, WR, WP, WY, avg_m1, avg_m2, avg_m3, avg_m4, delta_m1, delta_m2, delta_m3, delta_m4]
            lo = -np.inf
            hi = np.inf
            obs_lower_bound = np.array([[lo,lo,lo, lo,lo,lo, lo,lo,lo, lo,lo,lo, -1,-1,-1,-1, -2,-2,-2,-2] for i in range(self.NUM_DRONES)])
            obs_upper_bound = np.array([[hi,hi,hi, hi,hi,hi, hi,hi,hi, hi,hi,hi,  1, 1, 1, 1,  2, 2, 2, 2] for i in range(self.NUM_DRONES)])
            #### Add action buffer to observation space ################
            act_lo = -1
            act_hi = +1
            for i in range(self.ACTION_BUFFER_SIZE):
                if self.ACT_TYPE in [ActionType.RPM,ActionType.RPM2,ActionType.VEL,ActionType.RATE_PID]:
                    obs_lower_bound = np.hstack([obs_lower_bound, np.array([[act_lo,act_lo,act_lo,act_lo] for i in range(self.NUM_DRONES)])])
                    obs_upper_bound = np.hstack([obs_upper_bound, np.array([[act_hi,act_hi,act_hi,act_hi] for i in range(self.NUM_DRONES)])])
                elif self.ACT_TYPE==ActionType.PID:
                    obs_lower_bound = np.hstack([obs_lower_bound, np.array([[act_lo,act_lo,act_lo] for i in range(self.NUM_DRONES)])])
                    obs_upper_bound = np.hstack([obs_upper_bound, np.array([[act_hi,act_hi,act_hi] for i in range(self.NUM_DRONES)])])
                elif self.ACT_TYPE in [ActionType.ONE_D_RPM, ActionType.ONE_D_PID]:
                    obs_lower_bound = np.hstack([obs_lower_bound, np.array([[act_lo] for i in range(self.NUM_DRONES)])])
                    obs_upper_bound = np.hstack([obs_upper_bound, np.array([[act_hi] for i in range(self.NUM_DRONES)])])
            return spaces.Box(low=obs_lower_bound, high=obs_upper_bound, dtype=np.float32)
            ############################################################

        else:
            print("[ERROR] in BaseRLAviary._observationSpace()")
    
    ################################################################################

    def _computeObs(self):
        """Returns the current observation of the environment.

        Returns
        -------
        ndarray
            A Box() of shape (NUM_DRONES,H,W,4) or (NUM_DRONES,12) depending on the observation type.

        """
        if self.OBS_TYPE == ObservationType.RGB:
            if self.step_counter%self.IMG_CAPTURE_FREQ == 0:
                for i in range(self.NUM_DRONES):
                    self.rgb[i], self.dep[i], self.seg[i] = self._getDroneImages(i,
                                                                                 segmentation=False
                                                                                 )
                    #### Printing observation to PNG frames example ############
                    if self.RECORD:
                        self._exportImage(img_type=ImageType.RGB,
                                          img_input=self.rgb[i],
                                          path=self.ONBOARD_IMG_PATH+"drone_"+str(i),
                                          frame_num=int(self.step_counter/self.IMG_CAPTURE_FREQ)
                                          )
            return np.array([self.rgb[i] for i in range(self.NUM_DRONES)]).astype('float32')
        elif self.OBS_TYPE == ObservationType.KIN:
            if self.DRONE_MODEL == DroneModel.CF2XINVPEN:
                ############################################################
                #### OBS SPACE OF SIZE 16 for inverted pendulum
                obs_16 = np.zeros((self.NUM_DRONES,16))
                for i in range(self.NUM_DRONES):
                    #obs = self._clipAndNormalizeState(self._getDroneStateVector(i))
                    obs = self._getDroneStateVector(i)
                    obs_16[i, :] = np.hstack([self.TARGET_POS - obs[0:3], obs[7:10], obs[10:13], obs[13:16], obs[16:18], obs[18:20]]).reshape(16,) #Edited 2 
                ret = np.array([obs_16[i, :] for i in range(self.NUM_DRONES)]).astype('float32')
                #### Add action buffer to observation #######################
                for i in range(self.ACTION_BUFFER_SIZE):
                    ret = np.hstack([ret, np.array([self.action_buffer[i][j, :] for j in range(self.NUM_DRONES)])])
                return ret
                ############################################################
            else: # The drone model is not the inverted pendulum
                ############################################################
                #### OBS SPACE OF SIZE 12
                obs_12 = np.zeros((self.NUM_DRONES,12))
                for i in range(self.NUM_DRONES):
                    #obs = self._clipAndNormalizeState(self._getDroneStateVector(i))
                    obs = self._getDroneStateVector(i)
                    obs_12[i, :] = np.hstack([self.TARGET_POS - obs[0:3], obs[7:10], obs[10:13], obs[13:16]]).reshape(12,) #Edited 2 
                ret = np.array([obs_12[i, :] for i in range(self.NUM_DRONES)]).astype('float32')
                #### Add action buffer to observation #######################
                for i in range(self.ACTION_BUFFER_SIZE):
                    ret = np.hstack([ret, np.array([self.action_buffer[i][j, :] for j in range(self.NUM_DRONES)])])
                return ret
                ############################################################

        elif self.OBS_TYPE == ObservationType.REAL:
            if self.DRONE_MODEL == DroneModel.CF2XINVPEN:
                ############################################################
                #### OBS SPACE OF SIZE 16 for inverted pendulum
                obs_16 = np.zeros((self.NUM_DRONES,16))
                for i in range(self.NUM_DRONES):
                    #obs = self._clipAndNormalizeState(self._getDroneStateVector(i))
                    obs = self._getDroneStateVector(i)

                    if self.GoalPos == False:
                        obs_16[i, :] = np.hstack([self.TARGET_POS - obs[0:3], obs[7:10], obs[10:13], obs[13:16], obs[16:18], obs[18:20]]).reshape(16,) #Edited 2 
                    else:
                        obs_16[i, :] = np.hstack([obs[0:3]                  , obs[7:10], obs[10:13], obs[13:16], obs[16:18], obs[18:20]]).reshape(16,) #Edited 2
                
                ret = np.array([obs_16[i, :] for i in range(self.NUM_DRONES)]).astype('float32')
                #### Add action buffer to observation #######################
                for i in range(self.ACTION_BUFFER_SIZE):
                    ret = np.hstack([ret, np.array([self.action_buffer[i][j, :] for j in range(self.NUM_DRONES)])])
                return ret
                ############################################################
            else: # The drone model is not the inverted pendulum
                ############################################################
                #### OBS SPACE OF SIZE 12
                obs_12 = np.zeros((self.NUM_DRONES,12))
                for i in range(self.NUM_DRONES):
                    #obs = self._clipAndNormalizeState(self._getDroneStateVector(i))
                    obs = self._getDroneStateVector(i)

                    if self.GoalPos == False:
                        obs_12[i, :] = np.hstack([self.TARGET_POS - obs[0:3], obs[7:10]+self.rpy_offset, obs[10:13], obs[13:16]]).reshape(12,) #Edited 2 
                    else:
                        obs_12[i, :] = np.hstack([obs[0:3]                  , obs[7:10]+self.rpy_offset, obs[10:13], obs[13:16]]).reshape(12,) #Edited 2
                    
                    # Add noise to the observation x y z r p y
                    obs_12[i, 0] += np.random.normal(0, 0.00003)  # Add noise to X position
                    obs_12[i, 1] += np.random.normal(0, 0.00006)  # Add noise to Y position
                    obs_12[i, 2] += np.random.normal(0, 0.00003)  # Add noise to Z position
                    obs_12[i, 3] += np.random.normal(0, 0.001)  # Add noise to roll
                    obs_12[i, 4] += np.random.normal(0, 0.001)  # Add noise to pitch
                    obs_12[i, 5] += np.random.normal(0, 0.002)  # Add noise to yaw

                    #Calculate the velocity using a moving average filter
                    self.xVel = np.hstack(([obs_12[i, 0]], self.xVel[0:self.velBufSize - 1]))
                    self.yVel = np.hstack(([obs_12[i, 1]], self.yVel[0:self.velBufSize - 1]))
                    self.zVel = np.hstack(([obs_12[i, 2]], self.zVel[0:self.velBufSize - 1]))
                    self.yawRate = np.hstack(([obs_12[i, 5]], self.yawRate[0:self.velBufSize - 1]))
                    self.rollRate = np.hstack(([obs_12[i, 3]], self.rollRate[0:self.velBufSize - 1]))
                    self.pitchRate = np.hstack(([obs_12[i, 4]], self.pitchRate[0:self.velBufSize - 1]))

                    obs_12[i, 6] = np.mean(np.gradient(self.xVel)) / self.CTRL_TIMESTEP
                    obs_12[i, 7] = np.mean(np.gradient(self.yVel)) / self.CTRL_TIMESTEP
                    obs_12[i, 8] = np.mean(np.gradient(self.zVel)) / self.CTRL_TIMESTEP
                    obs_12[i, 9] = np.mean(np.gradient(self.yawRate)) / self.CTRL_TIMESTEP
                    obs_12[i, 10] = np.mean(np.gradient(self.rollRate)) / self.CTRL_TIMESTEP
                    obs_12[i, 11] = np.mean(np.gradient(self.pitchRate)) / self.CTRL_TIMESTEP
                

                ret = np.array([obs_12[i, :] for i in range(self.NUM_DRONES)]).astype('float32')
                #### Add action buffer to observation #######################
                for i in range(self.ACTION_BUFFER_SIZE):
                    ret = np.hstack([ret, np.array([self.action_buffer[i][j, :] for j in range(self.NUM_DRONES)])])
                return ret
                ############################################################

        elif self.OBS_TYPE == ObservationType.REAL2:
            if self.DRONE_MODEL == DroneModel.CF2XINVPEN:
                ############################################################
                #### OBS SPACE OF SIZE 16 for inverted pendulum
                print("[ERROR] Observation REAL2 not supported for inverted pendulum")
                ############################################################
            else: # The drone model is not the inverted pendulum
                ############################################################
                #### OBS SPACE OF SIZE 40
                obs_10 = np.zeros((self.NUM_DRONES, 10))
                obs_1000 = np.zeros((self.NUM_DRONES, 10*self.Real2_Size))
                for i in range(self.NUM_DRONES):
                    #obs = self._clipAndNormalizeState(self._getDroneStateVector(i))
                    obs = self._getDroneStateVector(i)
                    obs_10[i, :] = np.hstack([self.TARGET_POS-obs[0:3], obs[7:10]+self.rpy_offset, self.LastAction[i]]).reshape(10,) #Edited 2 
                    
                    # Add noise to the observation x y z r p y
                    obs_12[i, 0] += np.random.normal(0, 0.00003)  # Add noise to X position
                    obs_12[i, 1] += np.random.normal(0, 0.00006)  # Add noise to Y position
                    obs_12[i, 2] += np.random.normal(0, 0.00003)  # Add noise to Z position
                    obs_12[i, 3] += np.random.normal(0, 0.001)  # Add noise to roll
                    obs_12[i, 4] += np.random.normal(0, 0.0008)  # Add noise to pitch
                    obs_12[i, 5] += np.random.normal(0, 0.002)  # Add noise to yaw

                    obs_1000[i] = np.hstack((obs_10[i, :], self.LastObservation[i, :-10]))
                    self.LastObservation[i] = obs_1000[i].copy()

                ret = np.array([obs_1000[i, :] for i in range(self.NUM_DRONES)]).astype('float32')
                return ret
            
        elif self.OBS_TYPE == ObservationType.REAL3:
            if self.DRONE_MODEL == DroneModel.CF2XINVPEN:
                        ############################################################
                        #### OBS SPACE OF SIZE 16 for inverted pendulum
                        print("[ERROR] Observation REAL3 not supported for inverted pendulum")
                        ############################################################
            else: # The drone model is not the inverted pendulum
                 #### OBS SPACE OF SIZE 20
                Obs_out = np.zeros((self.NUM_DRONES,20))
                for i in range(self.NUM_DRONES):
                    obs = self._getDroneStateVector(i)
                    # Add last action features
                    avg_action = np.mean(self.LastActions[i, :5], axis=0)     # average of last 5
                    delta_action = self.LastActions[i, 0] - self.LastActions[i, 1]  # most recent change

                    xyz = obs[0:3]
                    Vxyz=obs[10:13]
                    rpy = obs[7:10]
                    Wrpy = obs[13:16]

                    # Add noise to the observation x y z r p y
                    xyz[0] += np.random.normal(0, 0.00003)  # Add noise to X position
                    xyz[1] += np.random.normal(0, 0.00006)  # Add noise to Y position
                    xyz[2] += np.random.normal(0, 0.00003)  # Add noise to Z position
                    rpy[0] += np.random.normal(0, 0.001)  # Add noise to roll
                    rpy[1] += np.random.normal(0, 0.0008)  # Add noise to pitch
                    rpy[2] += np.random.normal(0, 0.002)  # Add noise to yaw
                    if self.GoalPos == False:
                        Obs_out[i] = np.hstack([self.TARGET_POS - xyz, Vxyz, rpy + self.rpy_offset, Wrpy, avg_action, delta_action]).reshape(20,)
                    else:
                        Obs_out[i] = np.hstack([xyz, Vxyz, rpy + self.rpy_offset, Wrpy, avg_action, delta_action]).reshape(20,)
                ret = np.array([Obs_out[i, :] for i in range(self.NUM_DRONES)]).astype('float32')
                return ret

        else:
            print("[ERROR] in BaseRLAviary._computeObs()")
