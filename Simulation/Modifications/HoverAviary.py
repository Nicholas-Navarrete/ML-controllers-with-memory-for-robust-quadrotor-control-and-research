import numpy as np
import random
from gym_pybullet_drones.envs.BaseRLAviary import BaseRLAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics, ActionType, ObservationType
import pybullet as p

""" 
UPDATES by Nicholas Navarrete:         
self.EPISODE_LEN_SEC = 8 => 15
Added Randomizer to Target
Updated compute truncated to new bounds, The idea is that if we dont let it truncate so soon it can collect worse rewards
Added .reset() method at this level to add randomized target every reset.
Added X Y Z limits to init to be used in truncated and target generation

Added truncation from the pendulum angle

Added green sphere for target visualization
Updated rewards to include pendulum angle
Episode length =15 =>25

updated xylimit from 5 to 10 and added reward to angular velocity of pendululm

Inv Pen _13 has updated termination/trucation for pendulum angles

With Attitude rate changed the truncation drone angles form pi/4 to pi/2

Added observation REAL which is the real drone kinematic information (pose, linear and angular velocities) with emperical noise

Added an intial RPY and updated base aviary to randomize the initial rpy when houskeeping if an intial rpy was set
"""
class HoverAviary(BaseRLAviary):
    """Single agent RL problem: hover at position."""

    ################################################################################
    
    def __init__(self,
                 drone_model: DroneModel=DroneModel.CF2X,
                 initial_xyzs=None,
                 initial_rpys=None,
                 physics: Physics=Physics.PYB,
                 pyb_freq: int = 240,
                 ctrl_freq: int = 30,
                 gui=False,
                 record=False,
                 obs: ObservationType=ObservationType.KIN,
                 act: ActionType=ActionType.RPM,
                 randomize_init_rpy = False,
                 Inefficient_Motors = False,
                 Inflight_Motor_Variance = False,
                 kf_Variance = 1e-11,
                 GoalPos = False
                 ):
        """Initialization of a single agent RL environment.

        Using the generic single agent RL superclass.

        Parameters
        ----------
        drone_model : DroneModel, optional
            The desired drone type (detailed in an .urdf file in folder `assets`).
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
            The type of action space (1 or 3D; RPMS, thurst and torques, or waypoint with PID control)

        """
        mu=2
        sigma=1
        Targets = np.array([random.gauss(0,sigma),random.gauss(0,sigma),random.gauss(mu,sigma)])
        if(Targets[-1]<=0):
            Targets[-1]=0

        self.TARGET_POS = Targets


        self.XY_Limit = 3.7
        self.Z_Limit=7
        self.EPISODE_LEN_SEC = 25
        super().__init__(drone_model=drone_model,
                         num_drones=1,
                         initial_xyzs=initial_xyzs,
                         initial_rpys=initial_rpys,
                         physics=physics,
                         pyb_freq=pyb_freq,
                         ctrl_freq=ctrl_freq,
                         gui=gui,
                         record=record,
                         obs=obs,
                         act=act,
                         Inefficient_Motors=Inefficient_Motors,
                         randomize_init_rpy = randomize_init_rpy,
                         Inflight_Motor_Variance=Inflight_Motor_Variance,
                         kf_Variance = kf_Variance,
                         GoalPos = GoalPos
                         )
        # We are giving the targets after the super() because we are also declaring the targets in the parent class
        self.TARGET_POS = Targets

    ################################################################################
    
    def _computeReward(self):
        """Computes the current reward value.

        Returns
        -------
        float
            The reward.

        """
        if self.DRONE_MODEL == DroneModel.CF2XINVPEN:
            state = self._getDroneStateVector(0)
            Positions = state[0:3]
            Velocities = state[10:13]
            Rotations = state[7:10]
            angPos=state[16:17]
            angVel=state[17:18]
            Yaw = Rotations[-1]
            #Pos_Rot_States = np.concatenate(Positions , Rotations)
            # Testing with only height penalty 
            rpos = np.exp(-(2*np.linalg.norm(self.TARGET_POS-Positions))**2)
            rsigma = np.exp(-(0.2*np.linalg.norm(Yaw))**2)
            rvel = np.exp(-(2*np.linalg.norm(Velocities))**2)
            rangpos = np.exp(-(0.5*np.linalg.norm(angPos))**4)
            rangvel = np.exp(-(0.5*np.linalg.norm(angVel))**2)

            return 0.6*rpos + 0.2*rsigma + 0.1*rvel + 0.4*rangpos+ 0.05*rangvel
        else:
            state = self._getDroneStateVector(0)
            Positions = state[0:3]
            Velocities = state[10:13]
            Rotations = state[7:10]
            RotationalVelocities = state[13:16]
            Yaw = Rotations[-1]
            #Pos_Rot_States = np.concatenate(Positions , Rotations)

            rpos = np.exp(-(1.0*np.linalg.norm(self.TARGET_POS-Positions))**2)
            rsigma = np.exp(-(0.2*np.linalg.norm(Yaw))**2)
            rvel = np.exp(-(2*np.linalg.norm(Velocities))**2)
            rrotvel = np.exp(-(0.1*np.linalg.norm(RotationalVelocities))**2)

            # === Potential-based shaping term ===
            gamma = 0.99
            Phi_curr = -np.linalg.norm(self.TARGET_POS - Positions)
            Phi_prev = -np.linalg.norm(self.TARGET_POS - self.prev_pos)
            r_progress = gamma * Phi_curr - Phi_prev   # positive when moving closer
            self.prev_pos = Positions.copy()

            return 1.5*rpos + 0.2*rsigma + 0.15*rvel + 0*rrotvel + 1.0*r_progress

    ################################################################################
    
    def _computeTerminated(self):
        """Computes the current done value.

        Returns
        -------
        bool
            Whether the current episode is done.

        """
        state = self._getDroneStateVector(0)
        if np.linalg.norm(self.TARGET_POS-state[0:3]) < .0001:
            return True
        else:
            return False
        
    ################################################################################
    
    def _computeTruncated(self):
        """Computes the current truncated value.

        Returns
        -------
        bool
            Whether the current episode timed out.  

        """
        ##### Changing the truncation bounds It was 1.5 1.5 2

        state = self._getDroneStateVector(0)
        if (abs(state[0]) > self.XY_Limit or abs(state[1]) > self.XY_Limit or state[2] > self.Z_Limit # Truncate when the drone is too far away
             or abs(state[7]) > np.pi/2 or abs(state[8]) > np.pi/2 # Truncate when the drone is too tilted 
        ):
            return True
        if self.step_counter/self.PYB_FREQ > self.EPISODE_LEN_SEC:
            return True
        if self.DRONE_MODEL == DroneModel.CF2XINVPEN:
            pendulumAngles = state[16:18] #Truncate if the pendulum angles are greater than 90 deg
            if np.any(np.abs(pendulumAngles)>(np.pi/2)):
                return True
        else:
            return False

    ################################################################################
    
    def _computeInfo(self):
        """Computes the current info dict(s).

        Unused.

        Returns
        -------
        dict[str, int]
            Dummy value.

        """

        state = self._getDroneStateVector(0)
        if (abs(state[0]) > self.XY_Limit or abs(state[1]) > self.XY_Limit or state[2] > self.Z_Limit # Truncate when the drone is too far away
             or abs(state[7]) > np.pi/2 or abs(state[8]) > np.pi/2 # Truncate when the drone is too tilted 
        ):
            OutOfBounds = True
        else:
            OutOfBounds = False
        return {"out_of_bounds": OutOfBounds} 
    
    def reset(self,
              seed : int = None,
              options : dict = None):
        """Resets the environment.

        Parameters
        ----------
        seed : int, optional
            Random seed.
        options : dict[..], optional
            Additinonal options, unused

        Returns
        -------
        ndarray | dict[..]
            The initial observation, check the specific implementation of `_computeObs()`
            in each subclass for its format.
        dict[..]
            Additional information as a dictionary, check the specific implementation of `_computeInfo()`
            in each subclass for its format.

        """

        # TODO : initialize random number generator with seed
        if(seed):
            random.seed(seed)
        
        Targets = np.array([random.gauss(0,1),random.gauss(0,1),random.gauss(2,1)])
        if(Targets[-1]<=0):
            Targets[-1]=0
        if(np.abs(Targets[0])>self.XY_Limit):
            #    ( Percentage of limit ) * ( Limit ) *( Normalized Target Which provides the sign + or - )
            Targets[0] = 0.9*self.XY_Limit*(Targets[0]/np.abs(Targets[0]))
        if(np.abs(Targets[1])>self.XY_Limit):
            #    ( Percentage of limit ) * ( Limit ) *( Normalized Target Which provides the sign + or - )
            Targets[1] = 0.9*self.XY_Limit*(Targets[1]/np.abs(Targets[1]))


        self.TARGET_POS = Targets
        self.TARGET_POS = np.array([0,0,1.1])
        
        p.resetSimulation(physicsClientId=self.CLIENT)
        
        # Load Green target sphere
        self.TargetSphere = p.loadURDF("\GreenFloatingSphere.urdf",self.TARGET_POS,physicsClientId=self.CLIENT,useFixedBase=True)
        #### Housekeeping ##########################################
        self._housekeeping()
        #### Update and store the drones kinematic information #####
        self._updateAndStoreKinematicInformation()
        #### Start video recording #################################
        self._startVideoRecording()
        #### Return the initial observation ########################
        initial_obs = self._computeObs()
        initial_info = self._computeInfo()
        self.prev_pos = self._getDroneStateVector(0)[0:3].copy()
        return initial_obs, initial_info

    ################################################################################

    def UpdateTarget(self,NewTarget):
        """Updates the target position.

        Parameters
        ----------
        NewTarget : ndarray
            (3,)-shaped array containing the new target position.

        """
        self.TARGET_POS = NewTarget

        #Move the Green target sphere
        p.resetBasePositionAndOrientation(self.TargetSphere, self.TARGET_POS, [0,0,0,1], physicsClientId=self.CLIENT)

    ################################################################################