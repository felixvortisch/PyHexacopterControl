"""
  Physics simulation module using pybullet

  pybullet world frame is ENU EAST (X) NORTH (Y) UP (Z)
  pybullet body frame is FORWARD (X) LEFT (Y) UP (Z)

  (c) Jan Zwiener (jan@zwiener.org)
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pybullet as p
import pybullet_data
from geodetic_toolbox import quat_from_rpy, quat_to_matrix, quat_to_rpy, quat_invert

class SimRocketEnv(gym.Env):
    """
    Rocket simulation environment (physics simulation) with an
    OpenAI gym interface / gymnasium interface. pybullet is
    used for the heavy lifting under the hood.
    """
    def __init__(self, interactive=False, scale_obs_space=1.0):
        print("HexacopterControl")
        self.pybullet_initialized = False
        self.interactive = interactive
        self.reset_count = 0 # keep track of calls to reset() function
        self.time_sec = 0.0 # keep track of simulation time
        self.dt_sec = 1.0 / 120.0 # update rate of the simulation

        # <pybullet>
        self.debug_line_thrust = -1
        self.pybullet_body = -1
        self.pybullet_booster_index = -1
        self.pybullet_time_sec = self.time_sec
        self._pybullet_setup_environment() # one time setup of pybullet
        # </pybullet>

        # <vehicle specific>
        self.urdf_file = "./src/modelhexacopter.urdf"
        self.UMIN = 0.01 # min. control input
        self.UMAX =  1.0 # max. control input for thrust (= 100%)
        self.ACTUATORCOUNT = 6 # main thrust, 2x thrust vector, 2x attitude
        # 				Motor Nr: 1			2			3			4			5			6
        #					  Front left, CCW
        self.THRUST_VECTORS = np.array([[-3.74859930380329e-08, -2.65044773434346e-24, 3.74859930380329e-08, 3.74859930380329e-08, 7.95134320303039e-24, -3.74859930380329e-08],
                                        [-2.16425481713487e-08, -4.32850963426975e-08, -2.16425481713487e-08, 2.16425481713487e-08, 4.32850963426975e-08, 2.16425481713488e-08],
                                        [3.07989463947313e-07, 3.07989463947313e-07, 3.07989463947313e-07, 3.07989463947313e-07, 3.07989463947313e-07, 3.07989463947313e-07]])
        # 				Motor Nr: 1   2    3      4      5     6
        #					  Front left, CCW					
        self.MOTOR_COORDINATES = np.array([[0.39, 0.0, -0.39, -0.39, 0.0, 0.39],
                                           [0.225, 0.45, 0.225, -0.225, -0.45, -0.225],
                                           [0.115, 0.115, 0.115, 0.115, 0.115, 0.115]])
        self.U_TAU = 0.1
        self.GRAVITY = 9.81 # assume we want to land on Earth
        self.mass_kg = -99999999.9 # will be loaded and updated from URDF
        self.MIN_GROUND_DIST_M = 2.45 # shut off engine below this altitude
        # OFFSET between CoG and nozzle. Is there a way to get this from URDF?
        self.NOZZLE_OFFSET = -2.0
        self.ATT_THRUSTER_OFFSET = 2.0
        self.scale_obs_space = scale_obs_space
        # </vehicle specific>

        # <state vector config>
        self.state_cfg = {}
        self.state = np.zeros((15,))
        self.q = np.array([1.0, 0.0, 0.0, 0.0]) # attitude quaternion
        self.omega = np.array([0.0, 0.0, 0.0]) # angular rate (body)
        self.pos_n = np.array([0.0,0.0,0.0]) # East North Up Position (m)
        self.vel_n = np.array([0.0,0.0,0.0]) # East North Up Velocity (m/s)
        #self.u_current = np.array([ 0.01, 0.01, 0.01, 0.01, 0.01, 0.01])
        # </state vector config>
        self.roll_deg= 0.0    # helper: mirror attitude in euler angles
        self.pitch_deg = 0.0
        self.yaw_deg = 0.0
        self.h_error = 0.0
        # initialize state of the vehicle with the actual values
        state, _ = self.reset() # reset state and fill self.state vector
        # </state>
        self.engine_on = True # not part of state vector

        # Setup Gym environment interface settings
        self.action_space = spaces.Box(low=np.float32(self.UMIN),
                                       high=np.float32(self.UMAX),
                                       shape=(self.ACTUATORCOUNT,),
                                       dtype=np.float32)

        obs_hi = np.ones(state.shape[0]) * 2000.0
        self.observation_space = spaces.Box(low=-np.float32(obs_hi),
                                            high=np.float32(obs_hi),
                                            dtype=np.float32)

    def _pybullet_setup_environment(self):
        """
        Connect to pybullet environment.
        """
        assert self.pybullet_initialized is False
        self.pybullet_initialized = True

        self.PYBULLET_DT_SEC = 1.0/240.0

        # connect to pybullet and get the client id
        if self.interactive:
            print("\033[33mpybullet physics active.\033[0m")
            self.CLIENT = p.connect(p.GUI)
        else:
            self.CLIENT = p.connect(p.DIRECT)

    def reset(self, seed=14, options={}) -> float:
        """
        Gym interface. Reset the simulation.
        :return state (state vector), info dict.
        """
        np.random.seed(seed)

        self.engine_on = True
        # <state>
        self.pos_n = np.array([0.0, 0.0, 0.0]) # ENU
        self.vel_n = np.array([0.0, 0.0, 0.0]) * self.scale_obs_space # ENU

        # Maintain the attitude as quaternion and Euler angles. The source of truth is
        # the quaternion (self.q) and roll_deg, pitch_deg and yaw_deg will be updated
        # based on the quaternion. But here for initialization the Euler angles are
        # used to initialize the orientation (Euler angles are a bit more readable)
        self.roll_deg  = 0.0 * self.scale_obs_space
        self.pitch_deg = 0.0 * self.scale_obs_space
        self.yaw_deg   = 0.0
        # Attitude quaternion (transforming from body to navigation system
        # Careful: quaternion order: qw, qx,qy,qz (qw is the real part)
        self.q         = quat_from_rpy(np.deg2rad(self.roll_deg),
                                       np.deg2rad(self.pitch_deg),
                                       np.deg2rad(self.yaw_deg))

        roll_rate_rps  = np.deg2rad(0.0) * self.scale_obs_space
        pitch_rate_rps = np.deg2rad(0.0) * self.scale_obs_space
        yaw_rate_rps   = 0.0
        self.omega     = np.array([roll_rate_rps,
                                   pitch_rate_rps,
                                   yaw_rate_rps])
        self.h_error = 0.0
        self.setpoint = 0.0
        #self.u_current = np.array([ 0.3, 0.3, 0.3, 0.3, 0.3, 0.3])

        # </state>
        self._update_state() # create/update state vector

        # <simulation>
        self.time_sec = 0.0
        self.reset_count += 1
        self.epochs = 0
        # </simulation>

        self._pybullet_reset_environment()
        return self.state, {}

    def _pybullet_reset_environment(self):
        """
        Cleanup all pybullet objects, reset and restart simulation environment.
        """
        self.pybullet_time_sec = self.time_sec
        p.resetSimulation(physicsClientId=self.CLIENT) # remove all objects and reset

        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.CLIENT)

        p.setGravity( 0.0, 0.0, -self.GRAVITY, physicsClientId=self.CLIENT)
        p.setTimeStep(self.PYBULLET_DT_SEC, physicsClientId=self.CLIENT)

        plane = p.loadURDF("plane.urdf", physicsClientId=self.CLIENT)
        p.changeDynamics(plane, -1, lateralFriction=1, restitution=0.5, physicsClientId=self.CLIENT)

        # Experimental code to include terrain map:
        # terrainShape = p.createCollisionShape(
        #           shapeType = p.GEOM_HEIGHTFIELD,
        #           meshScale=[0.5,0.5,40.0],
        #           fileName = "heightmaps/wm_height_out.png",
        #           physicsClientId=self.CLIENT)
        # textureId = p.loadTexture("heightmaps/gimp_overlay_out.png",
        #                           physicsClientId=self.CLIENT)
        # terrain  = p.createMultiBody(0, terrainShape,
        #                              physicsClientId=self.CLIENT)
        # p.changeVisualShape(terrain, -1, textureUniqueId = textureId,
        #                     physicsClientId=self.CLIENT)

        initial_position_enu = [self.pos_n[0], self.pos_n[1], self.pos_n[2]]
        self.pybullet_body = p.loadURDF(self.urdf_file,
                                        initial_position_enu,
                                        physicsClientId=self.CLIENT)
        self.pybullet_booster_index = -1

        q_rosbody_to_enu = self.q
        # pybullet needs the scalar part at the end of the quaternion:
        qxyzw_rosbody_to_enu = [ q_rosbody_to_enu[1],   # img. part x
                                 q_rosbody_to_enu[2],   # img. part y
                                 q_rosbody_to_enu[3],   # img. part z
                                 q_rosbody_to_enu[0] ]  # real part
        p.resetBasePositionAndOrientation(self.pybullet_body,
                                          initial_position_enu,
                                          qxyzw_rosbody_to_enu,
                                          physicsClientId=self.CLIENT)
        self.mass_kg = self._get_total_mass(self.pybullet_body)

        self.debug_line_thrust = -1

    def _set_camera_follow_object(self, object_id, dist=4.5, pitch=-55, yaw=50):
        """
        Helper function to set camera to object.
        """
        pos, _ = p.getBasePositionAndOrientation(object_id,
                                                 physicsClientId=self.CLIENT)
        p.resetDebugVisualizerCamera(
            cameraDistance=dist,
            cameraYaw=yaw,
            cameraPitch=pitch,
            cameraTargetPosition=pos, physicsClientId=self.CLIENT
        )

    def _pybullet_physics(self, u):
        """
        Advance the physics simulation until self.pybullet_time_sec catches up
        with self.time_sec.
        :param u Control input vector
        """
        self._set_camera_follow_object(self.pybullet_body)

        pybullet_dt_sec = 0.0
        # advance pybullet simulation until current time
        while self.pybullet_time_sec < self.time_sec:
            # thrust dynamics (i.e. the thrust takes some time to react)
            #self.u_current 	   += np.transpose((np.transpose(u) - np.transpose(self.u_current))/self.U_TAU)
            #print("\nu1:")
            #print(u)
            #u = u*8880.0
           # self.u_current = u
            u1 = u[0]*8880.0*u[0]*8880.0
            u2 = u[1]*8880.0*u[1]*8880.0
            u3 = u[2]*8880.0*u[2]*8880.0
            u4 = u[3]*8880.0*u[3]*8880.0
            u5 = u[4]*8880.0*u[4]*8880.0
            u6 = u[5]*8880.0*u[5]*8880.0
            #print(u1)
           # print("u_current:")
            #print(self.u_current)
            #print("\nu:")
            #print(u)
            
            t = 5000.0*5000.0
            
            thrust_motor_1 = self.THRUST_VECTORS[:, 0] * u1 #Front left motor
            thrust_motor_2 = self.THRUST_VECTORS[:, 1] * u2 # left motor
            thrust_motor_3 = self.THRUST_VECTORS[:, 2] * u3 #Back left motor
            thrust_motor_4 = self.THRUST_VECTORS[:, 3] * u4 #Back right motor
            thrust_motor_5 = self.THRUST_VECTORS[:, 4] * u5 #right motor
            thrust_motor_6 = self.THRUST_VECTORS[:, 5] * u6 #Front right motor
            #print(self.THRUST_VECTORS[:, 0])
            
            #print(u[0]*u[0])
            #print("Thrust=(%6.2f,%6.2f,%6.2f,%6.2f,%6.2f,%6.2f N)"%
            #(thrust_motor_1[2], thrust_motor_2[2], thrust_motor_3[2], thrust_motor_4[2], thrust_motor_5[2], thrust_motor_6[2]),
            #end=" ")

            

            # Add force of rocket boost to pybullet simulation
            if self.engine_on:
                p.applyExternalForce(objectUniqueId=self.pybullet_body,
                                     linkIndex=6,
                                     forceObj=np.transpose(thrust_motor_1),
                                     posObj=np.transpose(self.MOTOR_COORDINATES[:, 0]),
                                     flags=p.LINK_FRAME,
                                     physicsClientId=self.CLIENT)
                p.applyExternalForce(objectUniqueId=self.pybullet_body,
                                     linkIndex=7,
                                     forceObj=np.transpose(thrust_motor_2),
                                     posObj=np.transpose(self.MOTOR_COORDINATES[:, 1]),
                                     flags=p.LINK_FRAME,
                                     physicsClientId=self.CLIENT)
                p.applyExternalForce(objectUniqueId=self.pybullet_body,
                                     linkIndex=8,
                                     forceObj=np.transpose(thrust_motor_3),
                                     posObj=np.transpose(self.MOTOR_COORDINATES[:, 2]),
                                     flags=p.LINK_FRAME,
                                     physicsClientId=self.CLIENT)
                p.applyExternalForce(objectUniqueId=self.pybullet_body,
                                     linkIndex=9,
                                     forceObj=np.transpose(thrust_motor_4),
                                     posObj=np.transpose(self.MOTOR_COORDINATES[:, 3]),
                                     flags=p.LINK_FRAME,
                                     physicsClientId=self.CLIENT)
                p.applyExternalForce(objectUniqueId=self.pybullet_body,
                                     linkIndex=10,
                                     forceObj=np.transpose(thrust_motor_5),
                                     posObj=np.transpose(self.MOTOR_COORDINATES[:, 4]),
                                     flags=p.LINK_FRAME,
                                     physicsClientId=self.CLIENT)
                p.applyExternalForce(objectUniqueId=self.pybullet_body,
                                     linkIndex=11,
                                     forceObj=np.transpose(thrust_motor_6),
                                     posObj=np.transpose(self.MOTOR_COORDINATES[:, 5]),
                                     flags=p.LINK_FRAME,
                                     physicsClientId=self.CLIENT)                                    

            

            p.stepSimulation(physicsClientId=self.CLIENT)
            self.pybullet_time_sec += self.PYBULLET_DT_SEC
            pybullet_dt_sec += self.PYBULLET_DT_SEC

        


        # <EXTRACT CURRENT STATE FROM PYBULLET>
        position, orientation = p.getBasePositionAndOrientation(
                self.pybullet_body, physicsClientId=self.CLIENT)
        linear_velocity, omega_enu = p.getBaseVelocity(
                self.pybullet_body, physicsClientId=self.CLIENT)
        self.pos_n = np.array([position[0], position[1], position[2]])
        self.vel_n = np.array([linear_velocity[0],
                               linear_velocity[1],
                               linear_velocity[2]])
        # reorder pybullet quaternion to our internal order:
        q_rosbody_to_enu = np.array([orientation[3],
                                     orientation[0],
                                     orientation[1],
                                     orientation[2]])
        self.q = q_rosbody_to_enu
        # transform the body rotation rates that are given in the ENU world
        # frame to the PyRocketCraft body frame.
        R_enu_to_rosbody = quat_to_matrix(quat_invert(q_rosbody_to_enu))
        omega_rosbody = R_enu_to_rosbody @ np.array([omega_enu[0],
                                                     omega_enu[1],
                                                     omega_enu[2]])
        self.omega = omega_rosbody
        # </EXTRACT CURRENT STATE FROM PYBULLET>

    def _update_state(self):
        """
        Internal helper function to update self.state vector based on
        attributes such as self.q, self.pos, etc.
        """
        euler          = quat_to_rpy(self.q)
        self.roll_deg  = np.rad2deg(euler[0])
        self.pitch_deg = np.rad2deg(euler[1])
        self.yaw_deg   = np.rad2deg(euler[2])

        # Produce state vector:
        state_index = 0
        
        self.state[state_index:(state_index+3)] = self.pos_n
        self.state_cfg["pos_n_index"] = state_index
        state_index += 3

        self.state[state_index:(state_index+4)] = self.q
        self.state_cfg["q_index"] = state_index
        state_index += 4
        
        self.state[state_index:(state_index+3)] = self.vel_n
        self.state_cfg["vel_n_index"] = state_index
        state_index += 3

        self.state[state_index:(state_index+3)] = self.omega
        self.state_cfg["omega_index"] = state_index
        state_index += 3
        
        self.state[state_index:(state_index+1)] = self.h_error
        self.state_cfg["h_error_index"] = state_index
        state_index += 1
        
        self.state[state_index:(state_index+1)] = self.setpoint
        self.state_cfg["setpoint_index"] = state_index
        state_index += 1

        #self.state[state_index:(state_index+6)] = self.u_current
        #self.state_cfg["u_index"] = state_index
        #state_index += 6


    def step(self, action: np.array):
        """
        Gym interface step function to simulate the system.
        :param action Control input to the simulation, i.e. motor/rotor
            setpoints between 0 and 1 (actually umin and umax to be precise)
        :return state (state vector), reward (score), done (simulation done?)
        """

        action = np.clip(action, self.UMIN, self.UMAX)


        done = False
        self.time_sec = self.time_sec + self.dt_sec
        try:
             self._pybullet_physics(action)
        except Exception as e:
            print("pybullet exception:", e)
            done = True

        self.epochs += 1
        self._update_state()

        #reward = self._calculate_reward()
        reward = 0.0
        if self.engine_on is False:
            done = True

        # Stop the non-interactive simulation if the attitude is way off
        if self.interactive is False:
            if np.abs(self.pitch_deg) > 90.0 or np.abs(self.roll_deg) > 90.0:
                reward -= 100.0
                done = True

        info = {}
        return self.state, reward, done, False, info

    def print_state(self):
        """
        Helper function to print the current state vector to stdout
        """

        print("ENU=(%6.2f,%6.2f,%6.2f m) V=(%6.1f,%6.1f,%6.1f m/s) RPY=(%6.1f,%6.1f,%6.1f °) o=(%6.1f,%6.1f,%6.1f °/s) " %
              ( self.pos_n[0], self.pos_n[1], self.pos_n[2],
                self.vel_n[0], self.vel_n[1], self.vel_n[2],
                self.roll_deg, self.pitch_deg, self.yaw_deg,
                np.rad2deg(self.omega[0]),
                np.rad2deg(self.omega[1]),
                np.rad2deg(self.omega[2]),
                ),
                end=" ")
        print("")

    def render(self):
        """
        Gym interface. Render current simulation status.
        """
        if self.interactive:
            self.print_state()

    def _calculate_reward(self):
        """
        Calculate the current reward score.
        """
        # Constants for reward calculation - these may need tuning
        POSITION_WEIGHT = 1.0
        VELOCITY_WEIGHT = 1.0
        ORIENTATION_WEIGHT = 1.0
        MAX_POS_REWARD = 50.0   # Maximum reward for position
        MAX_VEL_REWARD = 50.0   # Maximum reward for velocity
        MAX_ORI_REWARD = 2.0    # Maximum reward for orientation (cos(0) + cos(0))

        # Calculate the negative distance from the target position (0,0,0)
        target_pos = np.array([0, 0, 0])
        distance = np.linalg.norm(self.pos_n - target_pos)
        distance_reward = MAX_POS_REWARD - POSITION_WEIGHT * distance

        # Calculate the negative velocity magnitude
        velocity_magnitude = np.linalg.norm(self.vel_n)
        velocity_reward = MAX_VEL_REWARD - VELOCITY_WEIGHT * velocity_magnitude

        # Calculate orientation reward
        # Converting degrees to radians for cosine calculation
        roll_rad = np.radians(self.roll_deg)
        pitch_rad = np.radians(self.pitch_deg)
        orientation_reward = ORIENTATION_WEIGHT * (np.cos(roll_rad) + np.cos(pitch_rad))

        # Normalize rewards
        normalized_distance_reward = distance_reward / MAX_POS_REWARD
        normalized_velocity_reward = velocity_reward / MAX_VEL_REWARD
        normalized_orientation_reward = orientation_reward / MAX_ORI_REWARD

        # Total reward
        total_reward = 0.0
        total_reward += ( normalized_distance_reward
                        + normalized_velocity_reward
                        + normalized_orientation_reward )
        total_reward *= self.dt_sec # normalize

        # Shut off engine near the ground and give a huge reward bonus for
        # landing upright and with low velocity
        if self.pos_n[2] < self.MIN_GROUND_DIST_M:
            if self.engine_on is True:
                self.engine_on = False
                if self.interactive:
                    print("Engine off at altitude: %.3f (AGL)" % (self.pos_n[2]))
                if np.abs(self.roll_deg) < 5.0:
                    total_reward += 128.0
                if np.abs(self.pitch_deg) < 5.0:
                    total_reward += 256.0
                if np.linalg.norm(self.omega) < np.deg2rad(7.0):
                    total_reward += 512.0
                if velocity_magnitude < 2.0:
                    total_reward += 1024.0

        return total_reward

    def _get_total_mass(self, body_id):
        """
        Get total body mass from object tree.
        """
        # Start with the mass of the base link (index -1)
        total_mass = p.getDynamicsInfo(body_id, -1,
                                       physicsClientId=self.CLIENT)[0]

        # Add up the masses of all other links
        num_links = p.getNumJoints(body_id, physicsClientId=self.CLIENT)
        for i in range(num_links):
            total_mass += p.getDynamicsInfo(body_id, i,
                                            physicsClientId=self.CLIENT)[0]

        return total_mass
