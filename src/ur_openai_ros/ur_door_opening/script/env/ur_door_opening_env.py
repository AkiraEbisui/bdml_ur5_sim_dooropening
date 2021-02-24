#!/usr/bin/env python
'''
    By Akira Ebisui <shrimp.prawn.lobster713@gmail.com>
'''
# Python
import copy
import numpy as np
import math
import sys
import time
from matplotlib import pyplot as plt

# ROS 
import rospy
import rospkg
import tf
from tf.transformations import euler_from_quaternion, quaternion_from_euler

from joint_publisher import JointPub
from joint_traj_publisher import JointTrajPub

# Gazebo
from gazebo_msgs.srv import SetModelState, SetModelStateRequest, GetModelState
from gazebo_msgs.srv import GetWorldProperties
from gazebo_msgs.msg import LinkStates 
from gazebo_msgs.msg import ModelState

# For reset GAZEBO simultor
from gazebo_connection import GazeboConnection
from controllers_connection import ControllersConnection

# ROS msg
from geometry_msgs.msg import Pose, Point, Quaternion, Vector3, WrenchStamped
from sensor_msgs.msg import JointState, Image
from std_msgs.msg import String
from std_srvs.srv import SetBool, SetBoolResponse, SetBoolRequest
from std_srvs.srv import Empty

# Gym
import gym
from gym import utils, spaces
from gym.utils import seeding
from gym.envs.registration import register

# For inherit RobotGazeboEnv
from env import robot_gazebo_env_goal

# UR5 Utils
from env.ur_setups import setups
from env import ur_utils

from algorithm.ppo_gae import PPOGAEAgent

seed = rospy.get_param("/ML/seed")
obs_dim = rospy.get_param("/ML/obs_dim")
n_act = rospy.get_param("/ML/n_act")
epochs = rospy.get_param("/ML/epochs")
hdim = rospy.get_param("/ML/hdim")
policy_lr = rospy.get_param("/ML/policy_lr")
value_lr = rospy.get_param("/ML/value_lr")
max_std = rospy.get_param("/ML/max_std")
clip_range = rospy.get_param("/ML/clip_range")
n_step = rospy.get_param("/ML/n_step")
sub_step = rospy.get_param("/ML/sub_step")
add_act = rospy.get_param("/ML/add_act")
random_grasp = rospy.get_param("/ML/random_grasp")

change_sub = rospy.get_param("/act_params/change_sub")
sub_a0 = rospy.get_param("/act_params/sub_a0")
sub_a1 = rospy.get_param("/act_params/sub_a1")
sub_a2 = rospy.get_param("/act_params/sub_a2")
sub_a3 = rospy.get_param("/act_params/sub_a3")
sub_a4 = rospy.get_param("/act_params/sub_a4")
sub_a5 = rospy.get_param("/act_params/sub_a5")

sub2_a0 = rospy.get_param("/act_params/sub2_a0")
sub2_a1 = rospy.get_param("/act_params/sub2_a1")
sub2_a2 = rospy.get_param("/act_params/sub2_a2")
sub2_a3 = rospy.get_param("/act_params/sub2_a3")
sub2_a4 = rospy.get_param("/act_params/sub2_a4")
sub2_a5 = rospy.get_param("/act_params/sub2_a5")

knob_threshold = rospy.get_param("/threshold/knob_threshold")
panel_threshold = rospy.get_param("/threshold/panel_threshold")

knob_c = rospy.get_param("/reward_params/knob_c")
knob_bonus_c = rospy.get_param("/reward_params/knob_bonus_c")
panel_c = rospy.get_param("/reward_params/panel_c")
panel_bonus_c = rospy.get_param("/reward_params/panel_bonus_c")
tolerances_c = rospy.get_param("/reward_params/tolerances_c")
force_c = rospy.get_param("/reward_params/force_c")
force_c2 = rospy.get_param("/reward_params/force_c2")
taxel_c = rospy.get_param("/reward_params/taxel_c")
act_0_n = rospy.get_param("/reward_params/act_0_n")
act_1_n = rospy.get_param("/reward_params/act_1_n")
act_2_n = rospy.get_param("/reward_params/act_2_n")
act_3_n = rospy.get_param("/reward_params/act_3_n")
act_4_n = rospy.get_param("/reward_params/act_4_n")
act_5_n = rospy.get_param("/reward_params/act_5_n")
act_correct_c = rospy.get_param("/reward_params/act_correct_c")
catesian_xyz_c = rospy.get_param("/reward_params/catesian_xyz_c")
catesian_rpy_c = rospy.get_param("/reward_params/catesian_rpy_c")
cartesian_c = rospy.get_param("/reward_params/cartesian_c")

rospy.loginfo("register...")
#register the training environment in the gym as an available one
reg = gym.envs.register(
    id='URSimDoorOpening-v0',
    entry_point='env.ur_door_opening_env:URSimDoorOpening', # Its directory associated with importing in other sources like from 'ur_reaching.env.ur_sim_env import *' 
    #timestep_limit=100000,
    )
agent = PPOGAEAgent(obs_dim, n_act, epochs, hdim, policy_lr, value_lr, max_std, clip_range, seed)

class URSimDoorOpening(robot_gazebo_env_goal.RobotGazeboEnv):
    def __init__(self):
#        rospy.logdebug("Starting URSimDoorOpening Class object...")

        # Init GAZEBO Objects
        self.set_obj_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
        self.get_world_state = rospy.ServiceProxy('/gazebo/get_world_properties', GetWorldProperties)

        # Subscribe joint state and target pose
        rospy.Subscriber("/ft_sensor_topic", WrenchStamped, self.wrench_stamped_callback)
        rospy.Subscriber("/joint_states", JointState, self.joints_state_callback)
        rospy.Subscriber("/gazebo/link_states", LinkStates, self.link_state_callback)
        rospy.Subscriber("/robotiq/rightcam/image_raw_right", Image, self.r_image_callback)
        rospy.Subscriber("/robotiq/leftcam/image_raw_left", Image, self.l_image_callback)

        # Gets training parameters from param server
        self.running_step = rospy.get_param("/running_step")
        self.observations = rospy.get_param("/observations")
        
        # Joint limitation
        self.shp_max = rospy.get_param("/joint_limits_array/shp_max")
        self.shp_min = rospy.get_param("/joint_limits_array/shp_min")
        self.shl_max = rospy.get_param("/joint_limits_array/shl_max")
        self.shl_min = rospy.get_param("/joint_limits_array/shl_min")
        self.elb_max = rospy.get_param("/joint_limits_array/elb_max")
        self.elb_min = rospy.get_param("/joint_limits_array/elb_min")
        self.wr1_max = rospy.get_param("/joint_limits_array/wr1_max")
        self.wr1_min = rospy.get_param("/joint_limits_array/wr1_min")
        self.wr2_max = rospy.get_param("/joint_limits_array/wr2_max")
        self.wr2_min = rospy.get_param("/joint_limits_array/wr2_min")
        self.wr3_max = rospy.get_param("/joint_limits_array/wr3_max")
        self.wr3_min = rospy.get_param("/joint_limits_array/wr3_min")

        # cartesian_limits
        self.x_max = rospy.get_param("/cartesian_limits/x_max")
        self.x_min = rospy.get_param("/cartesian_limits/x_min")
        self.y_max = rospy.get_param("/cartesian_limits/y_max")
        self.y_min = rospy.get_param("/cartesian_limits/y_min")
        self.z_max = rospy.get_param("/cartesian_limits/z_max")
        self.z_min = rospy.get_param("/cartesian_limits/z_min")

        shp_init_value0 = rospy.get_param("/init_joint_pose0/shp")
        shl_init_value0 = rospy.get_param("/init_joint_pose0/shl")
        elb_init_value0 = rospy.get_param("/init_joint_pose0/elb")
        wr1_init_value0 = rospy.get_param("/init_joint_pose0/wr1")
        wr2_init_value0 = rospy.get_param("/init_joint_pose0/wr2")
        wr3_init_value0 = rospy.get_param("/init_joint_pose0/wr3")
        self.init_joint_pose0 = [shp_init_value0, shl_init_value0, elb_init_value0, wr1_init_value0, wr2_init_value0, wr3_init_value0]

        shp_init_value1 = rospy.get_param("/init_joint_pose1/shp")
        shl_init_value1 = rospy.get_param("/init_joint_pose1/shl")
        elb_init_value1 = rospy.get_param("/init_joint_pose1/elb")
        wr1_init_value1 = rospy.get_param("/init_joint_pose1/wr1")
        wr2_init_value1 = rospy.get_param("/init_joint_pose1/wr2")
        wr3_init_value1 = rospy.get_param("/init_joint_pose1/wr3")
        self.init_joint_pose1 = [shp_init_value1, shl_init_value1, elb_init_value1, wr1_init_value1, wr2_init_value1, wr3_init_value1]

        self.shp_init_value2 = rospy.get_param("/init_joint_pose2/shp")
        self.shl_init_value2 = rospy.get_param("/init_joint_pose2/shl")
        self.elb_init_value2 = rospy.get_param("/init_joint_pose2/elb")
        self.wr1_init_value2 = rospy.get_param("/init_joint_pose2/wr1")
        self.wr2_init_value2 = rospy.get_param("/init_joint_pose2/wr2")
        self.wr3_init_value2 = rospy.get_param("/init_joint_pose2/wr3")
        self.init_joint_pose2 = [self.shp_init_value2, self.shl_init_value2, self.elb_init_value2, self.wr1_init_value2, self.wr2_init_value2, self.wr3_init_value2]

        self.shp_after_pull = rospy.get_param("/after_pull_pose/shp")
        self.shl_after_pull = rospy.get_param("/after_pull_pose/shl")
        self.elb_after_pull = rospy.get_param("/after_pull_pose/elb")
        self.wr1_after_pull = rospy.get_param("/after_pull_pose/wr1")
        self.wr2_after_pull = rospy.get_param("/after_pull_pose/wr2")
        self.wr3_after_pull = rospy.get_param("/after_pull_pose/wr3")

        self.init_pos0 = self.init_joints_pose(self.init_joint_pose0)
        self.arr_init_pos0 = np.array(self.init_pos0, dtype='float32')
        self.init_pos1 = self.init_joints_pose(self.init_joint_pose1)
        self.arr_init_pos1 = np.array(self.init_pos1, dtype='float32')
        self.init_pos2 = self.init_joints_pose(self.init_joint_pose2)
        self.arr_init_pos2 = np.array(self.init_pos2, dtype='float32')

        # cartesian position
        init_pose1_x = rospy.get_param("/init_pose1/x")
        init_pose1_y = rospy.get_param("/init_pose1/y")
        init_pose1_z = rospy.get_param("/init_pose1/z")
        init_pose1_rpy_r = rospy.get_param("/init_pose1/rpy_r")
        init_pose1_rpy_p = rospy.get_param("/init_pose1/rpy_p")
        init_pose1_rpy_y = rospy.get_param("/init_pose1/rpy_y")
        self.init_pose1 = [init_pose1_x, init_pose1_y, init_pose1_z, init_pose1_rpy_r, init_pose1_rpy_p, init_pose1_rpy_y]
        self.arr_init_pose1 = np.array(self.init_pose1, dtype='float32')

        init_pose2_x = rospy.get_param("/init_pose2/x")
        init_pose2_y = rospy.get_param("/init_pose2/y")
        init_pose2_z = rospy.get_param("/init_pose2/z")
        init_pose2_rpy_r = rospy.get_param("/init_pose2/rpy_r")
        init_pose2_rpy_p = rospy.get_param("/init_pose2/rpy_p")
        init_pose2_rpy_y = rospy.get_param("/init_pose2/rpy_y")
        self.init_pose2 = [init_pose2_x, init_pose2_y, init_pose2_z, init_pose2_rpy_r, init_pose2_rpy_p, init_pose2_rpy_y]
        self.arr_init_pose2 = np.array(self.init_pose2, dtype='float32')

        # gripper position
        init_grp_pose1 = rospy.get_param("/init_grp_pose1")
        init_grp_pose2 = rospy.get_param("/init_grp_pose2")
        init_grp_pose3 = rospy.get_param("/init_grp_pose3")

        self.init_grp_pose1 = [init_grp_pose1, init_grp_pose1, -init_grp_pose1, -init_grp_pose1, init_grp_pose1, init_grp_pose1]
        self.init_grp_pose2 = [init_grp_pose2, init_grp_pose2, -init_grp_pose2, -init_grp_pose2, init_grp_pose2, init_grp_pose2]
        self.init_grp_pose3 = [init_grp_pose3, init_grp_pose3, -init_grp_pose3, -init_grp_pose3, init_grp_pose3, init_grp_pose3]

        init_g_pos1 = self.init_joints_pose(self.init_grp_pose1)
        self.arr_init_g_pos1 = np.array(init_g_pos1, dtype='float32')
        init_g_pos2 = self.init_joints_pose(self.init_grp_pose2)
        self.arr_init_g_pos2 = np.array(init_g_pos2, dtype='float32')
        init_g_pos3 = self.init_joints_pose(self.init_grp_pose3)
        self.arr_init_g_pos3 = np.array(init_g_pos3, dtype='float32')

        # Fill in the Done Episode Criteria list
        self.episode_done_criteria = rospy.get_param("/episode_done_criteria")
        
        # stablishes connection with simulator
        self._gz_conn = GazeboConnection()
        self._ctrl_conn = ControllersConnection(namespace="")
        
        # Controller type for ros_control
        self._ctrl_type =  rospy.get_param("/control_type")
        self.pre_ctrl_type =  self._ctrl_type

        # Use MoveIt or not
        self.moveit = rospy.get_param("/moveit")

	# Get the force and troque limit
        self.force_limit1 = rospy.get_param("/force_limit1")
        self.torque_limit1 = rospy.get_param("/torque_limit1")
        self.force_limit2 = rospy.get_param("/force_limit2")
        self.torque_limit2 = rospy.get_param("/torque_limit2")

        # Get tolerances of door_frame
        self.tolerances1 = rospy.get_param("/door_frame_tolerances1")
        self.tolerances2 = rospy.get_param("/door_frame_tolerances2")

        # Get observation parameters
        self.joint_n = rospy.get_param("/obs_params/joint_n")
        self.eef_n = rospy.get_param("/obs_params/eef_n")
        self.eef_rpy_n = rospy.get_param("/obs_params/eef_rpy_n")
        self.force_n = rospy.get_param("/obs_params/force_n")
        self.torque_n = rospy.get_param("/obs_params/torque_n")
        self.image_n = rospy.get_param("/obs_params/image_n")
        self.min_static_limit = rospy.get_param("/min_static_limit")
        self.max_static_limit = rospy.get_param("/max_static_limit")

        # We init the observations
        self.base_orientation = Quaternion()
        self.imu_link = Pose() # Quaternion()
        self.door = Pose() # Quaternion()
        self.door_frame = Pose() # Point()
        self.door_frame_ini = Pose() # Point()
        self.quat = Quaternion()
        self.imu_link_rpy = Vector3()
        self.imu_link_rpy_ini = Vector3()
        self.door_rpy = Vector3()
        self.door_rpy_ini = Vector3()
        self.link_state = LinkStates()
        self.wrench_stamped = WrenchStamped()
        self.joints_state = JointState()
        self.right_image = Image()
        self.right_image_ini = []
        self.left_image = Image()
        self.lift_image_ini = []
        self.end_effector = Pose() # Point()
        self.success = 0
        self.success_times = 0
        self.fail_door_frame_limit = 0
        self.fail_force_limit = 0
        self.fail_torque_limit = 0
        self.fail_taxel_limit = 0        

        # Arm/Control parameters
        self._ik_params = setups['UR5_6dof']['ik_params']
        
        # ROS msg type
#        self._joint_pubisher = JointPub()
        self._joint_traj_pubisher = JointTrajPub()

        # Gym interface and action
        self.action_space = spaces.Discrete(n_act)
        self.observation_space = obs_dim #np.arange(self.get_observations().shape[0])
        self.reward_range = (-np.inf, np.inf)
        self._seed()

        # Change the controller type 
#        set_joint_pos_server = rospy.Service('/set_position_controller', SetBool, self._set_pos_ctrl)
        set_joint_traj_pos_server = rospy.Service('/set_trajectory_position_controller', SetBool, self._set_traj_pos_ctrl)
#        set_joint_vel_server = rospy.Service('/set_velocity_controller', SetBool, self._set_vel_ctrl)
#        set_joint_traj_vel_server = rospy.Service('/set_trajectory_velocity_controller', SetBool, self._set_traj_vel_ctrl)


        self.pos_traj_controller = ['joint_state_controller',
                            'gripper_controller',
                            'pos_traj_controller']
        self.pos_controller = ["joint_state_controller",
                                "gripper_controller",
                                "ur_shoulder_pan_pos_controller",
                                "ur_shoulder_lift_pos_controller",
                                "ur_elbow_pos_controller",
                                "ur_wrist_1_pos_controller",
                                "ur_wrist_2_pos_controller",
                                "ur_wrist_3_pos_controller"]
        self.vel_traj_controller = ['joint_state_controller',
                            'gripper_controller',
                            'vel_traj_controller']
        self.vel_controller = ["joint_state_controller",
                                "gripper_controller",
                                "ur_shoulder_pan_vel_controller",
                                "ur_shoulder_lift_vel_controller",
                                "ur_elbow_vel_controller",
                                "ur_wrist_1_vel_controller",
                                "ur_wrist_2_vel_controller",
                                "ur_wrist_3_vel_controller"]

        # Helpful False
        self.stop_flag = False
        stop_trainning_server = rospy.Service('/stop_training', SetBool, self._stop_trainnig)
        start_trainning_server = rospy.Service('/start_training', SetBool, self._start_trainnig)

    def set_item(self, goal_x, goal_y, goal_z):
        state_msg = ModelState()

        state_msg.model_name = 'hinged_door'
        state_msg.pose.position.x = goal_x
        state_msg.pose.position.y = goal_y
        state_msg.pose.position.z = goal_z
        state_msg.pose.orientation.x = 1
        state_msg.pose.orientation.y = 0
        state_msg.pose.orientation.z = 0
        state_msg.pose.orientation.w = 0

        rospy.wait_for_service('/gazebo/set_model_state')
        try:
            set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
            resp = set_state(state_msg)
            print(state_msg)

        except rospy.ServiceException, e:
            print "Service call failed: %s" % e

    def check_stop_flg(self):
        if self.stop_flag is False:
            return False
        else:
            return True

    def _start_trainnig(self, req):
        rospy.logdebug("_start_trainnig!!!!")
        self.stop_flag = False
        return SetBoolResponse(True, "_start_trainnig")

    def _stop_trainnig(self, req):
        rospy.logdebug("_stop_trainnig!!!!")
        self.stop_flag = True
        return SetBoolResponse(True, "_stop_trainnig")

    def _set_pos_ctrl(self, req):
        rospy.wait_for_service('set_position_controller')
        self._ctrl_conn.stop_controllers(self.pos_traj_controller)
        self._ctrl_conn.start_controllers(self.pos_controller)
        self._ctrl_type = 'pos'
        return SetBoolResponse(True, "_set_pos_ctrl")

    def _set_traj_pos_ctrl(self, req):
        rospy.wait_for_service('set_trajectory_position_controller')
        self._ctrl_conn.stop_controllers(self.pos_controller)
        self._ctrl_conn.start_controllers(self.pos_traj_controller)    
        self._ctrl_type = 'traj_pos'
        return SetBoolResponse(True, "_set_traj_pos_ctrl")  

    def _set_vel_ctrl(self, req):
        rospy.wait_for_service('set_velocity_controller')
        self._ctrl_conn.stop_controllers(self.vel_traj_controller)
        self._ctrl_conn.start_controllers(self.vel_controller)
        self._ctrl_type = 'vel'
        return SetBoolResponse(True, "_set_vel_ctrl")

    def _set_traj_vel_ctrl(self, req):
        rospy.wait_for_service('set_trajectory_velocity_controller')
        self._ctrl_conn.stop_controllers(self.vel_controller)
        self._ctrl_conn.start_controllers(self.vel_traj_controller)    
        self._ctrl_type = 'traj_vel'
        return SetBoolResponse(True, "_set_traj_vel_ctrl")  

    # A function to initialize the random generator
    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def check_all_systems_ready(self):
        """
        We check that all systems are ready
        :return:
        """
        joint_states_msg = None
        while joint_states_msg is None and not rospy.is_shutdown():
            try:
                joint_states_msg = rospy.wait_for_message("/joint_states", JointState, timeout=0.1)
                self.joints_state = joint_states_msg
                rospy.logdebug("Current joint_states READY")
            except Exception as e:
                self._ctrl_conn.start_controllers(controllers_on="joint_state_controller")                
                rospy.logdebug("Current joint_states not ready yet, retrying==>"+str(e))
        
        link_states_msg = None
        while link_states_msg is None and not rospy.is_shutdown():
            try:
                link_states_msg = rospy.wait_for_message("/gazebo/link_states", LinkStates, timeout=0.1)
                self.link_states = link_states_msg
                rospy.logdebug("Reading link_states READY")
            except Exception as e:
                rospy.logdebug("Reading link_states not ready yet, retrying==>"+str(e))

        rospy.logdebug("ALL SYSTEMS READY")

    def check_cartesian_limits(self, sub_action):
        if self.moveit == 0:
            self.ee_xyz = Point()
            self.ee_xyz = self.get_xyz(sub_action)
        elif self.moveit == 1:
            self.ee_xyz = []
            self.ee_xyz = sub_action

        if sub_action[5] > 2.47 or sub_action[5] < 1.57:
            return False
        if self.x_min < self.ee_xyz[0] and self.ee_xyz[0] < self.x_max and self.y_min < self.ee_xyz[1] and self.ee_xyz[1] < self.y_max and self.z_min < self.ee_xyz[2] and self.ee_xyz[2] < self.z_max:
            return True
        elif self.x_min > self.ee_xyz[0] or self.ee_xyz[0] > self.x_max:
            print("over the x_cartesian limits", self.x_min, "<", self.ee_xyz[0], "<", self.x_max)
            return False
        elif self.y_min > self.ee_xyz[1] or self.ee_xyz[1] > self.y_max:
            print("over the y_cartesian limits", self.y_min, "<", self.ee_xyz[1], "<", self.y_max)
            return False
        elif self.z_min > self.ee_xyz[2] or self.ee_xyz[2] > self.z_max:
            print("over the z_cartesian limits", self.z_min, "<", self.ee_xyz[2], "<", self.z_max)
            return False

    def get_xyz(self, q):
        """Get x,y,z coordinates 
        Args:
            q: a numpy array of joints angle positions.
        Returns:
            xyz are the x,y,z coordinates of an end-effector in a Cartesian space.
        """
        mat = ur_utils.forward(q, self._ik_params)
        xyz = mat[:3, 3]
        return xyz

    def get_current_xyz(self):
        """Get x,y,z coordinates according to currrent joint angles
        Returns:
        xyz are the x,y,z coordinates of an end-effector in a Cartesian space.
        """
        joint_states = self.joints_state
        shp_joint_ang = joint_states.position[2]
        shl_joint_ang = joint_states.position[1]
        elb_joint_ang = joint_states.position[0]
        wr1_joint_ang = joint_states.position[9]
        wr2_joint_ang = joint_states.position[10]
        wr3_joint_ang = joint_states.position[11]
        
        q = [shp_joint_ang, shl_joint_ang, elb_joint_ang, wr1_joint_ang, wr2_joint_ang, wr3_joint_ang]
        mat = ur_utils.forward(q, self._ik_params)
        xyz = mat[:3, 3]
        return xyz

    def get_joint_value(self):
        """Get x,y,z coordinates according to currrent joint angles
        Returns:
        xyz are the x,y,z coordinates of an end-effector in a Cartesian space.
        """
        joint_states = self.joints_state
        shp_joint_ang = joint_states.position[2]
        shl_joint_ang = joint_states.position[1]
        elb_joint_ang = joint_states.position[0]
        wr1_joint_ang = joint_states.position[9]
        wr2_joint_ang = joint_states.position[10]
        wr3_joint_ang = joint_states.position[11]
        
        q = [shp_joint_ang, shl_joint_ang, elb_joint_ang, wr1_joint_ang, wr2_joint_ang, wr3_joint_ang]
        return q
            
    def get_orientation(self, q):
        """Get Euler angles 
        Args:
            q: a numpy array of joints angle positions.
        Returns:
            xyz are the x,y,z coordinates of an end-effector in a Cartesian space.
        """
        mat = ur_utils.forward(q, self._ik_params)
        orientation = mat[0:3, 0:3]
        roll = -orientation[1, 2]
        pitch = orientation[0, 2]
        yaw = -orientation[0, 1]
       
        return Vector3(roll, pitch, yaw)


    def cvt_quat_to_euler(self, quat):
        euler_rpy = Vector3()
        euler = euler_from_quaternion([self.quat.x, self.quat.y, self.quat.z, self.quat.w])

        euler_rpy.x = euler[0]
        euler_rpy.y = euler[1]
        euler_rpy.z = euler[2]
        return euler_rpy

    def init_joints_pose(self, init_pos):
        """
        We initialise the Position variable that saves the desired position where we want our
        joints to be
        :param init_pos:
        :return:
        """
        self.current_joint_pose =[]
        self.current_joint_pose = copy.deepcopy(init_pos)
        return self.current_joint_pose

    def get_euclidean_dist(self, p_in, p_pout):
        """
        Given a Vector3 Object, get distance from current position
        :param p_end:
        :return:
        """
        a = numpy.array((p_in.x, p_in.y, p_in.z))
        b = numpy.array((p_pout.x, p_pout.y, p_pout.z))

        distance = numpy.linalg.norm(a - b)

        return distance

    def joints_state_callback(self,msg):
        self.joints_state = msg

    def wrench_stamped_callback(self,msg):
        self.wrench_stamped = msg
        
    def link_state_callback(self, msg):
        self.link_state = msg
        self.end_effector = self.link_state.pose[12]
        self.imu_link = self.link_state.pose[5]
        self.door_frame = self.link_state.pose[1]
        self.door = self.link_state.pose[2]

    def r_image_callback(self, msg):
        self.right_image = msg

    def l_image_callback(self, msg):
        self.left_image = msg

    def get_observations(self):
        """
        Returns the state of the robot needed for OpenAI QLearn Algorithm
        The state will be defined by an array
        :return: observation
        """
        joint_states = self.joints_state
        eef_rpy = Vector3()

        self.force = self.wrench_stamped.wrench.force
        self.torque = self.wrench_stamped.wrench.torque
#        print("[force]", self.force.x, self.force.y, self.force.z)
#        print("[torque]", self.torque.x, self.torque.y, self.torque.z)

        shp_joint_ang = joint_states.position[2]
        shl_joint_ang = joint_states.position[1]
        elb_joint_ang = joint_states.position[0]
        wr1_joint_ang = joint_states.position[9]
        wr2_joint_ang = joint_states.position[10]
        wr3_joint_ang = joint_states.position[11]

        shp_joint_vel = joint_states.velocity[2]
        shl_joint_vel = joint_states.velocity[1]
        elb_joint_vel = joint_states.velocity[0]
        wr1_joint_vel = joint_states.velocity[9]
        wr2_joint_vel = joint_states.velocity[10]
        wr3_joint_vel = joint_states.velocity[11]

        q = [shp_joint_ang, shl_joint_ang, elb_joint_ang, wr1_joint_ang, wr2_joint_ang, wr3_joint_ang]
#        print("observation_q", q)
        self.eef_x, self.eef_y, self.eef_z = self.get_xyz(q)
        self.eef_rpy = self.get_orientation(q)

        delta_image_r, delta_image_l = self.get_image()
        self.cnn_image_r = agent.update_cnn(delta_image_r)
        self.cnn_image_l = agent.update_cnn(delta_image_l)
        self.cnn_image_r_list = self.cnn_image_r.tolist()
        self.cnn_image_l_list = self.cnn_image_l.tolist()
#        print("self.cnn_image_r_list", self.cnn_image_r_list)
#        print("self.cnn_image_r_list", self.cnn_image_r_list[0])

        observation = []
#        rospy.logdebug("List of Observations==>"+str(self.observations))
        for obs_name in self.observations:
            if obs_name == "shp_joint_ang":
                observation.append((shp_joint_ang - self.init_joint_pose2[0]) * self.joint_n)
            elif obs_name == "shl_joint_ang":
                observation.append((shl_joint_ang - self.init_joint_pose2[1]) * self.joint_n)
            elif obs_name == "elb_joint_ang":
                observation.append((elb_joint_ang - self.init_joint_pose2[2]) * self.joint_n)
            elif obs_name == "wr1_joint_ang":
                observation.append((wr1_joint_ang - self.init_joint_pose2[3]) * self.joint_n)
            elif obs_name == "wr2_joint_ang":
                observation.append((wr2_joint_ang - self.init_joint_pose2[4]) * self.joint_n)
            elif obs_name == "wr3_joint_ang":
                observation.append((wr3_joint_ang - self.init_joint_pose2[5]) * self.joint_n)
            elif obs_name == "shp_joint_vel":
                observation.append(shp_joint_vel)
            elif obs_name == "shl_joint_vel":
                observation.append(shl_joint_vel)
            elif obs_name == "elb_joint_vel":
                observation.append(elb_joint_vel)
            elif obs_name == "wr1_joint_vel":
                observation.append(wr1_joint_vel)
            elif obs_name == "wr2_joint_vel":
                observation.append(wr2_joint_vel)
            elif obs_name == "wr3_joint_vel":
                observation.append(wr3_joint_vel)
            elif obs_name == "eef_x":
                observation.append((self.eef_x - self.eef_x_ini) * self.eef_n)
            elif obs_name == "eef_y":
                observation.append((self.eef_y - self.eef_y_ini) * self.eef_n)
            elif obs_name == "eef_z":
                observation.append((self.eef_z - self.eef_z_ini) * self.eef_n)
            elif obs_name == "eef_rpy_x":
                observation.append((self.eef_rpy.x - self.eef_rpy_ini.x) * self.eef_rpy_n)
            elif obs_name == "eef_rpy_y":
                observation.append((self.eef_rpy.y - self.eef_rpy_ini.y) * self.eef_rpy_n)
            elif obs_name == "eef_rpy_z":
                observation.append((self.eef_rpy.z - self.eef_rpy_ini.z) * self.eef_rpy_n)
            elif obs_name == "force_x":
                observation.append((self.force.x - self.force_ini.x) / self.force_limit1 * self.force_n)
            elif obs_name == "force_y":
                observation.append((self.force.y - self.force_ini.y) / self.force_limit1 * self.force_n)
            elif obs_name == "force_z":
                observation.append((self.force.z - self.force_ini.z) / self.force_limit1 * self.force_n)
            elif obs_name == "torque_x":
                observation.append((self.torque.x - self.torque_ini.x) / self.torque_limit1 * self.torque_n)
            elif obs_name == "torque_y":
                observation.append((self.torque.y - self.torque_ini.y) / self.torque_limit1 * self.torque_n)
            elif obs_name == "torque_z":
                observation.append((self.torque.z - self.torque_ini.z) / self.torque_limit1 * self.torque_n)
            elif obs_name == "image_cnn":
                for x in range(0, 10):
                    observation.append(self.cnn_image_r_list[0][x])
#                    print("r_list", self.cnn_image_r_list[0][x])
                for x in range(0, 10):
                    observation.append(self.cnn_image_l_list[0][x])
#                    print("l_list", self.cnn_image_l_list[0][x])
            elif obs_name == "image_data":
                for x in range(0, 28):
                    observation.append((ord(r_image.data[x]) - ord(self.right_image_ini.data[x])) * self.image_n)
                for x in range(0, 28):
                    observation.append((ord(l_image.data[x]) - ord(self.left_image_ini.data[x])) * self.image_n)
            else:
                raise NameError('Observation Asked does not exist=='+str(obs_name))
#        print("observation", list(map(round, observation, [3]*len(observation))))
#        print("observation", observation)

        return observation

    def get_image(self):
        delta_image_r = []
        delta_image_l = []
        r_image = self.right_image
        l_image = self.left_image
        for x in range(0, 28):
            delta_image_r.append((ord(r_image.data[x]) - ord(self.right_image_ini.data[x])) * self.image_n)
        for x in range(0, 28):
            delta_image_l.append((ord(l_image.data[x]) - ord(self.left_image_ini.data[x])) * self.image_n)
        return delta_image_r, delta_image_l

    def clamp_to_joint_limits(self):
        """
        clamps self.current_joint_pose based on the joint limits
        self._joint_limits
        {
         "shp_max": shp_max,
         "shp_min": shp_min,
         ...
         }
        :return:
        """

        rospy.logdebug("Clamping current_joint_pose>>>" + str(self.current_joint_pose))
        shp_joint_value = self.current_joint_pose[0]
        shl_joint_value = self.current_joint_pose[1]
        elb_joint_value = self.current_joint_pose[2]
        wr1_joint_value = self.current_joint_pose[3]
        wr2_joint_value = self.current_joint_pose[4]
        wr3_joint_value = self.current_joint_pose[5]

        self.current_joint_pose[0] = max(min(shp_joint_value, self._joint_limits["shp_max"]), self._joint_limits["shp_min"])
        self.current_joint_pose[1] = max(min(shl_joint_value, self._joint_limits["shl_max"]), self._joint_limits["shl_min"])
        self.current_joint_pose[2] = max(min(elb_joint_value, self._joint_limits["elb_max"]), self._joint_limits["elb_min"])
        self.current_joint_pose[3] = max(min(wr1_joint_value, self._joint_limits["wr1_max"]), self._joint_limits["wr1_min"])
        self.current_joint_pose[4] = max(min(wr2_joint_value, self._joint_limits["wr2_max"]), self._joint_limits["wr2_min"])
        self.current_joint_pose[5] = max(min(wr3_joint_value, self._joint_limits["wr3_max"]), self._joint_limits["wr3_min"])

        rospy.logdebug("DONE Clamping current_joint_pose>>>" + str(self.current_joint_pose))


    def first_reset(self):
        self.quat = self.imu_link.orientation
        self.imu_link_rpy = self.cvt_quat_to_euler(self.quat)
        self.imu_link_rpy_ini = copy.deepcopy(self.imu_link_rpy)
        self.door_frame_ini = copy.deepcopy(self.door_frame)
        self.door_rpy_ini = copy.deepcopy(self.door_rpy)
#        if self.moveit ==0:
        for update in range(500):
            self._joint_traj_pubisher.jointTrajectoryCommand_reset(self.arr_init_pos0)
        time.sleep(1)
        for update in range(300):
       	    self._joint_traj_pubisher.jointTrajectoryCommand_reset(self.arr_init_pos1)
        time.sleep(1)
#        elif self.moveit ==1:
#            self._joint_traj_pubisher.MoveItJointTarget(self.init_pos1)

    def end_reset(self):
        for update in range(200):
      	    self._joint_traj_pubisher.GrpCommand(self.arr_init_g_pos1)
        time.sleep(0.1)
        for update in range(300):
            self._joint_traj_pubisher.jointTrajectoryCommand_reset(self.arr_init_pos2)
        time.sleep(0.5)
        for update in range(300):
       	    self._joint_traj_pubisher.jointTrajectoryCommand_reset(self.arr_init_pos1)
        time.sleep(1)

    # Resets the state of the environment and returns an initial observation.
    def reset(self):
        self.max_knob_rotation = 0
        self.max_door_rotation = 0
        self.max_wrist3 = 0
        self.min_wrist3 = 0
        self.max_wrist2 = 0
        self.min_wrist2 = 0
        self.max_wrist1 = 0
        self.min_wrist1 = 0
        self.max_elb = 0
        self.min_elb = 0
        self.max_shl = 0
        self.min_shl = 0
        self.max_shp = 0
        self.min_shp = 0
        self.max_force_x = 0
        self.min_force_x = 0
        self.max_force_y = 0
        self.min_force_y = 0
        self.max_force_z = 0
        self.min_force_z = 0
        self.max_torque_x = 0
        self.min_torque_x = 0
        self.max_torque_y = 0
        self.min_torque_y = 0
        self.max_torque_z = 0
        self.min_torque_z = 0
        self.max_taxel0 = 0
        self.min_taxel0 = 0
        self.max_taxel1 = 0
        self.min_taxel1 = 0
        self.max_door_tolerance = 0
        self.min_door_tolerance = 100
        self.max_act_correct_n = 0
        self.min_act_correct_n = 100
        self.max_eef_x = 0
        self.min_eef_x = 0
        self.max_eef_y = 0
        self.min_eef_y = 0
        self.max_eef_z = 0
        self.min_eef_z = 0
        self.max_eef_rpy_x = 0
        self.min_eef_rpy_x = 0
        self.max_eef_rpy_y = 0
        self.min_eef_rpy_y = 0
        self.max_eef_rpy_z = 0
        self.min_eef_rpy_z = 0
        self.act_correct_n = 0
        self.act_end = 0
        self.delta_force_x = 0
        self.delta_force_y = 0
        self.delta_force_z = 0
        self.delta_torque_x = 0
        self.delta_torque_y = 0
        self.delta_torque_z = 0

        self.success = 0

	# Go to initial position
	self._gz_conn.unpauseSim()
#        rospy.logdebug("set_init_pose init variable...>>>" + str(self.init_joint_pose0))

#        if self.moveit ==0:
        for update in range(200):
      	    self._joint_traj_pubisher.GrpCommand(self.arr_init_g_pos1)
        time.sleep(0.1)
        for update in range(300):
            self._joint_traj_pubisher.jointTrajectoryCommand_reset(self.arr_init_pos2)
        time.sleep(0.5)
        for update in range(10000):
            self._joint_traj_pubisher.jointTrajectoryCommand_reset(self.arr_init_pos1)
            if abs(self.imu_link_rpy_ini.z - self.imu_link_rpy.z) <= 0.01:
                break
#        elif self.moveit ==1:
#            self._joint_traj_pubisher.MoveItGrpOpen()
#            self._joint_traj_pubisher.MoveItJointTarget(self.init_pos2)
#            self._joint_traj_pubisher.MoveItJointTarget(self.init_pos1)

        # 0st: We pause the Simulator
#        rospy.logdebug("Pausing SIM...")
        self._gz_conn.pauseSim()

        # 1st: resets the simulation to initial values
#        rospy.logdebug("Reset SIM...")
        if self.moveit ==0 and random_grasp ==0:
            self._gz_conn.resetSim() # Comment out when you use MoveIt!

#        self.set_item(0.1, 0, 0)
#        self._gz_conn.resetSim()
        # 2nd: We Set the gravity to 0.0 so that we dont fall when reseting joints
        # It also UNPAUSES the simulation
#        rospy.logdebug("Remove Gravity...")
        self._gz_conn.change_gravity_zero()

        # EXTRA: Reset JoinStateControlers because sim reset doesnt reset TFs, generating time problems
#        rospy.logdebug("reset_ur_joint_controllers...")
        self._ctrl_conn.reset_ur_joint_controllers(self._ctrl_type)

        # 3rd: resets the robot to initial conditions
#        rospy.logdebug("set_init_pose init variable...>>>" + str(self.init_joint_pose1))
#        rospy.logdebug("set_init_pose init variable...>>>" + str(self.init_joint_pose2))

        self.force = self.wrench_stamped.wrench.force
        self.torque = self.wrench_stamped.wrench.torque
#        print("self.force", self.force)
#        print("self.torque", self.torque)

        self.force_ini = copy.deepcopy(self.force)
        self.torque_ini = copy.deepcopy(self.torque)
        if self.moveit == 0:
            self.previous_action = copy.deepcopy(self.arr_init_pos2)
        elif self.moveit == 1:
            self.previous_action = copy.deepcopy(self.init_pose2)
        # We save that position as the current joint desired position

        # 4th: We Set the init pose to the jump topic so that the jump control can update
        # We check the jump publisher has connection

        if self._ctrl_type == 'traj_pos':
            self._joint_traj_pubisher.check_publishers_connection()
        elif self._ctrl_type == 'pos':
            self._joint_pubisher.check_publishers_connection()
        elif self._ctrl_type == 'traj_vel':
            self._joint_traj_pubisher.check_publishers_connection()
        elif self._ctrl_type == 'vel':
            self._joint_pubisher.check_publishers_connection()
        else:
            rospy.logwarn("Controller type is wrong!!!!")
        
        # 5th: Check all subscribers work.
        # Get the state of the Robot defined by its RPY orientation, distance from
        # desired point, contact force and JointState of the three joints
#        rospy.logdebug("check_all_systems_ready...")
        self.check_all_systems_ready()

        # 6th: We restore the gravity to original
#        rospy.logdebug("Restore Gravity...")
        self._gz_conn.adjust_gravity()

        self.grasp_success = 0
        while self.grasp_success == 0:
    #        if self.moveit ==0:
            for update in range(200):
                self._joint_traj_pubisher.GrpCommand(self.arr_init_g_pos1)
            if random_grasp >= 1:
                rand = np.random.rand(6) # random number 0 to 1
                rand = 1 - rand * 2      # random number -1 to 1
#                print("rand", rand, type(rand))
                self.rand_list = rand.tolist()
#                print("rand_list", rand_list, type(rand_list))
                if random_grasp == 1:
                    self.rand_list = [self.rand_list[0] / 100, 0, 0, 0 ,0, 0]
                if random_grasp == 2:
                    self.rand_list = [self.rand_list[0] / 100, self.rand_list[1] / 100, 0, 0 ,0, 0]
#               print("rand_list", rand_list, type(rand_list))
                random_position = [self.rand_list[0] + self.init_pose2[0], self.rand_list[1] + self.init_pose2[1], self.rand_list[2] + self.init_pose2[2], self.rand_list[3] + self.init_pose2[3], self.rand_list[4] + self.init_pose2[4], self.rand_list[5] + self.init_pose2[5]]
#               print("init_pose2", self.init_pose2)
#               print("random_position", random_position, type(random_position))
                self._joint_traj_pubisher.MoveItCommand(random_position)

                arr_x_n = np.array([1, 1, -1, -1, 1, 1], dtype='float32')
                arr_y_n = np.array([1, 1, -1, -1, 1, 1], dtype='float32')
#                arr_n = np.array([0.005, 0.005, -0.005, -0.005, 0.005, 0.005], dtype='float32')
                arr_n = np.array([0.005, 0.005, -0.005, -0.005, 0.005, 0.005], dtype='float32')

                grasp_gap = self.arr_init_g_pos2 + arr_x_n * abs(self.rand_list[0]) * 2 + arr_x_n * abs(self.rand_list[1]) * 5
#                print("grasp_gap", grasp_gap)
#                print("arr_init_g_pos2", self.arr_init_g_pos2)
                for update in range(1000):
                    self._joint_traj_pubisher.GrpCommand(grasp_gap)
                    joint_states = self.joints_state
                    simple_gripper_left_driver_joint = joint_states.position[4]
                    if simple_gripper_left_driver_joint > grasp_gap[0] - 0.001:
                        break
                for update in range(100000):
                    grasp_gap = grasp_gap + arr_n
                    self._joint_traj_pubisher.GrpCommand(grasp_gap)
                    self.force = self.wrench_stamped.wrench.force
                    if self.force.x < -10:
                        self.grasp_success = 1
                        break
                    grasp_gap = grasp_gap - arr_n
                    self._joint_traj_pubisher.GrpCommand(grasp_gap)
                    self.force = self.wrench_stamped.wrench.force
                    if self.force.x < -10:
                        self.grasp_success = 1
                        break
#            for update in range(200):
#                if rand_list[0] > 0.007 or rand_list[0] < -0.007:
#                    self._joint_traj_pubisher.GrpCommand(self.arr_init_g_pos3)
#                else:
#                    self._joint_traj_pubisher.GrpCommand(self.arr_init_g_pos2)
#            time.sleep(1)
            else:
                self.rand_list = [0, 0, 0, 0 ,0, 0]
                for update in range(300):
       	            self._joint_traj_pubisher.jointTrajectoryCommand_reset(self.arr_init_pos2)
                time.sleep(0.1)
                grasp_gap = self.arr_init_g_pos2
                arr_n = np.array([0.005, 0.005, -0.005, -0.005, 0.005, 0.005], dtype='float32')
                for update in range(200):
                    self._joint_traj_pubisher.GrpCommand(grasp_gap)
                for update in range(300):
                    grasp_gap = grasp_gap + arr_n
                    self._joint_traj_pubisher.GrpCommand(grasp_gap)
                    self.force = self.wrench_stamped.wrench.force
                    if self.force.x < -10:
                        self.grasp_success = 1                        
                        break
                    grasp_gap = grasp_gap - arr_n
                    self._joint_traj_pubisher.GrpCommand(grasp_gap)
                    self.force = self.wrench_stamped.wrench.force
                    if self.force.x < -10:
                        self.grasp_success = 1                        
                        break
                time.sleep(1)
#        elif self.moveit ==1:
#            self._joint_traj_pubisher.MoveItJointTarget(self.init_pos2)
#            self._joint_traj_pubisher.MoveItGrpClose()
#            time.sleep(2)

#        ini_cartesian = self.check_cartesian_limits(self.arr_init_pos2)

        self.eef_x_ini, self.eef_y_ini, self.eef_z_ini = self.get_xyz(self.init_joint_pose2)
        self.eef_rpy_ini = self.get_orientation(self.init_joint_pose2)

        # 7th: pauses simulation
#        rospy.logdebug("Pause SIM...")
        self._gz_conn.pauseSim()

        self.right_image_ini = copy.deepcopy(self.right_image)
        self.left_image_ini = copy.deepcopy(self.left_image)

        # 8th: Get the State Discrete Stringuified version of the observations
#        rospy.logdebug("get_observations...")
        observation = self.get_observations()
#        print("[observations]", observation)

        return observation

    def _act(self, action):
        if self._ctrl_type == 'traj_pos':
            if self.moveit == 0:
                self.pre_ctrl_type = 'traj_pos'
                for update in range(3000):
                    self._joint_traj_pubisher.jointTrajectoryCommand(action)
#                observation = self.get_observations()
            elif self.moveit == 1:
                self._joint_traj_pubisher.MoveItCommand(action)
        elif self._ctrl_type == 'pos':
            self.pre_ctrl_type = 'pos'
            self._joint_pubisher.move_joints(action)
        elif self._ctrl_type == 'traj_vel':
            self.pre_ctrl_type = 'traj_vel'
            self._joint_traj_pubisher.jointTrajectoryCommand(action)
        elif self._ctrl_type == 'vel':
            self.pre_ctrl_type = 'vel'
            self._joint_pubisher.move_joints(action)
        else:
            self._joint_pubisher.move_joints(action)
        
    def training_ok(self):
        rate = rospy.Rate(1)
        while self.check_stop_flg() is True:                  
            rospy.logdebug("stop_flag is ON!!!!")
            self._gz_conn.unpauseSim()

            if self.check_stop_flg() is False:
                break 
            rate.sleep()
                
    def step(self, action, update):
        '''
        ('action: ', array([ 0.,  0. , -0., -0., -0. , 0. ], dtype=float32))        
        '''
#        rospy.logdebug("UR step func")	# define the logger
        self.training_ok()

        # Given the action selected by the learning algorithm,
        # we perform the corresponding movement of the robot
        # Act
        self._gz_conn.unpauseSim()
        current_joint_value = self.get_joint_value()
        arr_current_joint_value = np.array(current_joint_value)

        for x in range(1, sub_step + 1):
            self.cartesian_flag = 0
            self.sub_action = np.array(action) / sub_step * x

            if self.moveit == 0:
                if arr_current_joint_value[5] < self.arr_init_pos2[5] + change_sub:
                    self.sub_action[0] = self.sub_action[0] / sub_a0
                    self.sub_action[1] = self.sub_action[1] / sub_a1
                    self.sub_action[2] = self.sub_action[2] / sub_a2
                    self.sub_action[3] = self.sub_action[3] / sub_a3
                    self.sub_action[4] = self.sub_action[4] / sub_a4
                    self.sub_action[5] = self.sub_action[5] / sub_a5
                else:
                    self.sub_action[0] = self.sub_action[0] / sub2_a0
                    self.sub_action[1] = self.sub_action[1] / sub2_a1
                    self.sub_action[2] = self.sub_action[2] / sub2_a2
                    self.sub_action[3] = self.sub_action[3] / sub2_a3
                    self.sub_action[4] = self.sub_action[4] / sub2_a4
                    self.sub_action[5] = self.sub_action[5] / sub2_a5 

                if add_act == 0:
                    self.sub_action = self.sub_action + self.arr_init_pos2
                elif add_act == 1:
                    self.sub_action = self.sub_action + arr_current_joint_value
 #                   print("self.sub_action", self.sub_action)
 #                   print("current_joint_value", arr_current_joint_value)

# after rotate(shp,shl,elb,wr1,wr2,wr3)
#                self.sub_action[0] =  1.488122534496775
#                self.sub_action[1] = -1.4496597816566892
#                self.sub_action[2] =  2.4377209990850974
#                self.sub_action[3] =  2.168370898415174
#                self.sub_action[4] = -1.4670589583209175
#                self.sub_action[5] =  2.5709174034899702

# after pull
#                if update > 4:
#                    self.sub_action[0] =  1.612635433881735
#                    self.sub_action[1] = -1.4724165911604175
#                    self.sub_action[2] =  2.470747950622256
#                    self.sub_action[3] =  2.15165861084186
#                    self.sub_action[4] = -1.7013942143425345
#                    self.sub_action[5] =  2.575005366191597

            elif self.moveit == 1:
                self.sub_action[0] = self.sub_action[0] / sub_a0
                self.sub_action[1] = self.sub_action[1] / sub_a1
                self.sub_action[2] = self.sub_action[2] / sub_a2
                self.sub_action[3] = self.sub_action[3] / sub_a3
                self.sub_action[4] = self.sub_action[4] / sub_a4
                self.sub_action[5] = self.sub_action[5] / sub_a5
                self.sub_action = self.sub_action + self.arr_init_pose2
                self.sub_action = self.sub_action.tolist()

# grasp position(x,y,z,roll,pitch,yaw)
#                self.sub_action[0] = -0.08832874
#                self.sub_action[1] = 0.35898954
#                self.sub_action[2] = 0.27695617
#                self.sub_action[3] = 1.5746781585880325
#                self.sub_action[4] = 0.01488937165698871
#                self.sub_action[5] = 1.5931206693388063

# after rotate
#                self.sub_action[0] = -0.08832874
#                self.sub_action[1] = 0.35898954
#                self.sub_action[2] = 0.27695617
#                self.sub_action[3] = 2.5746781585880325
#                self.sub_action[4] = 0.01488937165698871
#                self.sub_action[5] = 1.5931206693388063

# after pull
#                if update > 7:
#                    self.sub_action[0] = -0.1126264565148
#                    self.sub_action[1] = 0.3348316039846
#                    self.sub_action[2] = 0.2770681230212
#                    self.sub_action[3] = 2.57467815858803
#                    self.sub_action[4] = 0.0145232756620397
#                    self.sub_action[5] = 1.4799236454285

#                print("update", update)
#                print("self.sub_action", self.sub_action)

            if self.check_cartesian_limits(self.sub_action) is True:
                self._act(self.sub_action)
                self.wrench_stamped
                self.force = self.wrench_stamped.wrench.force
                self.torque = self.wrench_stamped.wrench.torque
                self.delta_force_x = self.force.x - self.force_ini.x 
                self.delta_force_y = self.force.y - self.force_ini.y
                self.delta_force_z = self.force.z - self.force_ini.z
                self.delta_torque_x = self.torque.x - self.torque_ini.x
                self.delta_torque_y = self.torque.y - self.torque_ini.y
                self.delta_torque_z = self.torque.z - self.torque_ini.z

                if self.max_force_x < self.delta_force_x:
                    self.max_force_x = self.delta_force_x
                if self.min_force_x > self.delta_force_x:
                    self.min_force_x = self.delta_force_x
                if self.max_force_y < self.delta_force_y:
                    self.max_force_y = self.delta_force_y
                if self.min_force_y > self.delta_force_y:
                    self.min_force_y = self.delta_force_y
                if self.max_force_z < self.delta_force_z:
                    self.max_force_z = self.delta_force_z
                if self.min_force_z > self.delta_force_z:
                    self.min_force_z = self.delta_force_z
                if self.max_torque_x < self.delta_torque_x:
                    self.max_torque_x = self.delta_torque_x
                if self.min_torque_x > self.delta_torque_x:
                    self.min_torque_x = self.delta_torque_x
                if self.max_torque_y < self.delta_torque_y:
                    self.max_torque_y = self.delta_torque_y
                if self.min_torque_y > self.delta_torque_y:
                    self.min_torque_y = self.delta_torque_y
                if self.max_torque_z < self.delta_torque_z:
                    self.max_torque_z = self.delta_torque_z
                if self.min_torque_z > self.delta_torque_z:
                    self.min_torque_z = self.delta_torque_z

                self.min_static_taxel0 = 0
                self.min_static_taxel1 = 0
                self.max_static_taxel0 = 0
                self.max_static_taxel1 = 0
                r_image = self.right_image
                l_image = self.left_image
    
                for y in range(0, 28):
                    if self.min_static_taxel0 > (ord(r_image.data[y]) - ord(self.right_image_ini.data[y])) * self.image_n:
                        self.min_static_taxel0 = (ord(r_image.data[y]) - ord(self.right_image_ini.data[y])) * self.image_n
                    if self.min_static_taxel1 > (ord(l_image.data[y]) - ord(self.left_image_ini.data[y])) * self.image_n:
                        self.min_static_taxel1 = (ord(l_image.data[y]) - ord(self.left_image_ini.data[y])) * self.image_n
                    if self.max_static_taxel0 < (ord(r_image.data[y]) - ord(self.right_image_ini.data[y])) * self.image_n:
                        self.max_static_taxel0 = (ord(r_image.data[y]) - ord(self.right_image_ini.data[y])) * self.image_n
                    if self.max_static_taxel1 < (ord(l_image.data[y]) - ord(self.left_image_ini.data[y])) * self.image_n:
                        self.max_static_taxel1 = (ord(l_image.data[y]) - ord(self.left_image_ini.data[y])) * self.image_n
    #            print("min, max taxel", self.min_static_taxel0, self.max_static_taxel0, self.min_static_taxel1, self.max_static_taxel1)

    
                if self.force_limit2 < self.delta_force_x or self.delta_force_x < -self.force_limit2:
                    print(x, "force.x over the limit2", self.delta_force_x)
                    self.act_end = 1
                elif self.force_limit2 < self.delta_force_y or self.delta_force_y < -self.force_limit2:
                    print(x, "force.y over the limit2", self.delta_force_y)
                    self.act_end = 1
                elif self.force_limit2 < self.delta_force_z or self.delta_force_z < -self.force_limit2:
                    print(x, "force.z over the limit2", self.delta_force_z)
                    self.act_end = 1
                elif self.torque_limit2 < self.delta_torque_x or self.delta_torque_x < -self.torque_limit2:
                    print(x, "torque.x over the limit2", self.delta_torque_x)
                    self.act_end = 1
                elif self.torque_limit2 < self.delta_torque_y or self.delta_torque_y < -self.torque_limit2:
                    print(x, "torque.y over the limit2", self.delta_torque_y)
                    self.act_end = 1
                elif self.torque_limit2 < self.delta_torque_z or self.delta_torque_z < -self.torque_limit2:
                    print(x, "torque.z over the limit2", self.delta_torque_z)
                    self.act_end = 1
                elif self.min_static_taxel0 < self.min_static_limit and self.min_static_taxel1 < self.min_static_limit:
                    print(x, "slipped and break the for loop(min over)", self.min_static_taxel0, self.min_static_taxel1)
                    self.act_end = 1
                elif self.max_static_taxel0 > self.max_static_limit and self.max_static_taxel1 > self.max_static_limit:
                    print(x, "slipped and break the for loop(max over)", self.max_static_taxel0, self.max_static_taxel1)
                    self.act_end = 1
                elif abs(self.door_frame.position.x - self.door_frame_ini.position.x) > self.tolerances1 or abs(self.door_frame.position.y - self.door_frame_ini.position.y) > self.tolerances1 or abs(self.door_frame.position.z - self.door_frame_ini.position.z) > self.tolerances1:
                    print(x, "door frame position over the limit")
                    self.act_end = 1
                else:
                    self.previous_action = copy.deepcopy(self.sub_action)
                    self.act_correct_n += 1
                    print(x, "act correctly", self.act_correct_n)
                    if x == sub_step:
                        self.previous_action = copy.deepcopy(self.sub_action)
                        print("copy previous_action")   

                if self.force_limit1 < self.delta_force_x or self.delta_force_x < -self.force_limit1:
#                    self._act(self.previous_action)
                    print(x, "force.x over the limit1", self.delta_force_x)
                elif self.force_limit1 < self.delta_force_y or self.delta_force_y < -self.force_limit1:
#                    self._act(self.previous_action)
                    print(x, "force.y over the limit1", self.delta_force_y)
                elif self.force_limit1 < self.delta_force_z or self.delta_force_z < -self.force_limit1:
#                    self._act(self.previous_action)
                    print(x, "force.z over the limit1", self.delta_force_z)
                elif self.torque_limit1 < self.delta_torque_x or self.delta_torque_x < -self.torque_limit1:
#                    self._act(self.previous_action)
                    print(x, "torque.x over the limit1", self.delta_torque_x)
                elif self.torque_limit1 < self.delta_torque_y or self.delta_torque_y < -self.torque_limit1:
#                    self._act(self.previous_action)
                    print(x, "torque.y over the limit1", self.delta_torque_y)
                elif self.torque_limit1 < self.delta_torque_z or self.delta_torque_z < -self.torque_limit1:
#                    self._act(self.previous_action)
                    print(x, "torque.z over the limit1", self.delta_torque_x)

            else:
                self.cartesian_flag = 1
                print(x, "over the cartesian limits")
                self.act_end = 1
     
            observation = self.get_observations()
#            if observation[3] < -0.1 or observation[3] > 0.1:
#                print(x, "break the for loop(wr1_limit)", observation[3])
#                self.act_end = 1
#            if observation[2] < -0.1 or observation[2] > 0.1:
#                print(x, "break the for loop(elb_limit)", observation[2])
#                self.act_end = 1
#            if observation[1] < -0.1 or observation[1] > 0.1:
#                print(x, "break the for loop(shl_limit)", observation[1])
#                self.act_end = 1

            if self.act_end == 1:
   #             self._act(self.previous_action)
   #             print("act previous_action", self.previous_action)
                break

# [1.5338984518733714, -1.7979356133870397, 2.1357303575914095, -0.3226808371658887, 1.5854886701775008, 0.07412931630796482]
# [1.527437940043594, -1.798838363862096, 2.137143238430859, -0.3239121423979796, 1.5586089455341092, -0.5865493562517852]

        # Then we send the command to the robot and let it go for running_step seconds
        time.sleep(self.running_step)
        self._gz_conn.pauseSim()

        # We now process the latest data saved in the class state to calculate
        # the state and the rewards. This way we guarantee that they work
        # with the same exact data.
        # Generate State based on observations
        observation = self.get_observations()

        if self.max_wrist3 < observation[5]:
            self.max_wrist3 = observation[5]
        if self.min_wrist3 > observation[5]:
            self.min_wrist3 = observation[5]
        if self.max_wrist2 < observation[4]:
            self.max_wrist2 = observation[4]
        if self.min_wrist2 > observation[4]:
            self.min_wrist2 = observation[4]
        if self.max_wrist1 < observation[3]:
            self.max_wrist1 = observation[3]
        if self.min_wrist1 > observation[3]:
            self.min_wrist1 = observation[3]
        if self.max_elb < observation[2]:
            self.max_elb = observation[2]
        if self.min_elb > observation[2]:
            self.min_elb = observation[2]
        if self.max_shl < observation[1]:
            self.max_shl = observation[1]
        if self.min_shl > observation[1]:
            self.min_shl = observation[1]
        if self.max_shp < observation[0]:
            self.max_shp = observation[0]
        if self.min_shp > observation[0]:
            self.min_shp = observation[0]

        # finally we get an evaluation based on what happened in the sim
        reward = self.compute_dist_rewards(action, update)
        done = self.check_done(update)

        if self.act_end == 1:
            self._act(self.previous_action)
#            print("act previous_action", self.previous_action)

        return observation, reward, done, {}

    def compute_dist_rewards(self, action, update):
        self.quat = self.door.orientation
        self.door_rpy = self.cvt_quat_to_euler(self.quat)
        self.quat = self.imu_link.orientation
        self.imu_link_rpy = self.cvt_quat_to_euler(self.quat)

        self.knob_rotation_r = 0
        self.panel_rotation_r = 0
        self.door_frame_limit_r = 0
        self.force_limit_r = 0
        self.static_limit_r = 0
        self.action_limit_r = 0
        self.act_correct_r = 0
        self.catesian_xyz_r = 0
        self.catesian_rpy_r = 0
        self.cartesian_bonus_r = 0
        force_x_limit1_r = 0
        force_y_limit1_r = 0
        force_z_limit1_r = 0
        torque_x_limit1_r = 0
        torque_y_limit1_r = 0
        torque_z_limit1_r = 0
        force_x_limit2_r = 0
        force_y_limit2_r = 0
        force_z_limit2_r = 0
        torque_x_limit2_r = 0
        torque_y_limit2_r = 0
        torque_z_limit2_r = 0
        min_static_limit_r = 0
        max_static_limit_r = 0
        action5_limit_r = 0
        action4_limit_r = 0
        action3_limit_r = 0
        action2_limit_r = 0
        action1_limit_r = 0
        action0_limit_r = 0
        catesian_x = 0
        catesian_y = 0
        catesian_z = 0
        catesian_rpy_x = 0
        catesian_rpy_y = 0
        catesian_rpy_z = 0
        compute_rewards = 0.0001

        if self.max_act_correct_n < self.act_correct_n:
            self.max_act_correct_n = self.act_correct_n
        if self.min_act_correct_n > self.act_correct_n:
            self.min_act_correct_n = self.act_correct_n

        if self.max_eef_x < self.eef_x - self.eef_x_ini:
            self.max_eef_x = self.eef_x - self.eef_x_ini
        if self.min_eef_x > self.eef_x - self.eef_x_ini:
            self.min_eef_x = self.eef_x - self.eef_x_ini
        if self.max_eef_y < self.eef_y - self.eef_y_ini:
            self.max_eef_y = self.eef_y - self.eef_y_ini
        if self.min_eef_y > self.eef_y - self.eef_y_ini:
            self.min_eef_y = self.eef_y - self.eef_y_ini
        if self.max_eef_z < self.eef_z - self.eef_z_ini:
            self.max_eef_z = self.eef_z - self.eef_z_ini
        if self.min_eef_z > self.eef_z - self.eef_z_ini:
            self.min_eef_z = self.eef_z - self.eef_z_ini

        if self.max_eef_rpy_x < self.eef_rpy.x - self.eef_rpy_ini.x:
            self.max_eef_rpy_x = self.eef_rpy.x - self.eef_rpy_ini.x
        if self.min_eef_rpy_x > self.eef_rpy.x - self.eef_rpy_ini.x:
            self.min_eef_rpy_x = self.eef_rpy.x - self.eef_rpy_ini.x
        if self.max_eef_rpy_y < self.eef_rpy.y - self.eef_rpy_ini.y:
            self.max_eef_rpy_y = self.eef_rpy.y - self.eef_rpy_ini.y
        if self.min_eef_rpy_y > self.eef_rpy.y - self.eef_rpy_ini.y:
            self.min_eef_rpy_y = self.eef_rpy.y - self.eef_rpy_ini.y
        if self.max_eef_rpy_z < self.eef_rpy.z - self.eef_rpy_ini.z:
            self.max_eef_rpy_z = self.eef_rpy.z - self.eef_rpy_ini.z
        if self.min_eef_rpy_z > self.eef_rpy.z - self.eef_rpy_ini.z:
            self.min_eef_rpy_z = self.eef_rpy.z - self.eef_rpy_ini.z

        self.max_taxel0 = self.max_static_taxel0
        self.min_taxel0 = self.min_static_taxel0
        self.max_taxel1 = self.max_static_taxel1
        self.min_taxel1 = self.min_static_taxel1

        #1 rotation of knob(+), #2 bonus of knob rotation(+), #3 door panel open(+), 
        if self.imu_link_rpy.x < 0:
            self.knob_rotation_r = self.imu_link_rpy.x * knob_c
        elif 0 < self.imu_link_rpy.x < knob_threshold * 1 / 4:
            self.knob_rotation_r = self.imu_link_rpy.x * knob_c                     # 0.18 * 100 = 18 (0-18)
        elif knob_threshold / 4 <= self.imu_link_rpy.x < knob_threshold * 2 / 4:
            self.knob_rotation_r = self.imu_link_rpy.x * knob_c + knob_bonus_c      # 0.18 * 100 + 10 = 28 (28-45)
        elif knob_threshold * 2 / 4 <= self.imu_link_rpy.x < knob_threshold * 3 / 4:
            self.knob_rotation_r = self.imu_link_rpy.x * knob_c + knob_bonus_c * 2  # 0.35 * 100 + 10 * 2 = 55 (55-73)
        elif knob_threshold * 3 / 4 <= self.imu_link_rpy.x < knob_threshold:
            self.knob_rotation_r = self.imu_link_rpy.x * knob_c + knob_bonus_c * 3  # 0.53 * 100 + 10 * 3 = 83 (83-100)
            self.panel_rotation_r =  (self.imu_link_rpy_ini.z - self.imu_link_rpy.z) * panel_c
        elif knob_threshold < self.imu_link_rpy.x:
            self.knob_rotation_r = knob_threshold * knob_c + knob_bonus_c * 10      # 0.7 * 100 + 10 * 10 = 170 (170 * 32 = 5440)
            if (self.imu_link_rpy_ini.z - self.imu_link_rpy.z) < panel_threshold * 1 / 4:
                self.panel_rotation_r =  (self.imu_link_rpy_ini.z - self.imu_link_rpy.z) * 0.3 / 0.078 * panel_c + panel_bonus_c      # 0.075 * 2000 + 10 (10-160)
            elif panel_threshold * 1 / 4 <= (self.imu_link_rpy_ini.z - self.imu_link_rpy.z) < panel_threshold * 2 / 4:
                self.panel_rotation_r =  (self.imu_link_rpy_ini.z - self.imu_link_rpy.z) * 0.3 / 0.078 * panel_c + panel_bonus_c * 2  # 0.15 * 2000 + 10 * 2 (170-320)
            elif panel_threshold * 2 / 4 <= (self.imu_link_rpy_ini.z - self.imu_link_rpy.z) < panel_threshold * 3 / 4:
                self.panel_rotation_r =  (self.imu_link_rpy_ini.z - self.imu_link_rpy.z) * 0.3 / 0.078 * panel_c + panel_bonus_c * 3  # 0.225 * 2000 + 10 * 3 (330-510)
            elif panel_threshold * 3 / 4 <= (self.imu_link_rpy_ini.z - self.imu_link_rpy.z) < panel_threshold:
                self.panel_rotation_r =  (self.imu_link_rpy_ini.z - self.imu_link_rpy.z) * 0.3 / 0.078 * panel_c + panel_bonus_c * 5  # 0.3 * 2000 + 10 * 5 (530-650 * 32 = 20800)
            elif panel_threshold <= (self.imu_link_rpy_ini.z - self.imu_link_rpy.z):
                self.panel_rotation_r = panel_threshold * 0.3 / 0.078 * panel_c + panel_bonus_c * 300 + (n_step - update) * 750       # 0.3 * 2000 + 10 * 300 + n * 750 (3600 + 750*n)
                self.success = 1
                self.success_times += self.success
                print("success", self.success)

        print("##1 knob_rotation_r", self.knob_rotation_r, self.imu_link_rpy.x)
        print("##1 door_rotation_r", self.panel_rotation_r, self.imu_link_rpy_ini.z - self.imu_link_rpy.z)

        #5 movement of door frame(-)
#        if abs(self.door_frame.position.x - self.door_frame_ini.position.x) > self.tolerances1 or abs(self.door_frame.position.y - self.door_frame_ini.position.y) > self.tolerances1 or abs(self.door_frame.position.z - self.door_frame_ini.position.z) > self.tolerances1:
#            self.door_frame_limit_r = - ( tolerances_c * ( n_step - update ) / n_step + tolerances_c)
        if abs(self.door_frame.position.x - self.door_frame_ini.position.x) > self.tolerances1:
            self.door_frame_limit_r = - tolerances_c * abs(self.door_frame.position.x - self.door_frame_ini.position.x) * 1000
        if abs(self.door_frame.position.y - self.door_frame_ini.position.y) > self.tolerances1:
            self.door_frame_limit_r = - tolerances_c * abs(self.door_frame.position.y - self.door_frame_ini.position.y) * 1000
        if abs(self.door_frame.position.z - self.door_frame_ini.position.z) > self.tolerances1:
            self.door_frame_limit_r = - tolerances_c * abs(self.door_frame.position.z - self.door_frame_ini.position.z) * 1000
        print("##2 door_frame limit_r", self.door_frame_limit_r)

        if self.max_door_tolerance < abs(self.door_frame.position.x - self.door_frame_ini.position.x) * 1000:
            self.max_door_tolerance = abs(self.door_frame.position.x - self.door_frame_ini.position.x) * 1000
        if self.max_door_tolerance < abs(self.door_frame.position.y - self.door_frame_ini.position.y) * 1000:
            self.max_door_tolerance = abs(self.door_frame.position.y - self.door_frame_ini.position.y) * 1000
        if self.max_door_tolerance < abs(self.door_frame.position.z - self.door_frame_ini.position.z) * 1000:
            self.max_door_tolerance = abs(self.door_frame.position.z - self.door_frame_ini.position.z) * 1000

        if self.min_door_tolerance > abs(self.door_frame.position.x - self.door_frame_ini.position.x) * 1000:
            self.min_door_tolerance = abs(self.door_frame.position.x - self.door_frame_ini.position.x) * 1000
        if self.min_door_tolerance > abs(self.door_frame.position.y - self.door_frame_ini.position.y) * 1000:
            self.min_door_tolerance = abs(self.door_frame.position.y - self.door_frame_ini.position.y) * 1000
        if self.min_door_tolerance > abs(self.door_frame.position.z - self.door_frame_ini.position.z) * 1000:
            self.min_door_tolerance = abs(self.door_frame.position.z - self.door_frame_ini.position.z) * 1000
#        print("door_frame.position", self.door_frame.position)
#        print("door_frame_ini.position", self.door_frame_ini.position)

        #6 over force limit1(-)
        for obs_name in self.observations:
            if obs_name == "force_x":
                if self.force_limit2 < self.delta_force_x or self.delta_force_x < -self.force_limit2:
                    force_x_limit2_r = - (force_c2 * ( n_step - update ) / n_step + force_c2 )
                    print("# force_x limit2_r", force_x_limit2_r)
                elif self.force_limit1 < self.delta_force_x or self.delta_force_x < -self.force_limit1:
                    force_x_limit1_r = - (force_c * abs(abs(self.delta_force_x)-abs(self.force_limit1)) * ( n_step - update ) / n_step + force_c)
                    print("# force_x limit1_r", force_x_limit1_r)
                else:
                    force_x_limit1_r = 0
                    force_x_limit2_r = 0
            if obs_name == "force_x":
                if self.force_limit2 < self.delta_force_y or self.delta_force_y < -self.force_limit2:
                    force_y_limit2_r = - (force_c2 * ( n_step - update ) / n_step + force_c2)
                    print("# force_y limit2_r", force_y_limit2_r)
                elif self.force_limit1 < self.delta_force_y or self.delta_force_y < -self.force_limit1:
                    force_y_limit1_r = - (force_c * abs(abs(self.delta_force_y)-abs(self.force_limit1)) * ( n_step - update ) / n_step + force_c)
                    print("# force_y limit1_r", force_y_limit1_r)
                else:
                    force_y_limit1_r = 0
                    force_y_limit2_r = 0
            if obs_name == "force_z":
                if self.force_limit2 < self.delta_force_z or self.delta_force_z < -self.force_limit2:
                    force_z_limit2_r = - (force_c2 * ( n_step - update ) / n_step + force_c2)
                    print("# force_z limit2_r", force_z_limit2_r)
                elif self.force_limit1 < self.delta_force_z or self.delta_force_z < -self.force_limit1:
                    force_z_limit1_r = - (force_c * abs(abs(self.delta_force_z)-abs(self.force_limit1)) * ( n_step - update ) / n_step + force_c)
                    print("# force_z limit1_r", force_z_limit1_r)
                else:
                    force_z_limit1_r = 0
                    force_z_limit2_r = 0
            if obs_name == "torque_x":
                if self.torque_limit2 < self.delta_torque_x or self.delta_torque_x < -self.torque_limit2:
                    torque_x_limit2_r = - (force_c2 * ( n_step - update ) / n_step + force_c2)
                    print("# torque_x limit2_r", torque_x_limit2_r)
                elif self.torque_limit1 < self.delta_torque_x or self.delta_torque_x < -self.torque_limit1:
                    torque_x_limit1_r = - (force_c * abs(abs(self.delta_torque_x)-abs(self.torque_limit1)) * ( n_step - update ) / n_step + force_c)
                    print("# torque_x limit1_r", torque_x_limit1_r)
                else:
                    torque_x_limit1_r = 0
                    torque_x_limit2_r = 0
            if obs_name == "torque_y":
                if self.torque_limit2 < self.delta_torque_y or self.delta_torque_y < -self.torque_limit2:
                    torque_y_limit2_r = - (force_c2 * ( n_step - update ) / n_step + force_c2)
                    print("# torque_y limit2_r", torque_y_limit2_r)
                elif self.torque_limit1 < self.delta_torque_y or self.delta_torque_y < -self.torque_limit1:
                    torque_y_limit1_r = - (force_c * abs(abs(self.delta_torque_y)-abs(self.torque_limit1)) * ( n_step - update ) / n_step + force_c)
                    print("# torque_y limit1_r", torque_y_limit1_r)
                else:
                    torque_y_limit1_r = 0
                    torque_y_limit2_r = 0
            if obs_name == "torque_z":
                if self.torque_limit2 < self.delta_torque_z or self.delta_torque_z < -self.torque_limit2:
                    torque_z_limit2_r = - (force_c2 * ( n_step - update ) / n_step + force_c2)
                    print("# torque_z limit2_r", torque_z_limit2_r)
                elif self.torque_limit1 < self.delta_torque_z or self.delta_torque_z < -self.torque_limit1:
                    torque_z_limit1_r = - (force_c * abs(abs(self.delta_torque_z)-abs(self.torque_limit1)) * ( n_step - update ) / n_step + force_c)
                    print("# torque_z limit1_r", torque_z_limit1_r)
                else:
                    torque_z_limit1_r = 0
                    torque_z_limit2_r = 0
        self.force_limit_r = force_x_limit1_r + force_y_limit1_r + force_z_limit1_r + torque_x_limit1_r + torque_y_limit1_r + torque_z_limit1_r + force_x_limit2_r + force_y_limit2_r + force_z_limit2_r + torque_x_limit2_r + torque_y_limit2_r + torque_z_limit2_r
        print("##3 force_limit_r", self.force_limit_r)

        #7 release the knob(-)
        for obs_name in self.observations:
            if obs_name == "image_cnn" or "image_data":
                if self.min_static_taxel0 < self.min_static_limit and self.min_static_taxel1 < self.min_static_limit:
                    min_static_limit_r = - ( taxel_c * (n_step - update) / n_step + taxel_c)
                    print("##4 min_static limit_r", min_static_limit_r)
                if self.max_static_taxel0 > self.max_static_limit and self.max_static_taxel1 > self.max_static_limit:
                    max_static_limit_r = - ( taxel_c * (n_step - update) / n_step + taxel_c)
                    print("##4 max_static limit_r", max_static_limit_r)
        self.static_limit_r = min_static_limit_r + max_static_limit_r

        #8 joint(+)
        act_5_n_limit = self.wr3_init_value2        # 1.5733145480767723
        act_5_p_limit = self.wr3_after_pull + 0.2   # 2.5733145480767723
        act_4_n_limit = self.wr2_after_pull - 0.2   # -1.7477778260118484
        act_4_p_limit = self.wr2_init_value2        # -1.4691158467941916
        act_3_n_limit = self.wr1_after_pull - 0.2   # 2.1563878359790927
        act_3_p_limit = self.wr1_init_value2        # 2.177423014918695
        act_2_n_limit = self.elb_init_value2        # 2.413675198293162
        act_2_p_limit = self.elb_after_pull + 0.2   # 2.498128485003836
        act_1_n_limit = self.shl_after_pull - 0.2   # -1.4969974700328346
        act_1_p_limit = self.shl_init_value2        # -1.434487752926512
        act_0_n_limit = self.shp_init_value2        # 1.491407573528791
        act_0_p_limit = self.shp_after_pull + 0.2   # 1.648087725653139

        current_joint_value = self.get_joint_value()
        if act_5_n_limit < current_joint_value[5] and current_joint_value[5] < act_5_p_limit:
            action5_limit_r = (current_joint_value[5] - act_5_n_limit) * act_5_n
            print("##5 action5 limit_r", action5_limit_r)
        if act_4_n_limit < current_joint_value[4] and current_joint_value[4] < act_4_p_limit:
            action4_limit_r = - (current_joint_value[4] - act_4_p_limit) * act_4_n
            print("##5 action4 limit_r", action4_limit_r)
        if act_3_n_limit < current_joint_value[3] and current_joint_value[3] < act_3_p_limit:
            action3_limit_r = - (current_joint_value[3] - act_3_p_limit) * act_3_n
            print("##5 action3 limit_r", action3_limit_r)
        if act_2_n_limit < current_joint_value[2] and current_joint_value[2] < act_2_p_limit:
            action2_limit_r = (current_joint_value[2] - act_2_n_limit) * act_2_n
            print("##5 action2 limit_r", action2_limit_r)
        if act_1_n_limit < current_joint_value[1] and current_joint_value[1] < act_1_p_limit:
            action1_limit_r = - (current_joint_value[1] - act_1_p_limit) * act_1_n
            print("##5 action1 limit_r", action1_limit_r)
        if act_0_n_limit < current_joint_value[0] and current_joint_value[0] < act_0_p_limit:
            action0_limit_r = (current_joint_value[0] - act_0_n_limit) * act_0_n
            print("##5 action0 limit_r", action0_limit_r)
        self.action_limit_r = action5_limit_r + action4_limit_r + action3_limit_r + action2_limit_r + action1_limit_r + action0_limit_r
#        print("##5 action_limit_r.", current_joint_value[5], current_joint_value[4], current_joint_value[3], current_joint_value[2], current_joint_value[1], current_joint_value[0])
        print("##5 action_limit_r.", self.action_limit_r)

        #9 act_correct(+)
        self.act_correct_r = self.act_correct_n / sub_step * act_correct_c
        print("##6 act_correct_r", self.act_correct_r)

        self.eef_x_ini, self.eef_y_ini, self.eef_z_ini = self.get_xyz(self.init_joint_pose2)
        self.eef_rpy_ini = self.get_orientation(self.init_joint_pose2)

        #10 cartesian(+)
        catesian_x = (1 - abs(self.eef_x_ini - self.ee_xyz[0]) * 10) * catesian_xyz_c
        catesian_y = (1 - abs(self.eef_y_ini - self.ee_xyz[1]) * 10) * catesian_xyz_c
        catesian_z = (1 - abs(self.eef_z_ini - self.ee_xyz[2]) * 10) * catesian_xyz_c
        self.catesian_xyz_r = catesian_x + catesian_y + catesian_z
        print("##7-1 cartesian_xyz_r", catesian_x, catesian_y, catesian_z)
#        print("cartesian_xyz", self.ee_xyz[0], self.ee_xyz[1], self.ee_xyz[2])

        catesian_rpy_x = (1 - abs(self.eef_rpy_ini.x - self.eef_rpy.x) * 10) * catesian_rpy_c
        catesian_rpy_y = (1 - abs(self.eef_rpy_ini.y - self.eef_rpy.y) * 10) * catesian_rpy_c
        catesian_rpy_z = (1 - abs(self.eef_rpy_ini.z - self.eef_rpy.z) * 10) * catesian_rpy_c
        self.catesian_rpy_r = catesian_rpy_x + catesian_rpy_y + catesian_rpy_z

#        print("after rotate rpy", self.get_orientation([1.61263543388173, -1.47241659116041, 2.47074795062225, 2.15165861084186, -1.70139421434253, 2.57500536619159]))
#        print("##7-2 catesian_rpy_r", self.eef_rpy.x, self.eef_rpy.y, self.eef_rpy.z)
        print("##7-2 catesian_rpy_r", catesian_rpy_x, catesian_rpy_y, catesian_rpy_z)
#        print("catesian_rpy", self.eef_rpy.x, self.eef_rpy.y, self.eef_rpy.z)

        if self.cartesian_flag == 0:
            self.cartesian_bonus_r = cartesian_c
            print("##7-3 cartesian_bonus_r", self.cartesian_bonus_r)

        if self.max_knob_rotation < self.imu_link_rpy.x:
            self.max_knob_rotation = self.imu_link_rpy.x
        if self.max_door_rotation < self.imu_link_rpy_ini.z - self.imu_link_rpy.z:
            self.max_door_rotation = self.imu_link_rpy_ini.z - self.imu_link_rpy.z

        self.negative_r = self.door_frame_limit_r + self.force_limit_r + self.static_limit_r
        print("### negative_r", self.negative_r)
        self.action_r = self.action_limit_r + self.act_correct_r + self.catesian_xyz_r + self.catesian_rpy_r + self.cartesian_bonus_r

        compute_rewards = self.knob_rotation_r + self.panel_rotation_r + self.negative_r + self.action_r
        print("### total_compute_rewards", compute_rewards)

        return compute_rewards

    def check_done(self, update):
        if update > 1:
            observation = self.get_observations()
            if self.success == 1:
                print("####################################")
                print("############## success #############", update)
                print("####################################")
                return True
            elif abs(self.door_frame.position.x - self.door_frame_ini.position.x) > self.tolerances2 or abs(self.door_frame.position.y - self.door_frame_ini.position.y) > self.tolerances2 or abs(self.door_frame.position.z - self.door_frame_ini.position.z) > self.tolerances2:
                print("########## door frame position over the limit ##########", update)
                if self.success == 0:
                    self.fail_door_frame_limit += 1
                return True
            elif self.min_static_taxel0 < self.min_static_limit and self.min_static_taxel1 < self.min_static_limit:
                print("########## static_taxles over the min_limit ##########", update)
                if self.success == 0:
                    self.fail_taxel_limit += 1
#                return True
            elif self.max_static_taxel0 > self.max_static_limit and self.max_static_taxel1 > self.max_static_limit:
                print("########## static_taxles over the max_limit ##########", update)
                if self.success == 0:
                    self.fail_taxel_limit += 1
#                return True
            elif self.force_limit2 < self.delta_force_x or self.delta_force_x < -self.force_limit2:
                print("########## force.x over the limit2 ##########", self.delta_force_x)
#                return True
            elif self.force_limit2 < self.delta_force_y or self.delta_force_y < -self.force_limit2:
                print("########## force.y over the limit2 ##########", self.delta_force_y)
#                return True
            elif self.force_limit2 < self.delta_force_z or self.delta_force_z < -self.force_limit2:
                print("########## force.z over the limit2 ##########", self.delta_force_z)
#                return True
            elif self.torque_limit2 < self.delta_torque_x or self.delta_torque_x < -self.torque_limit2:
                print("########## torque.x over the limit2 ##########", self.delta_torque_x)
#                return True
            elif self.torque_limit2 < self.delta_torque_y or self.delta_torque_y < -self.torque_limit2:
                print("########## torque.y over the limit2 ##########", self.delta_torque_y)
#                return True
            elif self.torque_limit2 < self.delta_torque_z or self.delta_torque_z < -self.torque_limit2:
                print("########## torque.z over the limit2 ##########", self.delta_torque_z)
#                return True
            elif self.sub_action[5] > 2.57 or self.sub_action[5] < 1.57:
                print("########## action_limit ##########", self.sub_action[5])
#                return True
            elif observation[5] < -1 or observation[5] > 1:
                print("########## wr3_limit ##########", observation[5])
#                return True
            elif observation[4] < -1 or observation[4] > 1:
                print("########## wr2_limit ##########", observation[4])
#                return True
            elif observation[3] < -1 or observation[3] > 1:
                print("########## wr1_limit ##########", observation[3])
#                return True
            elif observation[2] < -1 or observation[2] > 1:
                print("########## elb_limit ##########", observation[2])
#                return True
            elif observation[1] < -1 or observation[1] > 1:
                print("########## shl_limit ##########", observation[1])
#                return True
            elif observation[0] < -1 or observation[0] > 1:
                print("########## shp_limit ##########", observation[0])
#                return True
            else:
                return False
        else :
            return False
