#!/usr/bin/env python
import math
import time
import copy
from std_msgs.msg import String
from std_msgs.msg import Float64
from geometry_msgs.msg import Vector3
from trajectory_msgs.msg import JointTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint
from .controllers_connection import ControllersConnection

# MoveIt
import os
import sys
import numpy as np
import rospy
import moveit_commander
from ur5_interface_for_door import UR5Interface
from robotiq_interface_for_door import RobotiqInterface

moveit = rospy.get_param("/moveit")
dt_reset = rospy.get_param("/act_params/dt_reset")
dt_act = rospy.get_param("/act_params/dt_act")
dt_grp = rospy.get_param("/act_params/dt_grp")

class JointTrajPub(object):
    def __init__(self):
        """
        Publish trajectory_msgs::JointTrajectory for velocity control
        """

        self._ctrl_conn = ControllersConnection(namespace="")
        current_controller_type =  rospy.get_param("/control_type")

        if (current_controller_type == "pos") or (current_controller_type == "traj_pos"):
        	self._ctrl_conn.load_controllers("pos_traj_controller")
        	self._joint_traj_pub = rospy.Publisher('/pos_traj_controller/command', JointTrajectory, queue_size=10)
        else:
        	self._ctrl_conn.load_controllers("vel_traj_controller")
        	self._joint_traj_pub = rospy.Publisher('/vel_traj_controller/command', JointTrajectory, queue_size=10)
       	self._grp_pub = rospy.Publisher('/gripper_controller/command', JointTrajectory, queue_size=10)

        if moveit == 1:
#            rospy.init_node("test_move_ur5_continuous", anonymous=True, disable_signals=True)
            self.ur5 = UR5Interface()
            self.grp = moveit_commander.MoveGroupCommander("gripper")

    def set_init_pose(self, init_pose):
    	"""
    	Sets joints to initial position [0,0,0]
    	:return: The init Pose
    	"""
    	self.check_publishers_connection()
    	self.move_joints(init_pose)

    def check_publishers_connection(self):
    	"""
    	Checks that all the publishers are working
    	:return:
    	"""
    	rate = rospy.Rate(1)  # 1hz
    	while (self._joint_traj_pub.get_num_connections() == 0):
    	    rospy.logdebug("No subscribers to vel_traj_controller yet so we wait and try again")
    	    try:
    	    	self._ctrl_conn.start_controllers(controllers_on="vel_traj_controller")
    	    	rate.sleep()
    	    except rospy.ROSInterruptException:
    	    	# This is to avoid error when world is rested, time when backwards.
    	    	pass
    	rospy.logdebug("_joint_traj_pub Publisher Connected")

    	rospy.logdebug("All Joint Publishers READY")

    def jointTrajectoryCommand(self, joints_array): # dtype=float32), <type 'numpy.ndarray'>
#    	rospy.loginfo("jointTrajectoryCommand")
    	try:    
#    	    rospy.loginfo (rospy.get_rostime().to_sec())
    	    while rospy.get_rostime().to_sec() == 0.0:
    	    	time.sleep(0.1)
#    	    	rospy.loginfo (rospy.get_rostime().to_sec())

    	    jt = JointTrajectory()
    	    jt.header.stamp = rospy.Time.now()
    	    jt.header.frame_id = "ur5"
    	    jt.joint_names.append("shoulder_pan_joint")
    	    jt.joint_names.append("shoulder_lift_joint")
    	    jt.joint_names.append("elbow_joint")
    	    jt.joint_names.append("wrist_1_joint")
    	    jt.joint_names.append("wrist_2_joint")
    	    jt.joint_names.append("wrist_3_joint")
    	    	    
    	    dt = dt_act 	#default 0.01
    	    p = JointTrajectoryPoint()	
    	    p.positions.append(1.488122534496775)   # 1.488122534496775
    	    p.positions.append(-1.4496597816566892) # -1.4496597816566892
    	    p.positions.append(joints_array[2])     # 2.4377209990850974
    	    p.positions.append(joints_array[3])
    	    p.positions.append(joints_array[4])
    	    p.positions.append(joints_array[5])
    	    jt.points.append(p)
    	    # set duration
    	    jt.points[0].time_from_start = rospy.Duration.from_sec(dt)
#            print("p.positions", p.positions)

    	    self._joint_traj_pub.publish(jt)

    	except rospy.ROSInterruptException: pass

    def jointTrajectoryCommand_reset(self, joints_array): # dtype=float32), <type 'numpy.ndarray'>
#    	rospy.loginfo("jointTrajectoryCommand")
    	try:    
#    	    rospy.loginfo (rospy.get_rostime().to_sec())
    	    while rospy.get_rostime().to_sec() == 0.0:
    	    	time.sleep(0.1)
#    	    	rospy.loginfo (rospy.get_rostime().to_sec())

    	    jt = JointTrajectory()
    	    jt.header.stamp = rospy.Time.now()
    	    jt.header.frame_id = "ur5"
    	    jt.joint_names.append("shoulder_pan_joint")
    	    jt.joint_names.append("shoulder_lift_joint")
    	    jt.joint_names.append("elbow_joint")
    	    jt.joint_names.append("wrist_1_joint")
    	    jt.joint_names.append("wrist_2_joint")
    	    jt.joint_names.append("wrist_3_joint")
    	    	    
    	    dt = dt_reset 	#default 0.01
    	    p = JointTrajectoryPoint()	
    	    p.positions.append(joints_array[0])
    	    p.positions.append(joints_array[1])
    	    p.positions.append(joints_array[2])
    	    p.positions.append(joints_array[3])
    	    p.positions.append(joints_array[4])
    	    p.positions.append(joints_array[5])
    	    jt.points.append(p)
    	    # set duration
    	    jt.points[0].time_from_start = rospy.Duration.from_sec(dt)

    	    self._joint_traj_pub.publish(jt)

    	except rospy.ROSInterruptException: pass

    def GrpCommand(self, joints_array): # dtype=float32), <type 'numpy.ndarray'>
#    	rospy.loginfo("GrpCommand")
    	try:    
#    	    rospy.loginfo (rospy.get_rostime().to_sec())
    	    while rospy.get_rostime().to_sec() == 0.0:
    	    	time.sleep(0.1)
#    	    	rospy.loginfo (rospy.get_rostime().to_sec())

    	    jt = JointTrajectory()
    	    jt.header.stamp = rospy.Time.now()
    	    jt.header.frame_id = "grp"
    	    jt.joint_names.append("simple_gripper_right_driver_joint")
    	    jt.joint_names.append("simple_gripper_left_driver_joint")
    	    jt.joint_names.append("simple_gripper_right_follower_joint")
    	    jt.joint_names.append("simple_gripper_left_follower_joint")
    	    jt.joint_names.append("simple_gripper_right_spring_link_joint")
    	    jt.joint_names.append("simple_gripper_left_spring_link_joint")
    	    	    
    	    dt = dt_grp 	#default 0.01
    	    p = JointTrajectoryPoint()	
    	    p.positions.append(joints_array[0])
    	    p.positions.append(joints_array[1])
    	    p.positions.append(joints_array[2])
    	    p.positions.append(joints_array[3])
    	    p.positions.append(joints_array[4])
    	    p.positions.append(joints_array[5])
    	    jt.points.append(p)

    	    # set duration
    	    jt.points[0].time_from_start = rospy.Duration.from_sec(dt)

    	    self._grp_pub.publish(jt)

    	except rospy.ROSInterruptException: pass

    def MoveItCommand(self, action):
        try:
            self.ur5 = UR5Interface()
            self.ur5.goto_pose_target(action)
        except rospy.ROSInterruptException: pass

    def MoveItJointTarget(self, joint_array):
        try:
            self.ur5 = UR5Interface()
            self.ur5.goto_joint_target(joint_array)
        except rospy.ROSInterruptException: pass

    def MoveItGrpOpen(self):
        try:
            self.grp = moveit_commander.MoveGroupCommander("gripper")
            self.grp.set_named_target('open')
            self.grp.go(wait=False)
        except rospy.ROSInterruptException: pass

    def MoveItGrpClose(self):
        try:
            self.grp = moveit_commander.MoveGroupCommander("gripper")
            self.grp.set_named_target('close0.31')
            self.grp.go(wait=False)
        except rospy.ROSInterruptException: pass

if __name__=="__main__":
    rospy.init_node('joint_publisher_node', log_level=rospy.WARN)
    joint_publisher = JointTrajPub()
    rate_value = 8.0
