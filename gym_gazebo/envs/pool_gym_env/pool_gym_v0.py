#!/usr/bin/env python3
import gym
import rospy
import roslaunch
import time
import numpy as np
from gym import utils, spaces
from gym_gazebo.envs import gazebo_env
from geometry_msgs.msg import Twist
from std_srvs.srv import Empty
from gym.utils import seeding
import copy
import math
import os
import cv2
import cv2.aruco as aruco
from cv_bridge import CvBridge, CvBridgeError
from std_msgs.msg import Float32
from sensor_msgs.msg import Image

class GazeboPoolv0Env(gazebo_env.GazeboEnv):
	def __init__(self):
		
		# init environment
		LAUNCH_PATH = '/home/fizzer/RL-Watercraft/gym_gazebo/envs/ros_ws/src/boat_gazebo/launch/boat.launch'
		gazebo_env.GazeboEnv.__init__(self, LAUNCH_PATH)

		# init gazebo services
		self.unpause = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
		self.pause = rospy.ServiceProxy('gazebo/pause_physics', Empty)
		self.reset_proxy = rospy.ServiceProxy('/gazebo/reset_world', Empty)

		# init pubs & subs
		self.left_pub = rospy.Publisher('/boat/thrusters/left_thrust_cmd', Float32, queue_size=10)
		self.right_pub = rospy.Publisher('/boat/thrusters/right_thrust_cmd', Float32, queue_size=10)
		rospy.Subscriber('/camera1/image_raw', Image, queue_size=10)

		# init aruco detector
		aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
		aruco_params = aruco.DetectorParameters()
		self.detector = aruco.ArucoDetector(aruco_dict, aruco_params)
		
		# idk
		self._seed()
		self.bridge = CvBridge()

		# define action and observation spaces
		act_high = np.array([1, 1])
		obs_high = np.array([np.inf, np.inf, np.inf, np.inf, np.pi, np.inf])
		self.action_space = spaces.Box(-act_high, act_high)
		self.observation_space = spaces.Box(-obs_high, obs_high)

		# position target (x y yaw x' y' yaw')
		self.target = np.array([5, 5, 0, 0, 0, 0])
		self.tol = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1])

	def get_state(self, prev_state, corners):

		# extract x and y at tag center
		corner_pts = corners[0][0]
		center = corner_pts.mean(axis=0)
		x,y = center.astype(int)

		# calc yaw
		tag_top = corner_pts[1] - corner_pts[0]
		a_rad = np.arctan2(-tag_top[1], tag_top[0])
		a = np.degrees(a_rad)

		# calc velocities

		if self.prev_time == 0:
			return [x,y,a,0,0,0]
		
		dt =  (self.time - self.prev_time).to_sec()
		x_vel = (x - prev_state[0]) / dt
		y_vel = (y - prev_state[1]) / dt
		a_vel = (a - prev_state[2]) / dt

		# return state
		return [x,y,a,x_vel,y_vel,a_vel]


	def done_check(self, state): #task
		# if (abs(state - self.target) < self.tol):
		# 	return True
		# else:
		return False
		
	def compute_reward(self, state, prev_state): #task
		reward = 0
		# for i in range(5):
		# 	if (abs(self.target[i]-state[i]) < self.tol[i]):
		# 		reward += 2
		# 	elif (abs(self.target[i]-state[i]) < abs(self.target[i]-prev_state[i])):
		# 		reward += 1
		# 	else: reward -= 1
		return reward

	def _seed(self, seed=None):
		self.np_random, seed = seeding.np_random(seed)
		return [seed]

	def step(self, action):
		speedL = action[0]
		speedR = action[1]

		# unpause physics
		rospy.wait_for_service('/gazebo/unpause_physics')
		try:
			self.unpause()
		except (rospy.ServiceException) as e:
			print ("/gazebo/unpause_physics service call failed")

		# execute action
		self.left_pub.publish(speedL)
		self.right_pub.publish(speedR)

		# get data from topic
		img_data = None
		while img_data is None:
			try:
				img_data = rospy.wait_for_message('/camera1/image_raw', Image, timeout=5)
			except:
				pass

		# process image
		try:

			self.prev_time = self.time
			self.time = img_data.header.stamp

			cv_image = self.bridge.imgmsg_to_cv2(img_data, "bgr8")
			gray = cv2.cvtColor(cv_image, cv2.COLOR_RGB2GRAY)

			corners, ids, rejected = self.detector.detectMarkers(gray)
			aruco.drawDetectedMarkers(cv_image, corners, ids)

			cv2.imshow("Camera Feed", cv_image)
			cv2.waitKey(1)

		except CvBridgeError as e:
			print(e)

        # pause physics
		# rospy.wait_for_service('/gazebo/pause_physics')
		# try:
		# 	self.pause()
		# except (rospy.ServiceException) as e:
		# 	print ("/gazebo/pause_physics service call failed")

		# get current state (x y yaw, x' y' yaw')
		self.prev_state = self.state
		if len(corners) > 0:
			self.state = self.get_state(self.prev_state,corners)
		else:
			pass

		# compute reward
		step_reward = self.compute_reward(self.state, self.prev_state)

		# check if done
		done = self.done_check(self.state)

		self.prev_state = self.state
		info = {}

      	# return state reward and done flag
		return self.state, step_reward, done, info

	def reset(self):

		# reset environment
		rospy.wait_for_service('/gazebo/reset_simulation')
		try:
			self.reset_proxy()
		except (rospy.ServiceException) as e:
			print ("/gazebo/reset_simulation service call failed")

		# unpause simulation to make observation
		rospy.wait_for_service('/gazebo/unpause_physics')
		try:
			self.unpause()
		except (rospy.ServiceException) as e:
			print ("/gazebo/unpause_physics service call failed")

		# get data from topic
		img_data = None
		while img_data is None:
			try:
				img_data = rospy.wait_for_message('/camera1/image_raw', Image, timeout=5)
			except:
				pass

		# process image
		try:

			self.prev_state = [0,0,0,0,0,0]
			self.prev_time = 0
			self.time = img_data.header.stamp

			cv_image = self.bridge.imgmsg_to_cv2(img_data, "bgr8")
			gray = cv2.cvtColor(cv_image, cv2.COLOR_RGB2GRAY)

			corners, ids, rejected = self.detector.detectMarkers(gray)
			aruco.drawDetectedMarkers(cv_image, corners, ids)

			cv2.imshow("Camera Feed", cv_image)
			cv2.waitKey(1)

		except CvBridgeError as e:
			print(e)

		# get current state (x y yaw, x' y' yaw')
		if len(corners) > 0:
			self.state = self.get_state(self.prev_state,corners)
		else:
			self.state = self.prev_state

		# pause physics
		rospy.wait_for_service('/gazebo/pause_physics')
		try:
			self.pause()
		except (rospy.ServiceException) as e:
			print ("/gazebo/pause_physics service call failed")

		return self.state

        
