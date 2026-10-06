#!/usr/bin/env python3
import gym
import rospy
import roslaunch
import time
import numpy as np
from gym import spaces
from gym_gazebo.envs import gazebo_env
from gazebo_msgs.msg import ModelStates
from std_srvs.srv import Empty
from gym.utils import seeding
import math
import os
from std_msgs.msg import Float32
from policy_contract import (
	gazebo_thrust_action,
	policy_observation,
)

class GazeboPoolv0Env(gazebo_env.GazeboEnv):
	def __init__(self):
		
		# init environment
		LAUNCH_PATH = os.path.abspath(os.path.join(
			os.path.dirname(__file__), '..', 'ros_ws', 'src', 'boat_gazebo', 'launch', 'boat.launch'))
		gazebo_env.GazeboEnv.__init__(self, LAUNCH_PATH)

		# init gazebo services
		self.unpause = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
		self.pause = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
		self.reset_proxy = rospy.ServiceProxy('/gazebo/reset_world', Empty)

		# init pubs & subs
		self.left_pub = rospy.Publisher('/boat/thrusters/left_thrust_cmd', Float32, queue_size=10)
		self.right_pub = rospy.Publisher('/boat/thrusters/right_thrust_cmd', Float32, queue_size=10)
		
		# misc
		self._seed()

		# define action and observation spaces
		self.action_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
		self.observation_space = spaces.Box(-1.0, 1.0, shape=(6,), dtype=np.float32)

		self.goal = np.array([5.0, 5.0], dtype=np.float32)
		self.position_scale_m = 10.0
		self.marker_offset_x_m = 0.5
		self.control_period_s = 0.1

		# timeout counter
		self.count = 0

	def get_state(self):
		model_states = rospy.wait_for_message('/gazebo/model_states', ModelStates, timeout=5)
		try:
			boat_index = model_states.name.index('boat')
		except ValueError as error:
			raise RuntimeError("Gazebo model_states does not contain the 'boat' model") from error

		pose = model_states.pose[boat_index]
		twist = model_states.twist[boat_index]
		q = pose.orientation
		yaw = math.atan2(
			2.0 * (q.w * q.z + q.x * q.y),
			1.0 - 2.0 * (q.y * q.y + q.z * q.z),
		)
		offset_x = self.marker_offset_x_m * math.cos(yaw)
		offset_y = self.marker_offset_x_m * math.sin(yaw)
		yaw_rate = twist.angular.z
		marker_x = pose.position.x + offset_x
		marker_y = pose.position.y + offset_y
		marker_velocity_x = twist.linear.x - yaw_rate * offset_y
		marker_velocity_y = twist.linear.y + yaw_rate * offset_x
		return policy_observation(
			marker_x,
			marker_y,
			yaw,
			marker_velocity_x,
			marker_velocity_y,
			yaw_rate,
			float(self.goal[0]),
			float(self.goal[1]),
			position_scale_m=self.position_scale_m,
		)


	def done_check(self, state, prev_state):

		goal_distance_m = self.position_scale_m * math.hypot(state[0], state[1])
		return goal_distance_m <= 0.5 or self.count >= 100
		
	def compute_reward(self, state, prev_state):

		previous_distance = self.position_scale_m * math.hypot(prev_state[0], prev_state[1])
		current_distance = self.position_scale_m * math.hypot(state[0], state[1])
		reward = previous_distance - current_distance
		if current_distance <= 0.5:
			reward += 10.0
		return reward

	def _seed(self, seed=None):
		self.np_random, seed = seeding.np_random(seed)
		return [seed]

	def step(self, action):
		action = gazebo_thrust_action(action)

		# unpause physics
		rospy.wait_for_service('/gazebo/unpause_physics')
		try:
			self.unpause()
		except (rospy.ServiceException) as e:
			print ("/gazebo/unpause_physics service call failed")

		# execute action
		self.left_pub.publish(float(action[0]))
		self.right_pub.publish(float(action[1]))
		rospy.sleep(self.control_period_s)
		self.prev_state = self.state
		self.state = self.get_state()
		self.pause()

		# compute reward
		step_reward = self.compute_reward(self.state, self.prev_state)

		# check if done
		done = self.done_check(self.state, self.prev_state)

		self.count += 1
		info = {}

      	# return state reward and done flag
		return self.state, step_reward, done, info

	def reset(self):

		# reset environment
		rospy.wait_for_service('/gazebo/reset_world')
		try:
			self.reset_proxy()
		except (rospy.ServiceException) as e:
			print ("/gazebo/reset_world service call failed")

		# unpause simulation to make observation
		rospy.wait_for_service('/gazebo/unpause_physics')
		try:
			self.unpause()
		except (rospy.ServiceException) as e:
			print ("/gazebo/unpause_physics service call failed")

		self.state = self.get_state()
		self.prev_state = self.state
		self.pause()

		self.count = 0

		return self.state

        
