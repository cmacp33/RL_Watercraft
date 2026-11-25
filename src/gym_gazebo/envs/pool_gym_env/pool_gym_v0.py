#!/usr/bin/env python3
import gym
import rospy
import roslaunch
import time
import numpy as np
from gym import spaces
from gym_gazebo.envs import gazebo_env
from gazebo_msgs.msg import ModelStates
from gazebo_msgs.srv import GetModelState
from std_srvs.srv import Empty
from gym.utils import seeding
import math
from collections import deque
import os
from std_msgs.msg import Float32
from policy_contract import (
	FORWARD_SPEED_SCALE_MPS,
	GOAL_RADIUS_M,
	YAW_RATE_SCALE_RADPS,
	POSITION_SCALE_M,
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
		# asks gzserver directly for the boat's current state (works while paused, never out of date)
		self.model_state_proxy = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)

		# init pubs & subs
		self.left_pub = rospy.Publisher('/boat/thrusters/left_thrust_cmd', Float32, queue_size=10)
		self.right_pub = rospy.Publisher('/boat/thrusters/right_thrust_cmd', Float32, queue_size=10)

		# Persistent subscriber keeping the newest model_states message. Opening a new subscriber
		# every step (rospy.wait_for_message) is slow, and physics keeps running meanwhile, which
		# stretched each 0.1 s control step to ~0.27 s of sim time.
		self._latest_model_states = None
		rospy.Subscriber('/gazebo/model_states', ModelStates, self._on_model_states, queue_size=1)

		# misc
		self._seed()

		# define action and observation spaces
		self.action_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
		self.observation_space = spaces.Box(-1.0, 1.0, shape=(6,), dtype=np.float32)

		self.goal = np.array([12.0, 12.0], dtype=np.float32)
		self.position_scale_m = POSITION_SCALE_M # shared with the real boat via policy_contract
		self.marker_offset_x_m = 0.020
		self.control_period_s = 0.1
		# episode length = max_episode_steps * control_period_s of sim time (300 -> 30 s);
		# must be long enough to cover the start-to-goal distance (~16.6 m) at the boat's speed
		self.max_episode_steps = 300

		# reward weights
		self.distance_weight = 2.0 # per metre closer to goal
		self.spin_weight = 0.5 # per spin rate. it should not spin.
		self.heading_weight = 0.5 # per radian turned toward goal
		self.step_penalty = 0.1 # per step
		self.goal_bonus = 30.0 # total paid for reaching the goal and holding it (see compute_reward)

		# The goal counts as reached only after the boat has stayed inside the goal radius for this
		# many steps in a row (x control_period_s = seconds). 1 = end the episode on first touch,
		# which taught the policy to arrive at full speed: fine in sim, but the real boat then
		# overshoots and cannot settle. Leaving the radius restarts the count.
		self.goal_hold_steps = 1
		self.hold_count = 0 # steps in a row inside the goal radius, up to now
		self.best_hold = 0 # longest hold so far this episode (the bonus is paid on new records only)

		# Goal heading (policy.heading_mode = 'target'): the boat must also face goal_heading, within
		# goal_heading_tolerance_rad, for a step to count toward the hold, and observations 2-3 are the
		# error to that heading instead of the bearing to the goal position. 'bearing' is the original task.
		self.heading_mode = 'bearing'
		self.goal_heading = 0.0 # rad, world frame (counterclockwise from +x)
		self.random_goal_headings = False # new uniform random goal heading each episode
		self.goal_heading_tolerance_rad = math.pi # pi = any heading counts
		# The heading reward is weighted by exp(-distance / this): close to the goal, facing the goal
		# heading pays; far away the distance reward does the steering.
		self.heading_distance_scale_m = 0.1

		# timeout counter
		self.count = 0

		# Domain randomization: per-episode random actuator/sensor conditions so the policy
		# copes with what calibration can't fix on the real boat (battery level, motor
		# mismatch, leftover deadband, latency, camera noise). Off by default; training
		# turns it on with --randomize. Reward and done always use the TRUE state.
		self.domain_randomization = False
		self.dr_ranges = {
			'motor_gain': (0.7, 1.15),      # per thruster, independent
			'deadband': (0.0, 0.10),        # |action| below this does nothing (normalized)
			'delay_steps': (0, 1),          # whole control steps of command latency
			'pos_noise_m': 0.03,            # camera position noise (std)
			'heading_noise_rad': math.radians(3.0),
			'speed_noise_mps': 0.05,
			'yaw_rate_noise_radps': 0.10,
		}
		self.dr_params = {}
		self._pending_actions = deque()

		# Random goals: a new goal each episode instead of the fixed self.goal, so the policy
		# learns to steer to any goal. The boat always resets to the spawn pose (0, 0, heading +x);
		# observations are boat-relative, so random goal directions also cover random start headings.
		self.random_goals = False
		self.goal_distance_range_m = (0.122, 0.486)  # from the spawn point; ~0.21 m/s -> <= ~2.5 s at 0.486 m
		self.goal_bounds_m = (0.53, 0.18)         # |x|, |y| limits: pool walls at x = +-0.65, y = +-0.30 m, minus the boat's half-width + 1.5 cm

		# where reset_world puts the boat (x, y in boat.launch); reset() checks it really got there
		self.spawn_xy_m = (0.0, 0.0)
		self.spawn_tolerance_m = 0.012
		self.reset_attempts = 5

	def sample_randomization(self):
		r = self.dr_ranges
		self.dr_params = {
			'left_gain': self.np_random.uniform(*r['motor_gain']),
			'right_gain': self.np_random.uniform(*r['motor_gain']),
			'deadband': self.np_random.uniform(*r['deadband']),
			'delay_steps': int(self.np_random.randint(r['delay_steps'][0], r['delay_steps'][1] + 1)),
		}
		self._pending_actions = deque([np.zeros(2, dtype=np.float32)] * self.dr_params['delay_steps'])

	def sample_goal(self):
		"""Uniform direction, uniform distance in goal_distance_range_m, inside the pool."""
		while True:
			distance = self.np_random.uniform(*self.goal_distance_range_m)
			angle = self.np_random.uniform(-math.pi, math.pi)
			goal = np.array([distance * math.cos(angle), distance * math.sin(angle)], dtype=np.float32)
			if np.all(np.abs(goal) <= self.goal_bounds_m):
				return goal

	def actuate(self, action):
		"""What the thrusters actually do with a commanded action (identity without randomization)."""
		if not self.domain_randomization:
			return action
		self._pending_actions.append(np.asarray(action, dtype=np.float32))
		delayed = self._pending_actions.popleft()
		p = self.dr_params
		out = delayed * np.array([p['left_gain'], p['right_gain']], dtype=np.float32)
		out[np.abs(delayed) < p['deadband']] = 0.0
		return np.clip(out, -1.0, 1.0)

	def observe(self, state):
		"""Observation handed to the policy: the true state plus camera-like noise."""
		if not self.domain_randomization:
			return state
		r = self.dr_ranges
		obs = np.array(state, dtype=np.float32)
		obs[0:2] += self.np_random.normal(0.0, r['pos_noise_m'], size=2) / self.position_scale_m
		heading = math.atan2(obs[2], obs[3]) + self.np_random.normal(0.0, r['heading_noise_rad'])
		obs[2], obs[3] = math.sin(heading), math.cos(heading)
		obs[4] += self.np_random.normal(0.0, r['speed_noise_mps']) / FORWARD_SPEED_SCALE_MPS
		obs[5] += self.np_random.normal(0.0, r['yaw_rate_noise_radps']) / YAW_RATE_SCALE_RADPS
		return np.clip(obs, -1.0, 1.0)

	def goal_distance_m(self):
		"""True distance from the marker to the goal (not affected by observation noise)."""
		return self.position_scale_m * math.hypot(self.state[0], self.state[1])

	def _on_model_states(self, msg):
		self._latest_model_states = msg

	def get_state(self, wait_fresh=True):
		"""Boat observation. wait_fresh=False uses the cached message (fast; physics must be paused)."""
		if wait_fresh or self._latest_model_states is None:
			model_states = rospy.wait_for_message('/gazebo/model_states', ModelStates, timeout=5)
		else:
			model_states = self._latest_model_states
		try:
			boat_index = model_states.name.index('boat')
		except ValueError as error:
			raise RuntimeError("Gazebo model_states does not contain the 'boat' model") from error

		return self._observation_from(model_states.pose[boat_index], model_states.twist[boat_index])

	def _observation_from(self, pose, twist):
		"""Normalized boat-relative observation from a world-frame pose and twist."""
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
			goal_heading_rad=float(self.goal_heading) if self.heading_mode == 'target' else None,
		)


	def update_hold(self, state):
		"""Count the steps in a row spent inside the goal radius (and, in 'target' heading mode,
		within the heading tolerance); call once per step, before the reward."""
		goal_distance_m = self.position_scale_m * math.hypot(state[0], state[1])
		inside = goal_distance_m <= GOAL_RADIUS_M
		if self.heading_mode == 'target':
			inside = inside and abs(math.atan2(state[2], state[3])) <= self.goal_heading_tolerance_rad
		self.hold_count = self.hold_count + 1 if inside else 0

	def goal_held(self):
		return self.hold_count >= self.goal_hold_steps

	def done_check(self, state, prev_state):

		return self.goal_held() or self.count >= self.max_episode_steps
		
	def compute_reward(self, state, prev_state):

		# distance to goal
		# state is distance to target in boat coordinates
		previous_dist = self.position_scale_m * math.hypot(prev_state[0], prev_state[1])
		current_dist = self.position_scale_m * math.hypot(state[0], state[1])
		# heading error magnitude in [0, pi], from sin (state[2]) and cos (state[3])
		heading_error_prev = abs(math.atan2(prev_state[2], prev_state[3]))
		heading_error_curr = abs(math.atan2(state[2], state[3]))

		# positive when closing distance / turning toward goal, negative when moving away / turning away
		distance_reward = self.distance_weight * (previous_dist - current_dist)
		if self.heading_mode == 'target':
			# Goal-heading error, weighted by closeness to the goal. Written as the change in a potential
			# (-weight * closeness * error), so driving around cannot farm it: any loop back to the same
			# state sums to zero, whatever the weights.
			closeness_prev = math.exp(-previous_dist / self.heading_distance_scale_m)
			closeness_curr = math.exp(-current_dist / self.heading_distance_scale_m)
			heading_reward = self.heading_weight * (closeness_prev * heading_error_prev
													- closeness_curr * heading_error_curr)
		else:
			heading_reward = self.heading_weight * (heading_error_prev - heading_error_curr)
		spin_penalty = self.spin_weight * abs(state[5]) # yaw_rate

		# small cost every step so standing still is never 0.0
		reward = distance_reward + heading_reward - self.step_penalty - spin_penalty

		# Goal bonus, paid in goal_hold_steps equal parts: one part each time the hold gets longer
		# than it has ever been this episode. Staying put collects the whole bonus and ends the
		# episode; leaving and coming back earns nothing until the old record is beaten, so the
		# total never exceeds goal_bonus (no reward farming by hovering in and out).
		# With goal_hold_steps = 1 this is the old one-time bonus on first touch.
		if self.hold_count > self.best_hold:
			self.best_hold = self.hold_count
			reward += self.goal_bonus / self.goal_hold_steps
		return reward

	def _seed(self, seed=None):
		self.np_random, seed = seeding.np_random(seed)
		return [seed]

	def step(self, action):
		action = self.actuate(gazebo_thrust_action(action))

		# unpause physics
		rospy.wait_for_service('/gazebo/unpause_physics')
		try:
			self.unpause()
		except (rospy.ServiceException) as e:
			print ("/gazebo/unpause_physics service call failed")

		# execute action
		self.left_pub.publish(float(action[0]))
		self.right_pub.publish(float(action[1]))
		step_start = rospy.get_time() # sim time (use_sim_time)
		rospy.sleep(self.control_period_s)
		self.pause() # pause physics right away so the step lasts ~control_period_s of sim time
		time.sleep(0.003) # wall-clock: let the last pre-pause model_states message arrive (rospy.sleep would hang while paused)
		self.prev_state = self.state
		self.state = self.get_state(wait_fresh=False)
		sim_dt = rospy.get_time() - step_start # > control_period_s if pausing lags behind fast physics

		# compute reward (the hold count comes first: the goal bonus depends on it)
		self.update_hold(self.state)
		step_reward = self.compute_reward(self.state, self.prev_state)

		# count this step before the done check, so an episode times out after exactly
		# max_episode_steps steps (it used to run one extra step)
		self.count += 1

		# check if done
		done = self.done_check(self.state, self.prev_state)
		# Two different ways an episode ends. Reaching the goal (and holding it for goal_hold_steps)
		# is a real end: nothing more can be earned. Running out of steps is not: the boat could
		# have carried on. Value-based methods (SAC/PPO) must only treat the first as terminal, so
		# the timeout is flagged here under the key gym's TimeLimit wrapper uses. An episode that
		# times out while the boat is inside the radius, but before the hold is complete, is a
		# timeout, not a reached goal.
		reached = self.goal_held()
		info = {'sim_dt': sim_dt, 'goal_dist_m': self.goal_distance_m(), 'applied_action': action,
				'reached': reached, 'in_goal': self.hold_count > 0, 'hold_steps': self.hold_count,
				# true (noise-free) observation 2-3 angle: to the goal heading in 'target' mode, else to the goal
				'heading_error_deg': math.degrees(math.atan2(self.state[2], self.state[3])),
				'TimeLimit.truncated': done and not reached}

      	# return state reward and done flag
		return self.observe(self.state), step_reward, done, info

	def _true_boat_state(self):
		"""Boat pose and twist straight from gzserver (service call, so it reflects right now)."""
		rospy.wait_for_service('/gazebo/get_model_state')
		deadline = time.time() + 15.0 # wall-clock; the boat may still be spawning right after launch
		while True:
			result = self.model_state_proxy('boat', 'world')
			if result.success:
				return result.pose, result.twist
			if time.time() > deadline:
				raise RuntimeError("get_model_state failed: %s" % result.status_message)
			time.sleep(0.2)

	def reset(self):

		# Pause first so nothing moves between the reset and the reading below.
		rospy.wait_for_service('/gazebo/pause_physics')
		self.pause()

		# Reset, then confirm with gzserver that the boat is back at the spawn point.
		# The first observation used to come from the /gazebo/model_states topic, which could
		# still deliver a message from before the reset: the episode then began with the previous
		# episode's final position, giving a wrong start distance and a huge first-step reward.
		rospy.wait_for_service('/gazebo/reset_world')
		for attempt in range(1, self.reset_attempts + 1):
			try:
				self.reset_proxy()
			except (rospy.ServiceException) as e:
				print ("/gazebo/reset_world service call failed")
			pose, twist = self._true_boat_state()
			offset_m = math.hypot(pose.position.x - self.spawn_xy_m[0], pose.position.y - self.spawn_xy_m[1])
			if offset_m <= self.spawn_tolerance_m:
				break
			print("reset: boat is %.2f m from spawn after reset_world (attempt %d/%d), retrying"
				  % (offset_m, attempt, self.reset_attempts))
		else:
			raise RuntimeError("Boat did not return to spawn after %d reset_world calls" % self.reset_attempts)

		# drop the cached topic message so step() can never reuse one from the previous episode
		self._latest_model_states = None

		# goal is read when building the observation, so pick it first
		if self.random_goals:
			self.goal = self.sample_goal()
		if self.random_goal_headings:
			self.goal_heading = self.np_random.uniform(-math.pi, math.pi)

		self.state = self._observation_from(pose, twist)
		self.prev_state = self.state

		self.count = 0
		self.hold_count = 0
		self.best_hold = 0
		if self.domain_randomization:
			self.sample_randomization()

		return self.observe(self.state)
