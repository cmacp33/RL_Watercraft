import cv2
import gym
import math
import rospy
import roslaunch
import time
import numpy as np

from cv_bridge import CvBridge, CvBridgeError
from gym import utils, spaces
from gym_gazebo.envs import gazebo_env
from geometry_msgs.msg import Twist
from std_srvs.srv import Empty

from sensor_msgs.msg import Image
from time import sleep

from gym.utils import seeding


class Gazebo_Linefollow_Env(gazebo_env.GazeboEnv):

    def __init__(self):
        # Launch the simulation with the given launchfile name
        LAUNCH_FILE = '/home/fizzer/enph353_gym-gazebo-noetic/gym_gazebo/envs/ros_ws/src/linefollow_ros/launch/linefollow_world.launch'
        gazebo_env.GazeboEnv.__init__(self, LAUNCH_FILE)
        self.vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        self.unpause = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
        self.pause = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
        self.reset_proxy = rospy.ServiceProxy('/gazebo/reset_world',
                                              Empty)

        self.action_space = spaces.Discrete(3)  # F,L,R
        self.reward_range = (-np.inf, np.inf)
        self.episode_history = []

        self._seed()

        self.bridge = CvBridge()
        self.timeout = 0  # Used to keep track of images with no line detected

    def drawCircle(self, img, center_pt):
        radius = 20
        (x, y) = center_pt
        color = (255, 0, 255)
        line_thickness = -1
        return cv2.circle(img, center_pt, radius, color, line_thickness)
    
    def find_path(self, cv_img):
        lower_blue = np.array([150, 115, 60])
        upper_blue = np.array([170, 130, 80])
        new_cv_img = cv2.inRange(cv_img, lower_blue, upper_blue)
        return new_cv_img

    def get_line_center(self, cv_img):

      new_cv_img = self.find_path(cv_img)
      contours, hierarchy = cv2.findContours(new_cv_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
      
      if len(contours) == 0:
          return -69
      
      contour = max(contours, key=cv2.contourArea)
      height, width = cv_img.shape[:2]
      x_contours = []
      target_y = height - 50
      tolerance = 25
      
      for point in contour:
          x, y = point[0]
          if abs(y - target_y) <= tolerance:
              x_contours.append(x)
      
      if len(x_contours) == 0:
          return -69
      
      left = np.min(x_contours)
      right = np.max(x_contours)
      middle = (left + right) // 2
      
      return middle


    def process_image(self, data):
        '''
            @brief Coverts data into a opencv image and displays it
            @param data : Image data from ROS

            @retval (state, done)
        '''
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

        NUM_BINS = 10
        state_list = [[1 if i == j else 0 for i in range(10)] for j in range(10)]
        state = [0] * 10
        done = False
        height, width = cv_image.shape[:2]
        for i in range (1, NUM_BINS):
            cv_image = cv2.line(cv_image, (math.ceil( i * width / 10), 0), (math.ceil( i * width / 10), height), (0, 0, 0), 1)
        cv_image = cv2.line(cv_image, (0, height - 50), (width, height - 50), (0, 0, 0), 1)

        middle = self.get_line_center(cv_image)
        if middle == -69:
            state = [0] * 10
            self.timeout += 1
            if self.timeout > 30:
                done = True
        else:
            cv_image = self.drawCircle(cv_image, (middle, height - 50))
            state = state_list[min(math.ceil(middle * NUM_BINS / width), len(state_list) - 1)]
            self.timeout = 0
        
        state_str = " ".join(map(str, state))
        cv_image = cv2.putText(cv_image, state_str, (0, height - 220), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2)
        cv_image = cv2.putText(cv_image, str(middle), (width - 50, height - 220), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2)

        cv2.imshow("Camera", cv_image)
        cv2.waitKey(1)

        return state, done

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        rospy.wait_for_service('/gazebo/unpause_physics')
        try:
            self.unpause()
        except (rospy.ServiceException) as e:
            print ("/gazebo/unpause_physics service call failed")

        self.episode_history.append(action)

        vel_cmd = Twist()

        if action == 0:  # FORWARD
            vel_cmd.linear.x = 0.4
            vel_cmd.angular.z = 0.0
        elif action == 1:  # LEFT
            vel_cmd.linear.x = 0.0
            vel_cmd.angular.z = 0.5
        elif action == 2:  # RIGHT
            vel_cmd.linear.x = 0.0
            vel_cmd.angular.z = -0.5

        self.vel_pub.publish(vel_cmd)

        data = None
        while data is None:
            try:
                data = rospy.wait_for_message('/pi_camera/image_raw', Image,
                                              timeout=5)
            except:
                pass

        rospy.wait_for_service('/gazebo/pause_physics')
        try:
            # resp_pause = pause.call()
            self.pause()
        except (rospy.ServiceException) as e:
            print ("/gazebo/pause_physics service call failed")

        state, done = self.process_image(data)

        # Set the rewards for your action
        if not done:
            if action == 0:  # FORWARD
                reward = 4
            elif action == 1:  # LEFT
                reward = 4
            else:
                reward = 2  # RIGHT
        else:
            reward = -200

        if not done:
            if state == [0,0,0,0,1,0,0,0,0,0] or state == [0,0,0,0,0,1,0,0,0,0]:
                reward += 12
            elif state == [0,0,0,1,0,0,0,0,0,0] or state == [0,0,0,0,0,0,1,0,0,0]:
                reward += 7
            elif state == [0,0,1,0,0,0,0,0,0,0] or state == [0,0,0,0,0,0,0,1,0,0]:
                reward += 5
            elif state == [0,1,0,0,0,0,0,0,0,0] or state == [0,0,0,0,0,0,0,0,1,0]:
                reward += 3
            elif state == [1,0,0,0,0,0,0,0,0,0] or state == [0,0,0,0,0,0,0,0,0,1]:
                reward += 1

        return state, reward, done, {}

    def reset(self):

        print("Episode history: {}".format(self.episode_history))
        self.episode_history = []
        print("Resetting simulation...")
        # Resets the state of the environment and returns an initial
        # observation.
        rospy.wait_for_service('/gazebo/reset_simulation')
        try:
            # reset_proxy.call()
            self.reset_proxy()
        except (rospy.ServiceException) as e:
            print ("/gazebo/reset_simulation service call failed")

        # Unpause simulation to make observation
        rospy.wait_for_service('/gazebo/unpause_physics')
        try:
            # resp_pause = pause.call()
            self.unpause()
        except (rospy.ServiceException) as e:
            print ("/gazebo/unpause_physics service call failed")

        # read image data
        data = None
        while data is None:
            try:
                data = rospy.wait_for_message('/pi_camera/image_raw',
                                              Image, timeout=5)
            except:
                pass

        rospy.wait_for_service('/gazebo/pause_physics')
        try:
            # resp_pause = pause.call()
            self.pause()
        except (rospy.ServiceException) as e:
            print ("/gazebo/pause_physics service call failed")

        self.timeout = 0
        state, done = self.process_image(data)

        return state


