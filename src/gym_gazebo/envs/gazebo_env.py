import gym
import rospy
#import roslaunch
import sys
import os
import signal
import subprocess
import time
from std_srvs.srv import Empty
import random
from rosgraph_msgs.msg import Clock

class GazeboEnv(gym.Env):
    """Superclass for all Gazebo environments.
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, launchfile):
        self.last_clock_msg = Clock()

        random_number = random.randint(10000, 15000)
        # self.port = "11311"#str(random_number) #os.environ["ROS_PORT_SIM"]
        # self.port_gazebo = "11345"#str(random_number+1) #os.environ["ROS_PORT_SIM"]
        self.port = str(random_number) #os.environ["ROS_PORT_SIM"]
        self.port_gazebo = str(random_number+1) #os.environ["ROS_PORT_SIM"]

        os.environ["ROS_MASTER_URI"] = "http://localhost:"+self.port
        os.environ["GAZEBO_MASTER_URI"] = "http://localhost:"+self.port_gazebo
        #
        # self.ros_master_uri = os.environ["ROS_MASTER_URI"];

        print("ROS_MASTER_URI=http://localhost:"+self.port + "\n")
        print("GAZEBO_MASTER_URI=http://localhost:"+self.port_gazebo + "\n")

        # self.port = os.environ.get("ROS_PORT_SIM", "11311")
        ros_path = os.path.dirname(subprocess.check_output(["which", "roscore"]))

        # NOTE: It doesn't make sense to launch a roscore because it will be done when spawing Gazebo, which also need
        #   to be the first node in order to initialize the clock.
        # # start roscore with same python version as current script
        # self._roscore = subprocess.Popen([sys.executable, os.path.join(ros_path, b"roscore"), "-p", self.port])
        # time.sleep(1)
        # print ("Roscore launched!")


        if launchfile.startswith("/"):
            fullpath = launchfile
        else:
            fullpath = os.path.join(os.path.dirname(__file__), "assets", "launch", launchfile)
        if not os.path.exists(fullpath):
            raise IOError("File "+fullpath+" does not exist")

        self._roslaunch = subprocess.Popen([sys.executable, os.path.join(ros_path, b"roslaunch"), "-p", self.port, fullpath])
        print ("Gazebo launched!")

        self.gzclient_pid = 0

        # rospy decides ONCE, inside init_node, whether rospy.sleep()/get_time() follow simulated
        # time or the wall clock, by reading the /use_sim_time parameter. roslaunch sets that
        # parameter a moment after it starts, so calling init_node straight away sometimes read it
        # too early and silently used the wall clock: a 0.1 s control step then lasted 0.1 s of
        # real time, i.e. up to ~1 s of physics when the sim runs 10x faster than real time.
        # Wait until the parameter is there before initialising the node.
        deadline = time.time() + 60.0
        while True:
            try:
                if rospy.get_param('/use_sim_time', False):
                    break
            except Exception:
                pass # master not up yet
            if time.time() > deadline:
                raise RuntimeError("/use_sim_time was not set within 60 s of launching " + fullpath)
            time.sleep(0.2)

        # Launch the simulation with the given launchfile name
        rospy.init_node('gym', anonymous=True)
        if rospy.rostime.is_wallclock():
            raise RuntimeError("rospy is using the wall clock instead of simulated time; "
                               "control steps would not last control_period_s of physics")

        ################################################################################################################
        # r = rospy.Rate(1)
        # self.clock_sub = rospy.Subscriber('/clock', Clock, self.callback, queue_size=1000000)
        # while not rospy.is_shutdown():
        #     print("initialization: ", rospy.rostime.is_rostime_initialized())
        #     print("Wallclock: ", rospy.rostime.is_wallclock())
        #     print("Time: ", time.time())
        #     print("Rospyclock: ", rospy.rostime.get_rostime().secs)
        #     # print("/clock: ", str(self.last_clock_msg))
        #     last_ros_time_ = self.last_clock_msg
        #     print("Clock:", last_ros_time_)
        #     # print("Waiting for synch with ROS clock")
        #     # if wallclock == False:
        #     #     break
        #     r.sleep()
        ################################################################################################################

    # def callback(self, message):
    #     """
    #     Callback method for the subscriber of the clock topic
    #     :param message:
    #     :return:
    #     """
    #     # self.last_clock_msg = int(str(message.clock.secs) + str(message.clock.nsecs)) / 1e6
    #     # print("Message", message)
    #     self.last_clock_msg = message
    #     # print("Message", message)

    def step(self, action):

        # Implement this method in every subclass
        # Perform a step in gazebo. E.g. move the robot
        raise NotImplementedError

    def reset(self):

        # Implemented in subclass
        raise NotImplementedError

    def _render(self, mode="human", close=False):

        if close:
            tmp = os.popen("ps -Af").read()
            proccount = tmp.count('gzclient')
            if proccount > 0:
                if self.gzclient_pid != 0:
                    os.kill(self.gzclient_pid, signal.SIGTERM)
                    os.wait()
            return

        tmp = os.popen("ps -Af").read()
        proccount = tmp.count('gzclient')
        if proccount < 1:
            subprocess.Popen("gzclient")
            self.gzclient_pid = int(subprocess.check_output(["pidof","-s","gzclient"]))
        else:
            self.gzclient_pid = 0

    def _close(self):

        # Kill gzclient, gzserver and roscore
        tmp = os.popen("ps -Af").read()
        gzclient_count = tmp.count('gzclient')
        gzserver_count = tmp.count('gzserver')
        roscore_count = tmp.count('roscore')
        rosmaster_count = tmp.count('rosmaster')

        if gzclient_count > 0:
            os.system("killall -9 gzclient")
        if gzserver_count > 0:
            os.system("killall -9 gzserver")
        if rosmaster_count > 0:
            os.system("killall -9 rosmaster")
        if roscore_count > 0:
            os.system("killall -9 roscore")

        if (gzclient_count or gzserver_count or roscore_count or rosmaster_count >0):
            os.wait()

    def _configure(self):

        # TODO
        # From OpenAI API: Provides runtime configuration to the enviroment
        # Maybe set the Real Time Factor?
        pass
    def _seed(self):

        # TODO
        # From OpenAI API: Sets the seed for this env's random number generator(s)
        pass
