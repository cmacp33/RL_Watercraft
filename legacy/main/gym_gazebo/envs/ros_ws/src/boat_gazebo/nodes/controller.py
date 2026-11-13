#!/usr/bin/env python3
import rospy
from std_msgs.msg import Float32

class Controller:
    def __init__(self):

        rospy.init_node('controller', anonymous=True)

        self.left_pub = rospy.Publisher('/boat/thrusters/left_thrust_cmd', Float32, queue_size=10)
        self.right_pub = rospy.Publisher('/boat/thrusters/right_thrust_cmd', Float32, queue_size=10)

        self.left_msg = Float32()
        self.right_msg = Float32()

        self.forward_thrust = 0.5
        self.turn_thrust = 0.01

        self.rate = rospy.Rate(1)

    def run(self):
        while not rospy.is_shutdown():

            self.left_thrust = self.forward_thrust - self.turn_thrust
            self.right_thrust = self.forward_thrust + self.turn_thrust

            self.left_msg.data = self.left_thrust
            self.right_msg.data = self.right_thrust

            self.left_pub.publish(self.left_msg)
            self.right_pub.publish(self.right_msg)

            rospy.loginfo("left thrust: %.3f, right thrust: %.3f", self.left_thrust, self.right_thrust)

            self.turn_thrust += 0

            self.rate.sleep()


if __name__ == "__main__":
    controller = Controller()
    controller.run()


