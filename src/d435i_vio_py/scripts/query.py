#!/usr/bin/env python3
# Subscriber node to test and print debug stuff to the console.
# TO analyse the topics published by the realsense

import rospy
from sensor_msgs.msg import Image
import cv2
from cv_bridge import CvBridge


class CameraSubscriber:
    def __init__(self):
        self.bridge  = CvBridge()
        self.rgb_sub = rospy.Subscriber("/camera/color/image_raw", Image, self.rgb_callback, queue_size=1)

    

    def rgb_callback(self, msg):
        # ROS Image -> OpenCV BGR
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        cv2.imshow("D435i Color", frame)
        key = cv2.waitKey(1)
        if key == 27:  # ESC to quit
            rospy.signal_shutdown("User exit")


        rospy.loginfo("Length of the rgb array is ", frame.shape)


def main():
    rospy.init_node("d435i_info_sub")
    viewer = CameraSubscriber()
    rospy.loginfo("Data test started.")
    rospy.spin()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()


