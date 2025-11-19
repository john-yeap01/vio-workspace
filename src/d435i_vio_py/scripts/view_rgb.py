#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import Image
import cv2
from cv_bridge import CvBridge


class ColorViewer:
    def __init__(self):
        self.bridge = CvBridge()
        self.sub = rospy.Subscriber(
            "/camera/color/image_raw",
            Image,
            self.callback,
            queue_size=1
        )

    def callback(self, msg):
        # ROS Image -> OpenCV BGR
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        cv2.imshow("D435i Color", frame)
        key = cv2.waitKey(1)
        if key == 27:  # ESC to quit
            rospy.signal_shutdown("User exit")

def main():
    rospy.init_node("d435i_color_viewer")
    viewer = ColorViewer()
    rospy.loginfo("Color viewer started.")
    rospy.spin()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
