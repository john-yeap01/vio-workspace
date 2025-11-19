#!/usr/bin/env python3
# remember to import the CvBridge after the cv2 library. There are some errors if it is not done in this way~!
# Summary : bridge needed to change ros image message into cv2 image, with encoding bgr8
# Ros frames come in each callback 
import rospy
from sensor_msgs.msg import Image
import cv2
from cv_bridge import CvBridge


class Preprocessor:
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
        raw = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        gray  = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5,5), 0)




        cv2.imshow("D435i Preprocessed", blur)
        key = cv2.waitKey(1)
        if key == 27:  # ESC to quit
            rospy.signal_shutdown("User exit")

def main():
    rospy.init_node("d435i_preprocessed_viewer")
    viewer = Preprocessor()
    rospy.loginfo("Preprocessed viewer started.")
    rospy.spin()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
