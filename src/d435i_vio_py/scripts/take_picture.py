#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import Image
import cv2
from cv_bridge import CvBridge
import os


class ColorViewer:
    def __init__(self):
        self.bridge = CvBridge()

        # path to images folder (one level above scripts/)
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.save_dir = os.path.join(os.path.dirname(script_dir), "images")

        # create folder if not exists
        os.makedirs(self.save_dir, exist_ok=True)

        self.sub = rospy.Subscriber(
            "/camera/color/image_raw",
            Image,
            self.callback,
            queue_size=1
        )

    def callback(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        cv2.imshow("D435i Color", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('p'):
            filename = f"d435i_{rospy.Time.now().to_nsec()}.jpg"
            filepath = os.path.join(self.save_dir, filename)
            cv2.imwrite(filepath, frame)
            rospy.loginfo(f"Saved image: {filepath}")

        if key == 27:  # ESC
            rospy.signal_shutdown("User exit")


def main():
    rospy.init_node("take_picture")
    ColorViewer()
    rospy.loginfo("Camera started. Press 'p' to save picture. ESC to quit.")

    rospy.spin()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
