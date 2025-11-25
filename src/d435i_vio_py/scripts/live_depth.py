#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import Float32
import cv2
from cv_bridge import CvBridge
import numpy as np


class CenterDistanceViewer:
    def __init__(self):
        self.bridge = CvBridge()

        self.color_topic = "/camera/color/image_raw"
        # Use depth aligned to color
        self.depth_topic = "/camera/aligned_depth_to_color/image_raw"

        rospy.loginfo(f"Subscribing to color: {self.color_topic}")
        rospy.loginfo(f"Subscribing to depth: {self.depth_topic}")

        self.color_image = None
        self.depth_image = None

        self.dist_pub = rospy.Publisher("/center_distance", Float32, queue_size=1)

        self.color_sub = rospy.Subscriber(
            self.color_topic,
            Image,
            self.color_callback,
            queue_size=1
        )

        self.depth_sub = rospy.Subscriber(
            self.depth_topic,
            Image,
            self.depth_callback,
            queue_size=1
        )

        self.printed_once = False

    def depth_callback(self, msg: Image) -> None:
        self.depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")

        if not self.printed_once:
            rospy.loginfo(f"[DEBUG] Depth encoding: {msg.encoding}")
            rospy.loginfo(f"[DEBUG] Depth shape: {self.depth_image.shape}, dtype: {self.depth_image.dtype}")
            self.printed_once = True

    def color_callback(self, msg: Image) -> None:
        self.color_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

        if self.depth_image is None:
            cv2.imshow("D435i Center Distance", self.color_image)
            key = cv2.waitKey(1)
            if key == 27:
                rospy.signal_shutdown("User exit")
            return

        color = self.color_image.copy()
        h, w, _ = color.shape

        dh, dw = self.depth_image.shape[:2]

        # Debug once: show size mismatch if any
        if not hasattr(self, "_size_debugged"):
            rospy.loginfo(f"[DEBUG] Color shape: {self.color_image.shape}")
            rospy.loginfo(f"[DEBUG] Depth shape: {self.depth_image.shape}")
            self._size_debugged = True

        cx = w // 2
        cy = h // 2

        x = int(np.clip(cx, 0, dw - 1))
        y = int(np.clip(cy, 0, dh - 1))

        # Use a small 5x5 window around the center for robustness
        win = 2
        y0 = max(0, y - win)
        y1 = min(dh, y + win + 1)
        x0 = max(0, x - win)
        x1 = min(dw, x + win + 1)

        depth_roi = self.depth_image[y0:y1, x0:x1]

        depth_value = None
        distance_m = np.nan

        if self.depth_image.dtype == np.uint16:
            valid = depth_roi[depth_roi > 0]
            if valid.size > 0:
                depth_value = np.median(valid)
                distance_m = float(depth_value) / 1000.0
        else:
            # Assume meters already
            valid = depth_roi[~np.isnan(depth_roi)]
            if valid.size > 0:
                depth_value = np.median(valid)
                distance_m = float(depth_value)

        if depth_value is not None:
            rospy.logdebug(f"[DEBUG] Center depth raw: {depth_value}, distance_m: {distance_m:.3f}")
        else:
            rospy.logdebug("[DEBUG] No valid depth in center ROI")

        cv2.circle(color, (cx, cy), 5, (0, 0, 255), -1)

        if np.isnan(distance_m) or distance_m <= 0.0:
            text = "--- m"
        else:
            text = f"{distance_m:.3f} m"

        cv2.putText(
            color, text, (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2
        )

        cv2.imshow("D435i Center Distance", color)
        key = cv2.waitKey(1)
        if key == 27:
            rospy.signal_shutdown("User exit")

        if not np.isnan(distance_m) and distance_m > 0.0:
            self.dist_pub.publish(Float32(data=distance_m))


def main():
    rospy.init_node("d435i_center_distance_viewer")
    viewer = CenterDistanceViewer()
    rospy.loginfo("Center distance viewer started.")
    rospy.spin()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
