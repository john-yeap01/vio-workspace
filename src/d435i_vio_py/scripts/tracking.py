#!/usr/bin/env python3
# Remember to import CvBridge after cv2.
# Summary: bridge needed to change ROS Image message into cv2 image (BGR8).
# ROS frames come in each callback.

import rospy
from sensor_msgs.msg import Image
import cv2
from cv_bridge import CvBridge
import numpy as np


class OpticalFlow:
    def __init__(self):
        self.bridge = CvBridge()
        self.sub = rospy.Subscriber(
            "/camera/color/image_raw",
            Image,
            self.callback,
            queue_size=1,
        )

        # Previous grayscale image and feature points
        self.prev_gray: np.ndarray | None = None
        self.prev_pts: np.ndarray | None = None

        # Lucas–Kanade optical flow parameters
        self.lk_params = dict(
            winSize=(15, 15),
            maxLevel=3,
            criteria=(
                cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                30,
                0.01,
            ),
        )

        # NEW: target feature counts for smooth "top-up"
        self.min_features = 200   # if fewer than this, add more
        self.max_features = 600   # hard cap so it doesn't explode

    def callback(self, msg: Image) -> None:
        # ROS Image -> OpenCV BGR
        raw: np.ndarray = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        gray: np.ndarray = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
        blur: np.ndarray = cv2.GaussianBlur(gray, (5, 5), 0.5)

        # ---------------------------------------------------------------------
        # INITIALISATION: no previous frame or no points yet
        # ---------------------------------------------------------------------
        if self.prev_gray is None or self.prev_pts is None:
            pts: np.ndarray | None = cv2.goodFeaturesToTrack(
                blur,
                maxCorners=self.max_features,  # NEW: tie to max_features
                qualityLevel=0.01,
                minDistance=7,
            )

            if pts is not None:
                # Draw detected points
                for p in pts:
                    x, y = p.ravel()
                    cv2.circle(raw, (int(x), int(y)), 3, (0, 255, 0), -1)

                # Store state for next frame
                self.prev_pts = pts
                self.prev_gray = gray
            else:
                # No points found; stay in init mode and try again next frame
                self.prev_pts = None
                self.prev_gray = None

            # Show frame (with or without points) and return
            cv2.imshow("D435i Optical Flow", raw)
            key = cv2.waitKey(1)
            if key == 27:
                rospy.signal_shutdown("User exit")
            return

        # ---------------------------------------------------------------------
        # TRACKING: we have prev_gray + prev_pts
        # ---------------------------------------------------------------------
        old_count = len(self.prev_pts)

        next_pts, status, err = cv2.calcOpticalFlowPyrLK(
            self.prev_gray,
            gray,
            self.prev_pts,
            None,
            **self.lk_params,
        )

        # If LK fails completely, reset and re-init next frame
        if next_pts is None or status is None:
            self.prev_gray = None
            self.prev_pts = None
            cv2.imshow("D435i Optical Flow", raw)
            key = cv2.waitKey(1)
            if key == 27:
                rospy.signal_shutdown("User exit")
            return

        status = status.reshape(-1)
        good_new = next_pts[status == 1]
        good_prev = self.prev_pts[status == 1]

        # If all tracks are lost, reset
        if len(good_new) == 0:
            self.prev_gray = None
            self.prev_pts = None
            cv2.imshow("D435i Optical Flow", raw)
            key = cv2.waitKey(1)
            if key == 27:
                rospy.signal_shutdown("User exit")
            return

        # NOTE: survival_ratio-based hard reset REMOVED
        # We will *top up* features instead of resetting when count drops.

        # ---------------------------------------------------------------------
        # DRAW TRACKS
        # ---------------------------------------------------------------------
        for new, prev in zip(good_new, good_prev):
            xnew, ynew = new.flatten()
            xprev, yprev = prev.flatten()

            # Draw previous point (small green circle)
            cv2.circle(raw, (int(xprev), int(yprev)), 2, (0, 255, 0), -1)

            # Draw motion vector as a red line from prev -> new
            cv2.line(
                raw,
                (int(xprev), int(yprev)),
                (int(xnew), int(ynew)),
                (0, 0, 255),
                1,
            )

            # Draw new position (slightly bigger circle if you like)
            cv2.circle(raw, (int(xnew), int(ynew)), 2, (0, 0, 255), -1)

        # ---------------------------------------------------------------------
        # UPDATE STATE FOR NEXT FRAME (smooth feature top-up)
        # ---------------------------------------------------------------------
        # Start with surviving tracks
        updated_pts = good_new.reshape(-1, 1, 2)
        num_existing = len(updated_pts)

        # If we have too few tracks, detect new ones and append
        if num_existing < self.min_features:
            # Mask to avoid detecting features on top of existing ones
            mask = np.full(gray.shape, 255, dtype=np.uint8)  # 255 = allowed

            for p in updated_pts:
                x, y = p.ravel()
                cv2.circle(mask, (int(x), int(y)), 7, 0, -1)  # 0 = blocked

            num_to_add = int(self.max_features - num_existing)
            if num_to_add > 0:
                new_pts = cv2.goodFeaturesToTrack(
                    blur,                  # use blurred image for stability
                    maxCorners=num_to_add,
                    qualityLevel=0.01,
                    minDistance=7,
                    mask=mask,
                )

                if new_pts is not None:
                    updated_pts = np.vstack((updated_pts, new_pts))

        # Final state for next frame
        self.prev_gray = gray
        self.prev_pts = updated_pts

        # SHOW THE FRAME
        cv2.imshow("D435i Optical Flow", raw)
        key = cv2.waitKey(1)
        if key == 27:  # ESC to quit
            rospy.signal_shutdown("User exit")


def main():
    rospy.init_node("optical_flow_calc")
    viewer = OpticalFlow()
    rospy.loginfo("LK optical track viewer started.")
    rospy.spin()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
