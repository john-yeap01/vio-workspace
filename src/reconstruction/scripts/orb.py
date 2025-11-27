#!/usr/bin/env python3
# ORB feature matching across 4 images + Essential + 3D camera trajectory
# Basis of VIO and SFM

import os
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # <-- add this line

# ------------------------------------------------------------------
# 0) RealSense intrinsics for COLOR camera (adjust resolution if needed)
# ------------------------------------------------------------------
COLOR_INTRINSICS = {
    # Color 848x480
    (848, 480): dict(
        fx=603.513916015625,
        fy=603.539123535156,
        cx=438.063507080078,
        cy=247.044128417969,
    ),
    # Color 640x480
    (640, 480): dict(
        fx=603.513916015625,
        fy=603.539123535156,
        cx=334.063537597656,
        cy=247.044128417969,
    ),
    # Color 1280x720
    (1280, 720): dict(
        fx=905.270935058594,
        fy=905.308715820312,
        cx=661.095275878906,
        cy=370.566192626953,
    ),
    # Color 1920x1080
    (1920, 1080): dict(
        fx=1357.90637207031,
        fy=1357.96301269531,
        cx=991.642944335938,
        cy=555.849304199219,
    ),
}

# ------------------------------------------------------------------
def visualize_camera_trajectory(cam_centers):
    """Plot camera centers in 3D with indices."""
    cam_centers = np.array(cam_centers)  # N x 3

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    xs = cam_centers[:, 0]
    ys = cam_centers[:, 1]
    zs = cam_centers[:, 2]

    ax.scatter(xs, ys, zs, marker="o")

    # connect in order
    ax.plot(xs, ys, zs, linestyle="-")

    # label each camera
    for i, (x, y, z) in enumerate(cam_centers):
        ax.text(x, y, z, str(i), fontsize=10)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Recovered camera trajectory (up to scale)")

    # make axes equal-ish
    max_range = np.ptp(cam_centers, axis=0).max()
    mid = cam_centers.mean(axis=0)
    for axis, m in zip([ax.set_xlim, ax.set_ylim, ax.set_zlim], mid):
        axis(m - max_range / 2, m + max_range / 2)

    plt.show()


# 1) Find the images ---------------------------------------------------------
script_dir = os.path.dirname(os.path.abspath(__file__))
img_dir = os.path.join(script_dir, "..", "..", "d435i_vio_py", "images")

image_paths = sorted(glob.glob(os.path.join(img_dir, "*.jpg")))

if len(image_paths) < 4:
    print("Need at least 4 images, found:", len(image_paths))
    print("Paths:", image_paths)
    raise SystemExit(1)

image_paths = image_paths[:4]
print("Using images:")
for p in image_paths:
    print("  ", p)

# 2) Load + gray -------------------------------------------------------------
imgs = []
grays = []

for path in image_paths:
    img = cv2.imread(path)
    if img is None:
        print("Failed to load:", path)
        raise SystemExit(1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgs.append(img)
    grays.append(gray)

h, w = grays[0].shape[:2]
print(f"Image resolution: {w}x{h}")

if (w, h) not in COLOR_INTRINSICS:
    print("WARNING: No intrinsics for this resolution; using uncalibrated E.")
    K = None
else:
    intr = COLOR_INTRINSICS[(w, h)]
    fx, fy, cx, cy = intr["fx"], intr["fy"], intr["cx"], intr["cy"]
    K = np.array(
        [[fx, 0.0, cx],
         [0.0, fy, cy],
         [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    print("Using camera matrix K:\n", K)

# 3) ORB features ------------------------------------------------------------
orb = cv2.ORB_create(nfeatures=1000)

keypoints = []
descriptors = []

for gray in grays:
    kp, des = orb.detectAndCompute(gray, None)
    keypoints.append(kp)
    descriptors.append(des)

# 4) BFMatcher + pose chaining ----------------------------------------------
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# world pose of camera 0: R = I, t = 0
R_w = np.eye(3, dtype=np.float64)
t_w = np.zeros((3, 1), dtype=np.float64)

# store camera centers (in world coords). camera 0 at origin, shape (3,1)
cam_centers = [np.array([0.0, 0.0, 0.0])]

for i in range(len(imgs) - 1):
    des1 = descriptors[i]
    des2 = descriptors[i + 1]
    kp1 = keypoints[i]
    kp2 = keypoints[i + 1]

    if des1 is None or des2 is None:
        print(f"No descriptors for pair {i}-{i+1}")
        continue

    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    if len(matches) < 8:
        print(f"Not enough matches for pair {i}-{i+1}: {len(matches)}")
        continue

    # visualize matches (optional)
    draw_matches = matches[:50]
    out = cv2.drawMatches(
        imgs[i], kp1,
        imgs[i + 1], kp2,
        draw_matches,
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )

    win_name = f"ORB matches: {i} -> {i+1}"
    cv2.imshow(win_name, out)
    print(f"Showing matches for pair {i}-{i+1} (press any key)")
    cv2.waitKey(0)
    cv2.destroyWindow(win_name)

    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])

    # Essential matrix
    if K is not None:
        E, mask = cv2.findEssentialMat(
            pts1, pts2,
            cameraMatrix=K,
            method=cv2.RANSAC,
            prob=0.999,
            threshold=1.0,
        )
    else:
        E, mask = cv2.findEssentialMat(
            pts1, pts2,
            method=cv2.RANSAC,
            prob=0.999,
            threshold=1.0,
        )

    if E is None:
        print(f"E could not be estimated for pair {i}-{i+1}")
        continue

    # Recover relative pose between cam_i and cam_{i+1}
    if K is not None:
        _, R_rel, t_rel, mask_pose = cv2.recoverPose(E, pts1, pts2, K)
    else:
        _, R_rel, t_rel, mask_pose = cv2.recoverPose(E, pts1, pts2)

    # Chain poses:
    # New world rotation & translation: [R_new | t_new] = [R_rel | t_rel] * [R_w | t_w]
    R_w = R_rel @ R_w
    t_w = R_rel @ t_w + t_rel

    # Camera center C = -R^T * t
    C = -R_w.T @ t_w
    cam_centers.append(C.ravel())

    print(f"\n=== Pair {i} -> {i+1} ===")
    print("R_rel:\n", R_rel)
    print("t_rel (up to scale):\n", t_rel.ravel())
    print("Camera center (world):\n", C.ravel())

cv2.destroyAllWindows()




if len(cam_centers) <=  1:
    print("Not enough poses to visualize.")
else:
    cam_centers = np.array(cam_centers)  # N x 3

    # Example: make Z_up (negate y), keep x, y as you like
    # OpenCV:  x right, y down, z forward
    # Target (for plotting): X right, Y forward, Z up
    R_vis = np.array([
        [1,  0,  0],   # X' =  X
        [0,  0,  1],   # Y' =  Z
        [0, -1,  0],   # Z' = -Y (up)
    ], dtype=np.float64)

    cam_centers_vis = (R_vis @ cam_centers.T).T  # N x 3

    visualize_camera_trajectory(cam_centers_vis)

