#!/usr/bin/env python3
# ORB feature matching across 4 images (in order taken) + Essential matrix with RealSense intrinsics

import os
import glob
import cv2
import numpy as np

# ------------------------------------------------------------------
# 0) RealSense intrinsics for COLOR camera (copy-pasted from your dump)
#    Keyed by (width, height)
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
    # Add other resolutions if needed...
}

# 1) Find the images ---------------------------------------------------------
script_dir = os.path.dirname(os.path.abspath(__file__))
img_dir = os.path.join(script_dir, "..", "..", "d435i_vio_py", "images")

# grab all jpgs and sort by name (timestamps in filename → order taken)
image_paths = sorted(glob.glob(os.path.join(img_dir, "*.jpg")))

if len(image_paths) < 4:
    print("Need at least 4 images, found:", len(image_paths))
    print("Paths:", image_paths)
    raise SystemExit(1)

# only use the first four for now
image_paths = image_paths[:4]
print("Using images:")
for p in image_paths:
    print("  ", p)

# 2) Load images + convert to gray ------------------------------------------
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

# figure out resolution and camera matrix K
h, w = grays[0].shape[:2]
print(f"Image resolution: {w}x{h}")

if (w, h) not in COLOR_INTRINSICS:
    print("WARNING: No intrinsics defined for this resolution; using normalized coords in findEssentialMat.")
    K = None
else:
    intr = COLOR_INTRINSICS[(w, h)]
    fx, fy, cx, cy = intr["fx"], intr["fy"], intr["cx"], intr["cy"]
    K = np.array([[fx, 0.0, cx],
                  [0.0, fy, cy],
                  [0.0, 0.0, 1.0]], dtype=np.float64)
    print("Using camera matrix K:\n", K)

# 3) ORB detector + descriptors for each image ------------------------------
orb = cv2.ORB_create(nfeatures=1000)

keypoints = []
descriptors = []

for gray in grays:
    kp, des = orb.detectAndCompute(gray, None)
    keypoints.append(kp)
    descriptors.append(des)

# 4) BFMatcher and match consecutive pairs ----------------------------------
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

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

    # draw top 50 matches
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
    print(f"Showing matches for pair {i}-{i+1} (press any key to continue)")
    cv2.waitKey(0)
    cv2.destroyWindow(win_name)

    # ---- Essential matrix from matched points -----------------------------
    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])

    if K is not None:
        # Use calibrated version
        E, mask = cv2.findEssentialMat(
            pts1,
            pts2,
            cameraMatrix=K,
            method=cv2.RANSAC,
            prob=0.999,
            threshold=1.0,
        )
    else:
        # Fallback: uncalibrated, assume normalized coords
        E, mask = cv2.findEssentialMat(
            pts1,
            pts2,
            method=cv2.RANSAC,
            prob=0.999,
            threshold=1.0,
        )

    if E is None:
        print(f"Essential matrix could not be estimated for pair {i}-{i+1}")
        continue
    

    # the mask indicates which of th eoriginal points were used in ransac to find the essential matrix
    inliers = int(mask.ravel().sum()) if mask is not None else 0
    print(f"\n=== Pair {i} -> {i+1} ===")
    print("Essential matrix E:\n", E)
    print(f"Inliers: {inliers} / {len(matches)}")

    # ---- Recover relative pose (R, t) -------------------------------------
    if K is not None:
        # recoverPose expects same pixel coords + K
        _, R, t, mask_pose = cv2.recoverPose(E, pts1, pts2, K)
        print("Rotation R:\n", R)
        print("Translation t (up to scale):\n", t.ravel())

cv2.destroyAllWindows()
