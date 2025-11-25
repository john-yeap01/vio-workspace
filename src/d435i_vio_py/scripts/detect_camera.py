#!/usr/bin/env python3
# remember to import the CvBridge after the cv2 library. There are some errors if it is not done in this way~!
# Summary: bridge needed to change ROS Image message into cv2 image.


import cv2

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    cv2.imshow("USB Camera", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
