import cv2
import numpy as np

cam = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)

if __name__ == "__main__":
    print(cam.isOpened())
    

    ret,frame = cam.read()
    print(ret, frame.shape)


    cv2.imshow("Cam", frame)
    cv2.waitKey(0)

    cam.release()
    cv2.destroyAllWindows()