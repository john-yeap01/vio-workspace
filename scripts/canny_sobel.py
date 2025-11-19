import cv2

cam = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)


if not cam.isOpened():
    print("Camera failed to oqpen")
    exit(1)

cam.set(cv2.CAP_PROP_FRAME_WIDTH, 500)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 300)

while True:
    ret, frame = cam.read()
    if not ret:
        print("Failed to grab frame")
        break

    # TODO: process frame here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray,  (5,5), 1)
    canny = cv2.Canny(blurred, 50, 100)
    sobel = cv2.Sobel(blurred, 2, 10, 10 )

    # cv2.imshow("Webcam", canny)
    cv2.imshow("Webcam",sobel)



    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
