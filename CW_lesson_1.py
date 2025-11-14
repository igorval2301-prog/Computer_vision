import cv2

# img = cv2.imread("images/1.jpg")
# img = cv2.resize(img, (500, 300))
# cv2.imshow("image", img)

cap = cv2.VideoCapture("video/video.mp4")
# cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (1000, 600))

    cv2.imshow("video", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
# cv2.waitKey(0)
cv2.destroyAllWindows()