import cv2
import numpy as np

img = cv2.imread("images/ihor.jpg")
img = cv2.resize(img, (img.shape[1] // 4, img.shape[0] // 4))
cv2.line(img, (420,100), (280,100), (0,255,0), 3)

cv2.line(img, (420, 100), (420, 290), (0, 255, 0), 3)

cv2.line(img, (280, 100), (280, 290), (0, 255, 0), 3)

cv2.line(img, (280,290), (420,290), (0,255,0), 3)

cv2. putText(img, "Ihor Kondratiuk", (230,320), cv2.QT_FONT_NORMAL, 1, (0,0,255))

mytree = cv2.imwrite("images/face_id.jpg" , img)










cv2.imshow("primitivi", img)
cv2.waitKey(0)
cv2.destroyAllWindows()