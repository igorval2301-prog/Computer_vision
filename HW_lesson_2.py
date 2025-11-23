import cv2
import numpy as np
image = cv2.imread('images/Ihor.jpg')
print(image.shape)
image = cv2.resize(image, (image.shape[1] // 4, image.shape[0] // 4))
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
print(image.shape)
image = cv2.Canny(image, 100, 100)

mytree = cv2.imwrite("images/white_ihor.jpg" , image)

# image = cv2.imread('images/email.jpg')
# print(image.shape)
# image = cv2.resize(image, (image.shape[1] // 4, image.shape[0] // 4))
# image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# image = cv2.Canny(image, 100 , 100)
# myemail = cv2.imwrite("images/white_email.jpg" , image)

cv2.imshow('image', image)
cv2.waitKey(0)