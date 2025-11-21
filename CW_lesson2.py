import cv2
import numpy as np
image = cv2.imread('images/tree.jpg')
print(image.shape)
# image = cv2.resize(image, (800, 500))
image = cv2.resize(image, (image.shape[1] // 1, image.shape[0] // 1))#пропорційне зменшення ділення на однакове число
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
print(image.shape)
image = cv2.Canny(image, 100 , 100)


kernel = np.ones((5,5), np.uint8)
image = cv2.dilate(image, kernel, iterations=1) #розширює світлі області

image = cv2.erode(image, kernel, iterations=1)

mytree = cv2.imwrite("images/treemy.jpg" , image)

cv2.imshow("image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()