import cv2
import numpy as np

img = cv2.imread("image/swat.jpg")

scale = 5
img = cv2.resize(img, (img.shape[1] // scale, img.shape[0] // scale))
original = img.copy()

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

lower_yellow = np.array([20, 100, 100])
upper_yellow = np.array([35, 255, 255])

lower_purple = np.array([90, 150, 50])
upper_purple = np.array([130, 255, 255])

lower_orange = np.array([0, 160, 120])
upper_orange = np.array([18, 255, 255])


mask_y = cv2.inRange(hsv, lower_yellow, upper_yellow)
mask_p = cv2.inRange(hsv, lower_purple, upper_purple)
mask_o = cv2.inRange(hsv, lower_orange, upper_orange)

mask = mask_y | mask_p | mask_o


contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

count = 1
for cnt in contours:
    area = cv2.contourArea(cnt)

    if area > 2500:
        x, y, w, h = cv2.boundingRect(cnt)


        cv2.rectangle(original, (x, y), (x + w, y + h), (0, 255, 0), 2)


        cv2.putText(original, f"Candy {count}", (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        count += 1

cv2.imshow("Masks", mask)
cv2.imshow("Final Result", original)

cv2.imwrite("image/Final Result.jpg", mask)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.waitKey(0)
cv2.destroyAllWindows()