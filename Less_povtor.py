import cv2
import numpy as np

img = cv2.imread("images/fridgik.jpg")
img = cv2.resize(img, (img.shape[1] // 4, img.shape[0] // 4))
img_copy = img.copy()

img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img = cv2.GaussianBlur(img, (9, 9), 1)
img = cv2.equalizeHist(img)

img_edges = cv2.Canny(img, 50, 150)
contours, hierarchy = cv2.findContours(img_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

sticker_count = 0
for cnt in contours:
    area = cv2.contourArea(cnt)
    if area < 40:
        continue

    x, y, w, h = cv2.boundingRect(cnt)

    aspect_ratio = w / h if h != 0 else 0
    if aspect_ratio > 10 or aspect_ratio < 0.1:
        continue

    sticker_count += 1
    cv2.rectangle(img_copy, (x, y), (x + w, y + h), (0, 255, 0), 2)

print(f'Знайдено {sticker_count} магнітиків')

cv2.imwrite("images/result.jpg", img_copy)
cv2.imshow("result", img_copy)
cv2.waitKey(0)
cv2.destroyAllWindows()