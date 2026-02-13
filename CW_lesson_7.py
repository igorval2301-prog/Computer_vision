import cv2
import numpy as np #matrix

cap = cv2.VideoCapture(0)

lower_red1 = np.array([0, 100, 100])
upper_red1 = np.array([10, 255, 255])

lower_red2 = np.array([170, 100, 100])
upper_red2 = np.array([180, 255, 255])

points = []# зберігаємо центри обєктів з кадру


while True:
    ret, frame = cap.read() # рет зчитання кадру тру фолс, фрейм - кадр
    if not ret:
        break


    frame = cv2.flip(frame, 1)
    hsw = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    mask1 = cv2.inRange(hsw, lower_red1, upper_red1)#для створення маски
    mask2 = cv2.inRange(hsw, lower_red2, upper_red2)


    mask = cv2.bitwise_or(mask1, mask2)#  поєднання масок


    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # спрощуємо контури

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 1000:
            cv2.drawContours(frame, [cnt], -1, (0, 255, 0), 2)
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])   # центр нашого елемента
                cy = int(M["m01"] / M["m00"])

                cv2.circle(frame, (cx,cy), 5, (255, 0, 0), -1)

                points.append((cx, cy))

                for i in range(1, len(points)):
                    if points[i - 1] is None or points[i] is None:
                        continue

                    cv2.line(frame, points[i - 1], points[i], (255, 0, 0), 2)



    cv2.imshow('video', frame)
    cv2.imshow('mask', mask)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()