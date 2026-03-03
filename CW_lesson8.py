import cv2
import numpy as np
# face_cascade = cv2.CascadeClassifier('data/haarcascad/haarcascade_frontalface_default.xml')
# eye_cascade = cv2.CascadeClassifier('data/haarcascad/haarcascade_eye.xml')
# smile_cascade = cv2.CascadeClassifier('data/haarcascad/haarcascade_smile.xml')

face_net = cv2.dnn.readNetFromCaffe("data/DNN/deploy.prototxt", "res10_300x300_ssd_iter_140000.caffemodel")


cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    # --------------------------dnn-----------------------
    (h,w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),(104.0, 177.0, 123.0))

    face_net.setInput(blob)
    detections = face_net.forward()

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x,y,x2,y2) = box.astype("int")


            x, y = max(0, x), max(0, y)
            x2 , y2 = min(w - 1, x2), min(h - 1, y2)


            cv2.rectangle(frame, (x, y), (x2, y2), (0, 255, 0), 2)


    cv2.imshow("DNN", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
         break
cap.release()
cv2.destroyAllWindows()


    # -------------------------cascade--------------------
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #
    # faces = face_cascade.detectMultiScale(gray,  1.1, 5, minSize=(30, 30))
    #
    # for (x, y, w, h) in faces:
    #     cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    #
    #     roi_gray = gray[y:y + h, x:x + w]
    #     roi_color = frame[y:y + h, x:x + w]
    #
    #     eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=10, minSize=(10, 10) )
    #
    #     for(ex, ey, ew, eh) in eyes:
    #         cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (255, 0, 0), 2)
    #
    #     smile = smile_cascade.detectMultiScale(roi_gray, 1.1, 5, minSize=(30, 16))
    #
    #     for (sx, sy, sw, sh) in smile:
    #         cv2.rectangle(roi_color, (sx, sy), (sx + sw, sy + sh), (0, 0, 255), 2)
    #
    # cv2.putText(frame, f"Faces detected: {len(faces)}",
    #             (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 1), 2)

#     cv2.imshow('Haar Face Tracking', frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#
# cap.release()
# cv2.destroyAllWindows()

