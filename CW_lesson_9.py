import cv2
import os
import numpy as np

input_folder = "image"
output_folder = "output"
formats = (".jpg", ".jpeg", ".png", ".webp", ".tiff")

os.makedirs(output_folder, exist_ok=True)


face_net = cv2.dnn.readNetFromCaffe("data/DNN/deploy.prototxt", "data/DNN/res10_300x300_ssd_iter_140000.caffemodel")
eye_cascade = cv2.CascadeClassifier('data/haarcascad/haarcascade_eye.xml')


files = sorted(os.listdir(input_folder))



for file in files:
    if not file.lower().endswith(formats):
        continue

    path = os.path.join(input_folder, file)
    img = cv2.imread(path)
    if img is None:
        continue

    img = cv2.imread('image/face.jpg')
    (h, w) = img.shape[:2]
    blob = cv2.dnn.blobFromImage(img, 1.0, (300, 300), (104.0, 177.0, 123.0))

    face_net.setInput(blob)
    detections = face_net.forward()

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x, y, x2, y2) = box.astype("int")

            cv2.rectangle(img, (x, y), (x2, y2), (0, 255, 0), 2)

            roi_color = img[y:y2, x:x2]
            roi_gray = cv2.cvtColor(roi_color, cv2.COLOR_BGR2GRAY)

            eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 10)

            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (255, 0, 0), 2)


    output_path = os.path.join(output_folder, file)
    cv2.imwrite(output_path, img)
    cv2.imshow("Processing", img)
    if cv2.waitKey(0) & 0xFF == ord('q'):
         break
cv2.destroyAllWindows()

