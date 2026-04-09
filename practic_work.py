# нейронка камера спостереження відтворюється вебка якщо в кадрі людина то детекція руху світиться кружечок зеленим і починається запис і зберігається кудись відео якщо людини нема червоним світиться кружечок запис не починається
import os
import cv2
import time
from ultralytics import YOLO

PROJECT_DIR = os.path.dirname(__file__)
OUT_DIR = os.path.join(PROJECT_DIR, 'recs')

os.makedirs(OUT_DIR, exist_ok=True)

cap = cv2.VideoCapture(0)
modelka = YOLO("yolov8s.pt")

CONF_THRESHOLD = 0.5

out = None
recording = False

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = modelka(frame, conf=CONF_THRESHOLD, verbose=False)

    people_count = 0
    PERSON_CLASS_ID = 0

    for r in results:
        if r.boxes is None:
            continue

        for box in r.boxes:
            cls = int(box.cls[0])

            if cls == PERSON_CLASS_ID:
                people_count += 1

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)


    if people_count > 0:

        cv2.circle(frame, (30, 30), 10, (0,255,0), -1)

        if not recording:
            print("start")
            name = os.path.join(OUT_DIR, f"video_{int(time.time())}.mp4")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(name, fourcc, 20.0, (frame.shape[1], frame.shape[0]))
            recording = True

    else:

        cv2.circle(frame, (30, 30), 10, (0,0,255), -1)

        if recording:
            print("stop")
            out.release()
            recording = False

    if recording:
        out.write(frame)

    cv2.imshow("YOLO", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
if out:
    out.release()
cv2.destroyAllWindows()