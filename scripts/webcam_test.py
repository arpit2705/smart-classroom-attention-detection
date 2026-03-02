from ultralytics import YOLO
import cv2
from head_pose import get_head_score
from gaze import get_gaze_score
from pose_estimation import get_pose_features
from attention_logic import calculate_attention

person_model = YOLO("yolov8n.pt")
attention_model = YOLO("../model/best.pt")

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    person_results = person_model(frame)

    for result in person_results:
        for box in result.boxes:

            if int(box.cls[0]) != 0:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            person_crop = frame[y1:y2, x1:x2]

            if person_crop.size == 0:
                continue

            att_results = attention_model(person_crop)

            if len(att_results[0].boxes) > 0:
                yolo_score = float(att_results[0].boxes[0].conf[0])
            else:
                yolo_score = 0.3

            head_score = get_head_score(person_crop)
            gaze_score = get_gaze_score(person_crop)
            pose_data = get_pose_features(frame, (x1, y1, x2, y2))

            final_score = calculate_attention(
                yolo_score,
                head_score,
                gaze_score,
                pose_data
            )

            label = "Attentive" if final_score > 0.6 else "Distracted"
            color = (0,255,0) if label == "Attentive" else (0,0,255)

            cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
            cv2.putText(frame,
                        f"{label} {final_score:.2f}",
                        (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, color, 2)

    cv2.imshow("Smart Classroom - Webcam", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()