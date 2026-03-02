import cv2
import torch
from ultralytics import YOLO

from attention_lstm import AttentionLSTM
from feature_extractor import extract_features
from sequence_buffer import SequenceBuffer
from tracker import SimpleTracker


# -----------------------------
# Load Models
# -----------------------------

# YOLO person detector (COCO pretrained)
person_model = YOLO("yolov8n.pt")

# Load trained LSTM model
lstm_model = AttentionLSTM()
lstm_model.load_state_dict(torch.load("attention_lstm.pt", map_location="cpu"))
lstm_model.eval()


# -----------------------------
# Initialize Tracker + Buffer
# -----------------------------

tracker = SimpleTracker()
buffer = SequenceBuffer(maxlen=50)


# -----------------------------
# Open Video
# -----------------------------

cap = cv2.VideoCapture("../sample.mp4")

if not cap.isOpened():
    print("Error: Cannot open video.")
    exit()


# -----------------------------
# Main Loop
# -----------------------------

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = person_model(frame)

    boxes_list = []

    # Collect only PERSON boxes (COCO class 0)
    for result in results:
        for box in result.boxes:

            cls = int(box.cls[0])

            if cls != 0:  # 0 = person
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            boxes_list.append((x1, y1, x2, y2))

    # Update tracker with detected boxes
    tracked_objects = tracker.update(boxes_list)

    # For each tracked student
    for student_id, center in tracked_objects.items():

        for (x1, y1, x2, y2) in boxes_list:

            box_center = ((x1 + x2) // 2, (y1 + y2) // 2)

            # Match box to tracked center
            if abs(box_center[0] - center[0]) < 15 and abs(box_center[1] - center[1]) < 15:

                # Phone detection flag (for now 0)
                phone_detected = 0

                # Extract behavioral features
                features = extract_features(
                    frame,
                    (x1, y1, x2, y2),
                    phone_detected
                )

                # Add to temporal buffer
                buffer.add(student_id, features)

                sequence = buffer.get_sequence(student_id)

                if sequence is not None:

                    seq_tensor = torch.tensor([sequence], dtype=torch.float32)
                    with torch.no_grad():
                        prediction = lstm_model(seq_tensor)

                    attention_score = float(prediction.item())

                else:
                    # Not enough frames yet
                    attention_score = 0.5

                label = "Attentive" if attention_score > 0.6 else "Distracted"
                color = (0, 255, 0) if label == "Attentive" else (0, 0, 255)

                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                cv2.putText(
                    frame,
                    f"ID:{student_id} {label} {attention_score:.2f}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2
                )

                break  # move to next student

    cv2.imshow("Smart Classroom - Video", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break


cap.release()
cv2.destroyAllWindows()