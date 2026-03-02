import numpy as np
from head_pose import get_head_score
from gaze import get_gaze_score
from pose_estimation import get_pose_features

def extract_features(frame, bbox, phone_detected):

    x1, y1, x2, y2 = bbox
    crop = frame[y1:y2, x1:x2]

    head_score = get_head_score(crop)
    gaze_score = get_gaze_score(crop)
    pose_data = get_pose_features(frame, bbox)

    if pose_data:
        spine_align = 1 if pose_data["body_forward"] else 0
        writing = 1 if pose_data["writing"] else 0
    else:
        spine_align = 0
        writing = 0

    feature_vector = np.array([
        head_score,
        gaze_score,
        spine_align,
        writing,
        phone_detected,
        x1 / frame.shape[1],
        y1 / frame.shape[0],
        x2 / frame.shape[1],
        y2 / frame.shape[0],
    ])

    return feature_vector