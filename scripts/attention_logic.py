def calculate_attention(yolo_score, head_score, gaze_score, pose_data):

    score = 0

    # YOLO confidence weight
    score += yolo_score * 0.4

    # Head orientation
    score += head_score * 0.2

    # Gaze
    score += gaze_score * 0.2

    # Writing posture (important!)
    if pose_data and pose_data["writing"]:
        score += 0.2

    return max(0.0, min(1.0, score))