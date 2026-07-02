import numpy as np


def euclidean_distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))


def calculate_ttc(object_future, ego_future):
    if len(object_future) < 2 or len(ego_future) < 2:
        return float("inf")

    obj_speed = euclidean_distance(object_future[1], object_future[0])
    ego_speed = euclidean_distance(ego_future[1], ego_future[0])

    relative_speed = obj_speed - ego_speed

    if relative_speed <= 0:
        return float("inf")

    distance = euclidean_distance(object_future[0], ego_future[0])

    return distance / relative_speed


def calculate_risk(object_future, ego_future, threshold=150):

    if object_future is None or ego_future is None:
        return 0.0, float("inf"), float("inf")

    if len(object_future) == 0 or len(ego_future) == 0:
        return 0.0, float("inf"), float("inf")

    # Ensure same length comparison
    min_len = min(len(object_future), len(ego_future))

    min_distance = float("inf")

    for i in range(min_len):
        dist = euclidean_distance(object_future[i], ego_future[i])
        if dist < min_distance:
            min_distance = dist

    # Normalize risk score safely
    risk_score = 0.0
    if min_distance != float("inf"):
        risk_score = max(0.0, 1.0 - (min_distance / threshold))

    ttc = calculate_ttc(object_future, ego_future)

    return risk_score, min_distance, ttc