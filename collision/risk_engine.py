import math
import numpy as np


def euclidean_distance(p1, p2):
    return float(np.linalg.norm(np.asarray(p1, dtype=np.float32) - np.asarray(p2, dtype=np.float32)))


def calculate_physical_ttc(distance_m, closing_speed_mps):
    """TTC in seconds for a lead object in front of ego.

    `closing_speed_mps` must be positive when ego is gaining on the
    object. A non-positive closing speed means there is no predicted
    collision from longitudinal closing motion.
    """
    if distance_m is None or closing_speed_mps is None:
        return float("inf")
    if not math.isfinite(distance_m) or not math.isfinite(closing_speed_mps):
        return float("inf")
    if distance_m <= 0.0:
        return 0.0
    if closing_speed_mps <= 0.1:
        return float("inf")
    return float(distance_m / closing_speed_mps)


def calculate_physical_risk(distance_m, closing_speed_mps):
    """Return (risk, distance_m, ttc_s) using physical CARLA units.

    The score is deliberately conservative but interpretable: it is the
    larger of a gap-risk term and a TTC-risk term. It is intended for the
    simulation controller, not as a certified automotive safety metric.
    """
    if distance_m is None or not math.isfinite(distance_m):
        return 0.0, float("inf"), float("inf")

    distance_m = max(0.0, float(distance_m))
    closing_speed_mps = max(0.0, float(closing_speed_mps or 0.0))
    ttc = calculate_physical_ttc(distance_m, closing_speed_mps)

    # Gap risk rises smoothly as the physical gap approaches zero.
    gap_risk = math.exp(-distance_m / 8.0)

    if math.isfinite(ttc):
        ttc_risk = math.exp(-ttc / 3.0)
    else:
        ttc_risk = 0.0

    risk = max(gap_risk, ttc_risk)
    return float(np.clip(risk, 0.0, 1.0)), distance_m, ttc


def calculate_ttc(object_future, ego_future):
    """Legacy image-space helper kept for compatibility with old tests.

    It is NOT used by the CARLA safety controller because image pixels do
    not carry physical time/velocity units.
    """
    if object_future is None or ego_future is None:
        return float("inf")
    if len(object_future) < 2 or len(ego_future) < 2:
        return float("inf")

    obj_speed = euclidean_distance(object_future[1], object_future[0])
    ego_speed = euclidean_distance(ego_future[1], ego_future[0])
    closing = ego_speed - obj_speed
    distance = euclidean_distance(object_future[0], ego_future[0])
    return calculate_physical_ttc(distance, closing)


def calculate_risk(object_future, ego_future, threshold=150):
    """Legacy tuple API used by the prediction/integration tests."""
    if object_future is None or ego_future is None:
        return 0.0, float("inf"), float("inf")
    if len(object_future) == 0 or len(ego_future) == 0:
        return 0.0, float("inf"), float("inf")

    min_len = min(len(object_future), len(ego_future))
    min_distance = min(
        euclidean_distance(object_future[i], ego_future[i])
        for i in range(min_len)
    )
    risk = float(np.clip(1.0 - min_distance / float(threshold), 0.0, 1.0))
    ttc = calculate_ttc(object_future, ego_future)
    return risk, min_distance, ttc
