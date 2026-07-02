import sys
import os

sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

import numpy as np
from adas_system.trajectory_buffer import TrajectoryBuffer
from prediction.predictor import predict_future
from collision.risk_engine import calculate_risk
from collision.warning_system import generate_warning

buffer = TrajectoryBuffer(max_len=10)

# =====================================
# SIMULATED TRACKS
# =====================================

track_id = 5

trajectory = [
    [320, 200],
    [322, 201],
    [324, 202],
    [326, 203],
    [328, 204],
    [330, 205],
    [332, 206],
    [334, 207],
    [336, 208],
    [338, 209]
]

for point in trajectory:

    buffer.update(
        track_id,
        point[0],
        point[1]
    )

# =====================================
# GET TRAJECTORY
# =====================================

past_traj = buffer.get_trajectory(
    track_id
)

print("\nPast Trajectory:")
print(past_traj)

# =====================================
# TRANSFORMER PREDICTION
# =====================================

future_traj = predict_future(
    past_traj
)

print("\nPredicted Future:")
print(future_traj)

# =====================================
# SIMULATED EGO PATH
# =====================================

ego_future = np.array([
    [320, 205],
    [321, 206],
    [322, 207],
    [323, 208],
    [324, 209]
])

# =====================================
# COLLISION RISK
# =====================================

risk_score, min_distance, ttc = calculate_risk(
    future_traj,
    ego_future
)

# =====================================
# WARNING
# =====================================
print("DEBUG TTC =", ttc)
print("DEBUG TTC TYPE =", type(ttc))

generate_warning(
    object_id=track_id,
    risk_score=risk_score,
    min_distance=min_distance,
    ttc=ttc
)