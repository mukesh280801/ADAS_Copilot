import numpy as np

from prediction.predictor import predict_future


def predict_if_ready(trajectory):

    if len(trajectory) < 10:
        return None

    trajectory = np.array(
        trajectory,
        dtype=np.float32
    )

    future = predict_future(
        trajectory
    )

    return future