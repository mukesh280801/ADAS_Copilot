import os
import threading
import torch
import numpy as np

from prediction.transformer_model import (
    TrajectoryTransformer
)

_MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
_WEIGHTS_PATH = os.path.join(_MODEL_DIR, "trajectory_transformer.pth")

# Confirmed against prediction/dataset_loader.py (training divides by
# /416.0) and tracking/carla_tracker_logger.py (the data-collection
# camera was spawned at 416x416) -- 416.0 is the correct constant.
# object_detection/carla_yolo.py's live camera has been corrected to
# also capture at 416x416 so live inference matches this normalization.
FRAME_SIZE = 416.0

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

_model = None
_model_lock = threading.Lock()


def _get_model():
    global _model

    if _model is not None:
        return _model

    with _model_lock:
        if _model is None:
            m = TrajectoryTransformer().to(device)
            try:
                state_dict = torch.load(
                    _WEIGHTS_PATH,
                    map_location=device
                )
                m.load_state_dict(state_dict)
            except FileNotFoundError:
                # This used to load at import time from a path
                # relative to the CURRENT WORKING DIRECTORY
                # ("prediction/trajectory_transformer.pth"), so
                # running the entry script from anywhere except the
                # exact project root crashed on import, before the
                # simulation even started. Now it's an absolute path
                # next to this file, loaded lazily on first use, and
                # a missing/renamed weights file degrades gracefully
                # (trajectory prediction just reports unavailable)
                # instead of taking the whole pipeline down with it.
                print(f"[predictor] weights not found at {_WEIGHTS_PATH}")
                return None
            except Exception as e:
                print(f"[predictor] failed to load weights: {e}")
                return None

            m.eval()
            _model = m

    return _model


def predict_future(trajectory):

    if trajectory is None or len(trajectory) < 10:
        return None

    model = _get_model()
    if model is None:
        return None

    trajectory = trajectory[-10:]

    trajectory = np.array(
        trajectory,
        dtype=np.float32
    )

    if not np.isfinite(trajectory).all():
        return None

    try:
        x = torch.tensor(
            trajectory / FRAME_SIZE,
            dtype=torch.float32
        ).unsqueeze(0).to(device)

        with torch.no_grad():
            pred = model(x)

        pred = pred.cpu().numpy()[0]
        pred = pred * FRAME_SIZE

    except Exception as e:
        print(f"[predictor] inference failed: {e}")
        return None

    if not np.isfinite(pred).all():
        return None

    return pred
