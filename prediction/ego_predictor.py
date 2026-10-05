import math
import numpy as np

# Must match the DETECTION camera spawned in object_detection/carla_yolo.py:
# Location(x=2.5, z=1.5), no rotation (facing straight ahead), fov=90,
# image 416x416. This is the same camera geometry the trajectory
# transformer's training data (tracking/carla_tracker_logger.py) was
# collected from, so the ego vehicle's own predicted path has to be
# expressed in that same pixel space to be comparable against the
# transformer's object-future predictions inside calculate_risk().
#
# The previous version of this function ignored the `vehicle` argument
# entirely and always returned five hardcoded points -- collision risk
# was never actually based on real ego motion, which is the root cause
# behind "collision prediction doesn't run consistently" / "ACC and AEB
# unstable".
CAMERA_LOCAL_X = 2.5
CAMERA_HEIGHT = 1.5
CAMERA_FOV_DEG = 90.0
IMG_W = 416
IMG_H = 416

_fx = (IMG_W / 2.0) / math.tan(math.radians(CAMERA_FOV_DEG) / 2.0)
_fy = _fx
_cx0 = IMG_W / 2.0
_cy0 = IMG_H / 2.0

# Rough curvature-per-steer-unit for a Model-3-like vehicle. CARLA's
# VehicleControl.steer is a normalized -1..1 command, not a physical
# wheel angle, so this is an approximation -- but it makes the
# predicted path actually bend the right way (and by a plausible
# amount) under steering, instead of always being a straight line
# regardless of what the car is doing.
STEER_TO_CURVATURE = 0.12


def predict_ego_future(vehicle, dt=0.4, steps=5):

    try:
        if vehicle is None or not vehicle.is_alive:
            return None
        velocity = vehicle.get_velocity()
        control = vehicle.get_control()
    except RuntimeError:
        # Camera callback can fire in the brief window while the
        # vehicle is being destroyed during shutdown.
        return None

    speed = math.sqrt(velocity.x ** 2 + velocity.y ** 2)
    curvature = control.steer * STEER_TO_CURVATURE

    points = []

    for i in range(1, steps + 1):
        forward = speed * dt * i
        lateral = 0.5 * curvature * (forward ** 2)

        cam_forward = forward - CAMERA_LOCAL_X
        # Avoid a divide-by-zero / behind-the-camera singularity for
        # the first point(s) when the vehicle is slow or stopped.
        cam_forward = max(cam_forward, 0.3)

        u = _fx * (lateral / cam_forward)
        v = _fy * (CAMERA_HEIGHT / cam_forward)

        px = float(np.clip(_cx0 + u, 0, IMG_W - 1))
        py = float(np.clip(_cy0 + v, 0, IMG_H - 1))

        points.append([px, py])

    return np.array(points, dtype=np.float32)
