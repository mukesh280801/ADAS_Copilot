import sys
import os

sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

import carla
import cv2
import numpy as np
from ultralytics import YOLO
import time

from adas_system.trajectory_buffer import TrajectoryBuffer
from prediction.live_predictor import predict_if_ready
from collision.risk_engine import calculate_risk
from collision.warning_system import generate_warning
from prediction.ego_predictor import predict_ego_future
from collision.risk_logger import log_risk
from lane_detection.lane_detector import detect_lanes
from lane_detection.lane_departure import check_lane_departure

# =====================================
# LOAD MODEL
# =====================================
model = YOLO("yolov8n.pt")

# =====================================
# BUFFER
# =====================================
buffer = TrajectoryBuffer(max_len=10)

frame_counter = 0
fps_start = time.time()
fps_frames = 0

actor_list = []
vehicle = None


# =====================================
# CONTROL SYSTEM
# =====================================
def apply_control(vehicle, risk_score, steering, closest_distance):

    control = vehicle.get_control()

    control.steer = steering

    # -----------------------------------
    # ADAPTIVE CRUISE CONTROL + AEB
    # -----------------------------------

    if risk_score >= 0.8:   

        # Emergency Brake
        control.throttle = 0.0
        control.brake = 1.0

    elif closest_distance < 40:

        # Follow vehicle
        control.throttle = 0.20
        control.brake = 0.20

    elif closest_distance < 70:

        # Reduce speed
        control.throttle = 0.35
        control.brake = 0.0

    else:

        # Cruise
        control.throttle = 0.55
        control.brake = 0.0

    vehicle.apply_control(control)


# =====================================
# CAMERA CALLBACK
# =====================================
def process_img(image):

    global frame_counter, fps_start, fps_frames

    frame_counter += 1
    fps_frames += 1

    if fps_frames % 30 == 0:
        fps = 30 / (time.time() - fps_start)
        print(f"FPS: {fps:.2f}")
        fps_start = time.time()

    if frame_counter % 2 != 0:
        return

    img = np.frombuffer(image.raw_data, dtype=np.uint8)
    img = img.reshape((image.height, image.width, 4))
    img = img[:, :, :3].copy()

    # =====================================
    # YOLO TRACKING
    # =====================================
    results = model.track(
        img,
        imgsz=320,
        persist=True,
        tracker="bytetrack.yaml",
        device=0,
        verbose=False
    )

    result = results[0]
    annotated = img.copy()
    annotated, lane_center = detect_lanes(annotated)

    departure, offset, steering = check_lane_departure(
        lane_center,
        annotated.shape[1]
    )

    risk_score = 0.0
    min_distance = 999.0

    highest_risk = 0.0
    closest_distance = float("inf")

    if result.boxes is not None and result.boxes.id is not None:

        ids = result.boxes.id.cpu().numpy().astype(int)
        boxes = result.boxes.xyxy.cpu().numpy()

        ego_future = predict_ego_future(vehicle)

        for track_id, box in zip(ids, boxes):

            x1, y1, x2, y2 = box

            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            buffer.update(track_id, cx, cy)
            trajectory = buffer.get_trajectory(track_id)

            if len(trajectory) >= 10 and frame_counter % 10 == 0:

                future = predict_if_ready(trajectory)

                if future is not None:

                    result_risk = calculate_risk(future, ego_future)

                    risk_score = result_risk[0]
                    min_distance = result_risk[1]
                    closest_distance = min(closest_distance, min_distance)
                    ttc = result_risk[2] if len(result_risk) > 2 else None
                    highest_risk = max(highest_risk, risk_score)

                    log_risk(track_id, risk_score, min_distance)

                    # =========================
                    # WARNING SYSTEM
                    # =========================
                    if risk_score >= 0.2:
                        generate_warning(
                            object_id=track_id,
                            risk_score=risk_score,
                            min_distance=min_distance,
                            ttc=ttc
                        )

                    # =========================
                    # CONTROL (IMPORTANT FIX)
                    # =========================

                    # =========================
                    # VISUALS
                    # =========================
                    if risk_score > 0.5:
                        color = (0, 0, 255)
                        label = "HIGH"
                    elif risk_score > 0.2:
                        color = (0, 255, 255)
                        label = "MEDIUM"
                    else:
                        color = (0, 255, 0)
                        label = "LOW"

                    cv2.rectangle(
                        annotated,
                        (int(x1), int(y1)),
                        (int(x2), int(y2)),
                        color,
                        2
                    )

                    cv2.putText(
                        annotated,
                        f"ID:{track_id}",
                        (int(x1), int(y1) - 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        color,
                        2
                    )

                    cv2.putText(
                        annotated,
                        f"Risk:{label}",
                        (int(x1), int(y1) - 15),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        color,
                        2
                    )

                    cv2.putText(
                        annotated,
                        f"Dist:{min_distance:.1f}",
                        (int(x1), int(y1)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        color,
                        2
                    )

                    if ttc is not None:
                        cv2.putText(
                            annotated,
                            f"TTC:{ttc:.2f}s",
                            (int(x1), int(y1)-60),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            color,
                            2
                        )

        if departure:
            cv2.putText(
                annotated,
                "LANE DEPARTURE!",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2
            )
        else:
            cv2.putText(
                annotated,
                "KEEP LANE",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )
    if closest_distance == float("inf"):
        closest_distance = 999.0

    apply_control(vehicle, highest_risk, steering, closest_distance)

    print(
        f"ACC | Closest={closest_distance:.1f} | "
        f"Risk={highest_risk:.2f} | "
        f"Steer={steering:.2f}"
    )

    display = cv2.addWeighted(annotated, 0.7, img, 0.3, 0)

    cv2.imshow("ADAS SYSTEM", display)
    cv2.waitKey(1)
    


# =====================================
# MAIN
# =====================================
try:

    client = carla.Client("localhost", 2000)
    client.set_timeout(30.0)

    print("Connected to CARLA")

    world = client.get_world()
    blueprint_library = world.get_blueprint_library()

    # VEHICLE
    vehicle_bp = blueprint_library.filter("model3")[0]
    spawn_points = world.get_map().get_spawn_points()

    spawn_point = None

    # Fixed Spawn Location (your old location)

    spawn_point = carla.Transform(
        carla.Location(
            x=-64.644844,
            y=24.471010,
            z=0.600000
        )
    )

    vehicle = world.try_spawn_actor(
        vehicle_bp,
        spawn_point
    )

    if vehicle is None:

        print("Old spawn occupied, trying nearby locations...")

        found = False

        offsets = [
            (2, 0),
            (-2, 0),
            (0, 2),
            (0, -2),
            (4, 0),
            (-4, 0)
        ]

        for dx, dy in offsets:

            nearby_spawn = carla.Transform(
                carla.Location(
                    x=-64.644844 + dx,
                    y=24.471010 + dy,
                    z=0.600000
                )
            )

            vehicle = world.try_spawn_actor(
                vehicle_bp,
                nearby_spawn
            )

            if vehicle is not None:
                spawn_point = nearby_spawn
                found = True
                break

        if not found:
            raise Exception("No free spawn near old location!")

    print("Spawn success at:", spawn_point.location)

    if spawn_point is None:
        raise Exception("No safe spawn point found!")

    actor_list.append(vehicle)

    # ❗ IMPORTANT FIX
    vehicle.set_autopilot(False)

    print("Vehicle Spawned")
    control = carla.VehicleControl()
    control.throttle = 0.45
    control.brake = 0.0
    vehicle.apply_control(control)

    # CAMERA
    camera_bp = blueprint_library.find("sensor.camera.rgb")
    camera_bp.set_attribute("image_size_x", "320")
    camera_bp.set_attribute("image_size_y", "320")
    camera_bp.set_attribute("fov", "90")
    camera_bp.set_attribute("sensor_tick", "0.05")

    camera_transform = carla.Transform(carla.Location(x=2.5, z=1.5))

    camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)
    actor_list.append(camera)

    camera.listen(lambda image: process_img(image))

    print("ADAS Running...")

    while True:
        time.sleep(1)

except KeyboardInterrupt:
    print("Stopped")

finally:
    for actor in actor_list:
        try:
            actor.destroy()
        except:
            pass

    cv2.destroyAllWindows()
    print("Cleaned up")