import carla
import cv2
import numpy as np
from ultralytics import YOLO
import csv
import os
import time

# =====================================
# CONFIG
# =====================================

CSV_FILE = "prediction/trajectory_dataset.csv"

os.makedirs("prediction", exist_ok=True)

# Create CSV file with header
if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "frame",
            "object_id",
            "class_name",
            "x_center",
            "y_center"
        ])

# =====================================
# YOLO MODEL
# =====================================

model = YOLO("yolov8n.pt")

actor_list = []
frame_count = 0

# =====================================
# SAVE TRACKS
# =====================================

def save_tracks(results):
    global frame_count

    frame_count += 1

    if results[0].boxes.id is None:
        return

    ids = results[0].boxes.id.cpu().numpy()
    boxes = results[0].boxes.xyxy.cpu().numpy()
    classes = results[0].boxes.cls.cpu().numpy()

    with open(CSV_FILE, mode="a", newline="") as f:

        writer = csv.writer(f)

        for track_id, box, cls in zip(ids, boxes, classes):

            x_center = (box[0] + box[2]) / 2
            y_center = (box[1] + box[3]) / 2

            class_name = model.names[int(cls)]

            writer.writerow([
                frame_count,
                int(track_id),
                class_name,
                round(float(x_center), 2),
                round(float(y_center), 2)
            ])

# =====================================
# CAMERA CALLBACK
# =====================================

def process_img(image):

    img = np.frombuffer(
        image.raw_data,
        dtype=np.uint8
    )

    img = img.reshape(
        (image.height,
         image.width,
         4)
    )

    img = img[:, :, :3]

    results = model.track(
        img,
        persist=True,
        tracker="bytetrack.yaml",
        device=0,
        verbose=False
    )

    save_tracks(results)

    annotated_frame = results[0].plot()

    cv2.imshow(
        "CARLA + YOLO + Tracking",
        annotated_frame
    )

    cv2.waitKey(1)

# =====================================
# MAIN
# =====================================

try:

    client = carla.Client(
        "localhost",
        2000
    )

    client.set_timeout(10.0)

    world = client.get_world()

    blueprint_library = world.get_blueprint_library()

    # Vehicle
    vehicle_bp = blueprint_library.filter(
        "model3"
    )[0]

    spawn_point = world.get_map().get_spawn_points()[0]

    vehicle = world.spawn_actor(
        vehicle_bp,
        spawn_point
    )

    vehicle.set_autopilot(True)

    actor_list.append(vehicle)

    print("✅ Vehicle Spawned")

    # Camera
    camera_bp = blueprint_library.find(
        "sensor.camera.rgb"
    )

    camera_bp.set_attribute(
        "image_size_x",
        "416"
    )

    camera_bp.set_attribute(
        "image_size_y",
        "416"
    )

    camera_bp.set_attribute(
        "fov",
        "90"
    )

    camera_transform = carla.Transform(
        carla.Location(
            x=2.5,
            z=1.5
        )
    )

    camera = world.spawn_actor(
        camera_bp,
        camera_transform,
        attach_to=vehicle
    )

    actor_list.append(camera)

    camera.listen(
        lambda image: process_img(image)
    )

    print("✅ Tracking Started")
    print("✅ Trajectory Logging Started")
    print(f"✅ Saving CSV -> {CSV_FILE}")

    # Run for 5 minutes
    time.sleep(300)

except KeyboardInterrupt:
    print("Stopped by User")

finally:

    print("Destroying Actors...")

    for actor in actor_list:
        if actor.is_alive:
            actor.destroy()

    cv2.destroyAllWindows()

    print("✅ Finished")