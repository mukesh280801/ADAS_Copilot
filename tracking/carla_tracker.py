import carla
import cv2
import numpy as np
from ultralytics import YOLO
import time

model = YOLO("yolov8n.pt")

actor_list = []

def process_img(image):
    img = np.array(image.raw_data)
    img = img.reshape((image.height, image.width, 4))
    img = img[:, :, :3]

    # Tracking enabled
    results = model.track(
        img,
        persist=True,
        tracker="bytetrack.yaml",
        device=0,
        verbose=False
    )

    annotated = results[0].plot()

    cv2.imshow("CARLA + YOLO + Tracking", annotated)
    cv2.waitKey(1)

try:
    client = carla.Client("localhost", 2000)
    client.set_timeout(10.0)

    world = client.get_world()
    blueprint_library = world.get_blueprint_library()

    vehicle_bp = blueprint_library.filter('model3')[0]
    spawn_point = world.get_map().get_spawn_points()[0]

    vehicle = world.spawn_actor(vehicle_bp, spawn_point)
    vehicle.set_autopilot(True)

    actor_list.append(vehicle)

    camera_bp = blueprint_library.find('sensor.camera.rgb')

    camera_bp.set_attribute('image_size_x', '416')
    camera_bp.set_attribute('image_size_y', '416')

    camera_transform = carla.Transform(
        carla.Location(x=2.5, z=1.5)
    )

    camera = world.spawn_actor(
        camera_bp,
        camera_transform,
        attach_to=vehicle
    )

    actor_list.append(camera)

    camera.listen(lambda image: process_img(image))

    print("✅ Tracking Started")

    time.sleep(120)

finally:
    for actor in actor_list:
        actor.destroy()

    cv2.destroyAllWindows()