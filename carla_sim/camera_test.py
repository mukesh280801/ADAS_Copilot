import carla
import cv2
import numpy as np
import time

actor_list = []

def process_img(image):
    img = np.array(image.raw_data)
    img = img.reshape((image.height, image.width, 4))
    img = img[:, :, :3]

    cv2.imshow("ADAS Camera Feed", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        return

try:
    client = carla.Client("localhost", 2000)
    client.set_timeout(10.0)

    world = client.get_world()

    blueprint_library = world.get_blueprint_library()

    # Vehicle
    vehicle_bp = blueprint_library.filter('model3')[0]

    spawn_point = world.get_map().get_spawn_points()[0]

    vehicle = world.spawn_actor(vehicle_bp, spawn_point)

    actor_list.append(vehicle)

    vehicle.set_autopilot(True)

    print("✅ Vehicle Spawned")

    # Camera
    camera_bp = blueprint_library.find('sensor.camera.rgb')

    camera_bp.set_attribute('image_size_x', '800')
    camera_bp.set_attribute('image_size_y', '600')
    camera_bp.set_attribute('fov', '110')

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

    print("✅ RGB Camera Attached")

    time.sleep(60)

finally:
    print("Destroying Actors...")

    for actor in actor_list:
        actor.destroy()

    cv2.destroyAllWindows()