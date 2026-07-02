import carla
import random
import time

def main():
    try:
        # Connect to CARLA
        client = carla.Client('localhost', 2000)
        client.set_timeout(10.0)

        world = client.get_world()

        print("✅ Connected to CARLA")

        # Get blueprint library
        blueprint_library = world.get_blueprint_library()

        # Select Tesla Model 3
        vehicle_bp = blueprint_library.filter('model3')[0]

        # Get spawn points
        spawn_points = world.get_map().get_spawn_points()

        # Random spawn point
        spawn_point = random.choice(spawn_points)

        # Spawn vehicle
        vehicle = world.spawn_actor(vehicle_bp, spawn_point)

        print(f"✅ Vehicle Spawned: {vehicle.type_id}")

        # Enable autopilot
        vehicle.set_autopilot(True)

        print("✅ Autopilot Enabled")

        # Keep simulation running
        time.sleep(30)

        vehicle.destroy()

        print("✅ Vehicle Destroyed")

    except Exception as e:
        print("❌ Error:", e)

if __name__ == "__main__":
    main()