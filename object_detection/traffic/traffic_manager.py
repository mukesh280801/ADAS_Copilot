import random
import carla



print("traffic_manager.py loaded")
def spawn_traffic(client, world, blueprint_library, ego_vehicle, number_of_vehicles=20):

    traffic_manager = client.get_trafficmanager(8000)

    traffic_manager.set_global_distance_to_leading_vehicle(2.5)
    traffic_manager.set_synchronous_mode(False)

    spawn_points = world.get_map().get_spawn_points()

    random.shuffle(spawn_points)

    ego_location = ego_vehicle.get_location()

    vehicles = []

    print("=" * 60)
    print("Spawning Traffic Vehicles...")
    print("=" * 60)

    for spawn in spawn_points:

        if len(vehicles) >= number_of_vehicles:
            break

        # Skip locations too close to ego vehicle
        if spawn.location.distance(ego_location) < 25:
            continue

        vehicle_blueprints = blueprint_library.filter("vehicle.*")
        bp = random.choice(vehicle_blueprints)

        # Random color
        if bp.has_attribute("color"):
            color = random.choice(
                bp.get_attribute("color").recommended_values
            )
            bp.set_attribute("color", color)

        npc = world.try_spawn_actor(bp, spawn)

        if npc is None:
            continue

        npc.set_autopilot(True, traffic_manager.get_port())

        vehicles.append(npc)

        print(
            f"[{len(vehicles):02d}] "
            f"{npc.type_id}  ->  "
            f"({spawn.location.x:.1f}, "
            f"{spawn.location.y:.1f})"
        )

    print("=" * 60)
    print(f"Total Traffic Spawned : {len(vehicles)}")
    print("=" * 60)

    return vehicles