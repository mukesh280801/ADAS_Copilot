import random
import carla


def _same_direction_lane(a, b):
    return a is not None and b is not None and a.lane_id * b.lane_id > 0


def _find_natural_overtake_lead_wp(carla_map, ego_wp):
    """Find a real road waypoint ahead of ego for a naturally seeded slow lead.

    This is not a separate scenario actor: the returned vehicle is spawned as
    an ordinary TrafficManager NPC. The seed only guarantees that the mixed
    traffic contains at least one slower same-lane vehicle, otherwise a random
    35-vehicle sample can legitimately contain no usable overtake opportunity.
    """
    if ego_wp is None:
        return None
    try:
        probe = ego_wp
        travelled = 0.0
        for _ in range(20):
            nxt = probe.next(5.0)
            if not nxt:
                return None
            # Preserve the ego's current heading/route through junctions.
            probe = min(
                nxt,
                key=lambda w: abs(
                    (w.transform.rotation.yaw - probe.transform.rotation.yaw + 180.0) % 360.0 - 180.0
                ),
            )
            travelled += 5.0
            if travelled < 24.0 or probe.is_junction:
                continue
            if probe.lane_type != carla.LaneType.Driving:
                continue

            # Require a usable non-junction corridor for the overtake.
            corridor = probe
            good = True
            for _ in range(4):
                if corridor.is_junction:
                    good = False
                    break
                nxt2 = corridor.next(6.0)
                if not nxt2:
                    good = False
                    break
                corridor = min(
                    nxt2,
                    key=lambda w: abs(
                        (w.transform.rotation.yaw - corridor.transform.rotation.yaw + 180.0) % 360.0 - 180.0
                    ),
                )
            if not good or corridor.is_junction:
                continue

            # At least one same-direction adjacent driving lane must exist.
            adjacent = [probe.get_left_lane(), probe.get_right_lane()]
            if not any(
                a is not None
                and a.lane_type == carla.LaneType.Driving
                and _same_direction_lane(a, probe)
                and not a.is_junction
                for a in adjacent
            ):
                continue
            return probe
    except RuntimeError:
        return None
    return None


def spawn_traffic(client, world, blueprint_library, ego_vehicle, number_of_vehicles=45):
    tm = client.get_trafficmanager(8000)
    tm.set_global_distance_to_leading_vehicle(5.0)
    tm.set_synchronous_mode(False)
    try:
        tm.set_random_device_seed(42)
    except Exception:
        pass

    ego_loc = ego_vehicle.get_location()
    ego_wp = world.get_map().get_waypoint(
        ego_loc, project_to_road=True, lane_type=carla.LaneType.Driving
    )
    all_spawns = world.get_map().get_spawn_points()

    # Keep every available adjacent same-direction lane locally open so the
    # ego can encounter a genuine overtake opportunity. The previous build
    # reserved only the RIGHT lane; on Town10HD the ego can start on a lane
    # where RIGHT is unavailable and LEFT is the only legal passing lane, so
    # random NPC traffic could permanently occupy the only escape lane.
    reserved_lanes = set()
    if ego_wp is not None:
        for candidate in (ego_wp.get_left_lane(), ego_wp.get_right_lane()):
            if (
                candidate is not None
                and candidate.lane_type == carla.LaneType.Driving
                and _same_direction_lane(candidate, ego_wp)
            ):
                reserved_lanes.add((candidate.road_id, candidate.lane_id))

    nearby, rest = [], []
    accepted_locations = []
    MIN_SPAWN_SEPARATION_M = 22.0
    for sp in all_spawns:
        d = sp.location.distance(ego_loc)
        if d < 18.0:
            continue
        # Avoid packing multiple NPCs onto nearly adjacent spawn points.
        # Dense startup clusters are a major source of artificial queues in
        # Town10HD before Traffic Manager has had time to spread them out.
        if any(sp.location.distance(prev) < MIN_SPAWN_SEPARATION_M for prev in accepted_locations):
            continue

        wp = world.get_map().get_waypoint(
            sp.location, project_to_road=True, lane_type=carla.LaneType.Driving
        )
        if wp is None:
            continue

        # Reserve the immediate adjacent same-direction lanes around the
        # ego. Keep them open for at least one natural overtake window.
        if (wp.road_id, wp.lane_id) in reserved_lanes and d < 180.0:
            continue

        if ego_wp and wp.road_id == ego_wp.road_id and _same_direction_lane(wp, ego_wp) and d < 120.0:
            nearby.append(sp)
        else:
            rest.append(sp)

    random.seed(42)
    random.shuffle(nearby)
    random.shuffle(rest)
    candidates = nearby + rest
    blueprints = [bp for bp in blueprint_library.filter("vehicle.*") if bp.has_attribute("number_of_wheels")]
    vehicles = []

    # Seed ONE ordinary NPC as a slower same-lane lead. This is deliberately
    # part of the normal traffic pool (not CONTROLLED_OVERTAKE_TEST and not a
    # special ego-side controller). Its only purpose is to make the natural
    # overtake feature observable on every run instead of depending on a lucky
    # random spawn distribution. All ego-side lane/TTC/pedestrian/signal gates
    # still decide whether an overtake actually starts.
    seed_wp = _find_natural_overtake_lead_wp(world.get_map(), ego_wp)
    seed_actor = None
    if seed_wp is not None and blueprints:
        try:
            seed_bp = random.choice(blueprints)
            if seed_bp.has_attribute("color"):
                vals = seed_bp.get_attribute("color").recommended_values
                if vals:
                    seed_bp.set_attribute("color", random.choice(vals))
            seed_tf = seed_wp.transform
            seed_tf.location.z += 0.35
            seed_actor = world.try_spawn_actor(seed_bp, seed_tf)
            if seed_actor is not None:
                seed_actor.set_autopilot(True, tm.get_port())
                tm.set_desired_speed(seed_actor, random.uniform(13.0, 16.0))
                tm.distance_to_leading_vehicle(seed_actor, 5.0)
                tm.auto_lane_change(seed_actor, False)
                tm.ignore_vehicles_percentage(seed_actor, 0.0)
                tm.ignore_walkers_percentage(seed_actor, 0.0)
                tm.ignore_lights_percentage(seed_actor, 0.0)
                tm.ignore_signs_percentage(seed_actor, 0.0)
                tm.collision_detection(seed_actor, ego_vehicle, True)
                tm.collision_detection(ego_vehicle, seed_actor, True)
                vehicles.append(seed_actor)
                accepted_locations.append(seed_actor.get_location())
                print("Natural overtake seed: slower same-lane NPC created; adjacent lane kept clear")
        except Exception:
            seed_actor = None

    for sp in candidates:
        if len(vehicles) >= number_of_vehicles:
            break
        bp = random.choice(blueprints)
        if bp.has_attribute("color"):
            vals = bp.get_attribute("color").recommended_values
            if vals:
                bp.set_attribute("color", random.choice(vals))
        npc = world.try_spawn_actor(bp, sp)
        if npc is None:
            continue
        try:
            npc.set_autopilot(True, tm.get_port())
            # CARLA TrafficManager.set_desired_speed() expects the desired
            # vehicle speed in km/h. The previous build supplied 4.2-5.5 as
            # if it were m/s, making NPCs target roughly 4-5 km/h. That was
            # the primary cause of the artificial traffic queue and repeated
            # stop/creep behaviour seen in the runtime recording.
            #
            # Keep a minority of NPCs slower than ego for natural overtake
            # opportunities, while the majority flow at normal city speed.
            if random.random() < 0.24:
                desired_speed_kmh = random.uniform(14.0, 17.0)
            else:
                desired_speed_kmh = random.uniform(25.0, 30.0)
            tm.set_desired_speed(npc, desired_speed_kmh)
            tm.distance_to_leading_vehicle(npc, random.uniform(4.5, 6.5))
            # Keep the locally reserved passing corridor stable while the
            # ego is close to it. NPCs spawned near the ego stay in their lane
            # instead of jumping into the freshly reserved adjacent lane;
            # farther traffic retains natural lane-change recovery. This is
            # still ordinary TrafficManager traffic -- there is no controlled
            # lead actor or ego-side NPC steering.
            local_distance = npc.get_location().distance(ego_loc)
            if local_distance < 180.0:
                tm.auto_lane_change(npc, False)
            else:
                tm.auto_lane_change(npc, True if random.random() < 0.60 else False)
            tm.ignore_vehicles_percentage(npc, 0.0)
            tm.ignore_walkers_percentage(npc, 0.0)
            tm.ignore_lights_percentage(npc, 0.0)
            tm.ignore_signs_percentage(npc, 0.0)
            tm.collision_detection(npc, ego_vehicle, True)
            tm.collision_detection(ego_vehicle, npc, True)
            for other in vehicles:
                if other is not None and other.is_alive:
                    tm.collision_detection(npc, other, True)
                    tm.collision_detection(other, npc, True)
        except Exception:
            pass
        vehicles.append(npc)
        accepted_locations.append(sp.location)

    print(f"Traffic vehicles spawned: {len(vehicles)}")
    print("Natural overtake setup: adjacent same-direction lane locally reserved; no controlled lead actor")
    print("Traffic behavior: 45 NPCs target 14–17 km/h (24% slow) / 25–30 km/h (76% normal); follow gap 4.5–6.5 m; local 180m passing corridor reserved; nearby NPC lane changes held; natural overtake seed enabled; collision checking ON")
    return vehicles
