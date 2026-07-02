def check_lane_departure(lane_center, frame_width, threshold=30):
    """
    Returns:
        departure : bool
        offset    : pixels
        steering  : steering command (-1 to 1)
    """

    if lane_center is None:
        return False, 0, 0.0

    vehicle_center = frame_width // 2

    offset = lane_center - vehicle_center

    departure = abs(offset) > threshold

    # Steering gain
    steering = -(offset / frame_width) * 2.0

    # Clamp steering
    steering = max(-1.0, min(1.0, steering))

    return departure, offset, steering