import cv2
import numpy as np

# Smoothing state persists across calls (this module is used as a
# singleton for the one live camera stream).
_smoothed_center = None
_ALPHA = 0.3

# If the color-filtered path hasn't produced a fresh detection in
# this many calls, fall back to the angle-filtered (no color mask)
# edges for that frame. Without this, a color threshold that happens
# to miss the actual paint brightness/saturation in a given scene's
# lighting makes left_xs/right_xs empty forever, which freezes
# _smoothed_center at a stale value permanently -- the car then
# steers on an outdated command indefinitely instead of reacting to
# the road at all. Verified in testing: Steer stayed bit-for-bit
# identical for 150+ consecutive frames, which is only possible if
# lane_center stopped updating entirely.
_STALE_LIMIT = 15
_stale_count = 0


def _extract_lines(masked, width, frame):
    left_xs = []
    right_xs = []

    lines = cv2.HoughLinesP(
        masked,
        1,
        np.pi / 180,
        threshold=50,
        minLineLength=40,
        maxLineGap=50
    )

    if lines is None:
        return left_xs, right_xs

    for line in lines:

        x1, y1, x2, y2 = line[0]

        dx = x2 - x1
        dy = y2 - y1
        angle = np.degrees(np.arctan2(abs(dy), abs(dx) + 1e-6))
        if angle < 25:
            # Real lane markings run roughly toward the vanishing
            # point ahead of the car; near-horizontal segments
            # (curbs, crosswalks, shadows) are almost never lane
            # markings and were a source of erratic jumps.
            continue

        cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)

        mid_x = (x1 + x2) / 2

        # Split by side and average the midpoint of the two sides,
        # rather than averaging every segment together (which lets
        # an imbalanced segment count on one side drag lane_center
        # off-center even when the lane itself is centered).
        if mid_x < width / 2:
            left_xs.append(mid_x)
        else:
            right_xs.append(mid_x)

    return left_xs, right_xs


def detect_lanes(frame):

    global _smoothed_center, _stale_count

    height, width = frame.shape[:2]

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(blur, 50, 150)

    roi = np.zeros_like(edges)

    polygon = np.array([
        [
            (0, height),
            (width, height),
            (width, int(height * 0.6)),
            (0, int(height * 0.6))
        ]
    ])

    cv2.fillPoly(roi, polygon, 255)

    edges_roi = cv2.bitwise_and(edges, roi)

    # Real lane markings are painted white or yellow with high
    # contrast against dark asphalt. Decorative pavement (tile
    # grout, patterned plazas, shop entrances) produces plenty of
    # Canny edges too, but they usually aren't bright white/yellow
    # paint -- this is what stops the car steering itself toward a
    # shop entrance whose floor tiles happen to form diagonal lines.
    # Thresholds widened from the first pass (which was too strict
    # and made real road paint fail the mask under some lighting).
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    white_mask = cv2.inRange(hsv, (0, 0, 150), (180, 90, 255))
    yellow_mask = cv2.inRange(hsv, (15, 60, 100), (40, 255, 255))
    paint_mask = cv2.bitwise_or(white_mask, yellow_mask)
    paint_mask = cv2.dilate(paint_mask, np.ones((5, 5), np.uint8), iterations=1)

    color_edges = cv2.bitwise_and(edges_roi, paint_mask)

    left_xs, right_xs = _extract_lines(color_edges, width, frame)

    if left_xs or right_xs:
        _stale_count = 0
    else:
        _stale_count += 1
        if _stale_count >= _STALE_LIMIT:
            # Color filter has found nothing for too long -- fall
            # back to angle-filtered-only edges so the car reacts to
            # *something* real on the road instead of driving on a
            # frozen, increasingly outdated lane_center.
            left_xs, right_xs = _extract_lines(edges_roi, width, frame)

    if left_xs and right_xs:
        raw_center = int((np.mean(left_xs) + np.mean(right_xs)) / 2)
    elif left_xs:
        raw_center = int(np.mean(left_xs) + width * 0.15)
    elif right_xs:
        raw_center = int(np.mean(right_xs) - width * 0.15)
    else:
        raw_center = None

    if raw_center is not None:
        if _smoothed_center is None:
            _smoothed_center = raw_center
        else:
            # Exponential smoothing instead of trusting each frame's
            # raw (noisy) estimate outright.
            _smoothed_center = int(
                _smoothed_center * (1 - _ALPHA) + raw_center * _ALPHA
            )

    lane_center = _smoothed_center

    if lane_center is not None:
        cv2.circle(
            frame,
            (lane_center, height - 30),
            8,
            (0, 0, 255),
            -1
        )

    return frame, lane_center
