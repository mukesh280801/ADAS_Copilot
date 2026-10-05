import cv2
import numpy as np

# Temporal state makes the visual lane estimate much less jittery than a
# frame-by-frame raw Hough average.
_prev_left = None
_prev_right = None


def _smooth(old, new, alpha=0.25):
    if new is None:
        return old
    if old is None:
        return float(new)
    return (1.0 - alpha) * float(old) + alpha * float(new)


def _line_x_at_y(line, y):
    x1, y1, x2, y2 = map(float, line)
    if abs(y2 - y1) < 1e-6:
        return None
    return x1 + (float(y) - y1) * (x2 - x1) / (y2 - y1)


def detect_lanes(frame):
    """Robust camera lane visualization with temporal smoothing.

    Public interface is intentionally unchanged:
        annotated_frame, lane_center = detect_lanes(frame)

    The detector emphasizes white/yellow road markings, rejects mostly
    horizontal crosswalk/road-edge segments, fits left/right lane candidates,
    and smooths the estimate over time.
    """
    global _prev_left, _prev_right

    img = frame.copy()
    h, w = img.shape[:2]

    # Focus on the road ahead. Ignore skyline/buildings where Hough edges are
    # abundant and frequently contaminate the old detector.
    roi_top = int(h * 0.54)
    roi = np.zeros((h, w), dtype=np.uint8)
    polygon = np.array([[
        (0, h),
        (w, h),
        (int(w * 0.92), roi_top),
        (int(w * 0.08), roi_top),
    ]], dtype=np.int32)
    cv2.fillPoly(roi, polygon, 255)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Bright/low-saturation mask catches white lane markings.
    white = cv2.inRange(
        hsv,
        np.array([0, 0, 145], dtype=np.uint8),
        np.array([180, 85, 255], dtype=np.uint8),
    )

    # Yellow center/road markings.
    yellow = cv2.inRange(
        hsv,
        np.array([12, 55, 90], dtype=np.uint8),
        np.array([42, 255, 255], dtype=np.uint8),
    )

    color_mask = cv2.bitwise_or(white, yellow)
    color_mask = cv2.bitwise_and(color_mask, roi)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 45, 135)
    edges = cv2.bitwise_and(edges, roi)

    # Requiring both color evidence and edge evidence reduces false lane
    # lines from building edges and vehicle contours.
    color_edges = cv2.bitwise_and(edges, color_mask)
    combined = cv2.bitwise_or(color_edges, cv2.bitwise_and(edges, color_mask))

    # Close small gaps in dashed lane markings.
    kernel = np.ones((5, 5), np.uint8)
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=1)

    lines = cv2.HoughLinesP(
        combined,
        1,
        np.pi / 180,
        threshold=28,
        minLineLength=max(35, int(w * 0.045)),
        maxLineGap=max(35, int(w * 0.075)),
    )

    left_candidates = []
    right_candidates = []

    if lines is not None:
        for raw in lines[:, 0]:
            x1, y1, x2, y2 = map(float, raw)
            dx = x2 - x1
            dy = y2 - y1
            if abs(dy) < 1e-6:
                continue

            slope = dy / dx if abs(dx) > 1e-6 else np.sign(dy) * 999.0

            # Reject crosswalks and almost-horizontal clutter.
            if abs(slope) < 0.35:
                continue

            xm = (x1 + x2) * 0.5
            y_bottom = h - 5
            xb = _line_x_at_y((x1, y1, x2, y2), y_bottom)
            if xb is None:
                continue

            # Only accept lines whose extrapolated bottom point is on the
            # plausible road area.
            if xb < -0.35 * w or xb > 1.35 * w:
                continue

            # Image-space perspective: left boundary slopes up-right
            # (negative dx/dy), right boundary slopes up-left.
            if slope < 0 and xb < w * 0.68:
                left_candidates.append((xb, abs(slope), (int(x1), int(y1), int(x2), int(y2))))
            elif slope > 0 and xb > w * 0.32:
                right_candidates.append((xb, abs(slope), (int(x1), int(y1), int(x2), int(y2))))

    def robust_x(cands):
        if not cands:
            return None
        # Prefer candidates with strong perspective slope, then median to
        # suppress isolated Hough detections.
        xs = np.array([c[0] for c in cands], dtype=np.float32)
        slopes = np.array([c[1] for c in cands], dtype=np.float32)
        weights = np.clip(slopes, 0.35, 4.0)
        return float(np.average(xs, weights=weights))

    left_x = robust_x(left_candidates)
    right_x = robust_x(right_candidates)

    # Physical plausibility checks.
    if left_x is not None and left_x >= w * 0.52:
        left_x = None
    if right_x is not None and right_x <= w * 0.48:
        right_x = None

    # If both sides exist, enforce a realistic lane width at the bottom.
    if left_x is not None and right_x is not None:
        lane_width = right_x - left_x
        if lane_width < 0.18 * w or lane_width > 0.90 * w:
            # Keep the more reliable side and recover the other from history.
            if _prev_left is not None and _prev_right is not None:
                left_x, right_x = _prev_left, _prev_right
            elif abs(lane_width - 0.55 * w) > abs(lane_width - 0.35 * w):
                right_x = None

    _prev_left = _smooth(_prev_left, left_x, 0.22)
    _prev_right = _smooth(_prev_right, right_x, 0.22)

    # Recover a missing side briefly from the previous lane width.
    if _prev_left is not None and _prev_right is None:
        _prev_right = _prev_left + 0.42 * w
    elif _prev_right is not None and _prev_left is None:
        _prev_left = _prev_right - 0.42 * w

    left_draw = _prev_left
    right_draw = _prev_right

    if left_draw is not None:
        lx = int(np.clip(left_draw, 0, w - 1))
        cv2.line(img, (lx, h), (int(w * 0.46), roi_top), (0, 255, 255), 3, cv2.LINE_AA)

    if right_draw is not None:
        rx = int(np.clip(right_draw, 0, w - 1))
        cv2.line(img, (rx, h), (int(w * 0.54), roi_top), (0, 255, 255), 3, cv2.LINE_AA)

    if left_draw is not None and right_draw is not None:
        lane_center = int(np.clip((left_draw + right_draw) * 0.5, 0, w - 1))
        cv2.line(
            img,
            (lane_center, h),
            (w // 2, roi_top),
            (0, 220, 80),
            2,
            cv2.LINE_AA,
        )
        cv2.circle(img, (lane_center, h - 55), 7, (0, 220, 80), -1)
    else:
        lane_center = int(w * 0.5)

    return img, lane_center
