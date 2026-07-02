import cv2
import numpy as np


def detect_lanes(frame):

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

    masked = cv2.bitwise_and(edges, roi)

    lines = cv2.HoughLinesP(
        masked,
        1,
        np.pi / 180,
        threshold=50,
        minLineLength=40,
        maxLineGap=50
    )

    lane_center = None

    if lines is not None:

        x_positions = []

        for line in lines:

            x1, y1, x2, y2 = line[0]

            cv2.line(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                3
            )

            x_positions.extend([x1, x2])

        lane_center = int(np.mean(x_positions))

    return frame, lane_center