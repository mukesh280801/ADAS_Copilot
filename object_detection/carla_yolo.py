"""ADAS Copilot - safety-first CARLA runner (Pedestrian Safety v3).

Design goals for this build:
- CARLA camera callbacks only capture frames; heavy work happens in one worker.
- One UI window owns all OpenCV rendering.
- YOLO/ByteTrack remains the perception/display layer.
- CARLA road geometry is used for the simulation controller so distance,
  lane identity, traffic lights and TTC are expressed in physical units.
- AEB is reserved for genuinely imminent hazards; a blocking same-direction
  lead vehicle is handled first by a safe adjacent-lane manoeuvre when possible.
"""

import math
import os
import queue
import random
import sys
import threading
import time
from collections import deque

import carla
import cv2
import numpy as np
import torch
from ultralytics import YOLO

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from collision.risk_engine import calculate_physical_risk
from collision.risk_logger import log_risk, shutdown_logger
from collision.warning_system import generate_warning
# Dashboard renderer is embedded below; this runner is self-contained.
from lane_detection.lane_detector import detect_lanes
from lane_detection.lane_departure import check_lane_departure
from object_detection.traffic.traffic_manager import spawn_traffic
from prediction.ego_predictor import predict_ego_future
from prediction.live_predictor import predict_if_ready
from adas_system.trajectory_buffer import TrajectoryBuffer

# ---------------- Embedded dashboard renderer ----------------
"""ADAS Copilot - premium standalone dashboard renderer."""
import math
import cv2
import numpy as np

W, H = 510, 830
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_B = cv2.FONT_HERSHEY_DUPLEX

BG = (9, 13, 19)
PANEL = (18, 25, 34)
PANEL2 = (23, 31, 42)
GRID = (39, 50, 64)
TEXT = (235, 242, 248)
MUTED = (135, 151, 168)
CYAN = (224, 204, 36)
GREEN = (92, 214, 126)
AMBER = (70, 190, 245)
RED = (72, 82, 238)
WHITE = (248, 250, 252)


def clamp(v, lo=0.0, hi=1.0):
    try:
        return max(lo, min(hi, float(v)))
    except Exception:
        return lo


def metric(v, unit=""):
    if v is None:
        return "--"
    try:
        if math.isinf(float(v)):
            return "INF"
        return f"{float(v):.1f}{unit}"
    except Exception:
        return str(v)


def panel(img, x1, y1, x2, y2, fill=PANEL, border=GRID, r=18):
    cv2.rectangle(img, (x1+r, y1), (x2-r, y2), fill, -1)
    cv2.rectangle(img, (x1, y1+r), (x2, y2-r), fill, -1)
    for x, y in ((x1+r,y1+r),(x2-r,y1+r),(x1+r,y2-r),(x2-r,y2-r)):
        cv2.circle(img, (x,y), r, fill, -1)
    cv2.rectangle(img, (x1+r,y1), (x2-r,y1), border, 1)
    cv2.rectangle(img, (x1+r,y2), (x2-r,y2), border, 1)
    cv2.rectangle(img, (x1,y1+r), (x1,y2-r), border, 1)
    cv2.rectangle(img, (x2,y1+r), (x2,y2-r), border, 1)


def text(img, s, xy, scale=.52, color=TEXT, thick=1, font=FONT):
    cv2.putText(img, str(s), xy, font, scale, color, thick, cv2.LINE_AA)


def pill(img, label, x, y, color, width=150, height=30):
    cv2.rectangle(img, (x,y), (x+width,y+height), color, -1)
    text(img, label, (x+12, y+21), .46, BG, 1, FONT_B)


def bar(img, x, y, w, h, value, fill):
    # BUG FIX: cv2.rectangle(img, (x,y,x+w,y+h), color, -1) was passing a
    # 4-element tuple as the "pt1" argument. OpenCV's Python bindings accept
    # that shape as the OTHER rectangle overload -- cv2.rectangle(img, rect,
    # color, thickness) where rect=(x,y,width,height) -- so it was silently
    # drawing a rectangle from (x,y) sized (x+w) x (y+h), i.e. far larger
    # than intended. This is what produced the oversized bar seen covering
    # the SAFETY MONITOR panel on screen. Fixed to the correct two-point
    # form: cv2.rectangle(img, pt1, pt2, color, thickness).
    cv2.rectangle(img, (x, y), (x + w, y + h), (31, 40, 52), -1)
    fw = int(w*clamp(value))
    if fw:
        cv2.rectangle(img, (x, y), (x + fw, y + h), fill, -1)


def status_color(action):
    a = str(action).upper()
    if "PEDESTRIAN" in a or "AEB" in a or "STOP" in a:
        return RED
    if "RED LIGHT" in a:
        return RED
    if "YELLOW" in a or "OVERTAKE" in a:
        return AMBER
    if "ACC" in a:
        return CYAN
    return GREEN


def draw_dashboard(frame=None, speed=0.0, fps=0.0, lane_status="KEEP", risk=0.0,
                   distance=float("inf"), ttc=float("inf"), steering="STRAIGHT",
                   throttle=0.0, brake=0.0, tracked_objects=0, lead_id=None,
                   action="CRUISE", traffic_light="NONE", lane_change="NONE",
                   pedestrian_id=None, pedestrian_distance=float("inf"),
                   pedestrian_conflict=False, yolo_person_visible=False):
    img = np.full((H,W,3), BG, np.uint8)
    accent = status_color(action)
    risk_v = clamp(risk)

    # Header
    panel(img, 14, 14, W-14, 92, PANEL2)
    text(img, "ADAS", (32, 47), .95, CYAN, 2, FONT_B)
    text(img, "COPILOT", (32, 72), .55, WHITE, 1, FONT_B)
    pill(img, str(action).upper()[:24], 292, 34, accent, 275, 30)

    # Speed / risk hero
    panel(img, 14, 106, W-14, 325, PANEL)
    cx, cy, r = 118, 215, 82
    cv2.circle(img, (cx,cy), r, (35,46,60), 9)
    cv2.ellipse(img, (cx,cy), (r,r), -90, 0, int(360*clamp(speed/60)), CYAN, 9, cv2.LINE_AA)
    text(img, f"{float(speed):.0f}", (74, 224), 1.55, WHITE, 2, FONT_B)
    text(img, "km/h", (88, 250), .48, MUTED, 1)
    text(img, "EGO SPEED", (67, 276), .42, MUTED, 1)

    text(img, "THREAT", (240, 142), .43, MUTED, 1, FONT_B)
    text(img, f"{risk_v:.2f}", (238, 184), 1.18, accent, 2, FONT_B)
    bar(img, 240, 201, 230, 14, risk_v, accent)  # width reduced from 330 -- 330 overflowed past the panel/canvas edge
    text(img, "LOW" if risk_v < .3 else "MEDIUM" if risk_v < .65 else "HIGH", (240, 239), .48, accent, 1, FONT_B)

    # Compact telemetry grid
    metrics = [
        ("TTC", metric(ttc," s")), ("LEAD", lead_id if lead_id is not None else "--"),
        ("GAP", metric(distance," m")), ("FPS", f"{float(fps):.1f}"),
    ]
    for i,(k,v) in enumerate(metrics):
        x = 240 + (i%2)*165
        y = 270 + (i//2)*55
        text(img, k, (x,y), .38, MUTED, 1)
        text(img, v, (x,y+25), .58, WHITE, 1, FONT_B)

    # Driving state
    panel(img, 14, 340, 294, 650, PANEL)
    text(img, "DRIVING STATE", (30, 370), .47, MUTED, 1, FONT_B)
    text(img, "LANE", (30, 401), .38, MUTED, 1)
    lane = str(lane_status).upper()
    text(img, lane, (30, 427), .64, GREEN if "KEEP" in lane else RED, 2, FONT_B)
    text(img, "STEERING", (30, 460), .38, MUTED, 1)
    text(img, str(steering).upper(), (30, 486), .58, WHITE, 1, FONT_B)
    text(img, "LANE CHANGE", (30, 518), .38, MUTED, 1)
    lc = str(lane_change).upper()
    text(img, lc, (30, 544), .56, AMBER if lc != "NONE" else MUTED, 1, FONT_B)
    text(img, "OBJECTS", (30, 578), .38, MUTED, 1)
    text(img, tracked_objects, (30, 608), .82, WHITE, 2, FONT_B)
    text(img, "PERCEPTION", (30, 638), .38, MUTED, 1)
    text(img, "YOLO + BYTETRACK", (30, 666), .48, CYAN, 1, FONT_B)

    # Safety panel
    panel(img, 320, 340, W-14, 650, PANEL)
    text(img, "SAFETY MONITOR", (338, 370), .47, MUTED, 1, FONT_B)
    text(img, "TRAFFIC LIGHT", (338, 401), .38, MUTED, 1)
    tl = str(traffic_light).upper()
    tlc = RED if tl=="RED" else AMBER if tl=="YELLOW" else GREEN if tl=="GREEN" else MUTED
    pill(img, tl, 338, 412, tlc, 112)

    text(img, "PEDESTRIAN", (338, 475), .38, MUTED, 1)
    if pedestrian_id is not None and math.isfinite(float(pedestrian_distance)):
        ptxt = f"ID {pedestrian_id}  {float(pedestrian_distance):.1f}m"
        text(img, ptxt + ("  |  CONFLICT" if pedestrian_conflict else ""), (338, 501), .50, RED if pedestrian_conflict else AMBER, 1, FONT_B)
    else:
        text(img, "CLEAR", (338, 501), .58, GREEN, 1, FONT_B)
    # Separate, DISPLAY-ONLY YOLO visual cue. Never feeds AEB/braking and
    # never borrows/fakes a CARLA pedestrian ID. Some CARLA bicycle
    # blueprints render the rider as part of the vehicle mesh with no
    # distinct walker.pedestrian actor, which is why "PEDESTRIAN" above
    # (CARLA ground truth) can correctly show CLEAR while a rider is
    # plainly visible on camera and the vehicle-AEB branch is braking for
    # the bike itself.
    text(img, ("CAM: PERSON VISIBLE" if yolo_person_visible else "CAM: no person"),
         (338, 519), .34, AMBER if yolo_person_visible else MUTED, 1)

    text(img, "SYSTEM", (338, 536), .38, MUTED, 1)
    pill(img, "SAFETY ARMED", 338, 548, GREEN, 165)
    text(img, "ACTIVE ACTION", (338, 606), .38, MUTED, 1)
    lines = [str(action).upper()[i:i+25] for i in range(0,len(str(action)),25)][:2]
    for i,line in enumerate(lines):
        text(img, line, (338, 632+i*24), .49, accent, 1, FONT_B)

    # Bottom actuator / health strip
    panel(img, 14, 665, W-14, 816, PANEL2)
    text(img, "ACTUATORS", (30, 693), .42, MUTED, 1, FONT_B)
    text(img, "THROTTLE", (30, 721), .36, MUTED, 1)
    bar(img, 30, 730, 380, 12, throttle, GREEN)
    text(img, f"{int(clamp(throttle)*100):02d}%", (440, 721), .38, WHITE, 1)
    text(img, "BRAKE", (30, 763), .36, MUTED, 1)
    bar(img, 30, 774, 380, 12, brake, RED)
    text(img, f"{int(clamp(brake)*100):02d}%", (440, 763), .38, WHITE, 1)
    text(img, "CONTROL", (30, 800), .36, MUTED, 1)
    text(img, f"{DISPLAY_HZ:.0f} FPS DISPLAY", (30, 817), .50, WHITE, 1, FONT_B)
    # OpenCV's Hershey fonts (used by cv2.putText) don't support the "•"
    # glyph -- it was rendering as "???" on screen (visible in your test
    # recording). Replaced with a plain "|" separator that renders correctly.
    text(img, "SAFETY ARBITER  |  ONLINE", (285, 817), .46, GREEN, 1, FONT_B)
    return img

# ---------------- End embedded dashboard renderer ----------------

# ---------------------------------------------------------------------------
# Runtime configuration
# ---------------------------------------------------------------------------
YOLO_DEVICE = 0 if torch.cuda.is_available() else "cpu"
YOLO_MODEL_PATH = os.environ.get("ADAS_YOLO_MODEL", "yolov8s.pt")

# HD camera stays full resolution for the operator view.
CAMERA_W, CAMERA_H = 1280, 720

# Only the YOLO inference copy is reduced. Detection coordinates are scaled
# back to the original 1280x720 frame before drawing.
YOLO_INPUT_W, YOLO_INPUT_H = 768, 432
YOLO_IMGSZ = 512
YOLO_CONF = 0.20

# Camera/UI are decoupled from perception. CARLA delivers 30 FPS HD frames;
# YOLO/ByteTrack runs at 15 FPS and always consumes the newest frame.
CAMERA_TICK = 1.0 / 30.0
DISPLAY_HZ = 30.0
PROCESS_HZ = 12.0
LANE_HZ = 15.0
CONTROL_HZ = 20.0

if torch.cuda.is_available():
    # Keep startup deterministic. The previous cuDNN autotune/warm-up path
    # could block the perception thread for the entire demo run.
    torch.backends.cudnn.benchmark = False

# Mixed-traffic validation: keep a moderate amount of normal traffic while
# reserving one deterministic slow lead for the overtake scenario.
SPAWN_TRAFFIC_VEHICLES = True
TRAFFIC_VEHICLE_COUNT = 45
# Keep normal traffic visible while the controlled lead guarantees a repeatable
# overtake trigger. Runtime lane safety still decides whether the maneuver is allowed.
ISOLATED_OVERTAKE_VALIDATION = False
CLEAR_EXISTING_VEHICLES = True
CONTROLLED_OVERTAKE_TEST = True
CONTROLLED_LEAD_DISTANCE = 18.0
CONTROLLED_LEAD_SPEED_DIFFERENCE = 45.0
# Deterministic validation speed. The controlled lead is intentionally
# slower than the ego, but it must not randomly accelerate under TM.
CONTROLLED_LEAD_TARGET_SPEED_MPS = 2.0

# Normal traffic is deliberately slower than the ego cruise speed so that
# overtaking occurs naturally when a slower same-direction vehicle is ahead.
# TrafficManager desired speed is configured in km/h. Ego typically cruises around 18–22 km/h.
# Normal traffic uses a small speed distribution below ego cruise so slower
# same-direction leads occur naturally; no controlled overtake actor is used.
TRAFFIC_TARGET_SPEED_MPS = None
CONTROLLED_OVERTAKE_MIN_STRAIGHT_M = 70.0
CONTROLLED_OVERTAKE_MIN_TARGET_FRONT_GAP_M = 70.0
CONTROLLED_OVERTAKE_MIN_TARGET_REAR_GAP_M = 45.0

# =====================================================================
# TEMPORARY OVERTAKE DIAGNOSTICS -- SAFE TO DELETE LATER
# =====================================================================
# Everything under this banner (this flag, _last_overtake_diag_t, and the
# _diagnose_lane() helper below) is read-only console logging added ONLY to
# explain why the overtake state machine does/doesn't trigger. None of it
# feeds vehicle.apply_control(), none of it changes any threshold, and none
# of it alters lane_is_safe()/choose_safe_lane()/update_lane_change()'s
# actual decisions -- it calls those same functions and prints what they
# already computed. Set OVERTAKE_DIAGNOSTICS = False (or delete this block
# and its call site inside update_lane_change) once the root cause is found.
OVERTAKE_DIAGNOSTICS = False
OVERTAKE_DIAGNOSTICS_PERIOD_S = 1.0  # throttle so the console stays readable
_last_overtake_diag_t = 0.0

# =====================================================================
# STOPPED-LEAD OVERTAKE PATH
# =====================================================================
# The normal moving-overtake trigger requires the ego to be moving and
# genuinely closing on a slower lead. Once the ego has already braked to a
# full stop directly behind a stationary lead, it can never satisfy the moving
# trigger again -- any throttle is immediately re-braked by
# vehicle_caution/vehicle_hard since the lead is still just as close. This
# left the ego permanently deadlocked at 0.00 m/s behind a stopped lead
# (observed: ego_speed and lead_distance pinned identically for 60+
# seconds) even with a verified SAFE adjacent lane sitting unused. These
# constants define a separate, narrower path for that exact case: only
# once both ego and lead have been mutually near-zero speed for a
# sustained hold time (so a brief traffic-flow sync, e.g. at a light,
# doesn't false-trigger) is a stopped lead treated as a blocking
# obstruction eligible for overtake. All other gates (lead_distance
# bounds, TTC, lane gap/pedestrian/junction/light checks) are unchanged
# and apply identically to this path.
STOPPED_LEAD_EGO_SPEED_MAX = 1.00  # m/s -- near-stopped ego can safely recover around a stopped lead
STOPPED_LEAD_SPEED_MAX = 0.3       # m/s -- lead considered "stopped"
STOPPED_LEAD_HOLD_SECONDS = 0.80   # brief stable hold before safe stopped-lead overtake
OVERTAKE_TIMEOUT_SECONDS = 12.0
OVERTAKE_STEER_MAX = 0.72
OVERTAKE_STEER_SLEW = 0.26

model = YOLO(YOLO_MODEL_PATH)
buffer = TrajectoryBuffer(max_len=10)

stop_event = threading.Event()
front_lock = threading.Lock()
front_latest = None
front_frame_id = -1
chase_lock = threading.Lock()
chase_latest = None

# Latest processed front frame and latest control state are consumed by the UI.
result_lock = threading.Lock()
latest_result = None

# Perception is read by the UI only as the newest completed snapshot. The
# camera itself never waits for YOLO.
perception_lock = threading.Lock()
latest_perception = {
    "overlay": None,
    "overlay_mask": None,
    "tracked_count": 0,
    "yolo_person_visible": False,
    "updated_at": 0.0,
    "image_id": -1,
    "state": "STARTING",
    "yolo_error": None,
    "track_active": False,
}

# Lane perception is independent from YOLO so a slow first YOLO/CUDA
# inference can never hide the visual lane overlay.
lane_lock = threading.Lock()
latest_lane = {
    "overlay": None,
    "overlay_mask": None,
    "updated_at": 0.0,
    "image_id": -1,
    "visual_mode": "NONE",
}

# Control/safety runs independently of YOLO. This preserves continuous ACC,
# AEB, traffic-light, pedestrian, route-steering and overtake control even
# while a perception frame is being processed.
control_snapshot_lock = threading.Lock()
control_snapshot = {
    "action": "CRUISE",
    "lane_change": "NONE",
    "steer": 0.0,
    "throttle": 0.0,
    "brake": 0.0,
    "lead_id": None,
    "lead_distance": float("inf"),
    "risk": 0.0,
    "distance": float("inf"),
    "ttc": float("inf"),
    "traffic_light": "NONE",
    "light_distance": float("inf"),
    "pedestrian_id": None,
    "pedestrian_distance": float("inf"),
    "pedestrian_lateral": 0.0,
    "pedestrian_conflict": False,
    "lane_status": "UNKNOWN",
    "lane_offset": 0.0,
    "lane_width": 0.0,
    "lane_heading_error": 0.0,
    "speed_kmh": 0.0,
    "vehicle_throttle": 0.0,
    "vehicle_brake": 0.0,
}

# CARLA handles
client = None
world = None
carla_map = None
vehicle = None
camera = None
chase_camera = None
obstacle_sensor = None
scenario_lead = None
actor_list = []

# Physical obstacle sensor is only an emergency backup, not the lead-vehicle
# detector. This prevents an adjacent vehicle from causing a blind full brake.
obstacle_lock = threading.Lock()
obstacle_state = {"distance": None, "actor_id": None, "last_update": 0.0}
OBSTACLE_TIMEOUT = 0.35
OBSTACLE_EMERGENCY_DISTANCE = 1.0

# Route cache
route_lock = threading.Lock()
route_points = deque()
ROUTE_STEP = 2.0
ROUTE_LOOKAHEAD = 8.0
ROUTE_MIN_POINTS = 35

# Control state
control_lock = threading.Lock()
control_state = {
    "action": "CRUISE",
    "lane_change": "NONE",
    "target_lane_id": None,
    "lane_change_until": 0.0,
    "last_steer": 0.0,
    "ped_mode": "NONE",
    "ped_blocked_since": 0.0,
    "ped_reverse_until": 0.0,
    "ped_escape_until": 0.0,
    # Vehicle overtaking state is kept separate from pedestrian escape state.
    "overtake_phase": "NONE",
    "overtake_original_lane_id": None,
    "overtake_target_lane_id": None,
    "overtake_direction": "NONE",
    "overtake_actor_id": None,
    "overtake_started": 0.0,
    "overtake_lane_entered": 0.0,
    # Tracks how long ego+lead have BOTH been near-zero speed, so a genuinely
    # stopped/blocking lead (not just a brief traffic-flow sync) can be
    # detected for the standstill overtake path below. 0.0 means "not
    # currently in a stopped pair".
    "stationary_pair_since": 0.0,
    # V20: stable physical lead association and overtake recovery.
    "overtake_candidate_id": None,
    "overtake_candidate_since": 0.0,
    "overtake_selected_lane_id": None,
    "overtake_selected_lane_since": 0.0,
    "last_lead_id": None,
    "last_lead_seen": 0.0,
    "overtake_cooldown_until": 0.0,
}

# ---------------------------------------------------------------------------
# Pedestrian / vulnerable-road-user safety
# ---------------------------------------------------------------------------
# Pedestrians are handled separately from vehicle overtaking.  The ego car
# must never steer toward a detected pedestrian merely because the opposite
# side looks more open.  A safe adjacent driving lane is required; otherwise
# the car brakes and waits until the pedestrian/corridor is clear.
PEDESTRIAN_MAX_DISTANCE = 32.0
PEDESTRIAN_FORWARD_CORRIDOR = 3.0
PEDESTRIAN_ROAD_OFFSET = 2.5
PEDESTRIAN_AVOID_MIN_DISTANCE = 16.0
PEDESTRIAN_REAR_GAP = 12.0
PEDESTRIAN_FRONT_GAP = 20.0
PEDESTRIAN_CLEAR_DISTANCE = 8.0
PEDESTRIAN_CLEAR_LATERAL = 3.8
PEDESTRIAN_LANE_MARGIN = 1.8
PEDESTRIAN_LANE_CHANGE_TIMEOUT = 5.0
# Hard safety envelope: inside this range we never swerve around a pedestrian.
PEDESTRIAN_HARD_BRAKE_DISTANCE = 18.0
PEDESTRIAN_SIDE_FORBID_LATERAL = 0.65
PEDESTRIAN_STEER_MARGIN = 0.03
PEDESTRIAN_REVERSE_DELAY = 1.5
PEDESTRIAN_REVERSE_DURATION = 2.2
PEDESTRIAN_REVERSE_THROTTLE = 0.16
PEDESTRIAN_REVERSE_MIN_CLEARANCE = 5.0
PEDESTRIAN_REVERSE_STOP_CLEARANCE = 2.8
PEDESTRIAN_ESCAPE_MIN_DISTANCE = 9.0

# Vehicle-to-vehicle longitudinal safety
# Brake early for a stopped/slow lead instead of waiting until the bumper is close.
VEHICLE_CAUTION_DISTANCE = 18.0
VEHICLE_BRAKE_DISTANCE = 10.0
VEHICLE_HARD_BRAKE_DISTANCE = 4.5
VEHICLE_HARD_BRAKE_TTC = 1.2
VEHICLE_CAUTION_TTC = 2.4
MIN_FOLLOW_GAP = 8.0
FOLLOW_TIME_GAP = 1.6

# ---------------------------------------------------------------------------
# Physical lane-status estimator
# ---------------------------------------------------------------------------
# The old dashboard status came directly from the Hough/camera lane detector.
# That produced false DEPARTURE states when road markings, shadows, curves,
# junction paint, or sidewalks looked like lane lines.  CARLA already knows
# the actual driving lane, so status is now derived from the ego waypoint
# centerline with hysteresis.  Camera lane detection remains visual-only.
LANE_STATUS_KEEP_RATIO = 0.34
LANE_STATUS_CAUTION_RATIO = 0.50
LANE_STATUS_DEPART_RATIO = 0.62
LANE_STATUS_RECOVER_RATIO = 0.44
LANE_STATUS_HEADING_LIMIT = 32.0
LANE_STATUS_HYSTERESIS_SEC = 0.35

lane_status_lock = threading.Lock()
lane_status_state = {
    "status": "UNKNOWN",
    "since": 0.0,
    "offset": 0.0,
    "lane_width": 0.0,
    "heading_error": 0.0,
}

# FPS
processed_count = 0
fps_window_start = time.monotonic()
processed_fps = 0.0


def actor_alive(actor):
    try:
        return actor is not None and actor.is_alive
    except RuntimeError:
        return False


def speed_mps(actor):
    try:
        v = actor.get_velocity()
        return math.sqrt(v.x * v.x + v.y * v.y + v.z * v.z)
    except RuntimeError:
        return 0.0


def _yaw_diff(a, b):
    return (b - a + 180.0) % 360.0 - 180.0


def _same_direction(wp_a, wp_b):
    return wp_a is not None and wp_b is not None and wp_a.road_id == wp_b.road_id and wp_a.lane_id * wp_b.lane_id > 0


def init_route():
    if not actor_alive(vehicle):
        return
    try:
        start = carla_map.get_waypoint(vehicle.get_location(), project_to_road=True, lane_type=carla.LaneType.Driving)
    except RuntimeError:
        return
    if start is None:
        return
    with route_lock:
        route_points.clear()
        route_points.append(start.transform.location)
        wp = start
        for _ in range(90):
            nxt = wp.next(ROUTE_STEP)
            if not nxt:
                break
            if len(nxt) == 1:
                wp = nxt[0]
            else:
                wp = min(nxt, key=lambda n: abs(_yaw_diff(wp.transform.rotation.yaw, n.transform.rotation.yaw)))
            route_points.append(wp.transform.location)


def update_route():
    if not actor_alive(vehicle):
        return
    try:
        loc = vehicle.get_location()
        wp = carla_map.get_waypoint(loc, project_to_road=True, lane_type=carla.LaneType.Driving)
    except RuntimeError:
        return
    if wp is None:
        return
    # If the car is far from the cached route, regenerate from its current lane.
    with route_lock:
        if route_points:
            nearest = min(math.hypot(p.x - loc.x, p.y - loc.y) for p in route_points)
        else:
            nearest = float("inf")
    if nearest > 12.0:
        init_route()
        return
    with route_lock:
        while route_points and math.hypot(route_points[0].x - loc.x, route_points[0].y - loc.y) < 3.0:
            route_points.popleft()
        if len(route_points) < ROUTE_MIN_POINTS:
            base = carla_map.get_waypoint(loc, project_to_road=True, lane_type=carla.LaneType.Driving)
            if base is not None:
                # Rebuild from current road position to avoid stale/behind targets.
                route_points.clear()
                route_points.append(base.transform.location)
                cur = base
                for _ in range(80):
                    nxt = cur.next(ROUTE_STEP)
                    if not nxt:
                        break
                    cur = min(nxt, key=lambda n: abs(_yaw_diff(cur.transform.rotation.yaw, n.transform.rotation.yaw)))
                    route_points.append(cur.transform.location)


def physical_lane_status():
    """Return the ego vehicle's lane status from CARLA road geometry.

    Returns: (status, signed_offset_m, lane_width_m, heading_error_deg).
    The vehicle is considered inside its actual CARLA driving lane based on
    distance from the lane centerline. A small hysteresis prevents one noisy
    waypoint/frame from flipping KEEP <-> DEPARTURE.
    """
    if not actor_alive(vehicle) or carla_map is None:
        return "UNKNOWN", 0.0, 0.0, 0.0
    try:
        tr = vehicle.get_transform()
        wp = carla_map.get_waypoint(
            tr.location, project_to_road=True, lane_type=carla.LaneType.Driving
        )
    except RuntimeError:
        return "UNKNOWN", 0.0, 0.0, 0.0
    if wp is None or wp.lane_width <= 0.1:
        return "UNKNOWN", 0.0, 0.0, 0.0

    lane_yaw = math.radians(wp.transform.rotation.yaw)
    dx = tr.location.x - wp.transform.location.x
    dy = tr.location.y - wp.transform.location.y
    # Signed lateral offset in the lane's local frame.
    offset = -dx * math.sin(lane_yaw) + dy * math.cos(lane_yaw)
    width = float(wp.lane_width)
    heading_error = abs(_yaw_diff(wp.transform.rotation.yaw, tr.rotation.yaw))

    ratio = abs(offset) / max(width, 0.1)
    now = time.monotonic()

    with lane_status_lock:
        previous = lane_status_state["status"]

        # Junctions can legitimately have larger heading error while the car
        # is still centered in the lane. Do not call that lane departure.
        if ratio <= LANE_STATUS_KEEP_RATIO:
            candidate = "KEEP"
        elif ratio <= LANE_STATUS_CAUTION_RATIO:
            candidate = "CAUTION"
        elif ratio <= LANE_STATUS_DEPART_RATIO:
            candidate = "DEPARTURE" if heading_error > LANE_STATUS_HEADING_LIMIT else "CAUTION"
        else:
            candidate = "DEPARTURE"

        # Hysteresis: once DEPARTURE is asserted, require a meaningful return
        # toward the lane center before clearing it.
        if previous == "DEPARTURE" and ratio <= LANE_STATUS_RECOVER_RATIO:
            candidate = "KEEP" if ratio <= LANE_STATUS_KEEP_RATIO else "CAUTION"
        elif previous == "KEEP" and candidate == "DEPARTURE":
            # Do not flash a false departure for a single frame.
            if now - lane_status_state["since"] < LANE_STATUS_HYSTERESIS_SEC:
                candidate = "KEEP"

        if candidate != previous:
            lane_status_state["since"] = now
        lane_status_state["status"] = candidate
        lane_status_state["offset"] = offset
        lane_status_state["lane_width"] = width
        lane_status_state["heading_error"] = heading_error
        return candidate, offset, width, heading_error


def route_steering():
    """Conservative waypoint-based steering.

    The visual lane detector is display/perception only; steering is tied to
    CARLA road waypoints so false Hough lines cannot drive the ego vehicle
    onto a sidewalk or into another actor.
    """
    if not actor_alive(vehicle):
        return 0.0
    try:
        tr = vehicle.get_transform()
        cur = carla_map.get_waypoint(
            tr.location, project_to_road=True, lane_type=carla.LaneType.Driving
        )
    except RuntimeError:
        return 0.0
    if cur is None:
        return 0.0

    # Use a short look-ahead on the current lane. At junctions choose the
    # continuation whose heading changes least; this avoids sudden swings.
    candidates = cur.next(10.0)
    if not candidates:
        return 0.0
    heading = tr.rotation.yaw
    target = min(
        candidates,
        key=lambda wp: abs(_yaw_diff(heading, wp.transform.rotation.yaw))
    )
    return steering_to_world_point(target.transform.location, gain=1.25)

def steering_to_world_point(target_location, gain=1.8):
    try:
        tr = vehicle.get_transform()
    except RuntimeError:
        return 0.0
    yaw = math.radians(tr.rotation.yaw)
    dx = target_location.x - tr.location.x
    dy = target_location.y - tr.location.y
    local_x = dx * math.cos(yaw) + dy * math.sin(yaw)
    local_y = -dx * math.sin(yaw) + dy * math.cos(yaw)
    if local_x < 0.5:
        return 0.0
    angle = math.atan2(local_y, local_x)
    return float(np.clip(gain * angle, -0.75, 0.75))


def get_lead_vehicle(max_distance=80.0):
    """
    Select the most relevant vehicle ahead of the ego vehicle using CARLA
    world geometry. Same-lane vehicles are preferred; at junctions, a
    same-direction vehicle on a connected road can also be considered.

    This function deliberately does NOT depend on YOLO pixels for safety.
    """
    if not actor_alive(vehicle):
        return None, float("inf")

    try:
        ego_tr = vehicle.get_transform()
        ego_loc = ego_tr.location
        ego_wp = carla_map.get_waypoint(
            ego_loc,
            project_to_road=True,
            lane_type=carla.LaneType.Driving,
        )
    except RuntimeError:
        return None, float("inf")

    if ego_wp is None:
        return None, float("inf")

    yaw = math.radians(ego_tr.rotation.yaw)

    # CONTROLLED-SCENARIO ASSOCIATION FIX
    # -----------------------------------
    # The deterministic demo lead is created by CARLA itself, so its actor
    # identity is more reliable than waiting for YOLO/ByteTrack or a generic
    # nearest-vehicle search to rediscover it.  In the failing run the lead
    # was physically present at ~38 m, but the generic selector temporarily
    # returned None; the ego therefore waited until the lead was already near
    # the hard-collision envelope.  Prefer the known scenario actor when it
    # is still physically ahead, same-lane, same-direction, and inside the
    # normal lead corridor.  This does NOT bypass lane safety, TTC, AEB, or
    # any overtake gate; it only fixes actor association.
    if CONTROLLED_OVERTAKE_TEST and actor_alive(scenario_lead):
        try:
            sloc = scenario_lead.get_location()
            sdx = sloc.x - ego_loc.x
            sdy = sloc.y - ego_loc.y
            sforward = sdx * math.cos(yaw) + sdy * math.sin(yaw)
            slateral = -sdx * math.sin(yaw) + sdy * math.cos(yaw)
            sheading = abs(_yaw_diff(ego_tr.rotation.yaw, scenario_lead.get_transform().rotation.yaw))
            swp = carla_map.get_waypoint(
                sloc, project_to_road=True, lane_type=carla.LaneType.Driving
            )
            if (
                swp is not None
                and sforward > 3.0
                and sforward <= max_distance
                and abs(slateral) <= 4.0
                and sheading <= 40.0
                and (
                    # Normal case: exact same CARLA road + lane.
                    (swp.road_id == ego_wp.road_id and swp.lane_id == ego_wp.lane_id)
                    # Robust case: CARLA has transitioned road_id/lane_id at
                    # a junction/road seam, but the controlled lead is still
                    # physically ahead, aligned, and on a same-direction
                    # driving lane. This remains a physical association
                    # check; it does not bypass TTC/AEB/lane safety.
                    or (
                        swp.lane_type == carla.LaneType.Driving
                        and ego_wp.lane_type == carla.LaneType.Driving
                        and swp.lane_id * ego_wp.lane_id > 0
                        and abs(slateral) <= 3.5
                    )
                )
            ):
                return scenario_lead, math.hypot(sdx, sdy)
        except RuntimeError:
            pass

    best_same_lane = None
    best_same_lane_dist = float("inf")
    best_fallback = None
    best_fallback_score = float("inf")

    for other in world.get_actors().filter("vehicle.*"):
        if other.id == vehicle.id or not actor_alive(other):
            continue

        try:
            other_tr = other.get_transform()
            loc = other_tr.location

            dx = loc.x - ego_loc.x
            dy = loc.y - ego_loc.y

            forward = dx * math.cos(yaw) + dy * math.sin(yaw)
            lateral = -dx * math.sin(yaw) + dy * math.cos(yaw)

            # Only vehicles genuinely ahead and inside the forward corridor.
            if forward <= 1.0 or forward > max_distance:
                continue
            if abs(lateral) > 3.2:
                continue

            distance = math.hypot(dx, dy)

            other_yaw = other_tr.rotation.yaw
            heading_error = abs(_yaw_diff(ego_tr.rotation.yaw, other_yaw))
            if heading_error > 40.0:
                continue

            other_wp = carla_map.get_waypoint(
                loc,
                project_to_road=True,
                lane_type=carla.LaneType.Driving,
            )
            if other_wp is None:
                continue

            # Best case: exactly the same road and lane.
            if (
                other_wp.road_id == ego_wp.road_id
                and other_wp.lane_id == ego_wp.lane_id
            ):
                if distance < best_same_lane_dist:
                    best_same_lane = other
                    best_same_lane_dist = distance
                continue

            # Junction/transition fallback. This prevents the lead selector
            # from disappearing just because CARLA changed road_id/lane_id
            # during an intersection or lane transition.
            connected_transition = (
                ego_wp.is_junction
                or other_wp.is_junction
                or other_wp.road_id == ego_wp.road_id
            )

            if connected_transition and abs(lateral) <= 2.2:
                score = distance + abs(lateral) * 4.0 + heading_error * 0.08
                if score < best_fallback_score:
                    best_fallback = other
                    best_fallback_score = score

        except RuntimeError:
            continue

    now = time.monotonic()
    if best_same_lane is not None:
        _remember_lead(best_same_lane, now)
        return best_same_lane, best_same_lane_dist

    if best_fallback is not None:
        try:
            d = ego_loc.distance(best_fallback.get_location())
            _remember_lead(best_fallback, now)
            return best_fallback, d
        except RuntimeError:
            pass

    recovered, recovered_dist = _recover_recent_lead()
    if recovered is not None:
        return recovered, recovered_dist

    return None, float("inf")

def adjacent_lane_candidates():
    if not actor_alive(vehicle):
        return []
    try:
        wp = carla_map.get_waypoint(vehicle.get_location(), project_to_road=True, lane_type=carla.LaneType.Driving)
    except RuntimeError:
        return []
    if wp is None:
        return []
    candidates = []
    for direction, candidate in (("LEFT", wp.get_left_lane()), ("RIGHT", wp.get_right_lane())):
        if candidate is None:
            continue
        if candidate.lane_type != carla.LaneType.Driving:
            continue
        if not _same_direction(candidate, wp):
            continue
        if candidate.lane_width < 2.5 or candidate.lane_width > 4.5:
            continue
        candidates.append((direction, candidate))
    return candidates


def _walker_is_on_driving_lane(walker, lane_wp):
    """Return True when a walker is physically close enough to a CARLA
    driving-lane centerline to be relevant to lane safety."""
    try:
        wloc = walker.get_location()
        return (
            wloc.distance(lane_wp.transform.location)
            <= PEDESTRIAN_ROAD_OFFSET
        )
    except RuntimeError:
        return False


# NOTE: an earlier, less strict `lane_is_safe(...)` used to be defined here
# (no junction/road-continuation check, different default gaps). Python
# silently kept only the later, stricter definition further below in this
# file (the one with the road-continuation probe), so the earlier version
# was 100% dead/unreachable code. It has been removed as duplicate-function
# cleanup; this is a no-op change -- every call site was already resolving
# to the surviving definition below at runtime.


def _lane_contains_pedestrian(target_wp, front_range=PEDESTRIAN_FRONT_GAP, rear_range=PEDESTRIAN_REAR_GAP):
    """Return True if a walker is occupying the target driving lane near ego."""
    try:
        ego_loc = vehicle.get_location()
        ego_tr = vehicle.get_transform()
    except RuntimeError:
        return True

    yaw = math.radians(ego_tr.rotation.yaw)
    for walker in world.get_actors().filter("walker.pedestrian.*"):
        if not actor_alive(walker):
            continue
        try:
            if not _walker_is_on_driving_lane(walker, target_wp):
                continue
            wloc = walker.get_location()
            dx = wloc.x - ego_loc.x
            dy = wloc.y - ego_loc.y
            longitudinal = dx * math.cos(yaw) + dy * math.sin(yaw)
            if -rear_range <= longitudinal <= front_range:
                return True
        except RuntimeError:
            continue
    return False


def get_pedestrian_hazard():
    """
    Find the nearest pedestrian who is actually inside/approaching the ego
    road corridor.

    Returns:
        (walker, distance, lateral)
    where lateral is positive to the vehicle's right and negative to its left.
    A pedestrian standing far on a sidewalk is ignored.
    """
    if not actor_alive(vehicle):
        return None, float("inf"), 0.0

    try:
        ego_tr = vehicle.get_transform()
    except RuntimeError:
        return None, float("inf"), 0.0

    yaw = math.radians(ego_tr.rotation.yaw)
    best = None
    best_distance = float("inf")
    best_lateral = 0.0

    for walker in world.get_actors().filter("walker.pedestrian.*"):
        if not actor_alive(walker):
            continue
        try:
            loc = walker.get_location()
            dx = loc.x - ego_tr.location.x
            dy = loc.y - ego_tr.location.y
            forward = dx * math.cos(yaw) + dy * math.sin(yaw)
            lateral = -dx * math.sin(yaw) + dy * math.cos(yaw)
            distance = math.hypot(dx, dy)

            if forward <= -1.0 or forward > PEDESTRIAN_MAX_DISTANCE:
                continue
            if abs(lateral) > PEDESTRIAN_FORWARD_CORRIDOR:
                continue

            # Require the pedestrian to be near an actual driving lane.
            wp = carla_map.get_waypoint(
                loc,
                project_to_road=True,
                lane_type=carla.LaneType.Driving,
            )
            if wp is None:
                continue

            road_offset = loc.distance(wp.transform.location)
            if road_offset > PEDESTRIAN_ROAD_OFFSET:
                continue

            # A walker is relevant when it is in the lane corridor or
            # moving toward it. This avoids braking for pedestrians well
            # outside the roadway.
            walker_speed = speed_mps(walker)
            approaching = walker_speed > 0.5
            if abs(lateral) <= PEDESTRIAN_FORWARD_CORRIDOR or approaching:
                if distance < best_distance:
                    best = walker
                    best_distance = distance
                    best_lateral = lateral
        except RuntimeError:
            continue

    return best, best_distance, best_lateral


def get_rear_clearance(max_distance=10.0):
    """Return nearest dynamic actor behind ego in a narrow reverse corridor."""
    if not actor_alive(vehicle):
        return 0.0
    try:
        tr = vehicle.get_transform()
        loc = tr.location
        yaw = math.radians(tr.rotation.yaw)
    except RuntimeError:
        return 0.0
    best = float(max_distance)
    actors = list(world.get_actors().filter("vehicle.*")) + list(world.get_actors().filter("walker.pedestrian.*"))
    for actor in actors:
        if not actor_alive(actor) or actor.id == vehicle.id:
            continue
        try:
            a = actor.get_location()
            dx, dy = a.x - loc.x, a.y - loc.y
            forward = dx * math.cos(yaw) + dy * math.sin(yaw)
            lateral = -dx * math.sin(yaw) + dy * math.cos(yaw)
            if forward >= 0.0 or abs(lateral) > 2.2:
                continue
            d = math.hypot(dx, dy)
            if d < best:
                best = d
        except RuntimeError:
            continue
    return best


def reset_pedestrian_state():
    with control_lock:
        control_state["ped_mode"] = "NONE"
        control_state["ped_blocked_since"] = 0.0
        control_state["ped_reverse_until"] = 0.0
        control_state["ped_escape_until"] = 0.0


def choose_safe_lane_for_pedestrian(pedestrian, pedestrian_lateral):
    """
    Select a same-direction adjacent driving lane that is safe for a
    pedestrian avoidance manoeuvre.

    If the pedestrian is already on one side, that side is explicitly
    forbidden. If both sides are blocked, return None so AEB/braking wins.
    """
    candidates = adjacent_lane_candidates()
    if not candidates:
        return None

    scored = []
    for direction, target_wp in candidates:
        # Never steer into the side occupied by the pedestrian.
        if pedestrian_lateral < -PEDESTRIAN_LANE_MARGIN and direction == "LEFT":
            continue
        if pedestrian_lateral > PEDESTRIAN_LANE_MARGIN and direction == "RIGHT":
            continue

        if not lane_is_safe(
            target_wp,
            front_gap=PEDESTRIAN_FRONT_GAP,
            rear_gap=PEDESTRIAN_REAR_GAP,
            pedestrian_front_gap=PEDESTRIAN_FRONT_GAP,
        ):
            continue

        try:
            ego_loc = vehicle.get_location()
            ego_yaw = math.radians(vehicle.get_transform().rotation.yaw)
        except RuntimeError:
            continue

        min_front = 60.0
        min_rear = 60.0

        # Score vehicle gaps.
        for other in world.get_actors().filter("vehicle.*"):
            if other.id == vehicle.id or not actor_alive(other):
                continue
            try:
                ow = carla_map.get_waypoint(
                    other.get_location(),
                    project_to_road=True,
                    lane_type=carla.LaneType.Driving,
                )
                if (
                    ow is None
                    or ow.road_id != target_wp.road_id
                    or ow.lane_id != target_wp.lane_id
                ):
                    continue

                dx = other.get_location().x - ego_loc.x
                dy = other.get_location().y - ego_loc.y
                longitudinal = dx * math.cos(ego_yaw) + dy * math.sin(ego_yaw)
                gap = math.hypot(dx, dy)

                if longitudinal >= 0:
                    min_front = min(min_front, gap)
                else:
                    min_rear = min(min_rear, gap)
            except RuntimeError:
                continue

        # Any nearby pedestrian makes this lane invalid (lane_is_safe already
        # checks this, but keep it explicit for robustness).
        if _lane_contains_pedestrian(
            target_wp,
            front_range=PEDESTRIAN_FRONT_GAP,
            rear_range=PEDESTRIAN_REAR_GAP,
        ):
            continue

        # Larger front/rear clearance is preferred.
        score = min_front + 0.5 * min_rear
        scored.append((score, direction, target_wp))

    return max(scored, key=lambda item: item[0]) if scored else None


def start_pedestrian_avoidance(pedestrian, pedestrian_distance, pedestrian_lateral):
    """Start a controlled same-direction lane change only when enough
    distance exists. Returns True when a safe target lane was selected."""
    if pedestrian is None or not math.isfinite(pedestrian_distance):
        return False

    if pedestrian_distance < PEDESTRIAN_AVOID_MIN_DISTANCE:
        return False

    selected = choose_safe_lane_for_pedestrian(pedestrian, pedestrian_lateral)
    if selected is None:
        return False

    _, direction, target_wp = selected

    with control_lock:
        control_state["target_lane_id"] = target_wp.lane_id
        control_state["lane_change_until"] = (
            time.monotonic() + PEDESTRIAN_LANE_CHANGE_TIMEOUT
        )
        control_state["lane_change"] = direction
        control_state["action"] = (
            "PEDESTRIAN AVOID " + direction
        )

    print(
        f"PEDESTRIAN -> avoid {direction} | "
        f"distance={pedestrian_distance:.1f}m | "
        f"lateral={pedestrian_lateral:.1f}m"
    )
    return True



def _lane_gap_metrics(target_wp):
    """Return conservative physical front/rear gaps for a candidate lane."""
    if target_wp is None or not actor_alive(vehicle):
        return 0.0, 0.0, 0.0
    try:
        target_loc = target_wp.transform.location
        target_yaw = math.radians(target_wp.transform.rotation.yaw)
        lane_width = max(float(target_wp.lane_width), 2.5)
    except RuntimeError:
        return 0.0, 0.0, 0.0

    front, rear = 80.0, 80.0
    for other in world.get_actors().filter("vehicle.*"):
        if other.id == vehicle.id or not actor_alive(other):
            continue
        try:
            loc = other.get_location()
            dx, dy = loc.x - target_loc.x, loc.y - target_loc.y
            longitudinal = dx * math.cos(target_yaw) + dy * math.sin(target_yaw)
            lateral = -dx * math.sin(target_yaw) + dy * math.cos(target_yaw)
            if abs(lateral) > max(1.65, lane_width * 0.62):
                continue
            gap = math.hypot(dx, dy)
            if longitudinal >= 0.0:
                front = min(front, gap)
            else:
                rear = min(rear, gap)
        except RuntimeError:
            continue
    return front, rear, min(front, rear)

def _lane_has_pedestrian(target_wp, front_range=28.0, rear_range=16.0):
    """Strict pedestrian exclusion for an overtaking target lane."""
    if target_wp is None:
        return True
    try:
        ego = vehicle.get_location()
        yaw = math.radians(vehicle.get_transform().rotation.yaw)
    except RuntimeError:
        return True
    for walker in world.get_actors().filter("walker.pedestrian.*"):
        if not actor_alive(walker):
            continue
        try:
            if not _walker_is_on_driving_lane(walker, target_wp):
                continue
            dx = walker.get_location().x - ego.x
            dy = walker.get_location().y - ego.y
            longitudinal = dx * math.cos(yaw) + dy * math.sin(yaw)
            if -rear_range <= longitudinal <= front_range:
                return True
        except RuntimeError:
            continue
    return False


def _lane_has_oncoming_threat(target_wp, max_distance=60.0, min_ttc=4.0):
    """Reject an overtake lane when an opposing/closing vehicle is approaching.

    This uses physical CARLA velocity projected onto the candidate lane
    direction. For an oncoming vehicle, its longitudinal velocity is negative,
    so relative closing speed becomes ego_speed + oncoming_speed rather than
    incorrectly subtracting the two speed magnitudes.
    """
    if target_wp is None or not actor_alive(vehicle):
        return True
    try:
        target_tr = target_wp.transform
        target_loc = target_tr.location
        yaw = math.radians(target_tr.rotation.yaw)
        ego_v = vehicle.get_velocity()
        ego_long = ego_v.x * math.cos(yaw) + ego_v.y * math.sin(yaw)
    except RuntimeError:
        return True

    for other in world.get_actors().filter("vehicle.*"):
        if other.id == vehicle.id or not actor_alive(other):
            continue
        try:
            loc = other.get_location()
            dx, dy = loc.x - target_loc.x, loc.y - target_loc.y
            longitudinal = dx * math.cos(yaw) + dy * math.sin(yaw)
            lateral = -dx * math.sin(yaw) + dy * math.cos(yaw)
            if longitudinal <= 0.0 or longitudinal > max_distance:
                continue
            if abs(lateral) > max(1.65, float(target_wp.lane_width) * 0.62):
                continue

            heading_error = abs(_yaw_diff(
                target_tr.rotation.yaw,
                other.get_transform().rotation.yaw,
            ))
            if heading_error < 75.0:
                continue

            other_v = other.get_velocity()
            other_long = other_v.x * math.cos(yaw) + other_v.y * math.sin(yaw)
            closing = ego_long - other_long
            if closing <= 0.2:
                continue

            gap = math.hypot(dx, dy)
            if gap / closing <= min_ttc:
                return True
        except RuntimeError:
            continue

    return False


def lane_is_safe(target_wp, front_gap=30.0, rear_gap=18.0, pedestrian_front_gap=28.0):
    """Conservative target-lane gate for automatic overtaking.

    A lane must have enough room both ahead and behind and must be free of
    pedestrians. The target lane must also have road continuation so the ego
    vehicle never changes into a lane that immediately terminates.
    """
    if target_wp is None or not actor_alive(vehicle):
        return False
    try:
        ego_wp = carla_map.get_waypoint(
            vehicle.get_location(), project_to_road=True, lane_type=carla.LaneType.Driving
        )
        if ego_wp is None or target_wp.is_junction:
            return False
        if not _same_direction(target_wp, ego_wp):
            return False
        # Require a meaningful continuation of the target lane.
        probe = target_wp
        for _ in range(4):
            if probe.is_junction:
                return False
            nxt = probe.next(8.0)
            if not nxt:
                return False
            probe = nxt[0]
            if probe.is_junction:
                return False
    except RuntimeError:
        return False

    front, rear, _ = _lane_gap_metrics(target_wp)
    if front < front_gap or rear < rear_gap:
        return False
    # Opposing traffic in the target lane is a dynamic collision threat even
    # when its raw distance is larger than the normal static front-gap gate.
    if _lane_has_oncoming_threat(target_wp, max_distance=60.0, min_ttc=4.0):
        return False
    if _lane_has_pedestrian(target_wp, front_range=pedestrian_front_gap, rear_range=16.0):
        return False
    return True


def _diagnose_lane(direction, candidate, ego_wp, front_gap=12.0, rear_gap=10.0, pedestrian_front_gap=28.0):
    """TEMPORARY DIAGNOSTICS ONLY -- read-only mirror of the exact checks in
    adjacent_lane_candidates() + lane_is_safe(), used ONLY to print out which
    specific check rejects a candidate overtake lane. This function makes no
    decisions of its own, changes no state, and is not called from anywhere
    in the actual control path -- only from the diagnostic print block inside
    update_lane_change(). Safe to delete once the overtake issue is found.
    """
    info = {"safe": False, "summary": ""}
    if candidate is None:
        info["summary"] = "exists=NO -> REJECTED (no adjacent lane on this side)"
        return info

    lane_type_ok = candidate.lane_type == carla.LaneType.Driving
    same_dir = _same_direction(candidate, ego_wp)
    width_ok = 2.5 <= candidate.lane_width <= 4.5
    junction = bool(candidate.is_junction)

    continuation = False
    front, rear = float("nan"), float("nan")
    pedestrian_present = None

    if lane_type_ok and same_dir and width_ok and not junction:
        try:
            probe = candidate
            continuation = True
            for _ in range(4):
                nxt = probe.next(8.0)
                if not nxt:
                    continuation = False
                    break
                probe = nxt[0]
                if probe.is_junction:
                    break
        except RuntimeError:
            continuation = False

        if continuation:
            front, rear, _ = _lane_gap_metrics(candidate)
            pedestrian_present = _lane_has_pedestrian(
                candidate, front_range=pedestrian_front_gap, rear_range=16.0
            )

    if not lane_type_ok:
        reason = "NOT A DRIVING LANE"
    elif not same_dir:
        reason = "OPPOSITE/CROSS DIRECTION"
    elif not width_ok:
        reason = f"LANE WIDTH OUT OF RANGE ({candidate.lane_width:.2f}m)"
    elif junction:
        reason = "TARGET LANE IS A JUNCTION"
    elif not continuation:
        reason = "NO ROAD CONTINUATION"
    elif front < front_gap:
        reason = f"FRONT GAP TOO SMALL ({front:.1f}m < {front_gap:.1f}m)"
    elif rear < rear_gap:
        reason = f"REAR GAP TOO SMALL ({rear:.1f}m < {rear_gap:.1f}m)"
    elif _lane_has_oncoming_threat(candidate, max_distance=60.0, min_ttc=4.0):
        reason = "ONCOMING VEHICLE / LOW TTC"
    elif pedestrian_present:
        reason = "PEDESTRIAN IN TARGET LANE"
    else:
        reason = "SAFE"

    info["safe"] = (reason == "SAFE")
    front_txt = f"{front:.1f}m" if math.isfinite(front) else "n/a"
    rear_txt = f"{rear:.1f}m" if math.isfinite(rear) else "n/a"
    ped_txt = "n/a" if pedestrian_present is None else str(pedestrian_present)
    info["summary"] = (
        f"exists=YES same_dir={same_dir} continuation={continuation} "
        f"front_gap={front_txt} rear_gap={rear_txt} pedestrian={ped_txt} "
        f"-> {reason}"
    )
    return info


def choose_safe_lane(front_gap=30.0, rear_gap=18.0):
    candidates = adjacent_lane_candidates()
    if not candidates:
        return None

    scored = []
    for direction, wp in candidates:
        if not lane_is_safe(wp, front_gap=front_gap, rear_gap=rear_gap):
            continue
        front, rear, minimum = _lane_gap_metrics(wp)
        # Prefer the lane with the larger limiting gap. Add a small front-gap
        # preference because the manoeuvre is an overtake, not a lane merge.
        score = minimum * 2.0 + front * 0.35 + rear * 0.20
        scored.append((score, direction, wp))
    return max(scored, key=lambda x: x[0]) if scored else None


def _clear_overtake_state():
    with control_lock:
        control_state["overtake_phase"] = "NONE"
        control_state["overtake_original_lane_id"] = None
        control_state["overtake_target_lane_id"] = None
        control_state["overtake_direction"] = "NONE"
        control_state["overtake_actor_id"] = None
        control_state["overtake_started"] = 0.0
        control_state["overtake_lane_entered"] = 0.0
        control_state["stationary_pair_since"] = 0.0
        control_state["overtake_candidate_id"] = None
        control_state["overtake_candidate_since"] = 0.0
        control_state["overtake_selected_lane_id"] = None
        control_state["overtake_selected_lane_since"] = 0.0


def _start_overtake(lead, selected=None):
    try:
        ego_wp = carla_map.get_waypoint(
            vehicle.get_location(), project_to_road=True, lane_type=carla.LaneType.Driving
        )
    except RuntimeError:
        return False
    if ego_wp is None or ego_wp.is_junction:
        return False

    if selected is None:
        selected = choose_safe_lane(front_gap=18.0, rear_gap=14.0)
    if selected is None:
        return False

    _, direction, target_wp = selected
    now = time.monotonic()
    with control_lock:
        control_state["overtake_phase"] = "OUTBOUND"
        control_state["overtake_original_lane_id"] = ego_wp.lane_id
        control_state["overtake_target_lane_id"] = target_wp.lane_id
        control_state["overtake_direction"] = direction
        control_state["overtake_actor_id"] = lead.id if lead is not None else None
        control_state["overtake_started"] = now
        control_state["overtake_lane_entered"] = 0.0
        control_state["lane_change"] = direction
        control_state["stationary_pair_since"] = 0.0
    print(f"OVERTAKE START -> {direction} | target lane={target_wp.lane_id} | lead={lead.id if lead else None}")
    return True


def _overtake_target_waypoint(lane_id):
    if lane_id is None or not actor_alive(vehicle):
        return None
    try:
        cur = carla_map.get_waypoint(
            vehicle.get_location(), project_to_road=True, lane_type=carla.LaneType.Driving
        )
    except RuntimeError:
        return None
    if cur is None:
        return None
    if cur.lane_id == lane_id:
        return cur
    for wp in (cur.get_left_lane(), cur.get_right_lane()):
        if wp is not None and wp.lane_id == lane_id and _same_direction(wp, cur):
            return wp
    return None


def _actor_is_passed(actor_id):
    if actor_id is None or world is None or not actor_alive(vehicle):
        return False
    try:
        actor = world.get_actor(actor_id)
        if actor is None or not actor_alive(actor):
            return True
        ego_tr = vehicle.get_transform()
        loc = actor.get_location()
        dx = loc.x - ego_tr.location.x
        dy = loc.y - ego_tr.location.y
        yaw = math.radians(ego_tr.rotation.yaw)
        forward = dx * math.cos(yaw) + dy * math.sin(yaw)
        lateral = -dx * math.sin(yaw) + dy * math.cos(yaw)
        # Once the ego is safely in the adjacent lane, the original lead is
        # considered passed as soon as the ego's reference point is clearly
        # ahead of the lead.  Requiring -6 m made a real lane change wait too
        # long for a full vehicle-length separation and could hit the
        # overtake timeout even though the pass was already physically valid.
        return forward < -2.5 and abs(lateral) < 10.0
    except RuntimeError:
        return True


def _stable_overtake_lead(lead, now, min_hold=0.30):
    """Require the same physical lead actor to persist before starting an overtake.

    CARLA ground-truth association is authoritative, but transient road/lane
    transitions can briefly make the selector return None. V20 therefore
    remembers the last physical lead for a short interval and requires a
    continuous actor identity hold before authorising a new moving overtake.
    """
    if lead is None or not actor_alive(lead):
        return False
    actor_id = lead.id
    with control_lock:
        if control_state["overtake_candidate_id"] != actor_id:
            control_state["overtake_candidate_id"] = actor_id
            control_state["overtake_candidate_since"] = now
            return False
        since = control_state["overtake_candidate_since"]
    return since > 0.0 and (now - since) >= min_hold


def _remember_lead(lead, now):
    if lead is None or not actor_alive(lead):
        return
    with control_lock:
        control_state["last_lead_id"] = lead.id
        control_state["last_lead_seen"] = now


def _recover_recent_lead(max_age=3.00, max_distance=80.0):
    """Recover a recently selected physical lead across a transient selector miss."""
    now = time.monotonic()
    with control_lock:
        actor_id = control_state["last_lead_id"]
        seen = control_state["last_lead_seen"]
    if actor_id is None or seen <= 0.0 or now - seen > max_age or world is None:
        return None, float("inf")
    try:
        actor = world.get_actor(actor_id)
        if actor is None or not actor_alive(actor) or not actor_alive(vehicle):
            return None, float("inf")
        ego_tr = vehicle.get_transform()
        tr = actor.get_transform()
        yaw = math.radians(ego_tr.rotation.yaw)
        dx = tr.location.x - ego_tr.location.x
        dy = tr.location.y - ego_tr.location.y
        forward = dx * math.cos(yaw) + dy * math.sin(yaw)
        lateral = -dx * math.sin(yaw) + dy * math.cos(yaw)
        if 1.0 < forward <= max_distance and abs(lateral) <= 3.5:
            if abs(_yaw_diff(ego_tr.rotation.yaw, tr.rotation.yaw)) <= 40.0:
                return actor, math.hypot(dx, dy)
    except RuntimeError:
        pass
    return None, float("inf")


def _target_lane_dynamic_clear(target_wp, front_min=5.0, rear_min=4.5, min_ttc=1.15):
    """Check an already-started lane change against dynamic target-lane traffic.

    The initial overtake gate is intentionally conservative (12/10 m at low
    speed). Once the manoeuvre has started, however, using that same static
    gate causes harmless 6-10 m gap fluctuations to abort the manoeuvre and
    trap the ego back in the original queue. During an active lane change we
    therefore use a true emergency envelope: hard-close gaps or an imminent
    target-lane TTC abort; otherwise the manoeuvre is allowed to complete.
    """
    if target_wp is None or not actor_alive(vehicle):
        return False
    try:
        ego_tr = vehicle.get_transform()
        target_tr = target_wp.transform
        yaw = math.radians(target_tr.rotation.yaw)
        ego_loc = ego_tr.location
    except RuntimeError:
        return False

    front = rear = 80.0
    ego_speed = speed_mps(vehicle)
    for other in world.get_actors().filter("vehicle.*"):
        if other.id == vehicle.id or not actor_alive(other):
            continue
        try:
            loc = other.get_location()
            dx = loc.x - ego_loc.x
            dy = loc.y - ego_loc.y
            longitudinal = dx * math.cos(yaw) + dy * math.sin(yaw)
            lateral = -dx * math.sin(yaw) + dy * math.cos(yaw)
            if abs(lateral) > max(1.65, float(target_wp.lane_width) * 0.62):
                continue
            gap = math.hypot(dx, dy)
            other_v = other.get_velocity()
            ego_v = vehicle.get_velocity()
            ego_long = ego_v.x * math.cos(yaw) + ego_v.y * math.sin(yaw)
            other_long = other_v.x * math.cos(yaw) + other_v.y * math.sin(yaw)
            if longitudinal >= 0.0:
                front = min(front, gap)
                # Physical relative closing speed. For an oncoming vehicle
                # other_long is negative, so this becomes ego + oncoming
                # speed instead of the old (and unsafe) ego - |other|.
                closing = ego_long - other_long
                if closing > 0.2 and gap / closing < min_ttc:
                    return False
            else:
                rear = min(rear, gap)
                closing_from_rear = other_long - ego_long
                if closing_from_rear > 0.2 and gap / closing_from_rear < min_ttc:
                    return False
        except RuntimeError:
            continue

    if front < front_min or rear < rear_min:
        return False
    if _lane_has_oncoming_threat(target_wp, max_distance=55.0, min_ttc=max(1.8, min_ttc)):
        return False
    if _lane_has_pedestrian(target_wp, front_range=20.0, rear_range=10.0):
        return False
    return True


def _recover_overtake_candidate(max_age=3.0, max_distance=80.0):
    """Recover the exact lead actor being evaluated for an overtake.

    During the low-speed/stopped-lead case the generic physical lead selector
    can briefly lose the actor while CARLA updates its waypoint/road segment.
    The overtake state machine must not forget the same actor in that short
    window, otherwise lane selection happens once and the next frame reports
    ``lead=None`` before the stable-start timer can complete.
    """
    now = time.monotonic()
    with control_lock:
        actor_id = control_state.get("overtake_candidate_id")
        candidate_since = control_state.get("overtake_candidate_since", 0.0)
    if actor_id is None or candidate_since <= 0.0 or now - candidate_since > max_age:
        return None, float("inf")
    if world is None or not actor_alive(vehicle):
        return None, float("inf")
    try:
        actor = world.get_actor(actor_id)
        if actor is None or not actor_alive(actor):
            return None, float("inf")
        ego_tr = vehicle.get_transform()
        tr = actor.get_transform()
        yaw = math.radians(ego_tr.rotation.yaw)
        dx = tr.location.x - ego_tr.location.x
        dy = tr.location.y - ego_tr.location.y
        forward = dx * math.cos(yaw) + dy * math.sin(yaw)
        lateral = -dx * math.sin(yaw) + dy * math.cos(yaw)
        heading_error = abs(_yaw_diff(ego_tr.rotation.yaw, tr.rotation.yaw))
        if 2.0 < forward <= max_distance and abs(lateral) <= 4.5 and heading_error <= 50.0:
            return actor, math.hypot(dx, dy)
    except RuntimeError:
        pass
    return None, float("inf")


def update_lane_change(lead, lead_distance, ttc, ped_conflict=False, light_state="NONE", light_distance=float("inf"), vehicle_hard=False, physical_ttc=float("inf"), front_actor=None, front_dist=float("inf"), front_lat=0.0):
    """Stateful, safety-gated vehicle overtake.

    Phases: OUTBOUND -> PASS -> RETURN. The target is never cleared merely
    because the ego enters the adjacent lane; it stays active until the lead
    has been passed and the original lane is verified safe.
    """
    now = time.monotonic()
    with control_lock:
        phase = control_state["overtake_phase"]
        target_lane_id = control_state["overtake_target_lane_id"]
        original_lane_id = control_state["overtake_original_lane_id"]
        actor_id = control_state["overtake_actor_id"]
        direction = control_state["overtake_direction"]
        started = control_state["overtake_started"]

    # Any vulnerable-road-user conflict or signal stop cancels starting a new
    # overtake. An already active manoeuvre is allowed only to finish safely
    # toward its current target; it never gets extended into a new manoeuvre.
    if phase == "NONE":
        # ---- TEMPORARY DIAGNOSTICS (overtake-only, read-only) ----
        # Prints why the overtake gate does/doesn't fire. Throttled to
        # OVERTAKE_DIAGNOSTICS_PERIOD_S so it doesn't flood the console at
        # 15Hz. This block never returns/raises and never touches
        # control_state, ped_conflict, light_state, or any threshold below
        # it -- the real gates immediately after this block are byte-for-byte
        # unchanged. Delete the whole block (down to the END marker) once
        # the overtake issue is diagnosed.
        if OVERTAKE_DIAGNOSTICS:
            global _last_overtake_diag_t
            _diag_now = time.monotonic()
            if _diag_now - _last_overtake_diag_t >= OVERTAKE_DIAGNOSTICS_PERIOD_S:
                _last_overtake_diag_t = _diag_now
                try:
                    _diag_ego_wp = carla_map.get_waypoint(
                        vehicle.get_location(), project_to_road=True,
                        lane_type=carla.LaneType.Driving
                    ) if actor_alive(vehicle) else None
                except RuntimeError:
                    _diag_ego_wp = None
                _diag_ego_speed = speed_mps(vehicle) if actor_alive(vehicle) else 0.0
                _diag_junction = _diag_ego_wp.is_junction if _diag_ego_wp is not None else None

                print("\n" + "-" * 60)
                print("OVERTAKE CHECK")
                print(f"  ego_speed     : {_diag_ego_speed:.2f} m/s")
                print(f"  junction      : {_diag_junction}")
                print(f"  ped_conflict  : {ped_conflict}")
                lt_txt = f"{light_distance:.1f}m" if math.isfinite(light_distance) else "inf"
                print(f"  light_state   : {light_state} (distance={lt_txt})")

                if lead is None:
                    print("  lead          : NONE (no lead vehicle detected)")
                    print("OVERTAKE DECISION: NO SAFE LANE (no lead to overtake)")
                    print("-" * 60)
                else:
                    _diag_lead_speed = speed_mps(lead)
                    _diag_closing = _diag_ego_speed - _diag_lead_speed
                    ld_txt = f"{lead_distance:.2f} m" if math.isfinite(lead_distance) else "inf"
                    ttc_txt = f"{ttc:.2f} s" if math.isfinite(ttc) else "inf"
                    print(f"  lead_speed    : {_diag_lead_speed:.2f} m/s")
                    print(f"  lead_distance : {ld_txt}")
                    print(f"  closing_speed : {_diag_closing:.2f} m/s")
                    print(f"  ttc           : {ttc_txt}")

                    if _diag_ego_wp is None:
                        print("LEFT LANE : ego waypoint unavailable")
                        print("RIGHT LANE: ego waypoint unavailable")
                        print("OVERTAKE DECISION: NO SAFE LANE (no ego waypoint)")
                    else:
                        _diag_left = _diagnose_lane("LEFT", _diag_ego_wp.get_left_lane(), _diag_ego_wp)
                        _diag_right = _diagnose_lane("RIGHT", _diag_ego_wp.get_right_lane(), _diag_ego_wp)
                        print(f"LEFT LANE : {_diag_left['summary']}")
                        print(f"RIGHT LANE: {_diag_right['summary']}")

                        if _diag_left["safe"] or _diag_right["safe"]:
                            _diag_front_gate = 12.0 if _diag_ego_speed < 3.0 else 18.0
                            _diag_rear_gate = 10.0 if _diag_ego_speed < 3.0 else 14.0
                            _diag_choice = choose_safe_lane(
                                front_gap=_diag_front_gate,
                                rear_gap=_diag_rear_gate
                            )
                            if _diag_choice is not None:
                                print(
                                    f"OVERTAKE DECISION: SELECT {_diag_choice[1]} "
                                    f"(gates={_diag_front_gate:.0f}m/{_diag_rear_gate:.0f}m)"
                                )
                            else:
                                print(
                                    f"OVERTAKE DECISION: NO SAFE LANE "
                                    f"(gates={_diag_front_gate:.0f}m/{_diag_rear_gate:.0f}m)"
                                )
                        else:
                            print("OVERTAKE DECISION: NO SAFE LANE")
                    print("-" * 60)
        # ---- END TEMPORARY DIAGNOSTICS ----

        if ped_conflict:
            return False
        # Only RED/YELLOW forbid starting a new overtake near a signal, per
        # the "never overtake through red lights" safety rule. A GREEN light
        # ahead is not itself a reason to block an otherwise-safe overtake;
        # it was previously included here by mistake, which meant any green
        # light within 45m silently disabled overtaking with no safety benefit.
        if light_state in {"RED", "YELLOW"} and math.isfinite(light_distance) and light_distance < 45.0:
            return False
        # Preserve the same physical lead through a short CARLA waypoint/road
        # transition. This is especially important for a stopped/near-stopped
        # lead: the previous build could select a safe lane, then lose the lead
        # for one frame, reset the stable gate, and leave ego creeping at 0-0.5
        # m/s forever without ever issuing OVERTAKE START.
        if lead is None or not math.isfinite(lead_distance):
            recovered_lead, recovered_distance = _recover_overtake_candidate()
            if recovered_lead is None:
                recovered_lead, recovered_distance = _recover_recent_lead()
            if recovered_lead is None:
                return False
            lead = recovered_lead
            lead_distance = recovered_distance
        try:
            ego_wp = carla_map.get_waypoint(
                vehicle.get_location(), project_to_road=True, lane_type=carla.LaneType.Driving
            )
            if ego_wp is None or ego_wp.is_junction:
                return False
            probe = ego_wp
            for _ in range(2):
                if probe.is_junction:
                    return False
                nxt = probe.next(8.0)
                if not nxt:
                    return False
                probe = nxt[0]
                if probe.is_junction:
                    return False
        except RuntimeError:
            return False

        ego_speed = speed_mps(vehicle)
        lead_speed = speed_mps(lead)
        closing = ego_speed - lead_speed
        # Overtake only when the lead is genuinely slowing/blocking the ego.
        #
        # THRESHOLD NOTE (data-driven, see diagnostic run): in dense traffic
        # (TRAFFIC_VEHICLE_COUNT=35) with frequent junctions interrupting
        # acceleration, the ego's observed cruising speed plateaus at ~4.60
        # m/s and NEVER reaches 5.0 m/s -- confirmed across an entire
        # diagnostic session where OVERTAKE DECISION correctly resolved to
        # SELECT LEFT (lane genuinely safe, 80m clear) but the manoeuvre
        # still could not start because this single gate never passed. 3.5
        # m/s keeps a full 1.1 m/s margin below that observed plateau (so a
        # near-stop/creep still cannot trigger an overtake) while actually
        # being reachable in this scenario.
        # Start early enough to complete the lateral manoeuvre before TTC
        # becomes critical. Safety gates remain unchanged: closing speed, TTC,
        # lane gaps, pedestrians, signals and junction protection still apply.
        # Do not start a lane change while the ego is still crawling.
        # The previous 0.8 m/s gate allowed an overtake to begin at ~0.67–0.8
        # m/s; in the observed run the target-lane vehicle then closed the
        # available gap before the ego had meaningful lateral progress.
        low_speed_overtake = ego_speed >= 1.20 and closing >= 0.35
        normal_speed_overtake = ego_speed >= 2.50 and closing >= 0.55
        moving_approach_ready = low_speed_overtake or normal_speed_overtake
        # Use the same timestamp for lead stability and the stopped-lead
        # hold timer.  The previous build referenced now_mono before it was
        # assigned, which could make the overtake gate fail/crash whenever a
        # physical lead was present.
        now_mono = time.monotonic()
        stable_lead_ready = _stable_overtake_lead(lead, now_mono, min_hold=0.30)

        # STOPPED-LEAD PATH (see STOPPED_LEAD_* constants above): handles a
        # lead that is fully stationary and blocking the lane, which the
        # moving-approach check above can never satisfy once the ego has
        # already braked to 0.00 m/s directly behind it.
        mutually_stopped = (
            ego_speed <= STOPPED_LEAD_EGO_SPEED_MAX
            and lead_speed <= STOPPED_LEAD_SPEED_MAX
        )
        with control_lock:
            if mutually_stopped:
                if control_state["stationary_pair_since"] == 0.0:
                    control_state["stationary_pair_since"] = now_mono
                stationary_since = control_state["stationary_pair_since"]
            else:
                control_state["stationary_pair_since"] = 0.0
                stationary_since = 0.0
        stopped_lead_ready = (
            mutually_stopped
            and stationary_since > 0.0
            and (now_mono - stationary_since) >= STOPPED_LEAD_HOLD_SECONDS
        )

        # lead_distance bounds, TTC, and every downstream lane/pedestrian/
        # junction/light check are unchanged and apply identically to
        # either path.
        # Moving overtakes need a larger starting envelope than the old 9 m
        # minimum. This prevents the previous late-start pattern where the
        # vehicle only began changing lanes at ~13–14 m and immediately hit
        # critical TTC. Stopped-lead recovery keeps its own 9 m lower bound.
        moving_distance_ready = 14.0 <= lead_distance <= 40.0
        stopped_distance_ready = 9.0 <= lead_distance <= 32.0
        if not (stopped_distance_ready if stopped_lead_ready else moving_distance_ready):
            return False
        if not (stopped_lead_ready or (moving_approach_ready and stable_lead_ready)):
            return False
        if math.isfinite(ttc) and ttc < 2.0:
            return False
        with control_lock:
            if now_mono < control_state["overtake_cooldown_until"]:
                return False

        # Low-speed stopped-lead recovery: when ego and lead are both stopped
        # for the full hold period, a 12m front / 10m rear adjacent-lane gap
        # is sufficient for a cautious escape. This is NOT used for moving
        # overtakes; moving overtakes retain the larger 18m/14m gate above.
        # Speed-dependent target-lane gate. Low-speed recovery uses 12/10 m;
        # once the ego is moving at normal speed we retain the stronger 18/14 m
        # requirement. This matches the physical risk envelope instead of
        # requiring a 24/16 m gap from a 1 m/s vehicle.
        if stopped_lead_ready:
            selected = choose_safe_lane(front_gap=14.0, rear_gap=10.0)
        elif ego_speed < 3.0:
            selected = choose_safe_lane(front_gap=20.0, rear_gap=12.0)
        else:
            selected = choose_safe_lane(front_gap=24.0, rear_gap=16.0)

        # Require the chosen target lane to remain safe for a short, stable
        # window before committing. This prevents the exact failure seen in
        # the supplied run: RIGHT was reported SAFE at ~16.7 m, the overtake
        # started immediately, and a target-lane vehicle reduced the gap to
        # ~8 m before the ego had completed the lateral move.
        with control_lock:
            if selected is None:
                control_state["overtake_selected_lane_id"] = None
                control_state["overtake_selected_lane_since"] = 0.0
                return False
            selected_lane_id = selected[2].lane_id
            if control_state["overtake_selected_lane_id"] != selected_lane_id:
                control_state["overtake_selected_lane_id"] = selected_lane_id
                control_state["overtake_selected_lane_since"] = now_mono
                return False
            selected_since = control_state["overtake_selected_lane_since"]

        if selected_since <= 0.0 or now_mono - selected_since < 0.25:
            return False
        return _start_overtake(lead, selected=selected)

    # Hard timeout: finish the manoeuvre or abort to a safe recovery rather
    # than wandering indefinitely in the adjacent lane.  ABORT is excluded
    # here deliberately: once ABORT has been entered it is handled exactly
    # once by the ABORT branch below.  Re-running this timeout while already
    # in ABORT was the direct cause of the repeated
    # "OVERTAKE TIMEOUT -> ABORT" spam and the stale controller state.
    if phase in {"OUTBOUND", "PASS"} and started and now - started > OVERTAKE_TIMEOUT_SECONDS:
        if original_lane_id is not None and lane_is_safe(_overtake_target_waypoint(original_lane_id), front_gap=18.0, rear_gap=12.0):
            with control_lock:
                control_state["overtake_phase"] = "RETURN"
                control_state["overtake_target_lane_id"] = original_lane_id
                control_state["lane_change"] = "RETURN"
            print("OVERTAKE TIMEOUT -> SAFE RETURN")
            return True
        # Do not leave the state machine stuck in OUTBOUND/PASS forever.
        # ABORT keeps the vehicle in the current lane until the original lane
        # is physically safe, then RETURN handles the lane restoration.
        with control_lock:
            control_state["overtake_phase"] = "ABORT"
            control_state["lane_change"] = "ABORT"
        print("OVERTAKE TIMEOUT -> ABORT, waiting for safe recovery")
        return True

    current_wp = None
    try:
        current_wp = carla_map.get_waypoint(
            vehicle.get_location(), project_to_road=True, lane_type=carla.LaneType.Driving
        )
    except RuntimeError:
        return True
    if current_wp is None:
        return True

    if phase == "OUTBOUND":
        if current_wp.lane_id == target_lane_id:
            with control_lock:
                control_state["overtake_phase"] = "PASS"
                control_state["overtake_lane_entered"] = now
                control_state["lane_change"] = direction
            return True
        # If target lane becomes occupied before entry, don't initiate a
        # second lane change; brake in the current lane and let it clear.
        target_wp = _overtake_target_waypoint(target_lane_id)
        if target_wp is None:
            return False
        # Once the manoeuvre starts, use a dynamic emergency envelope instead
        # of re-applying the initial 12/10 m entry gate. This prevents normal
        # 6-10 m gap fluctuations from causing an immediate abort while still
        # rejecting a genuinely imminent target-lane collision.
        # During OUTBOUND we keep a meaningful forward buffer in the target
        # lane.  A vehicle appearing 5-8 m ahead is no longer a usable passing
        # corridor even if its TTC is still technically non-critical.  Abort
        # early and recover safely instead of spending the full timeout beside
        # a newly occupied lane.
        if not _target_lane_dynamic_clear(
            target_wp, front_min=7.5, rear_min=5.0, min_ttc=1.50
        ):
            with control_lock:
                control_state["overtake_phase"] = "RETURN"
                control_state["overtake_target_lane_id"] = original_lane_id
                control_state["lane_change"] = "RETURN"
            print("OVERTAKE TARGET BLOCKED -> SAFE RETURN")
            return True
        return True

    if phase == "PASS":
        # Safety consistency gate: PASS confirmation must never be declared
        # while the independent physical collision monitor reports a genuine
        # hard AEB condition.  The previous implementation could print
        # "PASS CONFIRMED" in the same control cycle that TTC had already
        # fallen below the AEB threshold; the final arbitration correctly
        # braked, but the overtake state/log then claimed that the pass had
        # succeeded.  Keep the safety arbiter authoritative and hold the PASS
        # state until the hazard clears or the normal AEB interruption moves
        # the state to ABORT.
        if vehicle_hard:
            return True

        # A critical physical TTC that is not yet classified as vehicle_hard
        # is also not a valid pass-confirmation condition.  This avoids a
        # premature PASS -> RETURN transition while the ego is still closing
        # rapidly on an actor in its current collision corridor.
        if (
            front_actor is not None
            and front_dist < 6.0
            and abs(front_lat) <= 2.25
            and math.isfinite(physical_ttc)
            and physical_ttc <= 1.20
        ):
            return True

        # Stay in target lane until the original lead is clearly behind.
        if _actor_is_passed(actor_id) or (control_state["overtake_lane_entered"] and now - control_state["overtake_lane_entered"] > 3.0 and actor_id is None):
            original_wp = _overtake_target_waypoint(original_lane_id)
            if original_wp is not None and lane_is_safe(original_wp, front_gap=24.0, rear_gap=16.0):
                with control_lock:
                    control_state["overtake_phase"] = "RETURN"
                    control_state["overtake_target_lane_id"] = original_lane_id
                    control_state["lane_change"] = "RETURN"
                print("OVERTAKE PASS CONFIRMED -> target lead passed, returning to original lane")
                return True
        return True

    if phase == "ABORT":
        # ABORT is a transient cancellation state, NOT a driving mode.
        # Never hold throttle/brake/lateral control here.  Release the stale
        # overtake immediately and let the normal safety arbiter decide the
        # next longitudinal action (RED/YELLOW, AEB, ACC or CRUISE).
        #
        # This is especially important after a traffic-light transition: if
        # the signal is GREEN and no independent hazard exists, clearing the
        # stale ABORT lets the same frame fall through to ACC/CRUISE instead
        # of remaining stopped forever.
        print("OVERTAKE ABORT RECOVERY -> clearing stale overtake state")
        _clear_overtake_state()
        return False

    if phase == "RETURN":
        returned = current_wp.lane_id == original_lane_id

        # CARLA road/lane IDs can change across an intersection or road-segment
        # transition even though the vehicle is physically centered back in
        # the lane from which the manoeuvre started. The previous completion
        # test relied too heavily on the original numeric lane_id, which could
        # leave the state machine stuck in RETURN even when the vehicle was
        # visibly centered back in its lane.
        if not returned:
            try:
                original_wp = _overtake_target_waypoint(original_lane_id)
                if original_wp is not None:
                    d = current_wp.transform.location.distance(
                        original_wp.transform.location
                    )
                    heading_ok = (
                        abs(
                            _yaw_diff(
                                current_wp.transform.rotation.yaw,
                                original_wp.transform.rotation.yaw,
                            )
                        )
                        <= 25.0
                    )
                    returned = d <= 1.5 and heading_ok
            except RuntimeError:
                pass

        # Physical confirmation fallback for CARLA road/lane-ID transitions.
        # Require the ego to be centered in a driving lane, aligned with it,
        # and no longer inside the active target lane. This only clears an
        # already-approved RETURN phase; it never starts a lane change.
        if not returned and target_lane_id is not None:
            try:
                lane_status, lane_offset, lane_width, heading_error = physical_lane_status()
                returned = (
                    lane_status == "KEEP"
                    and lane_width > 2.5
                    and abs(lane_offset) <= max(0.55, lane_width * 0.18)
                    and heading_error <= 25.0
                    and current_wp.lane_id != target_lane_id
                )
            except RuntimeError:
                pass

        if returned:
            print("OVERTAKE COMPLETE -> returned to original lane")
            with control_lock:
                control_state["overtake_cooldown_until"] = now + 2.5
            _clear_overtake_state()
            return False
        return True

    return False


def lane_change_steer():
    """Drive toward the approved adjacent lane with an explicit lateral controller.

    The previous controller selected a waypoint only ~6 m ahead and relied on
    the normal route-steering slew limit. In the runtime recording the overtake
    state entered OUTBOUND, but the ego remained in its original lane until the
    12 s timeout. This controller uses target-lane lateral error + heading error
    and a longer look-ahead, while remaining bounded and safety-gated by the
    overtake state machine.
    """
    with control_lock:
        target_lane_id = control_state["overtake_target_lane_id"]
        ped_target = control_state["target_lane_id"] if control_state["ped_mode"] == "ESCAPE" else None
    target_lane_id = target_lane_id if target_lane_id is not None else ped_target
    if target_lane_id is None or not actor_alive(vehicle):
        return None

    try:
        ego_tr = vehicle.get_transform()
        cur = carla_map.get_waypoint(
            ego_tr.location, project_to_road=True, lane_type=carla.LaneType.Driving
        )
    except RuntimeError:
        return None
    if cur is None:
        return None

    target = None
    for candidate in (cur.get_left_lane(), cur.get_right_lane()):
        if candidate is not None and candidate.lane_id == target_lane_id and _same_direction(candidate, cur):
            target = candidate
            break
    if target is None:
        if cur.lane_id == target_lane_id:
            target = cur
        else:
            return None

    # Use a forward point on the target lane.  10-12 m gives the controller
    # enough preview to start the lateral move without making a sharp junction
    # cut.
    point = target.transform.location
    target_yaw = target.transform.rotation.yaw
    try:
        candidates = [wp for wp in target.next(11.0) if wp is not None and not wp.is_junction]
        if candidates:
            chosen = min(
                candidates,
                key=lambda wp: abs(_yaw_diff(wp.transform.rotation.yaw, ego_tr.rotation.yaw))
            )
            point = chosen.transform.location
            target_yaw = chosen.transform.rotation.yaw
    except RuntimeError:
        pass

    try:
        yaw = math.radians(ego_tr.rotation.yaw)
        dx = point.x - ego_tr.location.x
        dy = point.y - ego_tr.location.y
        local_x = dx * math.cos(yaw) + dy * math.sin(yaw)
        local_y = -dx * math.sin(yaw) + dy * math.cos(yaw)
        if local_x < 1.0:
            return 0.0

        # Lateral-angle term is the primary lane-change command. Heading term
        # helps the ego settle into the target lane instead of oscillating.
        angle_term = math.atan2(local_y, max(local_x, 2.0))
        heading_term = math.radians(_yaw_diff(target_yaw, ego_tr.rotation.yaw))
        steer = 1.55 * angle_term + 0.45 * heading_term

        # If CARLA's waypoint projection is temporarily almost centred, retain
        # a small deterministic command toward the selected adjacent lane.
        if cur.lane_id != target_lane_id and abs(local_y) < 0.35:
            # CARLA uses +steer for RIGHT and -steer for LEFT.  The previous
            # fallback had these signs reversed, so an approved RIGHT/LEFT
            # overtake could command the ego away from the target lane and
            # remain in OUTBOUND until the timeout.
            side = 1.0 if target_lane_id != cur.lane_id else 0.0
            right_wp, left_wp = cur.get_right_lane(), cur.get_left_lane()
            if right_wp is not None and right_wp.lane_id == target_lane_id:
                side = 1.0
            elif left_wp is not None and left_wp.lane_id == target_lane_id:
                side = -1.0
            steer = side * max(abs(steer), 0.22)

        return float(np.clip(steer, -OVERTAKE_STEER_MAX, OVERTAKE_STEER_MAX))
    except RuntimeError:
        return None

def upcoming_traffic_light():
    """Return the traffic light governing the ego lane, if it is ahead.

    CARLA's vehicle.get_traffic_light() is the primary source. A geometric
    fallback is used only when that association is unavailable.
    """
    if not actor_alive(vehicle):
        return "NONE", float("inf")
    try:
        tr = vehicle.get_transform()
        ego_wp = carla_map.get_waypoint(
            tr.location, project_to_road=True, lane_type=carla.LaneType.Driving
        )
    except RuntimeError:
        return "NONE", float("inf")

    # Primary CARLA association.
    try:
        light = vehicle.get_traffic_light()
    except RuntimeError:
        light = None
    if light is not None and actor_alive(light):
        try:
            loc = light.get_location()
            dx = loc.x - tr.location.x
            dy = loc.y - tr.location.y
            yaw = math.radians(tr.rotation.yaw)
            forward = dx * math.cos(yaw) + dy * math.sin(yaw)
            d = math.hypot(dx, dy)
            if forward > -2.0 and d < 80.0:
                state = str(light.get_state()).split(".")[-1].upper()
                return state, max(0.1, d)
        except RuntimeError:
            pass

    # Fallback: same-road, forward-facing light.
    best = ("NONE", float("inf"))
    yaw = math.radians(tr.rotation.yaw)
    for light in world.get_actors().filter("traffic.traffic_light"):
        if not actor_alive(light):
            continue
        try:
            loc = light.get_location()
            dx, dy = loc.x - tr.location.x, loc.y - tr.location.y
            forward = dx * math.cos(yaw) + dy * math.sin(yaw)
            d = math.hypot(dx, dy)
            if forward <= 0.0 or d > 65.0:
                continue
            lwp = carla_map.get_waypoint(
                loc, project_to_road=True, lane_type=carla.LaneType.Driving
            )
            lateral = abs(-dx * math.sin(yaw) + dy * math.cos(yaw))
            lateral_limit = max(7.0, float(ego_wp.lane_width) * 2.2) if ego_wp is not None else 7.0
            if lateral > lateral_limit:
                continue
            state = str(light.get_state()).split(".")[-1].upper()
            if d < best[1]:
                best = (state, d)
        except RuntimeError:
            continue
    return best

def on_obstacle(event):
    """
    Store the latest obstacle-sensor hit, including the actor id.

    The sensor is only a backup. It is NOT allowed to cause AEB merely
    because an object is close to the vehicle.
    """
    if stop_event.is_set() or event.other_actor is None:
        return

    try:
        if vehicle is not None and event.other_actor.id == vehicle.id:
            return

        if event.distance < 0.20:
            return

        with obstacle_lock:
            obstacle_state["distance"] = float(event.distance)
            obstacle_state["actor_id"] = int(event.other_actor.id)
            obstacle_state["last_update"] = time.monotonic()
    except (RuntimeError, AttributeError, TypeError, ValueError):
        pass


def get_obstacle_distance():
    """
    Return an obstacle distance only when the obstacle is plausibly inside
    the ego vehicle's forward collision corridor.

    Roadside/adjacent objects are ignored by the emergency backup.
    """
    with obstacle_lock:
        d = obstacle_state["distance"]
        actor_id = obstacle_state.get("actor_id")
        last_update = obstacle_state["last_update"]

    if (
        d is None
        or actor_id is None
        or time.monotonic() - last_update > OBSTACLE_TIMEOUT
    ):
        return None

    if not actor_alive(vehicle):
        return None

    try:
        actor = world.get_actor(actor_id)
        if actor is None or not actor_alive(actor):
            return None

        ego_tr = vehicle.get_transform()
        ego_wp = carla_map.get_waypoint(
            ego_tr.location,
            project_to_road=True,
            lane_type=carla.LaneType.Driving,
        )
        actor_loc = actor.get_location()

        dx = actor_loc.x - ego_tr.location.x
        dy = actor_loc.y - ego_tr.location.y
        yaw = math.radians(ego_tr.rotation.yaw)

        forward = dx * math.cos(yaw) + dy * math.sin(yaw)
        lateral = -dx * math.sin(yaw) + dy * math.cos(yaw)

        # The backup sensor is for objects directly in front, not roadside
        # objects or vehicles in a neighbouring lane.
        if forward <= 0.0 or forward > 3.0:
            return None
        if abs(lateral) > 1.35:
            return None

        actor_wp = carla_map.get_waypoint(
            actor_loc,
            project_to_road=True,
            lane_type=carla.LaneType.Driving,
        )

        if (
            ego_wp is not None
            and actor_wp is not None
            and not ego_wp.is_junction
            and not actor_wp.is_junction
            and (
                actor_wp.road_id != ego_wp.road_id
                or actor_wp.lane_id != ego_wp.lane_id
            )
        ):
            return None

        return float(d)

    except RuntimeError:
        return None


def get_front_collision_hazard(max_distance=22.0):
    """
    Conservative physical safety envelope.

    Returns the closest dynamic actor that is actually in the ego vehicle's
    forward collision corridor. This is intentionally independent of YOLO so
    a missed detection cannot make the controller drive through an actor.
    """
    if not actor_alive(vehicle):
        return None, float("inf"), 0.0, "NONE"

    try:
        ego_tr = vehicle.get_transform()
        ego_loc = ego_tr.location
        yaw = math.radians(ego_tr.rotation.yaw)
    except RuntimeError:
        return None, float("inf"), 0.0, "NONE"

    best = (None, float("inf"), 0.0, "NONE")

    def consider(actor, kind, lateral_limit, extra_margin=0.0):
        nonlocal best
        if not actor_alive(actor) or actor.id == vehicle.id:
            return
        try:
            loc = actor.get_location()
            dx = loc.x - ego_loc.x
            dy = loc.y - ego_loc.y
            forward = dx * math.cos(yaw) + dy * math.sin(yaw)
            lateral = -dx * math.sin(yaw) + dy * math.cos(yaw)
            dist = math.hypot(dx, dy)
            if forward <= 0.0 or forward > max_distance:
                return
            # Use longitudinal/lateral corridor rather than raw Euclidean
            # distance so actors in adjacent lanes are not treated as hits.
            if abs(lateral) > lateral_limit + extra_margin:
                return

            # Expand the safety envelope as the actor gets closer.
            dynamic_limit = lateral_limit + max(0.0, 1.2 - 0.08 * forward)
            if abs(lateral) > dynamic_limit:
                return

            if dist < best[1]:
                best = (actor, dist, lateral, kind)
        except RuntimeError:
            return

    # Vehicles: 2.25 m corridor catches a car/bus that is physically in
    # front of the ego, but still ignores clearly adjacent traffic.
    for other in world.get_actors().filter("vehicle.*"):
        consider(other, "VEHICLE", 2.25)

    # Pedestrians: use a slightly wider envelope because a walker can cross
    # the lane between perception updates.
    for walker in world.get_actors().filter("walker.pedestrian.*"):
        if not actor_alive(walker):
            continue
        try:
            ws = speed_mps(walker)
        except Exception:
            ws = 0.0
        consider(walker, "PEDESTRIAN", 1.75, 0.55 if ws > 0.5 else 0.0)

    return best


def apply_control(
    lead,
    lead_distance,
    risk,
    ttc,
    lane_steer,
    light_state,
    light_distance,
    pedestrian=None,
    pedestrian_distance=float("inf"),
    pedestrian_lateral=0.0,
):
    """Safety-first single arbitration point.

    Order is deliberately strict:
      1) pedestrian / vulnerable-road-user hazard
      2) physical front-vehicle hazard
      3) red/yellow traffic signal
      4) safe ACC following
      5) conservative road steering + cruise

    The controller does not promise mathematical 100% accuracy; instead it
    uses conservative physical CARLA ground-truth checks so a perception miss
    does not remove the safety brake.
    """
    if not actor_alive(vehicle):
        return {"steer":0.0,"throttle":0.0,"brake":1.0,"action":"STOPPED",
                "lane_change":"NONE","route_steer":0.0,"lead_id":None}

    try:
        ego_speed = speed_mps(vehicle)
    except Exception:
        ego_speed = 0.0

    # Never let Hough lane-departure output become the primary steering command.
    route = float(np.clip(route_steering(), -0.45, 0.45))

    # -------- Physical pedestrian check (independent of YOLO) --------
    p_actor, p_dist, p_lat = get_pedestrian_hazard()
    if p_actor is not None:
        pedestrian = p_actor
        pedestrian_distance = p_dist
        pedestrian_lateral = p_lat

    ped_conflict = (
        pedestrian is not None
        and math.isfinite(pedestrian_distance)
        and 0.0 < pedestrian_distance <= 28.0
        and abs(pedestrian_lateral) <= 2.35
    )

    # Pedestrian escape state machine. A blocked pedestrian is NEVER handled
    # by blind steering. We first stop, then (only if the rear is clear)
    # reverse briefly to create space, then re-evaluate a safe driving lane.
    now = time.monotonic()
    with control_lock:
        ped_mode = control_state["ped_mode"]
        ped_blocked_since = control_state["ped_blocked_since"]
        ped_reverse_until = control_state["ped_reverse_until"]
        ped_escape_until = control_state["ped_escape_until"]

    if not ped_conflict:
        reset_pedestrian_state()
        ped_mode = "NONE"
    else:
        if ped_mode == "NONE":
            with control_lock:
                control_state["ped_mode"] = "BRAKE_WAIT"
                control_state["ped_blocked_since"] = now
            ped_mode = "BRAKE_WAIT"
            ped_blocked_since = now

        if ped_mode == "BRAKE_WAIT" and ego_speed <= 0.7:
            if now - ped_blocked_since >= PEDESTRIAN_REVERSE_DELAY:
                rear_clearance = get_rear_clearance()
                if rear_clearance >= PEDESTRIAN_REVERSE_MIN_CLEARANCE:
                    with control_lock:
                        control_state["ped_mode"] = "REVERSE"
                        control_state["ped_reverse_until"] = now + PEDESTRIAN_REVERSE_DURATION
                    ped_mode = "REVERSE"
                    ped_reverse_until = now + PEDESTRIAN_REVERSE_DURATION

        if ped_mode == "REVERSE" and now >= ped_reverse_until:
            # Stop before deciding the forward escape direction.
            rear_clearance = get_rear_clearance()
            if rear_clearance >= PEDESTRIAN_REVERSE_STOP_CLEARANCE:
                selected = choose_safe_lane_for_pedestrian(pedestrian, pedestrian_lateral)
                if selected is not None:
                    _, direction, target_wp = selected
                    with control_lock:
                        control_state["target_lane_id"] = target_wp.lane_id
                        control_state["lane_change"] = direction
                        control_state["lane_change_until"] = now + 5.0
                        control_state["ped_mode"] = "ESCAPE"
                        control_state["ped_escape_until"] = now + 5.0
                    ped_mode = "ESCAPE"
                    ped_escape_until = now + 5.0

    # -------- Physical vehicle collision corridor --------
    front_actor, front_dist, front_lat, front_kind = get_front_collision_hazard(max_distance=35.0)
    vehicle_hazard = (
        front_actor is not None
        and front_kind == "VEHICLE"
        and math.isfinite(front_dist)
        and front_dist > 0.0
        and abs(front_lat) <= 2.15
    )
    front_speed = speed_mps(front_actor) if vehicle_hazard else 0.0

    # IMPORTANT: use relative longitudinal velocity, not speed-magnitude
    # subtraction. With an oncoming vehicle, the other vehicle's forward
    # component is negative, so closing speed is ego + oncoming speed.
    if vehicle_hazard:
        try:
            ego_tr = vehicle.get_transform()
            fyaw = math.radians(ego_tr.rotation.yaw)
            ego_v = vehicle.get_velocity()
            other_v = front_actor.get_velocity()
            ego_long = ego_v.x * math.cos(fyaw) + ego_v.y * math.sin(fyaw)
            other_long = other_v.x * math.cos(fyaw) + other_v.y * math.sin(fyaw)
            closing = max(0.0, ego_long - other_long)
        except RuntimeError:
            closing = max(0.0, ego_speed - front_speed)
    else:
        closing = 0.0

    physical_ttc = front_dist / closing if vehicle_hazard and closing > 0.2 else float("inf")
    front_oncoming = False
    if vehicle_hazard:
        try:
            front_oncoming = abs(_yaw_diff(
                vehicle.get_transform().rotation.yaw,
                front_actor.get_transform().rotation.yaw,
            )) >= 75.0
        except RuntimeError:
            front_oncoming = False

    # Traffic-light stopping distance. Be conservative and start braking early.
    valid_light = (
        light_state in {"RED", "YELLOW"}
        and math.isfinite(light_distance)
        and light_distance > 0.0
    )
    reaction_distance = ego_speed * 0.9
    braking_distance = (ego_speed * ego_speed) / max(2.0 * 5.0, 1.0)
    signal_stop_distance = reaction_distance + braking_distance + 4.0
    red_stop = valid_light and light_state == "RED" and light_distance <= max(12.0, signal_stop_distance)
    yellow_stop = valid_light and light_state == "YELLOW" and light_distance <= max(10.0, signal_stop_distance * 0.9)

    # Vehicle AEB/caution uses physical actor + closing speed, not a pixel box.
    # Caution braking must not fight a lead that is already faster than ego;
    # otherwise the controller oscillates between brake/creep and never builds
    # enough speed to reach the natural overtake envelope.
    # A distance-only hard AEB gate is unsafe for the controller logic in
    # intersections and dense traffic: a vehicle can be only 3-6 m ahead while
    # already moving substantially faster than ego (for example, traffic
    # clearing a green signal). In that case TTC is infinite and there is no
    # positive closing motion, so repeatedly applying full AEB creates the
    # exact green-light stop/deadlock seen in runtime.
    #
    # Hard AEB therefore requires an actual closing hazard:
    #   - critical TTC, OR
    #   - a genuinely slow/stopped obstacle inside the hard envelope while ego
    #     is moving toward it.
    # A stopped ego is never hard-braked merely because another moving vehicle
    # is physically close; ACC/signal arbitration handles that case.
    closing_positive_hard = ego_speed - front_speed > 0.20
    slow_front_hard = front_speed <= max(0.50, ego_speed * 0.35)
    vehicle_hard = vehicle_hazard and closing_positive_hard and (
        physical_ttc <= 1.6
        or (front_dist <= 6.5 and slow_front_hard)
    )
    closing_positive = closing
    vehicle_caution = vehicle_hazard and (
        (
            front_dist <= max(14.0, ego_speed * 1.5 + 5.0)
            and (
                closing_positive > 0.40
                or (front_dist < 8.0 and front_speed <= ego_speed + 0.20)
            )
        )
        # Oncoming traffic in the ego corridor gets an earlier physical
        # caution window instead of waiting for the hard-AEB envelope.
        or (front_oncoming and math.isfinite(physical_ttc) and physical_ttc <= 4.0)
    )

    # -------- Stateful vehicle overtaking --------
    lc_active = update_lane_change(
        lead, lead_distance, ttc,
        ped_conflict=ped_conflict,
        light_state=light_state,
        light_distance=light_distance,
        vehicle_hard=vehicle_hard,
        physical_ttc=physical_ttc,
        front_actor=front_actor,
        front_dist=front_dist,
        front_lat=front_lat,
    )

    # V20: a genuine vehicle AEB event interrupts the manoeuvre cleanly.
    # The final AEB arbitration remains unchanged and still has absolute
    # priority; this only prevents stale OUTBOUND/PASS state from surviving
    # an emergency stop and blocking future overtakes.
    if vehicle_hard:
        with control_lock:
            _phase_now = control_state["overtake_phase"]
            if _phase_now in {"OUTBOUND", "PASS"}:
                control_state["overtake_phase"] = "ABORT"
                control_state["lane_change"] = "ABORT"
                print("OVERTAKE INTERRUPTED -> VEHICLE AEB, switching to ABORT")

    with control_lock:
        overtake_phase = control_state["overtake_phase"]
        overtake_direction = control_state["overtake_direction"]
        overtake_target_lane_id = control_state["overtake_target_lane_id"]

    # A safety-approved outbound/pass manoeuvre is an evasive lateral action:
    # if the target lane is still clear, the current-lane lead must not force
    # an AEB stop before the ego has had a chance to leave that lane. This is
    # deliberately narrow: pedestrian conflicts always win, the target lane
    # must be physically safe, and a very close/critical TTC still invokes AEB.
    overtake_escape_window = False
    if (
        overtake_phase in {"OUTBOUND", "PASS"}
        and overtake_target_lane_id is not None
        and not ped_conflict
    ):
        try:
            target_wp = _overtake_target_waypoint(overtake_target_lane_id)
            target_safe = _target_lane_dynamic_clear(
                target_wp, front_min=5.0, rear_min=4.5, min_ttc=1.15
            )
            overtake_escape_window = (
                target_safe
                and front_dist > 4.5
                and (
                    not math.isfinite(physical_ttc)
                    or physical_ttc > 0.90
                )
            )
        except RuntimeError:
            overtake_escape_window = False

    if overtake_escape_window:
        vehicle_hard = False

    # A vulnerable-road-user conflict cancels any vehicle overtake mission.
    if ped_conflict and overtake_phase != "NONE":
        _clear_overtake_state()
        lc_active = False
        overtake_phase = "NONE"
        overtake_direction = "NONE"

    # -------- Final arbitration --------
    action = "CRUISE"
    steer = route
    throttle = 0.30 if ego_speed < 18.0 else 0.18  # cruise speed increased per request (was 0.24/0.16)
    brake = 0.0
    lane_change = overtake_direction if lc_active and overtake_phase in {"OUTBOUND", "PASS"} else ("RETURN" if lc_active and overtake_phase == "RETURN" else "NONE")

    # 1. Pedestrian always wins. If blocked for long enough, reverse ONLY
    # after a rear-clearance check; then use a verified safe adjacent lane.
    if ped_conflict and ped_mode == "REVERSE":
        rear_clearance = get_rear_clearance()
        if rear_clearance < PEDESTRIAN_REVERSE_STOP_CLEARANCE:
            action = "PEDESTRIAN BRAKE"
            steer = 0.0
            throttle = 0.0
            brake = 1.0
        else:
            action = "PEDESTRIAN REVERSE"
            throttle = 0.0
            brake = 0.0
            # Small reverse steering away from the pedestrian side. In reverse
            # the rear of the vehicle moves opposite the front-wheel steering.
            steer = 0.18 if pedestrian_lateral > 0.0 else -0.18

    elif ped_conflict and ped_mode == "ESCAPE":
        lc = lane_change_steer()
        if lc is not None and now < ped_escape_until:
            action = "PEDESTRIAN ESCAPE"
            steer = float(np.clip(lc, -0.35, 0.35))
            throttle = 0.10 if ego_speed < 8.0 else 0.16
            brake = 0.0
        else:
            action = "PEDESTRIAN BRAKE"
            steer = 0.0
            throttle = 0.0
            brake = 1.0

    elif ped_conflict:
        action = "PEDESTRIAN BRAKE"
        steer = 0.0
        throttle = 0.0
        brake = 1.0

    # 2. Physical stopped/slow vehicle in front. This is the only branch
    # that may ever override an in-progress overtake, and only for a
    # genuinely imminent hazard (vehicle_hard). Soft caution-level braking
    # (vehicle_caution) is intentionally NOT checked here anymore -- see
    # note on branch 5 below for why.
    elif vehicle_hard:
        action = "AEB VEHICLE"
        steer = 0.0
        throttle = 0.0
        brake = 1.0

    # 3. An already-approved RETURN may continue lateral alignment while the
    # signal controller owns longitudinal stopping. This prevents the state
    # machine from remaining in RETURN LANE simply because a red/yellow light
    # appears while the vehicle is completing its return. No new overtake is
    # initiated here.
    elif lc_active and overtake_phase == "RETURN" and (red_stop or yellow_stop):
        lc_steer = lane_change_steer()
        if lc_steer is not None:
            action = "RETURN LANE"
            steer = float(np.clip(lc_steer, -0.45, 0.45))
        else:
            action = "RETURN HOLD"
            steer = route
        throttle = 0.0
        if red_stop:
            brake = 1.0 if ego_speed > 1.0 else 0.65
        else:
            brake = 0.65 if ego_speed > 2.0 else 0.45

    # 4. Traffic signals. Stop before the intersection, keep steering stable.
    elif red_stop:
        action = "RED LIGHT"
        steer = route
        throttle = 0.0
        brake = 1.0 if ego_speed > 1.0 else 0.65
    elif yellow_stop:
        action = "YELLOW LIGHT"
        steer = route
        throttle = 0.0
        brake = 0.65 if ego_speed > 2.0 else 0.45


    # 4. Stateful safe overtake. It owns steering only while an approved
    # manoeuvre is active; otherwise ACC/route steering remains in control.
    #
    # ROOT-CAUSE FIX: this branch must be checked BEFORE the soft
    # vehicle_caution braking below. Previously vehicle_caution was checked
    # first (branch 2, together with vehicle_hard) and its distance window
    # (front_dist <= ~14-20m) heavily overlaps the exact 9-30m window in
    # which update_lane_change() authorises an overtake against the same
    # lead vehicle. That meant the instant an overtake became authorised
    # (overtake_phase == OUTBOUND), this same-frame vehicle_caution check
    # was almost always also true for the identical lead actor, so the
    # arbitration chain kept selecting "BRAKE FOR VEHICLE" every frame and
    # lane_change_steer() was never actually applied to the vehicle -- the
    # state machine advanced internally but the car just braked and sat
    # behind the lead until the 12s hard timeout aborted the manoeuvre.
    # Moving this branch above vehicle_caution (but still below the
    # genuine-emergency vehicle_hard and the traffic-light stop above) lets
    # an authorised, already-safety-checked lane change actually execute.
    # Emergency braking is untouched: vehicle_hard is still checked first except for the narrowly verified overtake escape window,
    # and the "absolute final guard" further below still forces AEB VEHICLE
    # (with lane_change forced to NONE) whenever vehicle_hard is true,
    # regardless of overtake state.
    elif lc_active:
        # ABORT should normally have been consumed by update_lane_change().
        # Keep this defensive fallback non-blocking in case another thread
        # observes the old state for one frame.
        if overtake_phase == "ABORT":
            action = "ABORT OVERTAKE"
            steer = route
            throttle = 0.0
            brake = 0.0
        else:
            action = "OVERTAKE" if overtake_phase in {"OUTBOUND", "PASS"} else "RETURN LANE"
            lc_steer = lane_change_steer()
            if lc_steer is None:
                # A temporary CARLA waypoint projection failure must not make
                # the ego vehicle freeze. Keep route steering and creep forward
                # while the target waypoint becomes available again.
                # Genuine imminent hazards are still handled by vehicle_hard,
                # pedestrian and traffic-light branches above.
                action = "OVERTAKE HOLD"
                steer = route
                throttle = 0.10
                brake = 0.0
            else:
                steer = float(np.clip(lc_steer, -0.45, 0.45))
                # Give the ego enough longitudinal acceleration to complete
                # the pass before the state timeout.  Safety arbitration above
                # still has priority for pedestrians, red/yellow signals and
                # genuine vehicle AEB.
                if overtake_phase == "OUTBOUND":
                    throttle = 0.42
                elif overtake_phase == "PASS":
                    throttle = 0.40
                else:
                    throttle = 0.30
                brake = 0.0

    # 5. Soft caution braking for a vehicle ahead, when there is no active
    # or authorised overtake in progress (lc_active is False here, since
    # branch 4 above already claimed every case where it was True). This is
    # exactly the original vehicle_caution behaviour, just re-scoped so it
    # no longer fires while an approved overtake is actively executing.
    elif vehicle_caution:
        action = "BRAKE FOR VEHICLE"
        steer = route
        throttle = 0.0
        if math.isfinite(physical_ttc):
            brake = float(np.clip(0.35 + (3.0 - physical_ttc) * 0.18, 0.30, 0.85))
        else:
            gap_error = max(0.0, 14.0 - front_dist)
            brake = float(np.clip(0.30 + gap_error * 0.035, 0.30, 0.65))

    # 6. ACC following based on physical lead.
    elif lead is not None and math.isfinite(lead_distance) and 0.0 < lead_distance <= 45.0:
        action = "ACC FOLLOW"
        desired_gap = max(9.0, ego_speed * 1.6)
        error = lead_distance - desired_gap
        steer = route
        if error < -3.0:
            throttle = 0.0
            brake = float(np.clip(0.20 + (-error) * 0.045, 0.20, 0.65))
        elif error < 3.0:
            throttle = 0.12
            brake = 0.0
        else:
            # Once the physical gap is open, let the ego recover speed instead
            # of creeping at 0.1–0.2 throttle indefinitely after a light/AEB.
            throttle = 0.30 if ego_speed < 8.0 else 0.22
            brake = 0.0

    # 7. Empty road / normal road.
    else:
        action = "CRUISE"
        steer = route
        throttle = 0.30 if ego_speed < 20.0 else 0.16  # cruise speed increased per request (was 0.24/0.12)
        brake = 0.0

    # Absolute final guard: no normal controller may steer into a pedestrian.
    # The only exception is the verified ESCAPE state, whose target lane was
    # checked against vehicles AND pedestrians before being activated.
    if ped_conflict and ped_mode != "ESCAPE":
        action = "PEDESTRIAN BRAKE"
        steer = 0.0
        throttle = 0.0
        brake = 1.0
        lane_change = "NONE"
    elif ped_conflict and ped_mode == "ESCAPE":
        lane_change = control_state.get("lane_change", "NONE")

    # Absolute final guard for imminent vehicle collision.
    if vehicle_hard and not ped_conflict:
        action = "AEB VEHICLE"
        steer = 0.0
        throttle = 0.0
        brake = 1.0
        lane_change = "NONE"

    # Steering slew limit only in non-emergency states.
    with control_lock:
        previous = control_state["last_steer"]
        if action in {"PEDESTRIAN BRAKE", "AEB VEHICLE"}:
            final_steer = 0.0
        else:
            if action in {"OVERTAKE", "OVERTAKE HOLD", "RETURN LANE", "RETURN HOLD"}:
                delta = OVERTAKE_STEER_SLEW
            else:
                delta = 0.08 if ego_speed > 10.0 else 0.11
            final_steer = float(np.clip(steer, previous - delta, previous + delta))
        control_state["last_steer"] = final_steer
        control_state["action"] = action
        control_state["lane_change"] = lane_change
        # Do NOT clear the vehicle overtake target here. The state machine
        # owns it until PASS -> RETURN -> COMPLETE. Pedestrian escape owns
        # target_lane_id only while ped_mode == ESCAPE.
        if control_state["overtake_phase"] == "NONE" and control_state["ped_mode"] != "ESCAPE":
            control_state["target_lane_id"] = None
            control_state["lane_change_until"] = 0.0

    try:
        control = vehicle.get_control()
        control.steer = float(np.clip(final_steer, -0.45, 0.45))
        control.throttle = float(np.clip(throttle, 0.0, 0.45))
        control.brake = float(np.clip(brake, 0.0, 1.0))
        control.hand_brake = False
        control.reverse = False
        vehicle.apply_control(control)
    except RuntimeError:
        stop_event.set()

    return {
        "steer": float(final_steer),
        "throttle": float(throttle),
        "brake": float(brake),
        "action": action,
        "lane_change": lane_change,
        "route_steer": route,
        "lead_id": lead.id if lead is not None else None,
        "pedestrian_id": pedestrian.id if pedestrian is not None else None,
        "pedestrian_distance": pedestrian_distance if pedestrian is not None else float("inf"),
        "front_actor_kind": front_kind,
        "front_actor_distance": front_dist,
        "front_vehicle_ttc": physical_ttc,
    }

def camera_callback(image):
    global front_latest, front_frame_id
    if stop_event.is_set():
        return
    try:
        raw = np.frombuffer(image.raw_data, dtype=np.uint8)
        frame = raw.reshape((image.height, image.width, 4))[:, :, :3].copy()
        with front_lock:
            front_latest = frame
            front_frame_id = int(image.frame)
    except Exception:
        pass


def chase_callback(image):
    global chase_latest
    if stop_event.is_set():
        return
    try:
        raw = np.frombuffer(image.raw_data, dtype=np.uint8)
        frame = raw.reshape((image.height, image.width, 4))[:, :, :3].copy()
        with chase_lock:
            chase_latest = frame
    except Exception:
        pass


def process_front_frame(frame, image_id):
    """Run YOLO/ByteTrack + visual lane annotation on one newest frame.

    Vehicle control is intentionally NOT executed here. The independent
    control loop owns all physical safety and actuator decisions.
    """
    global processed_count, processed_fps

    source = frame.copy()
    annotated = frame.copy()

    # Camera-lane visualization is handled by its own worker. Keeping it out
    # of the YOLO worker prevents a slow model initialization/inference from
    # blocking lane overlays.
    lane_status, lane_offset, lane_width, lane_heading_error = (
        physical_lane_status()
    )

    # Only the inference copy is reduced. The operator view remains HD.
    yolo_frame = cv2.resize(
        frame,
        (YOLO_INPUT_W, YOLO_INPUT_H),
        interpolation=cv2.INTER_AREA,
    )

    tracked_count = 0
    yolo_person_visible = False
    result = None
    yolo_error = None

    track_active = False
    perception_state = "LIVE"
    try:
        # FIRST CALL IS A REAL CAMERA FRAME.  Do not perform a synthetic
        # model.predict/model.track warm-up here: that was the V4 stall.
        results = model.track(
            yolo_frame,
            imgsz=YOLO_IMGSZ,
            persist=True,
            tracker="bytetrack.yaml",
            conf=YOLO_CONF,
            iou=0.50,
            device=YOLO_DEVICE,
            max_det=100,
            verbose=False,
        )
        result = results[0] if results else None
        track_active = result is not None and result.boxes is not None
    except Exception as exc:
        yolo_error = f"TRACK {type(exc).__name__}: {exc}"
        perception_state = "DETECTION FALLBACK"
        print(f"YOLO/ByteTrack error: {yolo_error}")

        # Keep visual perception alive even if the tracker rejects a frame.
        # This does not alter the physical safety controller.
        try:
            results = model.predict(
                yolo_frame,
                imgsz=YOLO_IMGSZ,
                conf=YOLO_CONF,
                iou=0.50,
                device=YOLO_DEVICE,
                max_det=100,
                verbose=False,
            )
            result = results[0] if results else None
            track_active = False
        except Exception as fallback_exc:
            yolo_error = (
                f"{yolo_error} | PREDICT {type(fallback_exc).__name__}: "
                f"{fallback_exc}"
            )
            print(f"YOLO detection fallback error: {fallback_exc}")
            result = None
            perception_state = "ERROR"

    if result is not None and result.boxes is not None:
        boxes = result.boxes.xyxy.cpu().numpy()
        ids_raw = (
            result.boxes.id.cpu().numpy().astype(int)
            if result.boxes.id is not None
            else None
        )
        classes = (
            result.boxes.cls.cpu().numpy().astype(int)
            if result.boxes.cls is not None
            else np.zeros(len(boxes), dtype=int)
        )
        confidences = (
            result.boxes.conf.cpu().numpy()
            if result.boxes.conf is not None
            else np.ones(len(boxes), dtype=float)
        )

        tracked_count = len(boxes)
        now = time.monotonic()

        # YOLO boxes are relative to 640x360. Scale to the original
        # 1280x720 frame before drawing.
        scale_x = frame.shape[1] / float(YOLO_INPUT_W)
        scale_y = frame.shape[0] / float(YOLO_INPUT_H)

        for idx, box in enumerate(boxes):
            raw_x1, raw_y1, raw_x2, raw_y2 = box.tolist()
            x1 = max(0, min(frame.shape[1] - 1, int(raw_x1 * scale_x)))
            y1 = max(0, min(frame.shape[0] - 1, int(raw_y1 * scale_y)))
            x2 = max(0, min(frame.shape[1] - 1, int(raw_x2 * scale_x)))
            y2 = max(0, min(frame.shape[0] - 1, int(raw_y2 * scale_y)))

            if x2 <= x1 or y2 <= y1:
                continue

            track_id = (
                int(ids_raw[idx])
                if ids_raw is not None and idx < len(ids_raw)
                else None
            )
            conf = float(confidences[idx]) if idx < len(confidences) else 0.0

            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            buffer.update(track_id, cx, cy, timestamp=now)

            try:
                cls_id = int(classes[idx])
                if isinstance(model.names, dict):
                    cls_name = str(model.names.get(cls_id, "object"))
                else:
                    cls_name = (
                        str(model.names[cls_id])
                        if 0 <= cls_id < len(model.names)
                        else "object"
                    )
            except Exception:
                cls_name = "object"

            if cls_name.lower() == "person":
                yolo_person_visible = True

            # Explicit object label makes YOLO/ByteTrack activity visible.
            color = (0, 220, 0)
            cv2.rectangle(
                annotated, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA
            )
            label = (
                f"{cls_name} {conf:.2f} | ID:{track_id}"
                if track_id is not None
                else f"{cls_name} {conf:.2f} | DET"
            )
            (tw, th), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1
            )
            label_y1 = max(0, y1 - th - 8)
            cv2.rectangle(
                annotated,
                (x1, label_y1),
                (min(frame.shape[1] - 1, x1 + tw + 8), y1),
                color,
                -1,
            )
            cv2.putText(
                annotated,
                label,
                (x1 + 4, max(14, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (5, 15, 5),
                1,
                cv2.LINE_AA,
            )

    # Safety/control is independent from perception.
    with control_snapshot_lock:
        snap = dict(control_snapshot)

    lead_id = snap["lead_id"]
    lead_distance = snap["lead_distance"]
    risk = snap["risk"]
    physical_distance = snap["distance"]
    ttc = snap["ttc"]
    light_state = snap["traffic_light"]
    light_distance = snap["light_distance"]
    pedestrian_id = snap["pedestrian_id"]
    pedestrian_distance = snap["pedestrian_distance"]
    pedestrian_lateral = snap["pedestrian_lateral"]

    if lead_id is not None:
        try:
            cv2.putText(
                annotated,
                f"LEAD {lead_id}  {lead_distance:.1f}m  "
                f"TTC:{'INF' if not math.isfinite(ttc) else f'{ttc:.1f}s'}",
                (18, annotated.shape[0] - 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.58,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )
        except Exception:
            pass

    lane_label = (
        "KEEP LANE" if lane_status == "KEEP"
        else "LANE CAUTION" if lane_status == "CAUTION"
        else "LANE DEPARTURE" if lane_status == "DEPARTURE"
        else "LANE UNKNOWN"
    )
    lane_color = (
        (0, 255, 0) if lane_status == "KEEP"
        else (0, 200, 255) if lane_status == "CAUTION"
        else (0, 0, 255) if lane_status == "DEPARTURE"
        else (180, 180, 180)
    )
    cv2.putText(
        annotated,
        lane_label,
        (20, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.68,
        lane_color,
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        annotated,
        f"LANE OFFSET: {lane_offset:+.2f}m  WIDTH: {lane_width:.1f}m",
        (20, 98),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        (220, 230, 240),
        1,
        cv2.LINE_AA,
    )

    if yolo_error:
        cv2.putText(
            annotated,
            "YOLO: RETRYING",
            (20, 126),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (0, 180, 255),
            2,
            cv2.LINE_AA,
        )

    if light_state != "NONE":
        cv2.putText(
            annotated,
            f"TRAFFIC LIGHT: {light_state} ({light_distance:.1f}m)",
            (20, 42),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 180, 255),
            2,
            cv2.LINE_AA,
        )

    if pedestrian_id is not None and math.isfinite(pedestrian_distance):
        ped_side = (
            "LEFT" if pedestrian_lateral < -0.6
            else "RIGHT" if pedestrian_lateral > 0.6
            else "CENTER"
        )
        cv2.putText(
            annotated,
            f"PEDESTRIAN: {pedestrian_distance:.1f}m {ped_side}",
            (20, 154),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (0, 165, 255),
            2,
            cv2.LINE_AA,
        )

    # Pre-compute the visual overlay so the UI can put it on the newest
    # camera frame instead of displaying a frozen 15-FPS frame.
    try:
        diff = cv2.absdiff(annotated, source)
        overlay_mask = np.max(diff, axis=2) > 8
        if float(np.mean(overlay_mask)) > 0.35:
            overlay_mask = np.max(diff, axis=2) > 20
    except Exception:
        overlay_mask = np.ones(
            (annotated.shape[0], annotated.shape[1]), dtype=bool
        )

    processed_count += 1
    elapsed = time.monotonic() - fps_window_start
    if elapsed >= 2.0:
        processed_fps = processed_count / elapsed
        processed_count = 0
        globals()["fps_window_start"] = time.monotonic()

    return {
        "overlay": annotated,
        "overlay_mask": overlay_mask,
        "tracked_count": tracked_count,
        "yolo_person_visible": yolo_person_visible,
        "yolo_error": yolo_error,
        "state": perception_state,
        "track_active": track_active,
        "updated_at": time.monotonic(),
        "image_id": image_id,
        "lane_status": lane_status,
        "lane_offset": lane_offset,
        "lane_width": lane_width,
        "lane_heading_error": lane_heading_error,
    }



def _build_lane_visual_overlay(source, detector_overlay, lane_center):
    """Strengthen camera-lane visualization without changing safety/control."""
    h, w = source.shape[:2]
    result = source.copy()

    if isinstance(detector_overlay, np.ndarray) and detector_overlay.shape == source.shape:
        diff = cv2.absdiff(detector_overlay, source)
        mask = (np.max(diff, axis=2) > 3).astype(np.uint8) * 255
        roi = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(roi, [np.array([
            [int(.08*w), h-1], [int(.92*w), h-1],
            [int(.68*w), int(.38*h)], [int(.32*w), int(.38*h)]
        ], dtype=np.int32)], 255)
        mask = cv2.bitwise_and(mask, roi)
        kernel = np.ones((3,3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        mask = cv2.dilate(mask, kernel, iterations=1)
        if cv2.countNonZero(mask) > max(120, int(.00015*h*w)):
            result[mask > 0] = detector_overlay[mask > 0]
            return result, mask.astype(bool), "DETECTOR"

    gray = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 60, 150)
    roi = np.zeros_like(edges)
    cv2.fillPoly(roi, [np.array([
        [int(.05*w), h-1], [int(.95*w), h-1],
        [int(.68*w), int(.40*h)], [int(.32*w), int(.40*h)]
    ], dtype=np.int32)], 255)
    edges = cv2.bitwise_and(edges, roi)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 45,
                            minLineLength=max(35, int(.055*w)), maxLineGap=35)

    cx = float(lane_center) if isinstance(lane_center, (int,float)) else w/2.0
    cx = max(.2*w, min(.8*w, cx))
    left, right = [], []
    if lines is not None:
        for x1,y1,x2,y2 in lines[:,0]:
            dx, dy = x2-x1, y2-y1
            if abs(dx) < 8 or abs(dy) < 12: continue
            slope = dy/float(dx)
            if abs(slope) < .35: continue
            length = float((dx*dx+dy*dy)**.5)
            mx = (x1+x2)*.5
            if mx < cx and slope < 0: left.append((length,(x1,y1,x2,y2)))
            elif mx > cx and slope > 0: right.append((length,(x1,y1,x2,y2)))

    mask = np.zeros((h,w), dtype=np.uint8)
    found = False
    for candidates in (left, right):
        if candidates:
            _, (x1,y1,x2,y2) = max(candidates, key=lambda z:z[0])
            cv2.line(result,(x1,y1),(x2,y2),(0,255,255),5,cv2.LINE_AA)
            cv2.line(mask,(x1,y1),(x2,y2),255,9,cv2.LINE_AA)
            found = True
    if found:
        lane_pixels = mask > 0
        return result, lane_pixels, "HOUGH-DISPLAY"
    return source.copy(), np.zeros((h,w), dtype=bool), "NONE"


def lane_worker_loop():
    """Continuously compute the newest camera-lane visualization."""
    last_id = -1
    target_interval = 1.0 / max(LANE_HZ, 1.0)

    while not stop_event.is_set():
        with front_lock:
            frame = None if front_latest is None else front_latest.copy()
            image_id = front_frame_id

        if frame is None or image_id == last_id:
            time.sleep(0.002)
            continue

        start = time.monotonic()
        last_id = image_id

        try:
            source = frame.copy()
            lane_overlay, lane_center = detect_lanes(source)

            # Keep the existing lane-departure estimator available without
            # coupling it to YOLO.
            try:
                check_lane_departure(
                    lane_center, lane_overlay.shape[1], threshold=55
                )
            except Exception:
                pass

            visual_lane, mask, visual_mode = _build_lane_visual_overlay(
                source, lane_overlay, lane_center
            )

            with lane_lock:
                global latest_lane
                latest_lane = {
                    "overlay": visual_lane,
                    "overlay_mask": mask,
                    "updated_at": time.monotonic(),
                    "image_id": image_id,
                    "visual_mode": visual_mode,
                }
        except Exception as exc:
            print(f"Lane detector error: {type(exc).__name__}: {exc}")

        remaining = target_interval - (time.monotonic() - start)
        if remaining > 0.0:
            time.sleep(min(0.005, remaining))


def worker_loop():
    """Run live YOLO/ByteTrack on the newest camera frame only.

    IMPORTANT: there is intentionally NO blocking model warm-up here.  The
    first real camera frame is the first YOLO call.  This avoids the V4 bug
    where a synthetic warm-up call (and CUDA synchronization) occupied the
    perception thread for the whole recording before any real frame reached
    YOLO/ByteTrack.

    If ByteTrack itself fails on a frame, a plain YOLO prediction is attempted
    immediately on that same frame so object boxes still reach the operator
    view.  The fallback is display-only; physical safety/control remains owned
    by the existing CARLA/geometry safety stack.
    """
    last_id = -1
    last_processed_time = 0.0
    target_interval = 1.0 / max(PROCESS_HZ, 1.0)
    first_live_call = True

    while not stop_event.is_set():
        with front_lock:
            frame = None if front_latest is None else front_latest.copy()
            image_id = front_frame_id

        if frame is None or image_id == last_id:
            time.sleep(0.001)
            continue

        remaining = target_interval - (time.monotonic() - last_processed_time)
        if remaining > 0.0:
            time.sleep(min(0.003, remaining))
            continue

        last_id = image_id
        last_processed_time = time.monotonic()

        try:
            output = process_front_frame(frame, image_id)
            if output is not None:
                with perception_lock:
                    global latest_perception
                    latest_perception = output
                if first_live_call:
                    print("YOLO live inference started on real camera frame")
                    first_live_call = False
        except Exception as exc:
            import traceback
            print(f"ADAS perception worker error: {type(exc).__name__}: {exc}")
            traceback.print_exc()
            with perception_lock:
                latest_perception["state"] = "ERROR"
                latest_perception["yolo_error"] = f"{type(exc).__name__}: {exc}"

        time.sleep(0.001)

def control_loop():
    """Run the existing physical safety arbitration independently of YOLO."""
    global control_snapshot

    target_interval = 1.0 / max(CONTROL_HZ, 1.0)

    while not stop_event.is_set():
        tick_start = time.monotonic()
        try:
            if not actor_alive(vehicle):
                stop_event.set()
                break

            # Scenario-only speed hold; all ego safety/control logic remains
            # unchanged below.
            maintain_controlled_lead_speed()

            lead, lead_distance = get_lead_vehicle()
            ego_speed = speed_mps(vehicle)
            lead_speed = speed_mps(lead) if lead is not None else 0.0
            closing_speed = max(0.0, ego_speed - lead_speed)

            risk, physical_distance, ttc = calculate_physical_risk(
                lead_distance if lead is not None else float("inf"),
                closing_speed,
            )
            light_state, light_distance = upcoming_traffic_light()
            pedestrian, pedestrian_distance, pedestrian_lateral = (
                get_pedestrian_hazard()
            )
            lane_status, lane_offset, lane_width, lane_heading_error = (
                physical_lane_status()
            )

            # Keep the original apply_control arbitration/state machine.
            # It still owns ACC, AEB, traffic lights, overtaking, pedestrian
            # escape/reverse and route steering exactly as before.
            control_info = apply_control(
                lead,
                lead_distance,
                risk,
                ttc,
                0.0,
                light_state,
                light_distance,
                pedestrian=pedestrian,
                pedestrian_distance=pedestrian_distance,
                pedestrian_lateral=pedestrian_lateral,
            )

            if lead is not None:
                try:
                    log_risk(lead.id, risk, physical_distance)
                    if risk >= 0.55 or (
                        math.isfinite(ttc) and ttc < 3.0
                    ):
                        generate_warning(
                            lead.id, risk, physical_distance, ttc
                        )
                except Exception:
                    pass

            snapshot = {
                "action": control_info["action"],
                "lane_change": control_info["lane_change"],
                "steer": control_info["steer"],
                "throttle": control_info["throttle"],
                "brake": control_info["brake"],
                "lead_id": lead.id if lead is not None else None,
                "lead_distance": lead_distance,
                "risk": risk,
                "distance": physical_distance,
                "ttc": ttc,
                "traffic_light": light_state,
                "light_distance": light_distance,
                "pedestrian_id": (
                    pedestrian.id if pedestrian is not None else None
                ),
                "pedestrian_distance": pedestrian_distance,
                "pedestrian_lateral": pedestrian_lateral,
                "pedestrian_conflict": (
                    pedestrian is not None
                    and math.isfinite(pedestrian_distance)
                    and pedestrian_distance < PEDESTRIAN_HARD_BRAKE_DISTANCE
                ),
                "lane_status": lane_status,
                "lane_offset": lane_offset,
                "lane_width": lane_width,
                "lane_heading_error": lane_heading_error,
                "speed_kmh": ego_speed * 3.6,
                "vehicle_throttle": control_info["throttle"],
                "vehicle_brake": control_info["brake"],
            }

            with control_snapshot_lock:
                control_snapshot = snapshot

        except Exception as exc:
            import traceback
            print(
                f"ADAS control loop error: {type(exc).__name__}: {exc}"
            )
            traceback.print_exc()

        elapsed = time.monotonic() - tick_start
        time.sleep(max(0.001, target_interval - elapsed))



def _overtake_demo_lane_candidate(ego_wp, start_distance):
    """Find a forward ego-lane waypoint with a long, same-direction
    adjacent driving lane that remains usable for the complete demo.

    This helper only selects the test scenario geometry. It never changes
    lane_is_safe(), choose_safe_lane(), or update_lane_change().
    """
    if ego_wp is None:
        return None
    try:
        nxts = ego_wp.next(start_distance)
    except RuntimeError:
        return None
    if not nxts:
        return None

    candidates = sorted(
        nxts,
        key=lambda w: abs(_yaw_diff(ego_wp.transform.rotation.yaw, w.transform.rotation.yaw))
    )
    for cand in candidates:
        if cand.is_junction or cand.lane_type != carla.LaneType.Driving:
            continue

        for direction, target in (("LEFT", cand.get_left_lane()), ("RIGHT", cand.get_right_lane())):
            if target is None:
                continue
            if target.lane_type != carla.LaneType.Driving:
                continue
            if not _same_direction(target, cand):
                continue
            if target.is_junction:
                continue
            if not (2.5 <= target.lane_width <= 4.5):
                continue

            # Verify the road segment between ego and the lead is also free of
            # junctions. The previous build only checked the road AFTER the
            # lead candidate, which allowed a lead to be spawned ~40m ahead
            # while ego entered a junction before reaching it.
            probe = ego_wp
            corridor_ok = True
            try:
                corridor_distance = max(0.0, float(start_distance))
                corridor_steps = max(2, int(math.ceil(corridor_distance / 6.0)))
                for _ in range(corridor_steps):
                    if probe.is_junction:
                        corridor_ok = False
                        break
                    nxts_corridor = probe.next(min(6.0, corridor_distance))
                    if not nxts_corridor:
                        corridor_ok = False
                        break
                    probe = min(
                        nxts_corridor,
                        key=lambda w: abs(_yaw_diff(
                            probe.transform.rotation.yaw,
                            w.transform.rotation.yaw
                        ))
                    )
                if probe.is_junction:
                    corridor_ok = False
            except RuntimeError:
                corridor_ok = False
            if not corridor_ok:
                continue

            # Verify that the target lane continues for the whole planned
            # overtaking window and does not immediately enter a junction.
            probe = target
            ok = True
            steps = max(10, int(CONTROLLED_OVERTAKE_MIN_STRAIGHT_M / 8.0))
            for _ in range(steps):
                try:
                    nxt = probe.next(8.0)
                except RuntimeError:
                    ok = False
                    break
                if not nxt:
                    ok = False
                    break
                probe = min(
                    nxt,
                    key=lambda w: abs(_yaw_diff(probe.transform.rotation.yaw, w.transform.rotation.yaw))
                )
                if probe.is_junction:
                    ok = False
                    break
            if not ok:
                continue

            # Also keep the ego lane itself out of junctions for the same
            # forward window. This prevents the controlled lead from being
            # spawned just before a junction even when the target lane is long.
            probe = cand
            for _ in range(steps):
                if probe.is_junction:
                    ok = False
                    break
                try:
                    nxt = probe.next(8.0)
                except RuntimeError:
                    ok = False
                    break
                if not nxt:
                    ok = False
                    break
                probe = min(
                    nxt,
                    key=lambda w: abs(_yaw_diff(probe.transform.rotation.yaw, w.transform.rotation.yaw))
                )
            if not ok:
                continue

            return cand, direction, target
    return None


def configure_slow_traffic(traffic):
    """Keep normal traffic under traffic_manager.py speed control.

    TrafficManager owns NPC longitudinal speed. The desired speed is set
    in traffic_manager.py in km/h (CARLA API units); this function only reports
    the active configuration and never overrides it.
    """
    if not traffic:
        return

    alive = sum(1 for actor in traffic if actor is not None and actor.is_alive)
    print(
        f"Traffic speed control: {alive}/{len(traffic)} vehicles | "
        "TrafficManager desired speed=14–18 km/h slow / 25–30 km/h normal"
    )


def spawn_controlled_overtake_lead():
    """Create a deterministic, repeatable slow lead for a real overtake demo.

    Scenario generation is strengthened here only: the lead is placed farther
    ahead, on a straight non-junction section, with a verified same-direction
    adjacent lane and large initial gaps. The existing overtake state machine
    and all safety gates remain untouched.
    """
    global scenario_lead
    if not CONTROLLED_OVERTAKE_TEST or not actor_alive(vehicle):
        return None

    try:
        ego_wp = carla_map.get_waypoint(
            vehicle.get_location(),
            project_to_road=True,
            lane_type=carla.LaneType.Driving,
        )
        if ego_wp is None or ego_wp.is_junction:
            return None

        # Search several forward positions. If traffic occupies one target
        # lane, another clean section is selected instead of forcing the
        # state machine into a blocked scenario.
        selected = None
        for start_distance in (18.0, 20.0, 22.0, 24.0, 26.0, 28.0):
            candidate = _overtake_demo_lane_candidate(ego_wp, start_distance)
            if candidate is None:
                continue
            cand, direction, target_lane = candidate

            # The target lane must already be comfortably clear. These are
            # scenario-setup margins; lane_is_safe() still performs the real
            # runtime safety decision later.
            front_gap, rear_gap, _ = _lane_gap_metrics(target_lane)
            if front_gap < CONTROLLED_OVERTAKE_MIN_TARGET_FRONT_GAP_M:
                continue
            if rear_gap < CONTROLLED_OVERTAKE_MIN_TARGET_REAR_GAP_M:
                continue
            if _lane_has_pedestrian(target_lane, front_range=35.0, rear_range=25.0):
                continue

            selected = (cand, direction, target_lane)
            break

        if selected is None:
            print("CONTROLLED OVERTAKE LEAD: no long clear same-direction lane found")
            return None

        chosen, target_direction, target_lane = selected
        tr = vehicle.get_transform()
        loc = chosen.transform.location
        dx = loc.x - tr.location.x
        dy = loc.y - tr.location.y
        yaw = math.radians(tr.rotation.yaw)
        forward = dx * math.cos(yaw) + dy * math.sin(yaw)
        if forward < 18.0:
            print(f"CONTROLLED OVERTAKE LEAD: candidate too close ({forward:.1f}m)")
            return None

        bps = (
            bp_global.filter("vehicle.*")
            if 'bp_global' in globals() and bp_global is not None
            else world.get_blueprint_library().filter("vehicle.*")
        )
        if not bps:
            return None

        preferred = [
            b for b in bps
            if any(k in b.id.lower() for k in ("sedan", "model3", "audi", "tesla", "lincoln"))
        ]
        lead_bp = random.choice(preferred or list(bps))
        if lead_bp.has_attribute("color"):
            lead_bp.set_attribute(
                "color",
                random.choice(lead_bp.get_attribute("color").recommended_values),
            )

        spawn_tf = chosen.transform
        spawn_tf.location.z += 0.35
        lead = world.try_spawn_actor(lead_bp, spawn_tf)
        if lead is None:
            print("CONTROLLED OVERTAKE LEAD: spawn blocked at selected clean waypoint")
            return None

        # IMPORTANT: do not let Traffic Manager choose the lead speed.
        # Previous runs showed 45% speed reduction still producing wildly
        # varying lead speeds (0.28 -> 5.39 m/s), which made the overtake
        # scenario non-deterministic. The controlled lead is a validation
        # actor, so hold it at a fixed physical speed while preserving all
        # ego-side safety logic.
        lead.set_autopilot(False)
        try:
            fwd = spawn_tf.get_forward_vector()
            lead.set_target_velocity(
                carla.Vector3D(
                    fwd.x * CONTROLLED_LEAD_TARGET_SPEED_MPS,
                    fwd.y * CONTROLLED_LEAD_TARGET_SPEED_MPS,
                    0.0,
                )
            )
        except RuntimeError:
            pass

        scenario_lead = lead
        actor_list.append(lead)
        print(
            f"CONTROLLED OVERTAKE LEAD: {lead.type_id} | "
            f"distance={forward:.1f}m | target_speed={CONTROLLED_LEAD_TARGET_SPEED_MPS:.1f}m/s "
            f"({CONTROLLED_LEAD_TARGET_SPEED_MPS * 3.6:.1f}km/h) | "
            f"target={target_direction} | target_front={front_gap:.1f}m | target_rear={rear_gap:.1f}m"
        )

        print(
            "CONTROLLED OVERTAKE SCENARIO: straight road + same-direction "
            "adjacent lane + clear target gaps"
        )
        print(
            "CONTROLLED LEAD ASSOCIATION: CARLA actor-id priority enabled "
            "(same-lane physical validation required)"
        )
        return lead
    except Exception as exc:
        print(f"Controlled overtake lead not spawned: {exc}")
        return None


def maintain_controlled_lead_speed():
    """Keep only the validation lead at a gentle constant speed.

    This is scenario generation, not ADAS control. It prevents CARLA Traffic
    Manager from accelerating the lead back toward ego speed during the test.
    """
    if not CONTROLLED_OVERTAKE_TEST or not actor_alive(scenario_lead):
        return
    try:
        tr = scenario_lead.get_transform()
        fwd = tr.get_forward_vector()
        scenario_lead.set_target_velocity(
            carla.Vector3D(
                fwd.x * CONTROLLED_LEAD_TARGET_SPEED_MPS,
                fwd.y * CONTROLLED_LEAD_TARGET_SPEED_MPS,
                0.0,
            )
        )
    except RuntimeError:
        pass


def clear_existing_vehicles():
    """Remove existing CARLA vehicle actors for a clean ADAS safety demo."""
    if world is None:
        return 0
    removed = 0
    try:
        for actor in world.get_actors().filter("vehicle.*"):
            try:
                if actor.is_alive:
                    actor.destroy()
                    removed += 1
            except RuntimeError:
                pass
    except RuntimeError:
        pass
    return removed


def choose_spawn_point():
    """Choose an ego spawn that is suitable for the controlled overtake demo.

    Prefer a long straight section with a same-direction adjacent driving lane.
    Fall back to the previous non-junction spawn rule if the map cannot provide
    such a section. No controller or safety threshold is changed here.
    """
    points = carla_map.get_spawn_points()
    random.shuffle(points)

    for sp in points:
        try:
            wp = carla_map.get_waypoint(
                sp.location,
                project_to_road=True,
                lane_type=carla.LaneType.Driving,
            )
            if wp is None or wp.is_junction:
                continue
            if not (2.5 <= wp.lane_width <= 4.5):
                continue
            if _overtake_demo_lane_candidate(wp, 22.0) is not None:
                return sp
        except Exception:
            continue

    # During controlled overtake validation, NEVER fall back to an arbitrary
    # spawn. An arbitrary fallback is exactly what allowed earlier runs to
    # start in a city section where the controlled lead could not be created.
    if CONTROLLED_OVERTAKE_TEST:
        print("CONTROLLED OVERTAKE: no valid straight two-lane spawn found")
        return None

    for sp in points:
        try:
            wp = carla_map.get_waypoint(
                sp.location,
                project_to_road=True,
                lane_type=carla.LaneType.Driving,
            )
            if wp is None or wp.is_junction:
                continue
            if not (2.5 <= wp.lane_width <= 4.5):
                continue
            return sp
        except Exception:
            continue
    return points[0] if points else None


def cleanup():
    stop_event.set()
    for sensor in (camera, chase_camera, obstacle_sensor):
        try:
            if sensor is not None and sensor.is_alive:
                sensor.stop()
        except RuntimeError:
            pass
    time.sleep(0.25)
    try:
        shutdown_logger()
    except Exception:
        pass
    for actor in reversed(actor_list):
        try:
            if actor.is_alive:
                actor.destroy()
        except RuntimeError:
            pass
    cv2.destroyAllWindows()



def _fit_image(image, width, height, background=(10, 14, 20)):
    """Resize without distortion and center inside the requested rectangle."""
    canvas = np.full((height, width, 3), background, dtype=np.uint8)
    if image is None or image.size == 0:
        return canvas
    ih, iw = image.shape[:2]
    scale = min(width / float(iw), height / float(ih))
    nw, nh = max(1, int(iw * scale)), max(1, int(ih * scale))
    resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_AREA)
    x = (width - nw) // 2
    y = (height - nh) // 2
    canvas[y:y + nh, x:x + nw] = resized
    return canvas


def _label_box(canvas, title, x, y, w, accent=(80, 220, 255)):
    cv2.rectangle(canvas, (x, y), (x + w, y + 30), (15, 21, 29), -1)
    cv2.rectangle(canvas, (x, y), (x + w, y + 30), accent, 1)
    cv2.putText(
        canvas, title, (x + 12, y + 21),
        cv2.FONT_HERSHEY_DUPLEX, 0.52, (235, 242, 248), 1, cv2.LINE_AA
    )


def compose_operator_view(front, chase, dashboard, telemetry=None):
    """Full-frame operator layout: front view + chase/telemetry + live dashboard.

    The previous layout intentionally capped the front camera at ~600 px and
    left the remaining lower-left area unused. This version uses that space
    for the secondary chase view and live system telemetry without changing
    any ADAS control logic.
    """
    OUT_W, OUT_H = 1600, 900
    margin, gap, dash_w = 8, 10, 540
    left_w = OUT_W - 2 * margin - gap - dash_w
    out = np.full((OUT_H, OUT_W, 3), (7, 11, 16), dtype=np.uint8)
    lx, rx = margin, margin + left_w + gap

    _label_box(out, "FRONT CAMERA  |  HD PERCEPTION", lx, margin, left_w)
    _label_box(out, "ADAS COPILOT  |  LIVE SAFETY", rx, margin, dash_w)

    fy = margin + 34
    # Keep the camera undistorted. The unused vertical space becomes the
    # dedicated chase/telemetry area below it.
    front_h = min(580, int(left_w / (16 / 9)))
    front_panel = _fit_image(front, left_w, front_h)
    out[fy:fy + front_h, lx:lx + left_w] = front_panel

    bottom_y = fy + front_h + gap
    bottom_h = OUT_H - bottom_y - margin
    bottom_left_w = int(left_w * 0.52) - gap // 2
    bottom_right_w = left_w - bottom_left_w - gap

    # Chase view fills the lower-left area instead of floating over the front.
    chase_panel = _fit_image(
        chase, bottom_left_w, bottom_h, background=(12, 16, 22)
    )
    cv2.rectangle(
        out,
        (lx - 1, bottom_y - 1),
        (lx + bottom_left_w + 1, bottom_y + bottom_h + 1),
        (35, 45, 56), 1, cv2.LINE_AA
    )
    cv2.rectangle(
        out,
        (lx, bottom_y),
        (lx + bottom_left_w, bottom_y + 26),
        (18, 25, 34), -1
    )
    cv2.putText(
        out, "CHASE  |  SECONDARY",
        (lx + 10, bottom_y + 18),
        cv2.FONT_HERSHEY_DUPLEX, .46, (225, 232, 240), 1, cv2.LINE_AA
    )
    out[bottom_y + 27:bottom_y + bottom_h, lx:lx + bottom_left_w] = chase_panel[27:, :]

    # Live telemetry occupies the remaining lower-left space. These values
    # come from the same control/perception snapshot shown in the dashboard.
    tx = lx + bottom_left_w + gap
    tw = bottom_right_w
    th = bottom_h
    cv2.rectangle(
        out, (tx, bottom_y), (tx + tw, bottom_y + th),
        (18, 25, 34), -1
    )
    cv2.rectangle(
        out, (tx, bottom_y), (tx + tw, bottom_y + th),
        (39, 50, 64), 1, cv2.LINE_AA
    )
    cv2.putText(
        out, "LIVE SYSTEM TELEMETRY",
        (tx + 14, bottom_y + 24),
        cv2.FONT_HERSHEY_DUPLEX, .50, (225, 232, 240), 1, cv2.LINE_AA
    )

    telemetry = telemetry or {}
    cards = [
        ("SPEED", f'{float(telemetry.get("speed", 0.0)):.0f} km/h', (224, 204, 36)),
        ("ACTION", str(telemetry.get("action", "CRUISE"))[:18], (92, 214, 126)),
        ("LANE", str(telemetry.get("lane", "UNKNOWN"))[:14], (92, 214, 126)),
        ("TTC", "INF" if not math.isfinite(float(telemetry.get("ttc", float("inf"))))
                else f'{float(telemetry.get("ttc", 0.0)):.1f} s', (224, 204, 36)),
        ("LEAD GAP", "INF" if not math.isfinite(float(telemetry.get("gap", float("inf"))))
                    else f'{float(telemetry.get("gap", 0.0)):.1f} m', (224, 204, 36)),
        ("OBJECTS", str(int(telemetry.get("objects", 0))), (224, 204, 36)),
        ("LIGHT", str(telemetry.get("light", "NONE"))[:10], (92, 214, 126)),
        ("OVERTAKE", str(telemetry.get("lane_change", "NONE"))[:12], (70, 190, 245)),
    ]

    inner_x = tx + 12
    inner_y = bottom_y + 40
    cols = 2
    rows = 4
    card_gap = 8
    card_w = (tw - 2 * 12 - card_gap) // cols
    card_h = (th - 52 - 12 - (rows - 1) * card_gap) // rows

    for i, (label, value, color) in enumerate(cards):
        row, col = divmod(i, cols)
        x = inner_x + col * (card_w + card_gap)
        y = inner_y + row * (card_h + card_gap)
        cv2.rectangle(out, (x, y), (x + card_w, y + card_h), (23, 31, 42), -1)
        cv2.rectangle(out, (x, y), (x + card_w, y + card_h), (39, 50, 64), 1, cv2.LINE_AA)
        cv2.putText(out, label, (x + 9, y + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, .34, (135, 151, 168), 1, cv2.LINE_AA)
        cv2.putText(out, value, (x + 9, y + min(card_h - 8, 42)),
                    cv2.FONT_HERSHEY_DUPLEX, .48, color, 1, cv2.LINE_AA)

    # Right dashboard uses the full remaining height.
    dy = margin + 34
    dash_h = OUT_H - dy - margin
    dash_panel = _fit_image(
        dashboard, dash_w, dash_h, background=(9, 13, 19)
    )
    out[dy:dy + dash_h, rx:rx + dash_w] = dash_panel

    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    global client, world, carla_map, vehicle, camera, chase_camera, obstacle_sensor
    global actor_list, front_latest, chase_latest, bp_global, scenario_lead

    client = carla.Client("localhost", 2000)
    client.set_timeout(20.0)
    world = client.get_world()
    carla_map = world.get_map()

    settings = world.get_settings()
    if settings.no_rendering_mode:
        settings.no_rendering_mode = False
        world.apply_settings(settings)

    bp = world.get_blueprint_library()
    bp_global = bp

    if CLEAR_EXISTING_VEHICLES:
        removed = clear_existing_vehicles()
        if removed:
            print(f"Cleared existing CARLA vehicles: {removed}")

    vehicle_bp = bp.filter("model3")[0]
    spawn = choose_spawn_point()
    if spawn is None:
        raise RuntimeError("No CARLA spawn point available")

    vehicle = world.try_spawn_actor(vehicle_bp, spawn)
    if vehicle is None:
        raise RuntimeError("Could not spawn ego vehicle")
    actor_list.append(vehicle)
    world.wait_for_tick()
    vehicle.set_autopilot(False)
    init_route()

    # Short hand-brake settle; controller starts after this.
    vehicle.apply_control(carla.VehicleControl(hand_brake=True))
    time.sleep(0.6)

    if SPAWN_TRAFFIC_VEHICLES and not (CONTROLLED_OVERTAKE_TEST and ISOLATED_OVERTAKE_VALIDATION):
        traffic = spawn_traffic(
            client,
            world,
            bp,
            vehicle,
            number_of_vehicles=TRAFFIC_VEHICLE_COUNT,
        )
        actor_list.extend(traffic)
        print(f"Traffic vehicles spawned: {len(traffic)}")
        configure_slow_traffic(traffic)
    else:
        if CONTROLLED_OVERTAKE_TEST and ISOLATED_OVERTAKE_VALIDATION:
            print("Traffic vehicles: ISOLATED OVERTAKE VALIDATION (random traffic disabled)")
        else:
            print("Traffic vehicles: DISABLED")

    if CONTROLLED_OVERTAKE_TEST:
        spawn_controlled_overtake_lead()

    # HD front camera: 1280x720 at 30 FPS. YOLO sees only a 640x360 copy.
    camera_bp = bp.find("sensor.camera.rgb")
    camera_bp.set_attribute("image_size_x", str(CAMERA_W))
    camera_bp.set_attribute("image_size_y", str(CAMERA_H))
    camera_bp.set_attribute("fov", "92")
    camera_bp.set_attribute("sensor_tick", str(CAMERA_TICK))
    camera = world.spawn_actor(
        camera_bp,
        carla.Transform(
            carla.Location(x=2.2, z=1.75),
            carla.Rotation(pitch=-2.0),
        ),
        attach_to=vehicle,
    )
    actor_list.append(camera)
    camera.listen(camera_callback)

    # Cosmetic chase camera.
    chase_bp = bp.find("sensor.camera.rgb")
    chase_bp.set_attribute("image_size_x", "640")
    chase_bp.set_attribute("image_size_y", "360")
    chase_bp.set_attribute("fov", "95")
    chase_bp.set_attribute("sensor_tick", str(CAMERA_TICK))
    chase_camera = world.spawn_actor(
        chase_bp,
        carla.Transform(
            carla.Location(x=-7.0, z=3.2),
            carla.Rotation(pitch=-12.0),
        ),
        attach_to=vehicle,
        attachment_type=carla.AttachmentType.SpringArmGhost,
    )
    actor_list.append(chase_camera)
    chase_camera.listen(chase_callback)

    # Emergency physical obstacle backup.
    obstacle_bp = bp.find("sensor.other.obstacle")
    obstacle_bp.set_attribute("distance", "12")
    obstacle_bp.set_attribute("hit_radius", "0.35")
    obstacle_bp.set_attribute("only_dynamics", "False")
    obstacle_sensor = world.spawn_actor(
        obstacle_bp,
        carla.Transform(carla.Location(x=2.4, z=0.8)),
        attach_to=vehicle,
    )
    actor_list.append(obstacle_sensor)
    obstacle_sensor.listen(on_obstacle)

    # Control never waits for YOLO.
    controller = threading.Thread(
        target=control_loop,
        name="ADAS-Control",
        daemon=True,
    )
    controller.start()

    lane_worker = threading.Thread(
        target=lane_worker_loop,
        name="ADAS-Lane",
        daemon=True,
    )
    lane_worker.start()

    worker = threading.Thread(
        target=worker_loop,
        name="ADAS-Perception",
        daemon=True,
    )
    worker.start()

    print("=" * 72)
    print("ADAS COPILOT RUNNING")
    print(
        f"Front camera: {CAMERA_W}x{CAMERA_H} @ "
        f"{1.0 / CAMERA_TICK:.0f} FPS | "
        f"YOLO input: {YOLO_INPUT_W}x{YOLO_INPUT_H} | "
        f"imgsz: {YOLO_IMGSZ}"
    )
    print(
        f"Perception target: {PROCESS_HZ:.0f} FPS | "
        f"Control target: {CONTROL_HZ:.0f} Hz | "
        f"Display target: {DISPLAY_HZ:.0f} FPS"
    )
    print(
        f"Traffic: {TRAFFIC_VEHICLE_COUNT} vehicles | "
        "speed control: TrafficManager 14–18 km/h slow / 25–30 km/h normal | "
        f"Controlled slow lead: {CONTROLLED_OVERTAKE_TEST}"
    )
    print(
        "Safety: pedestrian > vehicle AEB > traffic light > "
        "safe overtake > ACC > cruise"
    )
    print("Perception: lane worker + latest-frame YOLO/ByteTrack | NO blocking warm-up")
    print(f"Detection model: {YOLO_MODEL_PATH} | conf={YOLO_CONF:.2f}")
    print("Lane visual: detector overlay + display-only road-line fallback")
    print("Validation logging: safety/overtake/pedestrian state transitions")
    print("Press Q or ESC in ADAS window to stop")
    print("=" * 72)

    WINDOW_NAME = "ADAS COPILOT | SAFETY MONITOR"
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, 1600, 900)
    cv2.moveWindow(WINDOW_NAME, 20, 20)

    ui_interval = 1.0 / max(DISPLAY_HZ, 1.0)

    while not stop_event.is_set():
        loop_start = time.monotonic()

        with front_lock:
            raw_front = (
                None if front_latest is None else front_latest.copy()
            )
        with chase_lock:
            chase = None if chase_latest is None else chase_latest.copy()
        with perception_lock:
            perception = {
                k: (v.copy() if isinstance(v, np.ndarray) else v)
                for k, v in latest_perception.items()
            }
        with lane_lock:
            lane_view = {
                k: (v.copy() if isinstance(v, np.ndarray) else v)
                for k, v in latest_lane.items()
            }
        with control_snapshot_lock:
            snap = dict(control_snapshot)

        if raw_front is not None:
            # Current HD camera frame is always shown. Only the newest
            # perception overlay is reused between YOLO frames.
            live_front = raw_front.copy()
            overlay = perception.get("overlay")
            overlay_mask = perception.get("overlay_mask")

            # Apply the newest lane overlay first, then the newest YOLO
            # overlay. Either subsystem can update independently.
            lane_overlay = lane_view.get("overlay")
            lane_mask = lane_view.get("overlay_mask")
            if (
                isinstance(lane_overlay, np.ndarray)
                and isinstance(lane_mask, np.ndarray)
                and lane_overlay.shape == live_front.shape
                and lane_mask.shape[:2] == live_front.shape[:2]
            ):
                live_front[lane_mask] = lane_overlay[lane_mask]

            if (
                isinstance(overlay, np.ndarray)
                and isinstance(overlay_mask, np.ndarray)
                and overlay.shape == live_front.shape
                and overlay_mask.shape[:2] == live_front.shape[:2]
            ):
                live_front[overlay_mask] = overlay[overlay_mask]
            else:
                state_text = str(perception.get("state") or "STARTING")
                if state_text == "ERROR":
                    state_text = "YOLO ERROR / RETRYING"
                cv2.putText(
                    live_front,
                    f"PERCEPTION: {state_text}",
                    (24, 42),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.78,
                    (0, 220, 255),
                    2,
                    cv2.LINE_AA,
                )

            # Keep the perception state visible even when the last completed
            # overlay is being reused between 15-FPS inference frames.
            perception_state = str(perception.get("state") or "STARTING")
            state_color = (0, 255, 0) if perception_state == "LIVE" else (0, 220, 255)
            cv2.putText(
                live_front,
                f"PERCEPTION: {perception_state}",
                (24, 32),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.68,
                state_color,
                2,
                cv2.LINE_AA,
            )

            lane_visual_mode = str(lane_view.get("visual_mode") or "NONE")
            lane_visual_color = (0, 255, 0) if lane_visual_mode != "NONE" else (0, 180, 255)
            cv2.putText(
                live_front,
                f"LANE VISUAL: {lane_visual_mode}",
                (24, 182),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                lane_visual_color,
                1,
                cv2.LINE_AA,
            )

            # Dashboard is cheap enough to regenerate at 30 FPS, so its
            # actuator/safety values stay live while YOLO runs at 15 FPS.
            dashboard = draw_dashboard(
                live_front,
                speed=float(snap["speed_kmh"]),
                fps=processed_fps,
                lane_status=snap["lane_status"],
                risk=snap["risk"],
                distance=snap["distance"],
                ttc=snap["ttc"],
                steering=(
                    "RIGHT" if snap["steer"] > 0.05
                    else "LEFT" if snap["steer"] < -0.05
                    else "STRAIGHT"
                ),
                throttle=float(snap["vehicle_throttle"]),
                brake=float(snap["vehicle_brake"]),
                tracked_objects=int(perception.get("tracked_count", 0)),
                lead_id=snap["lead_id"],
                action=snap["action"],
                traffic_light=snap["traffic_light"],
                lane_change=snap["lane_change"],
                pedestrian_id=snap["pedestrian_id"],
                pedestrian_distance=snap["pedestrian_distance"],
                pedestrian_conflict=snap["pedestrian_conflict"],
                yolo_person_visible=bool(
                    perception.get("yolo_person_visible", False)
                ),
            )

            operator_view = compose_operator_view(
                live_front,
                chase,
                dashboard,
                telemetry={
                    "speed": snap["speed_kmh"],
                    "action": snap["action"],
                    "lane": snap["lane_status"],
                    "ttc": snap["ttc"],
                    "gap": snap["distance"],
                    "objects": perception.get("tracked_count", 0),
                    "light": snap["traffic_light"],
                    "lane_change": snap["lane_change"],
                },
            )
            cv2.imshow(WINDOW_NAME, operator_view)

        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q")):
            stop_event.set()
            break

        if not actor_alive(vehicle):
            stop_event.set()
            break

        remaining = ui_interval - (time.monotonic() - loop_start)
        if remaining > 0:
            time.sleep(min(0.004, remaining))




if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Stopped by user")
    except Exception as exc:
        print(f"FATAL: {exc}")
        raise
    finally:
        cleanup()
        print("Cleaned up")
