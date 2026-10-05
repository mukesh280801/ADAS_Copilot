import time
import threading

# adas_system/voice_alert.py already has a working pyttsx3-based
# speak() function (background thread, de-dup cooldown) -- it just
# was never actually imported/called from anywhere. Import failure
# (e.g. pyttsx3 not installed) degrades gracefully: the rest of the
# warning system keeps working with text-only output.
try:
    from adas_system.voice_alert import speak
    _voice_available = True
except Exception as e:
    print(f"[warning_system] voice alerts unavailable: {e}")
    _voice_available = False

    def speak(message):
        pass

# Every processed frame with risk_score >= 0.2 printed a full warning
# block from inside the camera callback thread -- that's sustained
# console I/O sitting directly in the hot path, contributing to the
# reported FPS instability. Non-critical warnings are now throttled
# per object; HIGH-risk / critical-TTC warnings are never throttled.
_last_warned = {}
_lock = threading.Lock()
_COOLDOWN_SECONDS = 1.5


def get_risk_level(score):
    if score >= 0.8:
        return "HIGH"
    elif score >= 0.5:
        return "MEDIUM"
    else:
        return "LOW"


def generate_warning(object_id, risk_score, min_distance, ttc=None):

    level = get_risk_level(risk_score)
    critical = level == "HIGH" or (ttc is not None and ttc < 2.0)

    now = time.time()
    with _lock:
        last = _last_warned.get(object_id, 0)
        if not critical and (now - last) < _COOLDOWN_SECONDS:
            return
        _last_warned[object_id] = now

    print("\n" + "=" * 40)
    print(f"OBJECT ID      : {object_id}")
    print(f"RISK SCORE     : {risk_score:.2f}")
    print(f"MIN DISTANCE   : {min_distance:.2f}")

    if ttc is not None:
        print(f"TTC            : {ttc:.2f} s")

    print(f"RISK LEVEL     : {level}")

    # ADAS DECISION LOGIC
    if ttc is not None and ttc < 2.0:
        action = "BRAKE IMMEDIATELY (TTC CRITICAL)"
        voice_message = "Warning. Collision risk critical. Brake immediately."

    elif level == "HIGH":
        action = "BRAKE IMMEDIATELY"
        voice_message = "Warning. Brake immediately."

    elif level == "MEDIUM":
        action = "SLOW DOWN"
        voice_message = "Caution. Slow down."

    else:
        action = "SAFE"
        voice_message = None

    print(f"ACTION         : {action}")
    print("=" * 40)

    if voice_message and _voice_available:
        speak(voice_message)


if __name__ == "__main__":

    generate_warning(
        object_id=5,
        risk_score=0.91,
        min_distance=4.5,
        ttc=None
    )
