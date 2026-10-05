import csv
import os
import threading
import queue
import atexit
from datetime import datetime

FILE_NAME = "risk_events.csv"

# The original log_risk() opened and closed the CSV file on every
# single call -- a blocking disk write happening directly inside the
# camera callback thread on every risk-scored detection. That's a
# direct, concrete contributor to the reported FPS instability.
# Logging is now handed off to a background writer thread through a
# queue so process_img() never blocks on disk I/O.
_log_queue = queue.Queue()
_writer_thread = None
_stop_event = threading.Event()
_start_lock = threading.Lock()


def _writer_loop():
    file_exists = os.path.exists(FILE_NAME)

    with open(FILE_NAME, "a", newline="") as f:
        writer = csv.writer(f)

        if not file_exists:
            writer.writerow(
                ["timestamp", "object_id", "risk_score", "min_distance"]
            )
            f.flush()

        while not _stop_event.is_set() or not _log_queue.empty():
            try:
                row = _log_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            writer.writerow(row)
            f.flush()
            _log_queue.task_done()


def _ensure_writer_started():
    global _writer_thread
    if _writer_thread is not None:
        return
    with _start_lock:
        if _writer_thread is None:
            _writer_thread = threading.Thread(target=_writer_loop, daemon=True)
            _writer_thread.start()


def log_risk(
    object_id,
    risk_score,
    min_distance
):
    _ensure_writer_started()

    row = [
        datetime.now().strftime("%H:%M:%S"),
        object_id,
        round(risk_score, 4),
        round(min_distance, 4)
    ]

    try:
        _log_queue.put_nowait(row)
    except queue.Full:
        pass


def shutdown_logger():
    """Call during cleanup so buffered rows are flushed and the
    writer thread exits instead of being killed mid-write."""
    _stop_event.set()
    if _writer_thread is not None:
        _writer_thread.join(timeout=2.0)


atexit.register(shutdown_logger)
