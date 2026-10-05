from collections import defaultdict, deque
import time


class TrajectoryBuffer:
    """Short history buffer with timestamps for each tracked object."""

    def __init__(self, max_len=10):
        self.max_len = max_len
        self.buffer = defaultdict(lambda: deque(maxlen=max_len))

    def update(self, object_id, x, y, timestamp=None):
        if timestamp is None:
            timestamp = time.monotonic()
        self.buffer[object_id].append((float(timestamp), float(x), float(y)))

    def get_trajectory(self, object_id, include_timestamps=False):
        items = list(self.buffer[object_id])
        if include_timestamps:
            return items
        return [(x, y) for _, x, y in items]

    def get_all_ids(self):
        return list(self.buffer.keys())

    def clear(self, object_id=None):
        if object_id is None:
            self.buffer.clear()
        else:
            self.buffer.pop(object_id, None)
