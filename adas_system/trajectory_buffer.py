from collections import defaultdict, deque

class TrajectoryBuffer:

    def __init__(self, max_len=10):

        self.max_len = max_len

        self.buffer = defaultdict(
            lambda: deque(maxlen=max_len)
        )

    def update(
        self,
        object_id,
        x,
        y
    ):

        self.buffer[object_id].append(
            (x, y)
        )

    def get_trajectory(
        self,
        object_id
    ):

        return list(
            self.buffer[object_id]
        )

    def get_all_ids(self):

        return list(
            self.buffer.keys()
        )