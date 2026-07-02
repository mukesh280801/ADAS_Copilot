from adas_system.trajectory_buffer import TrajectoryBuffer

buf = TrajectoryBuffer(max_len=10)

for i in range(15):
    buf.update(5, i, i + 10)

print("Trajectory:")
print(buf.get_trajectory(5))