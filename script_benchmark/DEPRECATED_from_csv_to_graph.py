import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("/home/simone/docker_simulator/rosbag/REAL_ROBOT_rough_GIOVEDI/incline_5deg/benchmark_vel_x_incline_5deg_v1/rosbag_data.csv")

t = df["__time"] - df["__time"].min()   # tempo relativo

# cmd_vel linear
plt.plot(t, df["/cmd_vel/linear/x"], label="linear_x")
plt.plot(t, df["/cmd_vel/linear/y"], label="linear_y")
plt.plot(t, df["/cmd_vel/linear/z"], label="linear_z")
plt.legend(); plt.title("/cmd_vel linear"); plt.show()

# rigid body position
plt.plot(t, df["/rigid_bodies/rigidbodies[3]/pose/position/x"], label="x")
plt.plot(t, df["/rigid_bodies/rigidbodies[3]/pose/position/y"], label="y")
plt.plot(t, df["/rigid_bodies/rigidbodies[3]/pose/position/z"], label="z")
plt.legend(); plt.title("Rigid body[3] position"); plt.show()

# joint velocities
for idx in [8,9,10,11]:
    plt.plot(t, df[f"/state_broadcaster/joints_state/velocity[{idx}]"], label=f"vel[{idx}]")
plt.legend(); plt.title("Joint velocities"); plt.show()
