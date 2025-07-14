# rlg_quad_controller Package
## TODO LIST:

- [ ] Add launch file and configuration file
- [ ] Test building a model
- [ ] Test on gazebo
- [ ] Setup services to stop stuff
- [ ] Infer model build params from rl_games config file
- [ ] Setup logging (inference time, cpu load, memory usage ...)
  
## rlg_quad_controller: inference_node

This package contains a ROS 2 node named `inference_controller` that controls a quadrupedal robot using an end-to-end reinforcement learning policy trained with `rl_games`. The node uses a PyTorch model to infer the target positions of the robot's joints and publishes these positions to a specified topic.

## Node: InferenceController

### Parameters

The `InferenceController` node uses the following parameters:

- `model_path` (string, default: ''): The path to the PyTorch model file (.pth) used for inference.
- `joint_state_topic` (string, default: '/joint_states'): The topic name where the current joint states are published.
- `joint_target_pos_topic` (string, default: '/joint_target_poss'): The topic name where the target joint positions are published.
- `cmd_vel_topic` (string, default: '/cmd_vel'): The topic name where the command velocities are published.
- `imu_topic` (string, default: '/imu'): The topic name where the IMU data is published.
- `rate` (integer, default: 10 [Hz] ): The rate at which the inference is performed.

### Published Topics

- `/joint_target_poss` (`JointsCommand`): The target positions for the robot's joints, inferred from the PyTorch model.

### Subscribed Topics

> these topics may vary depending on the observation space of the model.
- `/joint_states` (`JointsStates`): The current states of the robot's joints.
- `/cmd_vel` (`Twist`): The command velocities for the robot.
- `/imu` (`Imu`): The IMU data from the robot.

## Usage
To add a model, put the .pth file and the config.yaml file in the models directory, then add the names in the launch file.



## Dependencies

This package depends on the following ROS 2 packages:

- `rclpy`
- `sensor_msgs`
- `geometry_msgs`
- `pi3hat_moteus_int_msgs`

... TODO 

It also requires `PyTorch` and `numpy` Python libraries.

## Installation

TODO

## Running the Node

TODO
