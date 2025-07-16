# Dockerized Gazebo Harmonic Simulator 
## introduction
This repository contains a docker compose structure to simulate robots in Gazebo Sim Fortress supported by ROS2 humble on ubuntu 22.04.
The Repository purpose is to provide an easy set up to allow robotic software developer to test their own controller in simulation.
## prerequisite  

Docker Engine: follow the docker engine installation [here](https://docs.docker.com/engine/install/ubuntu/) and then enable the docker usage as non-root user as specified [here](https://docs.docker.com/engine/install/linux-postinstall/)

Install Nvidia Driver: TODO

Nvidia-Container-Toolkit: follow the nvidia-container-toolkit following [this](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)

## Build 

``` bash compose_build.bash ```

## Start Compose 

``` docker compose up -d ```

``` docker compose exec ros2_humble_sim bash```

This command launches a docker with ros inside it. The first time you open it, you have to run 

``` colcon build```


## NVIDIA stuff test 

To test the correct isntallation of the nvidia support you have to start the container and call

``` nvidia-smi```


## LUNCH ROBOT SIMULATION

#### TERMINAL 1:
This terminal is used to run Gazebo

``` source install/setup.bash ```

``` ros2 launch mulinex_ignition gz_harmonic_sim_w_rbt.launch.py ```
#### TERMINAL 2:
This terminal is used to run the node to move the robot

``` source install/setup.bash ```

``` ros2 run mulinex_ignition getup_node ```


## TROUBLESHOOT
If ``` colcon build ``` does not work the first time you launch it during the installation, try to do: 

``` docker rm -f ros2_humble_simulator``` 

``` docker compose up -d``` 

and then, again: ``` docker compose exec ros2_humble_sim bash```
