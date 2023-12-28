#!/usr/bin/env bash

set -e

echo "install CUDA"
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub
sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/ /"
sudo apt install --no-install-recommends -y cuda-libraries-dev-11-3 cuda-compiler-11-3 cuda-nvtx-11-3 libcudnn8-dev

echo "install ROS"
sudo sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'
curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | sudo apt-key add -
sudo apt update
sudo apt install --no-install-recommends -y ros-noetic-ros-base
echo "source /opt/ros/noetic/setup.bash" >> ~/.bashrc

echo "install vcstool, rosdep, colcon"
sudo pip3 install -U vcstool rosdep colcon-common-extensions
sudo rosdep init
rosdep update
