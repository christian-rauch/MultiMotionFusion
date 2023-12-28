#!/usr/bin/env bash

set -e

sudo apt update
sudo -E apt install -y software-properties-common
sudo add-apt-repository -y universe
sudo apt update
sudo -E apt install -y curl wget

echo "install CUDA"
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
rm cuda-keyring_1.1-1_all.deb
sudo apt update
sudo apt install --no-install-recommends -y cuda-libraries-dev-11-8 cuda-compiler-11-8 cuda-nvtx-11-8 libcudnn8-dev

echo "install ROS"
sudo sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'
curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | sudo apt-key add -
sudo apt update
sudo -E apt install --no-install-recommends -y ros-noetic-ros-base ros-dev-tools
echo "source /opt/ros/noetic/setup.bash" >> ~/.bashrc

echo "initialise rosdep"
sudo rosdep init
rosdep update
