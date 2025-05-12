#!/usr/bin/env bash

set -e

sudo apt update
sudo -E apt install -y software-properties-common
sudo add-apt-repository -y universe
sudo apt update
sudo -E apt install -y curl wget

source /etc/lsb-release

if [ "$DISTRIB_RELEASE" = "20.04" ]; then
    CUDA_REPO_VER="ubuntu2004"
    ROS_VER=""
    ROS_DIST="noetic"
elif [ "$DISTRIB_RELEASE" = "22.04" ]; then
    CUDA_REPO_VER="ubuntu2204"
    ROS_VER="2"
    ROS_DIST="humble"
elif [ "$DISTRIB_RELEASE" == "24.04" ]; then
    CUDA_REPO_VER="ubuntu2404"
    ROS_VER="2"
    ROS_DIST="jazzy"
else
    echo "unsupported Ubuntu distribution ($DISTRIB_RELEASE)"
    exit 1
fi

echo "install CUDA"
wget https://developer.download.nvidia.com/compute/cuda/repos/${CUDA_REPO_VER}/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
rm cuda-keyring_1.1-1_all.deb
sudo apt update
sudo apt install --no-install-recommends -y cuda-toolkit-12

echo "install ROS ${ROS_VER}"
sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros${ROS_VER}/ubuntu $(lsb_release -sc) main" | sudo tee /etc/apt/sources.list.d/ros${ROS_VER}.list > /dev/null
sudo apt update
sudo -E apt install --no-install-recommends -y ros-${ROS_DIST}-ros-base ros-dev-tools
echo "source /opt/ros/${ROS_DIST}/setup.bash" >> ~/.bashrc

echo "initialise rosdep"
sudo rosdep init
rosdep update
