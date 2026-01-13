#!/usr/bin/env bash

set -e

which sudo || apt update && apt install -y sudo
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

CUDA_VER="13-3"

echo "install CUDA"
wget https://developer.download.nvidia.com/compute/cuda/repos/${CUDA_REPO_VER}/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
rm cuda-keyring_1.1-1_all.deb
sudo apt update
sudo apt install --no-install-recommends -y cuda-libraries-dev-${CUDA_VER} cuda-compiler-${CUDA_VER} cuda-nvtx-${CUDA_VER}

echo "install ROS ${ROS_VER}"
if [ "${ROS_VER}" = "" ]; then
    sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros${ROS_VER}/ubuntu $(lsb_release -sc) main" | sudo tee /etc/apt/sources.list.d/ros${ROS_VER}.list > /dev/null
elif [ "${ROS_VER}" = "2" ]; then
    sudo apt install --no-install-recommends -y curl
    export ROS_APT_SOURCE_VERSION=$(curl -s https://api.github.com/repos/ros-infrastructure/ros-apt-source/releases/latest | grep -F "tag_name" | awk -F\" '{print $4}')
    curl -L -o /tmp/ros2-apt-source.deb "https://github.com/ros-infrastructure/ros-apt-source/releases/download/${ROS_APT_SOURCE_VERSION}/ros2-apt-source_${ROS_APT_SOURCE_VERSION}.$(. /etc/os-release && echo ${UBUNTU_CODENAME:-${VERSION_CODENAME}})_all.deb"
    sudo apt install --no-install-recommends -y /tmp/ros2-apt-source.deb
else
    echo "unsupported ROS version"
    exit 1
fi
sudo apt update
sudo -E apt install --no-install-recommends -y ros-${ROS_DIST}-ros-base ros-dev-tools
echo "source /opt/ros/${ROS_DIST}/setup.bash" >> ~/.bashrc

echo "initialise rosdep"
sudo rosdep init
rosdep update --include-eol-distros
