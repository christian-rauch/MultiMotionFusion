#!/usr/bin/env bash

set -e

source /etc/lsb-release

if [ "$DISTRIB_RELEASE" = "20.04" ]; then
    ROS_DIST="noetic"
elif [ "$DISTRIB_RELEASE" = "22.04" ]; then
    ROS_DIST="humble"
else
    echo "unsupported Ubuntu distribution"
    exit 1
fi

source /opt/ros/${ROS_DIST}/setup.bash

echo "setup workspace"
mkdir -p ~/mmf_ws/
cd ~/mmf_ws/
vcs import << EOF
repositories:
  src/MultiMotionFusion:
    type: git
    url: https://github.com/christian-rauch/MultiMotionFusion.git
    version: master
  src/Pangolin:
    type: git
    url: https://github.com/stevenlovegrove/Pangolin.git
    version: v0.8
  src/densecrf:
    type: git
    url: https://github.com/christian-rauch/densecrf.git
    version: master
  src/gSLICr:
    type: git
    url: https://github.com/christian-rauch/gSLICr.git
    version: colcon
  src/super_point_inference:
    type: git
    url: https://github.com/christian-rauch/super_point_inference.git
    version: master
  src/torch_cpp:
    type: git
    url: https://github.com/christian-rauch/torch_cpp.git
    version: master
EOF
rosdep install --from-paths src --ignore-src -y

echo "build workspace"
cd ~/mmf_ws/
colcon build --cmake-args -D CMAKE_BUILD_TYPE=Release -D CMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc
