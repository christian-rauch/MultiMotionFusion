#!/usr/bin/env bash

set -e

echo "setup workspace"
source /opt/ros/noetic/setup.bash
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
source /opt/ros/noetic/setup.bash
export CUDACXX=/usr/local/cuda-11.3/bin/nvcc
cd ~/mmf_ws/
colcon build --cmake-args "-DCMAKE_BUILD_TYPE=Release"
