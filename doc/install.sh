#!/usr/bin/env bash

set -e

source /etc/lsb-release

if [ "$DISTRIB_RELEASE" == "20.04" ]; then
    ROS_DIST="noetic"
elif [ "$DISTRIB_RELEASE" == "22.04" ]; then
    ROS_DIST="humble"
elif [ "$DISTRIB_RELEASE" == "24.04" ]; then
    ROS_DIST="jazzy"
else
    echo "unsupported Ubuntu distribution ($DISTRIB_RELEASE)"
    exit 1
fi

source /opt/ros/${ROS_DIST}/setup.bash

# determine repo from environment variables inside CI or use defaults
MMF_REPO_URL="${GITHUB_SERVER_URL:-https://github.com}/${GITHUB_REPOSITORY:-christian-rauch/MultiMotionFusion}"
MMF_BRANCH="${GITHUB_HEAD_REF:-master}"

echo "setup workspace"
mkdir -p ~/mmf_ws/
cd ~/mmf_ws/
vcs import << EOF
repositories:
  src/MultiMotionFusion:
    type: git
    url: ${MMF_REPO_URL}.git
    version: ${MMF_BRANCH}
  src/Pangolin:
    type: git
    url: https://github.com/stevenlovegrove/Pangolin.git
    version: v0.9.1
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
