#!/usr/bin/env bash
MultiMotionFusion -run -dim 640x480 \
  -topic_colour /rgb/image_raw/compressed \
  -topic_depth /depth_to_rgb/image_raw/filtered/compressed \
  -topic_info /rgb/camera_info \
  -model ~/mmf_ws/install/super_point_inference/share/weights/SuperPointNet.pt \
  -init kp -icp_refine \
  -fs \
  -l $1
