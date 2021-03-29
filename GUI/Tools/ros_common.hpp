#ifdef ROSCOMMON

#pragma once

#include <opencv2/core.hpp>
#include <sensor_msgs/CameraInfo.h>

bool
intrinsics_crop_target(const sensor_msgs::CameraInfo::ConstPtr &ci, const cv::Size &target_dimensions, cv::Rect &crop_roi);

#endif // ROSCOMMON
