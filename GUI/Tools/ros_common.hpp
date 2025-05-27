#ifdef ROSCOMMON

#pragma once

#include <opencv2/core.hpp>

#if __has_include(<image_geometry/pinhole_camera_model.h>)
#include <image_geometry/pinhole_camera_model.h>
#elif __has_include(<image_geometry/pinhole_camera_model.hpp>)
#include <image_geometry/pinhole_camera_model.hpp>
#else
#error "no 'pinhole_camera_model' header"
#endif

#include <Core/FrameData.h>

#if defined(ROS1)
    #include <sensor_msgs/CameraInfo.h>
    using namespace sensor_msgs;
    using CICPtr = CameraInfo::ConstPtr;
#elif defined(ROS2)
    #include <sensor_msgs/msg/camera_info.hpp>
    using namespace sensor_msgs::msg;
    using CICPtr = CameraInfo::ConstSharedPtr;
#endif

class ImageCropTarget {
public:
  ImageCropTarget() = default;

  ImageCropTarget(const CICPtr &camera_info, const cv::Size &target_dimensions);

  void map_target(FrameData &data);

private:
  image_geometry::PinholeCameraModel camera_model;
  cv::Rect crop_roi;
  cv::Size target_dimensions;
};

#endif // ROSCOMMON
