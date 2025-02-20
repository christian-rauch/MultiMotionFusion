#ifdef ROSCOMMON

#include "ros_common.hpp"
#include <Eigen/Core>
#include <Utils/Resolution.h>
#include <Utils/Intrinsics.h>
#include <iostream>

static std::tuple<cv::Rect, double>
get_crop_roi(const cv::Size &source_dimensions, const cv::Size &target_dimensions) {
  // scale ratios
  const double r_w = double(source_dimensions.width) / double(target_dimensions.width);
  const double r_h = double(source_dimensions.height) / double(target_dimensions.height);

  // centred ROI with aspect ratio of target dimensions
  cv::Rect crop_roi;

  // scale of cropped dimensions = source / target
  double scale = 0;

  // aspect_ratio = width / height;

  if (r_w>r_h) {
    // crop width (left, right)
    crop_roi.width = source_dimensions.height * target_dimensions.aspectRatio();
    crop_roi.height = source_dimensions.height;
    crop_roi.x = (source_dimensions.width-crop_roi.width) / 2;
    crop_roi.y = 0;
    scale = r_h;
  }
  else if (r_h>r_w) {
    // crop height (top, bottom)
    crop_roi.width = source_dimensions.width;
    crop_roi.height = source_dimensions.width / target_dimensions.aspectRatio();
    crop_roi.x = 0;
    crop_roi.y = (source_dimensions.height-crop_roi.height) / 2;
    scale = r_w;
  }
  else {
    // equal aspect ratio, no crop
    crop_roi = cv::Rect(cv::Size(), source_dimensions);
  }

  return {crop_roi, scale};
}

ImageCropTarget::ImageCropTarget(const CameraInfo::ConstPtr &camera_info,
                                 const cv::Size &target_dimensions)
  : target_dimensions(target_dimensions)
{
  camera_model.fromCameraInfo(camera_info);

  // Projection/camera matrix
  //     [fx'  0  cx' Tx]
  // P = [ 0  fy' cy' Ty]
  //     [ 0   0   1   0]

  // source intrinsics
  const cv::Size source_dimensions(camera_info->width, camera_info->height);
  // 'P' row-major 3x4 projection matrix: (fx, 0, cx, Tx, 0, fy, cy, Ty, 0, 0, 1, 0)
#if defined(ROS1)
  const Eigen::Vector2d src_f = {camera_info->P[0], camera_info->P[5]}; // focal length
  const Eigen::Vector2d src_c = {camera_info->P[2], camera_info->P[6]}; // centre
#elif defined(ROS2)
  const Eigen::Vector2d src_f = {camera_info->p[0], camera_info->p[5]}; // focal length
  const Eigen::Vector2d src_c = {camera_info->p[2], camera_info->p[6]}; // centre
#endif

  // source field-of-view (FoV)
  const Eigen::Vector2d src_d = {camera_info->width, camera_info->height}; // dimension
  const Eigen::Vector2d src_fov = (2 * (src_d.array() / (2*src_f.array())).atan()) * (180/M_PI);

  std::cout << "source sensor properties:" << std::endl;
  std::cout << "  dimensions:    " << source_dimensions << " px" << std::endl;
  std::cout << "  image centre:  (" << src_c.x() << " , " << src_c.y() << ") px" << std::endl;
  std::cout << "  focal length:  (" << src_f.x() << " , " << src_f.y() << ") px" << std::endl;
  std::cout << "  field of view: " << src_fov.x() << "°(H) x " << src_fov.y() << "°(V)" << std::endl;

  // target intrinsics
  Eigen::Vector2d tgt_f;
  Eigen::Vector2d tgt_c;

  int width;
  int height;

  if (!target_dimensions.empty() && target_dimensions!=source_dimensions) {
    // crop and scale to target dimension
    double scale = 1;
    std::tie(crop_roi, scale) = get_crop_roi(source_dimensions, target_dimensions);

    width = target_dimensions.width;
    height = target_dimensions.height;
    tgt_c = (src_c - Eigen::Vector2d(crop_roi.x, crop_roi.y)) / scale;
    tgt_f = src_f / scale;

    const Eigen::Vector2d tgt_fov = (2 * (Eigen::Array2d({width, height}) / (2*tgt_f.array())).atan()) * (180/M_PI);

    std::cout << "new target sensor properties:" << std::endl;
    std::cout << "  dimensions:    " << target_dimensions << " px" << std::endl;
    std::cout << "  image centre:  (" << tgt_c.x() << " , " << tgt_c.y() << ") px" << std::endl;
    std::cout << "  focal length:  (" << tgt_f.x() << " , " << tgt_f.y() << ") px" << std::endl;
    std::cout << "  field of view: " << tgt_fov.x() << "°(H) x " << tgt_fov.y() << "°(V)" << std::endl;
  }
  else {
    width = camera_info->width;
    height = camera_info->height;
    tgt_f = src_f;
    tgt_c = src_c;
  }

  // set global camera intrinsics
  Resolution::setResolution(width, height);
  Intrinsics::setIntrinics(tgt_f.x(), tgt_f.y(), tgt_c.x(), tgt_c.y());
}

void
ImageCropTarget::map_target(FrameData &data) {
  // rectify images
  if (!camera_model.distortionCoeffs().empty()) {
    camera_model.rectifyImage(data.rgb, data.rgb, cv::INTER_LINEAR);
    camera_model.rectifyImage(data.depth, data.depth, cv::INTER_NEAREST);
  }

  // crop and scale to target dimension
  if (!crop_roi.empty()) {
    cv::resize(data.rgb(crop_roi), data.rgb, this->target_dimensions, 0, 0, cv::INTER_LINEAR);
    cv::resize(data.depth(crop_roi), data.depth, this->target_dimensions, 0, 0, cv::INTER_NEAREST);
  }
}

#endif
