#pragma once

#include "PointTracker.hpp"
#include "../Cuda/containers/device_array.hpp"
#include <queue>
#include <opencv2/core.hpp>
#include<Eigen/Geometry>

namespace motion {

struct Triangle {
  std::array<tracker::TrackPtr, 3> tracks;
};

std::vector<Triangle> triangulate(const tracker::Tracks &tracks);

class DenseMotionMetric {
public:
  DenseMotionMetric(const CameraModel &intrinsics, const size_t history);

  void addRGBD(const cv::Mat &rgb,
               const DeviceArray2D<float> &vmap,
               const DeviceArray2D<float> &nmap);

  cv::Mat projectionError(const tracker::Tracks &tracks) const;

  std::tuple<const cv::Mat &, const DeviceArray2D<float> &, const DeviceArray2D<float> &>
  getRGBD(const int &idx) const;

private:
  const CameraModel intrinsics;

  size_t index;

  std::vector<DeviceArray2D<float>> vmaps;
  std::vector<DeviceArray2D<float>> nmaps;
  std::vector<cv::Mat> rgbs;
};

} // namespace motion
