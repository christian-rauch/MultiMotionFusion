#pragma once
#include <vector>
#include <opencv2/core/types.hpp>
#include <Eigen/Core>
#include "Cuda/types.cuh"


namespace tracker {

struct Keypoint {
  // time (ns)
  uint64_t timestamp;
  // 2D coordinate in image plane
  cv::Point xy;
  // 3D coordinate in camera frame
  Eigen::RowVector3d coordinate;
  // feature vector
  Eigen::RowVectorXd descriptor;
};

typedef std::shared_ptr<Keypoint> KeypointPtr;

typedef std::vector<KeypointPtr> Track;
typedef std::shared_ptr<Track> TrackPtr;
typedef std::shared_ptr<const Track> TrackCPtr;
typedef std::vector<TrackPtr> Tracks;

class PointTracker {
public:
  PointTracker();

  PointTracker(const CameraModel &intrinsics);

  const Tracks& getTracks() const;

  void addKeypoints(const Eigen::MatrixX2d &coordinates, const Eigen::MatrixXd &descriptors, const uint64_t timestamp, const cv::Mat &depth, const float min_feature_distance = {}, const size_t &history = {});

  cv::Mat drawTracks(const cv::Mat &image, const size_t length = {}) const;

  void prune(const size_t &min_kps, const uint64_t &min_time);

private:
  CameraModel intrinsics;

  Tracks tracks;

  std::vector<KeypointPtr> getLastActiveKeypoints(const size_t &history) const;
};

} // namespace tracker
