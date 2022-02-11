#include "PointTracker.hpp"
#include <opencv2/core/eigen.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/viz/types.hpp>
#include <opencv2/imgproc.hpp>
#include <set>
#include <random>
#include <algorithm>

namespace tracker {

PointTracker::PointTracker()
{
  //
}

PointTracker::PointTracker(const CameraModel &intrinsics) : intrinsics(intrinsics)
{
  //
}

const Tracks &PointTracker::getTracks() const
{
  return tracks;
}

void
PointTracker::addKeypoints(const Eigen::MatrixX2d &coordinates,
                           const Eigen::MatrixXd &descriptors,
                           const uint64_t timestamp,
                           const cv::Mat &depth,
                           const float min_feature_distance,
                           const size_t &history)
{
  const auto construct_kp =
      [intr=intrinsics, &timestamp](const Eigen::Vector2d &coordinate, const Eigen::VectorXd &descriptor, const cv::Mat &depth) -> KeypointPtr
  {
    // extract keypoint coordinate
    cv::Vec2d xy_norm;
    cv::eigen2cv(coordinate, xy_norm);
    const cv::Point xy(xy_norm.mul(cv::Vec2i(depth.cols, depth.rows)));
    const float z = depth.at<float>(xy);

    Eigen::Vector3d v;
    if (z>0) {
      // project keypoint coordinate to camera frame
      v.x() = double(z * (xy.x - intr.cx) / intr.fx);
      v.y() = double(z * (xy.y - intr.cy) / intr.fy);
      v.z() = double(z);
    }
    else {
      v.setConstant(std::numeric_limits<double>::quiet_NaN());
    }

    return std::make_shared<Keypoint>(Keypoint{timestamp, xy, v, descriptor});
  };

  assert(coordinates.rows()==descriptors.rows());
  assert(tracks.empty() || (*(*tracks.begin())->begin())->descriptor.cols()==descriptors.cols());

  if(tracks.empty()) {
    // add without matching
    for(int ik=0; ik<coordinates.rows(); ik++) {
      tracks.emplace_back(std::make_shared<Track>(1, construct_kp(coordinates.row(ik), descriptors.row(ik), depth)));
    }
  }
  else {
    const std::vector<KeypointPtr> active_kp = getLastActiveKeypoints(history);

    // set all tracks to inactive by default
    for (TrackPtr &track : tracks) {
      track->emplace_back(nullptr);
    }

    if(descriptors.rows()>0) {
      // match with previous keypoint set
      cv::Mat previous;
      cv::Mat current;

      Eigen::MatrixXf prev_tmp(int(tracks.size()), int(descriptors.cols()));
      size_t n_current = 0;
      // map from valid index in 'prev_tmp' to full range in 'active_kp'
      std::unordered_map<size_t, size_t> map_valid_prev;
      for (size_t i=0; i<active_kp.size(); i++) {
        if (active_kp[i]) {
          map_valid_prev[n_current] = i;
          prev_tmp.row(int(n_current)) = active_kp[i]->descriptor.cast<float>();
          n_current++;
        }
      }

      // convert valid descriptors from Eigen to OpenCV
      cv::eigen2cv(Eigen::MatrixXf(prev_tmp.topRows(int(n_current))), previous);
      cv::eigen2cv(descriptors, current);

      // convert to float for cv matching
      current.convertTo(current, CV_32F);

      // pairwise match between previous and current set of active keypoints
      std::vector<cv::DMatch> matches;
      cv::BFMatcher(cv::NORM_L2, true).match(current, previous, matches); // "query", "train"

      // keep track of current keypoints that are not matched to the previous keypoints
      std::set<int> unmatched;
      for(int i=0; i<coordinates.rows(); i++)
        unmatched.insert(i);

      for(const cv::DMatch &match : matches) {
        if (min_feature_distance<std::numeric_limits<float>::epsilon() || match.distance <= min_feature_distance) {
          // map matched keypoint back to full set of global tracks
          tracks[map_valid_prev.at(size_t(match.trainIdx))]->back() = construct_kp(coordinates.row(match.queryIdx), descriptors.row(match.queryIdx), depth);
          unmatched.erase(match.queryIdx);
        }
      }

      // add unmatched keypoints as new tracks
      const size_t curr_length = (*tracks.begin())->size();
      for(const int &id : unmatched) {
        tracks.emplace_back(std::make_shared<Track>(curr_length, nullptr));
        tracks.back()->back() = construct_kp(coordinates.row(id), descriptors.row(id), depth);
      }
    } // has current descriptors
  } // has previous tracks

#ifndef NDEBUG
  // check that all tracks have the same length, including inactive keypoints
  for(const TrackPtr &track : tracks) {
    assert(track->size()==(*tracks.begin())->size());
  }
#endif
}

cv::Mat
PointTracker::drawTracks(const cv::Mat &image, const size_t length) const
{
  std::uniform_real_distribution<double> u(0,1);
  std::default_random_engine g;

  cv::Mat img_tracks;

  // convert to grey scale
  if(image.channels()>1) {
    cv::cvtColor(image, img_tracks, cv::COLOR_RGB2GRAY);
  }
  else {
    img_tracks = image;
  }

  // convert to colour to store coloured tracks on grey image
  cv::cvtColor(img_tracks, img_tracks, cv::COLOR_GRAY2RGB);

  for (size_t it=0; it<tracks.size(); it++) {
    // sample random colour with track-specific seed
    g.seed(it);
    const cv::viz::Color c(u(g)*255, u(g)*255, u(g)*255);

    // draw track segments
    const TrackCPtr track = tracks[it];
    const size_t start = (length>0 && length<track->size()) ? (track->size()-length) : 0;
    for (size_t ik=start; ik<(track->size()-1); ik++) {
      if ((*track)[ik]!=nullptr && (*track)[ik+1]!=nullptr) {
        cv::line(img_tracks, (*track)[ik]->xy, (*track)[ik+1]->xy, c, 2);
      }
    }
  }

  return img_tracks;
}

void
PointTracker::prune(const size_t &min_kps, const uint64_t &min_time)
{
  Tracks tracks_pruned;

  for (TrackPtr &track : tracks) {
    const size_t nvalid = std::count_if(track->begin(), track->end(), [](const KeypointPtr &kp){return kp!=nullptr;});
    uint64_t last_stamp = 0;
    for (const KeypointPtr &kp : *track) {
      if (kp!=nullptr) {
        last_stamp = kp->timestamp;
      }
    }

    if (nvalid < min_kps && last_stamp < min_time) {
      // delete all keypoints of this track and the track itself
      for (KeypointPtr &kp : *track) {
        if (kp!=nullptr) {
          kp.reset();
          kp = nullptr;
        }
      }
      track->clear();
      track.reset();
      track = nullptr;
    }
    else {
      // keep this track
      tracks_pruned.push_back(track);
    }
  }

  tracks = tracks_pruned;
}

std::vector<KeypointPtr>
PointTracker::getLastActiveKeypoints(const size_t &history) const
{
  std::vector<KeypointPtr> active(tracks.size(), nullptr);

  // find the last active keypoint of each track within history
  for (size_t i=0; i<tracks.size(); i++) {
    // iterate from last element towards front for given history size
    for (auto it = tracks[i]->rbegin();
         it != tracks[i]->rend() &&
         active[i] == nullptr &&
         (!history || std::distance(tracks[i]->rbegin(), it)<int(history));
         it++)
    {
      active[i] = (*it);
    }
  }

  return active;
}

} // namespace tracker
