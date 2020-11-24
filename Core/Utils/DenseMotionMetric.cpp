#include "DenseMotionMetric.hpp"
#include "RigidRANSAC.h"
#include "cuda_runtime.h"
#include "../Core/Cuda/cudafuncs.cuh"
#include <opencv2/imgproc.hpp>
#include <unordered_map>

namespace motion {

// custom hash function for 'cv::Point_' types
struct PointHash {
  template <typename T>
  std::size_t operator()(const cv::Point_<T> &pt) const {
    return std::hash<T>()(pt.x) ^ (std::hash<T>()(pt.y) << 1);
  }
};

std::vector<Triangle> triangulate(const tracker::Tracks &tracks)
{
  std::vector<cv::Point2f> pts;
  std::unordered_map<cv::Point2l, tracker::TrackPtr, PointHash> coordinate_track;
  for (const tracker::TrackPtr &track : tracks) {
    if (track->front()!=nullptr && track->back()!=nullptr &&
        track->front()->coordinate.array().isFinite().all() &&
        track->back()->coordinate.array().isFinite().all())
    {
      const cv::Point pt = track->back()->xy;
      pts.push_back(pt);
      coordinate_track[pt] = track;
    }
  }

  cv::Subdiv2D subdiv(cv::boundingRect(pts));
  subdiv.insert(pts);

  std::vector<cv::Vec6f> triangles;
  subdiv.getTriangleList(triangles);

  std::vector<Triangle> triplets;
  for (const cv::Vec6f &tri : triangles) {
    const cv::Point2l v0(std::lrint(tri[0]),std::lrint(tri[1]));
    const cv::Point2l v1(std::lrint(tri[2]),std::lrint(tri[3]));
    const cv::Point2l v2(std::lrint(tri[4]),std::lrint(tri[5]));
    triplets.push_back(Triangle{.tracks = {coordinate_track[v0], coordinate_track[v1], coordinate_track[v2]}});
  }

  return triplets;
}

DenseMotionMetric::DenseMotionMetric(const CameraModel &intrinsics, const size_t history) :
  intrinsics(intrinsics), index(0), vmaps(history), nmaps(history)
{

}

void
DenseMotionMetric::addDepth(const DeviceArray2D<float> &vmap,
                            const DeviceArray2D<float> &nmap)
{
  vmap.copyTo(vmaps[index]);
  nmap.copyTo(nmaps[index]);

  // advance and wrap index
  index++;
  index = index % vmaps.size();
}

cv::Mat
DenseMotionMetric::projectionError(const tracker::Tracks &tracks) const
{
  if (tracks.empty()) { return {}; }

  // index of newest and oldest data within ring buffer
  // the modulo operator (%) has different behaviour in C and Python
  const size_t id_newest = size_t(((int(index-1) % int(vmaps.size())) + int(vmaps.size())) % int(vmaps.size()));
  const bool full = (vmaps.back().rows() * vmaps.back().cols()) > 0;
  const size_t id_oldest = full ? index : 0;

  // new
  const size_t jk = (*tracks.begin())->size() - 1;
  // old
  const size_t ik = jk - (full ? vmaps.size()-1 : id_newest);

  const size_t ntracks = tracks.size();
  Eigen::MatrixX3f p0s, p1s;
  p0s.resize(int(ntracks), Eigen::NoChange);
  p1s.resize(int(ntracks), Eigen::NoChange);

  int nvalid = 0;
  for (size_t it=0; it<ntracks; it++) {
    if ((*tracks[it])[ik] && (*tracks[it])[jk]) {
      const Eigen::RowVector3d &p0 = (*tracks[it])[ik]->coordinate;
      const Eigen::RowVector3d &p1 = (*tracks[it])[jk]->coordinate;
      if (p0.array().isFinite().all() && p1.array().isFinite().all()) {
        p0s.row(nvalid) = p0.cast<float>();
        p1s.row(nvalid) = p1.cast<float>();
        nvalid++;
      }
    }
  }
  p0s.conservativeResize(nvalid, Eigen::NoChange);
  p1s.conservativeResize(nvalid, Eigen::NoChange);

  if (nvalid<3) { return {}; }

  // RANSAC estimation
  RigidRANSAC rrs(10, 0.03f, 0.6f);
  const Eigen::Isometry3f T_01 = rrs.estimate(p0s, p1s).transformation;

  DeviceArray2D<float> err;
  projectionError2(vmaps[id_oldest], nmaps[id_oldest], vmaps[id_newest], nmaps[id_newest],
                   T_01.inverse().matrix(), intrinsics, err);

  cv::Mat_<float> err_img(err.rows(), err.cols());
  err.download(err_img.data, err_img.step);

  return std::move(err_img);
}

} // namespace motion
