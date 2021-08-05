#ifdef ROSBAG

#pragma once
#include "LogReader.h"
#include <rosbag/bag.h>
#include <rosbag/view.h>
#include <sensor_msgs/CompressedImage.h>
#include <tf2/buffer_core.h>
#include <Utils/GroundTruthOdometryInterface.hpp>
#include <Eigen/Geometry>

class RosBagReader : public LogReader, public GroundTruthOdometryInterface {
public:
  RosBagReader(const std::string bagfile_path,
               const std::string topic_colour,
               const std::string topic_depth,
               const std::string topic_camera_info,
               const bool flipColors = false,
               const cv::Size target_dimensions = {},
               const std::string frame_gt_camera = {});

  ~RosBagReader();

  void getNext() override;

  int getNumFrames() override;

  bool hasMore() override;

  bool rewind() override;

  void getPrevious() override;

  void fastForward(int frame) override;

  const std::string getFile() override;

  void setAuto(bool value) override;

  FrameData getFrameData() override;

  Eigen::Matrix4f getIncrementalTransformation(uint64_t timestamp) override;

private:
  typedef std::tuple<ros::Time, ros::Time, ros::Time> sync_tuple_t;

  rosbag::Bag bag;

  const cv::Size target_dimensions;
  cv::Rect crop_roi;
  std::function<void(cv::Mat &)> scale_colour;
  std::function<void(cv::Mat &)> scale_depth;

  // topics
  const std::string topic_colour;
  const std::string topic_depth;
  const std::string topic_camera_info;

  // synchronised (time, colour, depth) tuples
  std::vector<sync_tuple_t> matches;
  std::vector<sync_tuple_t>::const_iterator iter_sync;

  // ground truth camera poses in root frame
  std::string frame_gt_root;
  std::string frame_gt_camera;
  tf2::BufferCore tf_buffer;
  bool has_tf;
  std::map<uint64_t, Eigen::Isometry3d> poses;
  uint64_t ref_time = 0;

  FrameData data;

  bool add_all_tf_msgs(const std::string &topic, const bool tf_static);

  void sync();
};

#endif
