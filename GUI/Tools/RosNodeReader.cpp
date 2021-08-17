#ifdef ROSNODE

#include "RosNodeReader.hpp"
#include <sensor_msgs/CameraInfo.h>
#include <cv_bridge/cv_bridge.h>
#include <tf2_eigen/tf2_eigen.h>


RosNodeReader::RosNodeReader(const uint32_t synchroniser_queue_size,
                             const bool flipColors, const cv::Size &target_dimensions,
                             const std::string frame_gt_camera) :
  LogReader(std::string(), flipColors),
  frame_gt_camera(frame_gt_camera)
{
  n = std::make_unique<ros::NodeHandle>();
  it = std::make_unique<image_transport::ImageTransport>(*n);

  tf_listener = std::make_unique<tf2_ros::TransformListener>(tf_buffer);

  sub_colour.subscribe(*it, ros::names::resolve("colour"), 1);
  sub_depth.subscribe(*it, ros::names::resolve("depth"), 1);

  sync = std::make_unique<message_filters::Synchronizer<ApproximateTimePolicy>>(ApproximateTimePolicy(synchroniser_queue_size));
  sync->connectInput(sub_colour, sub_depth);
  sync->registerCallback(&RosNodeReader::on_rgbd, this);

  // wait for single CameraInfo message to get intrinsics
  std::cout << "waiting for 'sensor_msgs/CameraInfo' message on '" + ros::names::resolve("camera_info") + "'" << std::endl;
  sensor_msgs::CameraInfo::ConstPtr ci = ros::topic::waitForMessage<sensor_msgs::CameraInfo>("camera_info", *n);

  image_crop_target = ImageCropTarget(ci, target_dimensions);

  width = Resolution::getInstance().width();
  height = Resolution::getInstance().height();
  numPixels = width * height;

  ref_pose.matrix().array() = 0;

  spinner = std::make_unique<ros::AsyncSpinner>(1);
  spinner->start();
}

RosNodeReader::~RosNodeReader() {
  spinner->stop();
}

void RosNodeReader::on_rgbd(const sensor_msgs::Image::ConstPtr& msg_colour, const sensor_msgs::Image::ConstPtr& msg_depth) {
  mutex.lock();
  const std_msgs::Header hdr_colour = msg_colour->header;
  data.timestamp = int64_t(hdr_colour.stamp.toNSec());
  data.rgb = cv_bridge::toCvCopy(msg_colour, "rgb8")->image;

  data.depth = cv_bridge::toCvCopy(msg_depth)->image;
  if (!data.depth.empty() && data.depth.type() == CV_16U) {
    // convert from 16 bit integer millimeter to 32 bit float meter
    data.depth.convertTo(data.depth, CV_32F, 1e-3);
  }

  // scale and crop images in place
  image_crop_target.map_target(data);
  mutex.unlock();

  // use provided ground truth camera frame or colour optical frame from images
  if (frame_gt_camera.empty())
    frame_gt_camera = hdr_colour.frame_id;

  // find root frame
  if (frame_gt_root.empty()) {
    std::string parent = frame_gt_camera;
    while (tf_buffer._getParent(parent, {}, parent));
    frame_gt_root = parent;
  }
}

void RosNodeReader::getNext() {}

int RosNodeReader::getNumFrames() {
  return std::numeric_limits<int>::max();
}

bool RosNodeReader::hasMore() {
  return sub_colour.getNumPublishers()>0 && sub_depth.getNumPublishers()>0;
}

bool RosNodeReader::rewind() {
  return false;
}

void RosNodeReader::getPrevious() {}

void RosNodeReader::fastForward(int /*frame*/) {}

const std::string RosNodeReader::getFile() {
  return {};
}

void RosNodeReader::setAuto(bool /*value*/) {
  // TODO: implement dynamic reconfigure of ROS driver
}

FrameData RosNodeReader::getFrameData() {
  const std::lock_guard<std::mutex> lock(mutex);
  return data;
}

Eigen::Matrix4f RosNodeReader::getIncrementalTransformation(uint64_t timestamp) {
  // camera pose at requested time with respect to root frame
  ros::Time time;
  time.fromNSec(timestamp);
  Eigen::Isometry3d pose;
  try {
    pose = tf2::transformToEigen(tf_buffer.lookupTransform(frame_gt_root, frame_gt_camera, time));
  } catch (tf2::ExtrapolationException &) {
    // there is no transformation available
    return {};
  }

  // store the first requested pose as reference
  if (!ref_pose.matrix().any()) {
    ref_pose = pose;
  }

  // provide the camera poses with respect to the reference pose
  return (ref_pose.inverse() * pose).matrix().cast<float>();
}

#endif
