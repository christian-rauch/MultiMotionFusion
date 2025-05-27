#ifdef ROSREADER

#include "RosNodeReader.hpp"

#if __has_include(<cv_bridge/cv_bridge.h>)
#include <cv_bridge/cv_bridge.h>
#elif __has_include(<cv_bridge/cv_bridge.hpp>)
#include <cv_bridge/cv_bridge.hpp>
#else
#error "no 'cv_bridge' header"
#endif

#if defined(ROS1)
    #include <tf2_eigen/tf2_eigen.h>
    #include <sensor_msgs/CameraInfo.h>
    #include <std_msgs/Header.h>
    using namespace sensor_msgs;
    using namespace std_msgs;
#elif defined(ROS2)
    #include <tf2_eigen/tf2_eigen.hpp>
    #include <rclcpp/wait_for_message.hpp>
    #include <sensor_msgs/msg/camera_info.hpp>
    #include <std_msgs/msg/header.hpp>
    using namespace sensor_msgs::msg;
    using namespace std_msgs::msg;
#endif


RosNodeReader::RosNodeReader(const uint32_t synchroniser_queue_size,
                             const bool flipColors, const cv::Size &target_dimensions,
                             const std::string frame_gt_camera) :
  LogReader(std::string(), flipColors),
  frame_gt_camera(frame_gt_camera)
{
#if defined(ROS1)
  n = std::make_unique<ros::NodeHandle>();
  it = std::make_unique<image_transport::ImageTransport>(*n);
#elif defined(ROS2)
  n = std::make_shared<rclcpp::Node>("MMF");
  const std::string transport = n->declare_parameter("image_transport", "raw");
#endif

  tf_listener = std::make_unique<tf2_ros::TransformListener>(tf_buffer);

#if defined(ROS1)
  sub_colour.subscribe(*it, resolve("colour"), 1);
  sub_depth.subscribe(*it, resolve("depth"), 1);
#elif defined(ROS2)
  sub_colour.subscribe(n.get(), resolve("colour"), transport);
  sub_depth.subscribe(n.get(), resolve("depth"), transport);
#endif

  sync = std::make_unique<message_filters::Synchronizer<ApproximateTimePolicy>>(ApproximateTimePolicy(synchroniser_queue_size));
  sync->connectInput(sub_colour, sub_depth);
  sync->registerCallback(&RosNodeReader::on_rgbd, this);

  // wait for single CameraInfo message to get intrinsics
  std::cout << "waiting for 'sensor_msgs/CameraInfo' message on '" + resolve("camera_info") + "'" << std::endl;

#if defined(ROS1)
  CameraInfo::ConstPtr ci = ros::topic::waitForMessage<CameraInfo>("camera_info", *n);
#elif defined(ROS2)
  CameraInfo::SharedPtr ci = std::make_shared<CameraInfo>();
  if(!rclcpp::wait_for_message(*ci, n, "camera_info")) {
    throw std::runtime_error("error while waiting for message");
  }
#endif

  image_crop_target = ImageCropTarget(ci, target_dimensions);

  width = Resolution::getInstance().width();
  height = Resolution::getInstance().height();
  numPixels = width * height;

  ref_pose.matrix().array() = 0;
}

void RosNodeReader::on_rgbd(const ImgCPtr &msg_colour, const ImgCPtr &msg_depth) {
  mutex.lock();
  const Header hdr_colour = msg_colour->header;
  data.frame_name = hdr_colour.frame_id;
#if defined(ROS1)
  data.timestamp = int64_t(hdr_colour.stamp.toNSec());
#elif defined(ROS2)
  data.timestamp = int64_t(hdr_colour.stamp.sec * 1e9 + hdr_colour.stamp.nanosec);
#endif
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
#if defined(ROS1)
  ros::Time time;
  time.fromNSec(timestamp);
#elif defined(ROS2)
  tf2::TimePoint time{std::chrono::nanoseconds(timestamp)};
#endif
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

std::string RosNodeReader::resolve(const std::string &topic) {
#if defined(ROS1)
  return ros::names::resolve(topic);
#elif defined(ROS2)
  return this->n->get_node_base_interface()->resolve_topic_or_service_name(topic, false);
#endif
}

#endif
