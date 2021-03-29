#ifdef ROSNODE

#include "RosNodeReader.hpp"
#include "ros_common.hpp"
#include <sensor_msgs/CameraInfo.h>
#include <cv_bridge/cv_bridge.h>


RosNodeReader::RosNodeReader(const uint32_t synchroniser_queue_size,
                             const bool flipColors, const cv::Size &target_dimensions) :
  LogReader(std::string(), flipColors), target_dimensions(target_dimensions)
{
  n = std::make_unique<ros::NodeHandle>();
  it = std::make_unique<image_transport::ImageTransport>(*n);

  sub_colour.subscribe(*it, ros::names::resolve("colour"), 1);
  sub_depth.subscribe(*it, ros::names::resolve("depth"), 1);

  sync = std::make_unique<message_filters::Synchronizer<ApproximateTimePolicy>>(ApproximateTimePolicy(synchroniser_queue_size));
  sync->connectInput(sub_colour, sub_depth);
  sync->registerCallback(&RosNodeReader::on_rgbd, this);

  // wait for single CameraInfo message to get intrinsics
  std::cout << "waiting for 'sensor_msgs/CameraInfo' message on '" + ros::names::resolve("camera_info") + "'" << std::endl;
  sensor_msgs::CameraInfo::ConstPtr ci = ros::topic::waitForMessage<sensor_msgs::CameraInfo>("camera_info", *n);

  if (intrinsics_crop_target(ci, target_dimensions, this->crop_roi)) {
    this->scale_colour = [this](cv::Mat &img) {
      img = img(this->crop_roi);
      cv::resize(img, img, this->target_dimensions, 0, 0, CV_INTER_LINEAR);
    };

    this->scale_depth = [this](cv::Mat &img) {
      img = img(this->crop_roi);
      cv::resize(img, img, this->target_dimensions, 0, 0, CV_INTER_NN);
    };
  }

  width = Resolution::getInstance().width();
  height = Resolution::getInstance().height();
  numPixels = width * height;

  spinner = std::make_unique<ros::AsyncSpinner>(1);
  spinner->start();
}

RosNodeReader::~RosNodeReader() {
  spinner->stop();
}

void RosNodeReader::on_rgbd(const sensor_msgs::Image::ConstPtr& msg_colour, const sensor_msgs::Image::ConstPtr& msg_depth) {
  mutex.lock();
  data.timestamp = int64_t(msg_colour->header.stamp.toNSec());

  data.rgb = cv_bridge::toCvCopy(msg_colour, "rgb8")->image;

  data.depth = cv_bridge::toCvCopy(msg_depth)->image;
  if (!data.depth.empty() && data.depth.type() == CV_16U) {
    // convert from 16 bit integer millimeter to 32 bit float meter
    data.depth.convertTo(data.depth, CV_32F, 1e-3);
  }

  // scale and crop images in place
  if (scale_colour) {
    scale_colour(data.rgb);
  }
  if (scale_depth) {
    scale_depth(data.depth);
  }
  mutex.unlock();
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

#endif
