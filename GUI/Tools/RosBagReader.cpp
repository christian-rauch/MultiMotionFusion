#ifdef ROSBAG

#include "RosBagReader.hpp"
#include "ros_common.hpp"
#include <sensor_msgs/CameraInfo.h>
#include <cv_bridge/cv_bridge.h>
#include <Eigen/Core>

RosBagReader::RosBagReader(const std::string bagfile_path,
                           const std::string topic_colour,
                           const std::string topic_depth,
                           const std::string topic_camera_info,
                           const bool flipColors,
                           const cv::Size target_dimensions) :
  LogReader(bagfile_path, flipColors), target_dimensions(target_dimensions),
  topic_colour(topic_colour), topic_depth(topic_depth), topic_camera_info(topic_camera_info)
{
  bag.open(bagfile_path, rosbag::bagmode::Read);

  // fetch intrinsic parameters
  rosbag::View topic_view_ci(bag, rosbag::TopicQuery({topic_camera_info}));
  if (topic_view_ci.size() == 0)
    throw std::runtime_error("No messages on camera_info topic '"+topic_camera_info+"'");

  if (const auto m = topic_view_ci.begin()->instantiate<sensor_msgs::CameraInfo>()) {
    if (intrinsics_crop_target(m, target_dimensions, this->crop_roi)) {
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
  }
  else {
    throw std::runtime_error("topic '" + topic_camera_info + "' must only contain messages of type 'sensor_msgs/CameraInfo'");
  }

  calibrationFile = {};

  topic_view.addQuery(bag, rosbag::TopicQuery({topic_colour, topic_depth}));

  if (topic_view.size() == 0)
    throw std::runtime_error("None of the requested topics contain messages.");

  iter_msg = topic_view.begin();
}

RosBagReader::~RosBagReader() {
  bag.close();
}

void RosBagReader::getNext() {
  // reset data
  data = {};

  for(bool has_colour = false, has_depth = false;
      !(has_colour & has_depth) & hasMore();
      iter_msg++)
  {
    if(iter_msg->getTopic()==topic_colour) {
      if (const auto m = iter_msg->instantiate<sensor_msgs::CompressedImage>()) {
        data.timestamp = int64_t(m->header.stamp.toNSec());
        data.rgb = cv_bridge::toCvCopy(m, "rgb8")->image;
      }
      else if (const auto m = iter_msg->instantiate<sensor_msgs::Image>()) {
        data.timestamp = int64_t(m->header.stamp.toNSec());
        data.rgb = cv_bridge::toCvCopy(m, "rgb8")->image;
      }
      else {
        throw std::runtime_error("colour topic '" + topic_colour + "' must only contain messages of type 'sensor_msgs/Image' or 'sensor_msgs/CompressedImage'");
      }
      has_colour = true;
    }
    else if(iter_msg->getTopic()==topic_depth) {
      if (const auto m = iter_msg->instantiate<sensor_msgs::CompressedImage>()) {
        data.depth = cv_bridge::toCvCopy(m)->image;
      }
      else if (const auto m = iter_msg->instantiate<sensor_msgs::Image>()) {
        data.depth = cv_bridge::toCvCopy(m)->image;
      }
      else {
        throw std::runtime_error("depth topic '" + topic_depth + "' must only contain messages of type 'sensor_msgs/Image' or 'sensor_msgs/CompressedImage'");
      }
      if (!data.depth.empty() && data.depth.type() == CV_16U) {
        // convert from 16 bit integer millimeter to 32 bit float meter
        data.depth.convertTo(data.depth, CV_32F, 1e-3);
      }
      has_depth = true;
    }
  }

  if (hasMore() && data.rgb.empty())
    throw std::runtime_error("no images on colour topic '" + topic_colour + "'");

  if (hasMore() && data.depth.empty())
    throw std::runtime_error("no images on depth topic '" + topic_depth + "'");

  // scale and crop images in place
  if (scale_colour && !data.rgb.empty()) {
    scale_colour(data.rgb);
  }
  if (scale_depth && !data.depth.empty()) {
    scale_depth(data.depth);
  }
};

int RosBagReader::getNumFrames() {
  // get number of colour image frames
  return int(rosbag::View(bag, rosbag::TopicQuery(topic_colour)).size());
};

bool RosBagReader::hasMore() {
  return iter_msg != topic_view.end();
};

bool RosBagReader::rewind() {
  iter_msg = topic_view.begin();
  return true;
};

void RosBagReader::getPrevious() {};

void RosBagReader::fastForward(int frame) {
  for(int i = 0; i<frame && hasMore(); i++) { getNext(); }
};

const std::string RosBagReader::getFile() {
  return file;
};

void RosBagReader::setAuto(bool /*value*/) {
  // ignore since auto exposure and auto white balance settings do not apply to log
};

FrameData RosBagReader::getFrameData() {
  return data;
};

#endif
