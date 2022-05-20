#ifdef ROSBAG

#include "RosBagReader.hpp"
#include <sensor_msgs/CameraInfo.h>
#include <cv_bridge/cv_bridge.h>
// TODO: replace by std::format (C++20)
#include <boost/format.hpp>

RosBagReader::RosBagReader(const std::string bagfile_path,
                           const std::string topic_colour,
                           const std::string topic_depth,
                           const std::string topic_camera_info,
                           const bool flipColors, const double scale) :
  LogReader(bagfile_path, flipColors), scale(scale), do_scale(std::rint(scale) != 1),
  topic_colour(topic_colour), topic_depth(topic_depth), topic_camera_info(topic_camera_info)
{
  bag.open(bagfile_path, rosbag::bagmode::Read);

  width = 0;
  height = 0;
  calibrationFile = {};

  topic_view.addQuery(bag, rosbag::TopicQuery({topic_colour, topic_depth, topic_camera_info}));

  if (topic_view.size() == 0)
    throw std::runtime_error("None of the requested topics contain messages.");

  iter_msg = topic_view.begin();
}

RosBagReader::~RosBagReader() {
  bag.close();
}

void RosBagReader::getNext() {
  for(bool has_colour = false, has_depth = false;
      !(has_colour & has_depth) & hasMore();
      iter_msg++)
  {
    if(iter_msg->getTopic()==topic_colour) {
      if (const auto m = iter_msg->instantiate<sensor_msgs::CompressedImage>()) {
        msg_colour = *m;
      }
      else {
        throw std::runtime_error("colour topic '" + topic_colour + "' must only contain messages of type 'sensor_msgs/CompressedImage'");
      }
      has_colour = true;
    }
    else if(iter_msg->getTopic()==topic_depth) {
      if (const auto m = iter_msg->instantiate<sensor_msgs::CompressedImage>()) {
        msg_depth = *m;
      }
      else {
        throw std::runtime_error("depth topic '" + topic_depth + "' must only contain messages of type 'sensor_msgs/CompressedImage'");
      }
      has_depth = true;
    }
    else if(calibrationFile.empty() && iter_msg->getTopic()==topic_camera_info) {
      // image dimensions
      if (const auto m = iter_msg->instantiate<sensor_msgs::CameraInfo>()) {
        width = int(scale * m->width);
        height = int(scale * m->height);
      }
      else {
        throw std::runtime_error("topic '" + topic_camera_info + "' must only contain messages of type 'sensor_msgs/CameraInfo'");
      }
      // row-major 3x4 projection matrix
      const auto &P = iter_msg->instantiate<sensor_msgs::CameraInfo>()->P;
      // store intrinsics, format: "<f_x> <f_y> <c_x> <c_y> <w> <h>"
      calibrationFile = (boost::format("%lg %lg %lg %lg %lg %lg")
                         % (scale*P[0]) % (scale*P[5]) // focal length
                         % (scale*P[2]) % (scale*P[6]) // image centre
                         % width % height              // image dimension
                        ).str();
    }
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
  FrameData data;

  if(!msg_colour.data.empty()) {
    data.rgb = cv_bridge::toCvCopy(msg_colour, "rgb8")->image;
  }

  if(!msg_depth.data.empty()) {
    data.depth = cv_bridge::toCvCopy(msg_depth)->image;
    if (data.depth.type() == CV_16U) {
      // convert from 16 bit integer millimeter to 32 bit float meter
      data.depth.convertTo(data.depth, CV_32F, 1e-3);
    }
  }

  if (do_scale) {
    cv::resize(data.rgb, data.rgb, cv::Size(), scale, scale, CV_INTER_NN);
    cv::resize(data.depth, data.depth, cv::Size(), scale, scale, CV_INTER_NN);
  }

  data.timestamp = int64_t(msg_colour.header.stamp.toNSec());

  return data;
};

#endif
