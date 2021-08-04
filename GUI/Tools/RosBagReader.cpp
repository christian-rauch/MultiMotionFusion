#ifdef ROSBAG

#include "RosBagReader.hpp"
#include "ros_common.hpp"
#include <sensor_msgs/CameraInfo.h>
#include <cv_bridge/cv_bridge.h>
#include <tf2_msgs/TFMessage.h>
#include <tf2/exceptions.h>
#include <tf2_eigen/tf2_eigen.h>

RosBagReader::RosBagReader(const std::string bagfile_path,
                           const std::string topic_colour,
                           const std::string topic_depth,
                           const std::string topic_camera_info,
                           const bool flipColors,
                           const cv::Size target_dimensions,
                           const std::string frame_gt_camera) :
  LogReader(bagfile_path, flipColors), target_dimensions(target_dimensions),
  topic_colour(topic_colour), topic_depth(topic_depth), topic_camera_info(topic_camera_info),
  frame_gt_camera(frame_gt_camera),
  tf_buffer(ros::Duration(std::numeric_limits<int>::max()))
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

  // cache all transformations
  const bool has_tf_dyn = add_all_tf_msgs("/tf", false);
  const bool has_tf_sta = add_all_tf_msgs("/tf_static", true);
  has_tf = has_tf_dyn || has_tf_sta;

  if (!frame_gt_camera.empty() && !tf_buffer._frameExists(frame_gt_camera))
    throw std::runtime_error("provided ground truth camera frame '" + frame_gt_camera + "' does not exist");

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

  std_msgs::Header hdr_colour;
  std_msgs::Header hdr_depth;

  for(bool has_colour = false, has_depth = false;
      !(has_colour & has_depth) & hasMore();
      iter_msg++)
  {
    if(iter_msg->getTopic()==topic_colour) {
      if (const auto m = iter_msg->instantiate<sensor_msgs::CompressedImage>()) {
        hdr_colour = m->header;
        data.rgb = cv_bridge::toCvCopy(m, "rgb8")->image;
      }
      else if (const auto m = iter_msg->instantiate<sensor_msgs::Image>()) {
        hdr_colour = m->header;
        data.rgb = cv_bridge::toCvCopy(m, "rgb8")->image;
      }
      else {
        throw std::runtime_error("colour topic '" + topic_colour + "' must only contain messages of type 'sensor_msgs/Image' or 'sensor_msgs/CompressedImage'");
      }

      data.timestamp = int64_t(hdr_colour.stamp.toNSec());

      if (has_tf) {
        // use provided ground truth camera frame or colour optical frame from images
        if (frame_gt_camera.empty())
          frame_gt_camera = hdr_colour.frame_id;

        // find root frame
        if (frame_gt_root.empty()) {
          std::string parent = frame_gt_camera;
          while (tf_buffer._getParent(parent, {}, parent));
          frame_gt_root = parent;
        }

        try {
          poses[data.timestamp] = tf2::transformToEigen(tf_buffer.lookupTransform(frame_gt_root, frame_gt_camera, hdr_colour.stamp));
          has_colour = true;
        } catch (tf2::ExtrapolationException &) {
          // we have to wait for a colour frame with a matching ground truth transformation
          has_colour = false;
        }
      }
      else {
        has_colour = true;
      }
    }
    else if(iter_msg->getTopic()==topic_depth) {
      if (const auto m = iter_msg->instantiate<sensor_msgs::CompressedImage>()) {
        hdr_depth = m->header;
        data.depth = cv_bridge::toCvCopy(m)->image;
      }
      else if (const auto m = iter_msg->instantiate<sensor_msgs::Image>()) {
        hdr_depth = m->header;
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

  if (hasMore() && (hdr_colour.frame_id != hdr_depth.frame_id))
    throw std::runtime_error("colour and depth images are not registered into the same frame");

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

Eigen::Matrix4f RosBagReader::getIncrementalTransformation(uint64_t timestamp) {
  if (!has_tf)
    throw std::runtime_error("rosbag has no ground truth camera poses");

  if (!timestamp)
    return {};

  // provide camera poses with respect to the reference pose (first pose in the trajectory)
  if (ref_time==0) {
    ref_time = timestamp;
  }
  return (poses.at(ref_time).inverse() * poses.at(timestamp)).matrix().cast<float>();
};

bool RosBagReader::add_all_tf_msgs(const std::string &topic, const bool tf_static) {
  size_t nvalid = 0;
  for (const rosbag::MessageInstance &m : rosbag::View(bag, rosbag::TopicQuery(topic))) {
    if (const tf2_msgs::TFMessage::ConstPtr &mi = m.instantiate<tf2_msgs::TFMessage>()) {
      for (const geometry_msgs::TransformStamped &tf : mi->transforms) {
        if (tf_buffer.setTransform(tf, {}, tf_static)) {
          nvalid++;
        }
      }
    }
  }
  return nvalid>0;
}

#endif
