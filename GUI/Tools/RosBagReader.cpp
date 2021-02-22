#ifdef ROSBAG

#include "RosBagReader.hpp"
#include <sensor_msgs/CameraInfo.h>
#include <cv_bridge/cv_bridge.h>
#include <Eigen/Core>

static std::tuple<cv::Rect, double>
get_crop_roi(const cv::Size &source_dimensions, const cv::Size &target_dimensions) {
  // scale ratios
  const double r_w = double(source_dimensions.width) / double(target_dimensions.width);
  const double r_h = double(source_dimensions.height) / double(target_dimensions.height);

  // centred ROI with aspect ratio of target dimensions
  cv::Rect crop_roi;

  // scale of cropped dimensions = source / target
  double scale;

  // aspect_ratio = width / height;

  if (r_w>r_h) {
    // crop width (left, right)
    crop_roi.width = source_dimensions.height * target_dimensions.aspectRatio();
    crop_roi.height = source_dimensions.height;
    crop_roi.x = (source_dimensions.width-crop_roi.width) / 2;
    crop_roi.y = 0;
    scale = r_h;
  }
  else if (r_h>r_w) {
    // crop height (top, bottom)
    crop_roi.width = source_dimensions.width;
    crop_roi.height = source_dimensions.width / target_dimensions.aspectRatio();
    crop_roi.x = 0;
    crop_roi.y = (source_dimensions.height-crop_roi.height) / 2;
    scale = r_w;
  }
  else {
    // equal aspect ratio, no crop
    crop_roi = cv::Rect(cv::Size(), source_dimensions);
  }

  return {crop_roi, scale};
}

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
    // source intrinsics
    const cv::Size source_dimensions(m->width, m->height);
    // 'P' row-major 3x4 projection matrix: (fx, 0, cx, Tx, 0, fy, cy, Ty, 0, 0, 1, 0)
    const Eigen::Vector2d src_f = {m->P[0], m->P[5]}; // focal length
    const Eigen::Vector2d src_c = {m->P[2], m->P[6]}; // centre

    // source field-of-view (FoV)
    const Eigen::Vector2d src_d = {m->width, m->height}; // dimension
    const Eigen::Vector2d src_fov = (2 * (src_d.array() / (2*src_f.array())).atan()) * (180/M_PI);

    std::cout << "source sensor properties:" << std::endl;
    std::cout << "  dimensions:    " << source_dimensions << " px" << std::endl;
    std::cout << "  image centre:  (" << src_c.x() << " , " << src_c.y() << ") px" << std::endl;
    std::cout << "  focal length:  (" << src_f.x() << " , " << src_f.y() << ") px" << std::endl;
    std::cout << "  field of view: " << src_fov.x() << "°(H) x " << src_fov.y() << "°(V)" << std::endl;

    // target intrinsics
    Eigen::Vector2d tgt_f;
    Eigen::Vector2d tgt_c;

    if (!target_dimensions.empty() && target_dimensions!=source_dimensions) {
      // crop and scale to target dimension
      double scale = 1;
      std::tie(crop_roi, scale) = get_crop_roi(source_dimensions, target_dimensions);

      width = target_dimensions.width;
      height = target_dimensions.height;
      tgt_c = (src_c - Eigen::Vector2d(crop_roi.x, crop_roi.y)) / scale;
      tgt_f = src_f / scale;

      const Eigen::Vector2d tgt_fov = (2 * (Eigen::Array2d({width, height}) / (2*tgt_f.array())).atan()) * (180/M_PI);

      std::cout << "new target sensor properties:" << std::endl;
      std::cout << "  dimensions:    " << target_dimensions << " px" << std::endl;
      std::cout << "  image centre:  (" << tgt_c.x() << " , " << tgt_c.y() << ") px" << std::endl;
      std::cout << "  focal length:  (" << tgt_f.x() << " , " << tgt_f.y() << ") px" << std::endl;
      std::cout << "  field of view: " << tgt_fov.x() << "°(H) x " << tgt_fov.y() << "°(V)" << std::endl;

      this->scale_colour = [this](cv::Mat &img) {
        img = img(this->crop_roi);
        cv::resize(img, img, this->target_dimensions, 0, 0, CV_INTER_LINEAR);
      };

      this->scale_depth = [this](cv::Mat &img) {
        img = img(this->crop_roi);
        cv::resize(img, img, this->target_dimensions, 0, 0, CV_INTER_NN);
      };
    }
    else {
      width = m->width;
      height = m->height;
      tgt_f = src_f;
      tgt_c = src_c;
    }

    numPixels = width * height;

    // set global camera intrinsics
    Resolution::setResolution(width, height);
    Intrinsics::setIntrinics(tgt_f.x(), tgt_f.y(), tgt_c.x(), tgt_c.y());
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
