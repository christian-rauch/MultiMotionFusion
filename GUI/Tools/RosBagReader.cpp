#ifdef ROSBAG

#include "RosBagReader.hpp"
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
  LogReader(bagfile_path, flipColors),
  topic_colour(topic_colour), topic_depth(topic_depth), topic_camera_info(topic_camera_info),
  frame_gt_camera(frame_gt_camera),
  tf_buffer(ros::Duration(std::numeric_limits<int>::max()))
{
  bag.open(bagfile_path, rosbag::bagmode::Read);

  // fetch intrinsic parameters
  rosbag::View topic_view_ci(bag, rosbag::TopicQuery({topic_camera_info}));
  if (topic_view_ci.size() == 0)
    throw std::runtime_error("No messages on camera_info topic '"+topic_camera_info+"'");

  if (const sensor_msgs::CameraInfo::ConstPtr m = topic_view_ci.begin()->instantiate<sensor_msgs::CameraInfo>()) {
    image_crop_target = ImageCropTarget(m, target_dimensions);
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

  sync();

  iter_sync = matches.cbegin();
}

RosBagReader::~RosBagReader() {
  bag.close();
}

void RosBagReader::getNext() {
  // reset data
  data = {};

  std_msgs::Header hdr_colour;
  std_msgs::Header hdr_depth;

  while (hasMore()) {
    const ros::Time &cmsgtime = std::get<1>(*iter_sync);
    const ros::Time &dmsgtime = std::get<2>(*iter_sync);
    rosbag::View cv(bag, cmsgtime, cmsgtime);
    rosbag::View dv(bag, dmsgtime, dmsgtime);

    // the log timestamp might not be unique, iterate over all messages of that
    // timestamp and find the correct image message from the colour / depth topic
    for (const rosbag::MessageInstance &m : cv) {
      if (m.getTopic() == topic_colour) {
        if (const auto im = m.instantiate<sensor_msgs::CompressedImage>()) {
          hdr_colour = im->header;
          data.rgb = cv_bridge::toCvCopy(im, "rgb8")->image;
        }
        else if (const auto im = m.instantiate<sensor_msgs::Image>()) {
          hdr_colour = im->header;
          data.rgb = cv_bridge::toCvCopy(im, "rgb8")->image;
        }
      }
    }

    for (const rosbag::MessageInstance &m : dv) {
      if (m.getTopic() == topic_depth) {
        if (const auto im = m.instantiate<sensor_msgs::CompressedImage>()) {
          hdr_depth = im->header;
          data.depth = cv_bridge::toCvCopy(im)->image;
        }
        else if (const auto im = m.instantiate<sensor_msgs::Image>()) {
          hdr_depth = im->header;
          data.depth = cv_bridge::toCvCopy(im)->image;
        }
      }
    }

    if (data.depth.type() == CV_16U) {
      // convert from 16 bit integer millimeter to 32 bit float meter
      data.depth.convertTo(data.depth, CV_32F, 1e-3);
    }

    data.timestamp = int64_t(hdr_colour.stamp.toNSec());

    iter_sync++;

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
        break;
      } catch (tf2::ExtrapolationException &e) {
        // we have to iterate over synchronised message pairs until we find a colour message with a matching transformation
        data = {};
      }
    }
    else {
      break;
    }
  }

  if (hasMore() && (hdr_colour.frame_id != hdr_depth.frame_id))
    throw std::runtime_error("colour and depth images are not registered into the same frame");

  // scale and crop images in place
  image_crop_target.map_target(data);
}

int RosBagReader::getNumFrames() {
  // get number of colour and depth image pairs
  return matches.size();
}

bool RosBagReader::hasMore() {
  return iter_sync != matches.cend();
}

bool RosBagReader::rewind() {
  iter_sync = matches.cbegin();
  return true;
}

void RosBagReader::getPrevious() {}

void RosBagReader::fastForward(int frame) {
  for(int i = 0; i<frame && hasMore(); i++) { getNext(); }
}

const std::string RosBagReader::getFile() {
  return file;
}

void RosBagReader::setAuto(bool /*value*/) {
  // ignore since auto exposure and auto white balance settings do not apply to log
}

FrameData RosBagReader::getFrameData() {
  return data;
}

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
}

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

void RosBagReader::sync() {
  // store tuples of message header time and logged timestamp
  // the message header will be used to match colour and depth image
  // the logged timestamp will be used to access logged messages directly
  std::map<ros::Time, ros::Time> index_colour;
  std::map<ros::Time, ros::Time> index_depth;
  for (const rosbag::MessageInstance &m : rosbag::View(bag, rosbag::TopicQuery({topic_colour, topic_depth}))) {
    std_msgs::Header hdr;
    if (const auto &im = m.instantiate<sensor_msgs::CompressedImage>())
      hdr = im->header;
    else if (const auto &im = m.instantiate<sensor_msgs::Image>())
      hdr = im->header;

    if (m.getTopic()==topic_colour)
      index_colour[hdr.stamp] = m.getTime();
    else if (m.getTopic()==topic_depth)
      index_depth[hdr.stamp] = m.getTime();
  }

  std::cout << "colour images: " << index_colour.size() << std::endl;
  std::cout << "depth images: " << index_depth.size() << std::endl;

  if (index_colour.empty())
    throw std::runtime_error("no images on colour topic '" + topic_colour + "'");

  if (index_depth.empty())
    throw std::runtime_error("no images on depth topic '" + topic_depth + "'");

  // sort by all colour - depth distances
  typedef std::tuple<int64_t, ros::Time, ros::Time> time_tuple_t;
  std::vector<time_tuple_t> time_diff;
  for (const auto &[ctime, chash] : index_colour) {
    for (const auto &[dtime, dhash] : index_depth) {
      time_diff.emplace_back(abs((ctime-dtime).toNSec()), ctime, dtime);
    }
  }
  std::sort(time_diff.begin(), time_diff.end(),
            [](const time_tuple_t &a, const time_tuple_t &b) { return std::get<0>(a) < std::get<0>(b); });

  // sort by colour timestamp
  for (const auto &[diff, ctime, dtime] : time_diff) {
    // keep first match with smallest time distance
    if (index_colour.count(ctime) && index_depth.count(dtime)) {
      matches.emplace_back(ctime, index_colour.at(ctime), index_depth.at(dtime));
    }
    // remove all other associations with larger time distances
    index_colour.erase(ctime);
    index_depth.erase(dtime);
  }
  std::sort(matches.begin(), matches.end(),
            [](const sync_tuple_t &a, const sync_tuple_t &b) { return std::get<0>(a) < std::get<0>(b); });

  std::cout << "synchronised " << matches.size() << " image pairs" << std::endl;
}

#endif
