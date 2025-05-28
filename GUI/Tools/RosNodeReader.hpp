#ifdef ROSNODE

#pragma once
#include "LogReader.h"
#include <message_filters/sync_policies/approximate_time.h>
#include "ros_common.hpp"
#include <Utils/GroundTruthOdometryInterface.hpp>
#include <Eigen/Geometry>
#include <tf2_ros/transform_listener.h>

#if defined(ROS1)
    #include <ros/ros.h>
    #include <sensor_msgs/Image.h>
    using namespace sensor_msgs;
    #include <image_transport/subscriber_filter.h>
    using ImgCPtr = Image::ConstPtr;
#elif defined(ROS2)
    #include <rclcpp/rclcpp.hpp>
    #include <sensor_msgs/msg/image.h>
    using namespace sensor_msgs::msg;
    #include <image_transport/subscriber_filter.hpp>
    using ImgCPtr = Image::ConstSharedPtr;
#endif


class RosNodeReader : public LogReader, public GroundTruthOdometryInterface {
public:
  RosNodeReader(const uint32_t synchroniser_queue_size,
                const bool flipColors = false, const cv::Size &target_dimensions = {},
                const std::string frame_gt_camera = {});

  void on_rgbd(const ImgCPtr& msg_colour, const ImgCPtr& msg_depth);

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

#if defined(ROS2)
  rclcpp::Node::SharedPtr n;
#endif

private:
  typedef message_filters::sync_policies::ApproximateTime<Image, Image> ApproximateTimePolicy;

  // node
#if defined(ROS1)
  std::unique_ptr<ros::NodeHandle> n;
#endif
  std::unique_ptr<image_transport::ImageTransport> it;

  // ground truth camera poses in root frame
  std::string frame_gt_root;
  std::string frame_gt_camera;
  tf2::BufferCore tf_buffer;
  std::unique_ptr<tf2_ros::TransformListener> tf_listener;
  Eigen::Isometry3d ref_pose;

  // topics
  image_transport::SubscriberFilter sub_colour;
  image_transport::SubscriberFilter sub_depth;
  std::unique_ptr<message_filters::Synchronizer<ApproximateTimePolicy>> sync;

  // scale&crop to target dimensions
  ImageCropTarget image_crop_target;

  std::mutex mutex;
  FrameData data;

  std::string resolve(const std::string &topic);
};

#endif
