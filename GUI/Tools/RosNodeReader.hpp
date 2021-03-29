#ifdef ROSNODE

#pragma once
#include "LogReader.h"
#include <sensor_msgs/Image.h>
#include <ros/ros.h>
#include <image_transport/subscriber_filter.h>
#include <message_filters/sync_policies/approximate_time.h>


class RosNodeReader : public LogReader {
public:
  RosNodeReader(const uint32_t synchroniser_queue_size,
                const bool flipColors = false, const cv::Size &target_dimensions = {});

  ~RosNodeReader();

  void on_rgbd(const sensor_msgs::ImageConstPtr& msg_colour, const sensor_msgs::ImageConstPtr& msg_depth);

  void getNext() override;

  int getNumFrames() override;

  bool hasMore() override;

  bool rewind() override;

  void getPrevious() override;

  void fastForward(int frame) override;

  const std::string getFile() override;

  void setAuto(bool value) override;

  FrameData getFrameData() override;

private:
  typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::Image> ApproximateTimePolicy;

  // node
  std::unique_ptr<ros::NodeHandle> n;
  std::unique_ptr<image_transport::ImageTransport> it;
  std::unique_ptr<ros::AsyncSpinner> spinner;

  // topics
  image_transport::SubscriberFilter sub_colour;
  image_transport::SubscriberFilter sub_depth;
  std::unique_ptr<message_filters::Synchronizer<ApproximateTimePolicy>> sync;

  // optional scale&crop to target dimensions
  cv::Rect crop_roi;
  const cv::Size target_dimensions;
  std::function<void(cv::Mat &)> scale_colour;
  std::function<void(cv::Mat &)> scale_depth;

  std::mutex mutex;
  FrameData data;
};

#endif
