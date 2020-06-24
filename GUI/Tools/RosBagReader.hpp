#ifdef ROSBAG

#pragma once
#include "LogReader.h"
#include <rosbag/bag.h>
#include <rosbag/view.h>
#include <sensor_msgs/CompressedImage.h>

class RosBagReader : public LogReader {
public:
  RosBagReader(std::string file, bool flipColors = false, const double scale = 1.);

  ~RosBagReader();

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
  rosbag::Bag bag;
  rosbag::View topic_view;
  rosbag::View::iterator iter_msg;

  const double scale;

  const bool do_scale;

  // raw image messages
  sensor_msgs::CompressedImage msg_colour;
  sensor_msgs::CompressedImage msg_depth;
};

#endif
