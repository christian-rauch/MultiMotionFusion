#pragma once

#include <memory>
#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <tf2_ros/transform_broadcaster.h>
#include <opencv2/core.hpp>
#include <Model/Model.h>


class RosStatePublisher {
public:
    RosStatePublisher(const std::string &camera_frame);

    void pub_segmentation(const cv::Mat &segmentation, const int64_t timestamp_ns);

    void pub_models(const ModelList &models, const int64_t timestamp_ns);

    void reset();

    void send_status_message(const std::string &message);

private:
    std::unique_ptr<ros::NodeHandle> n;
    std::unique_ptr<image_transport::ImageTransport> it;
    tf2_ros::TransformBroadcaster broadcaster;

    image_transport::Publisher pub_segm;
    ros::Publisher pub_camera_info;
    std::unordered_map<uint8_t, ros::Publisher> pub_camera_info_depth;
    std::unordered_map<uint8_t, ros::Publisher> pub_model_pc;
    std::unordered_map<uint8_t, ros::Publisher> pub_model_proj_colour;
    std::unordered_map<uint8_t, ros::Publisher> pub_model_proj_depth;
    ros::Publisher pub_status_message;

    const std::string camera_frame;
};
