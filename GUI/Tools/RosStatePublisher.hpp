#pragma once

#include <memory>
#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <opencv2/core.hpp>
#include <Model/Model.h>


class RosStatePublisher {
public:
    RosStatePublisher(const std::string &camera_frame);

    void pub_segmentation(const cv::Mat &segmentation, const int64_t timestamp_ns);

    void pub_models(const ModelList &models, const int64_t timestamp_ns);

private:
    std::unique_ptr<ros::NodeHandle> n;
    std::unique_ptr<image_transport::ImageTransport> it;

    image_transport::Publisher pub_segm;
    std::unordered_map<uint8_t, ros::Publisher> pub_model_pc;

    const std::string camera_frame;
};
