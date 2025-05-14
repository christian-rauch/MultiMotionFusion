#pragma once

#include <memory>
#if defined(ROS1)
    #include <ros/ros.h>
    #include <image_transport/image_transport.h>
#elif defined(ROS2)
    #include <rclcpp/rclcpp.hpp>
    #include <image_transport/image_transport.hpp>
    #include <sensor_msgs/msg/image.hpp>
    #include <sensor_msgs/msg/compressed_image.hpp>
    #include <sensor_msgs/msg/point_cloud2.hpp>
    #include <geometry_msgs/msg/pose_stamped.hpp>
    #include <std_msgs/msg/string.hpp>
    using namespace sensor_msgs::msg;
    using namespace std_msgs::msg;
    using namespace geometry_msgs::msg;
#endif
#include <tf2_ros/transform_broadcaster.h>
#include <opencv2/core.hpp>
#include <Model/Model.h>


class RosStatePublisher {
public:
    RosStatePublisher();

    void pub_segmentation(const cv::Mat &segmentation, const int64_t timestamp_ns, const std::string &camera_frame);

    void pub_models(const ModelList &models, const int64_t timestamp_ns, const std::string &camera_frame);

    void reset();

    void send_status_message(const std::string &message);

private:
#if defined(ROS1)
    std::unique_ptr<ros::NodeHandle> n;
    tf2_ros::TransformBroadcaster broadcaster;
#elif defined(ROS2)
    rclcpp::Node::SharedPtr n;
    std::unique_ptr<tf2_ros::TransformBroadcaster> broadcaster;
#endif
    std::unique_ptr<image_transport::ImageTransport> it;

    image_transport::Publisher pub_segm;
#if defined(ROS1)
    ros::Publisher pub_camera_info;
    std::unordered_map<uint8_t, ros::Publisher> pub_camera_info_depth;
    std::unordered_map<uint8_t, ros::Publisher> pub_model_pc;
    std::unordered_map<uint8_t, ros::Publisher> pub_model_proj_colour;
    std::unordered_map<uint8_t, ros::Publisher> pub_model_proj_depth;
    ros::Publisher pub_status_message;
    ros::Publisher pub_pose_map;
#elif defined(ROS2)
    rclcpp::Publisher<CameraInfo>::SharedPtr pub_camera_info;
    std::unordered_map<uint8_t, rclcpp::Publisher<CameraInfo>::SharedPtr> pub_camera_info_depth;
    std::unordered_map<uint8_t, rclcpp::Publisher<PointCloud2>::SharedPtr> pub_model_pc;
    std::unordered_map<uint8_t, rclcpp::Publisher<CompressedImage>::SharedPtr> pub_model_proj_colour;
    std::unordered_map<uint8_t, rclcpp::Publisher<CompressedImage>::SharedPtr> pub_model_proj_depth;
    rclcpp::Publisher<String>::SharedPtr pub_status_message;
    rclcpp::Publisher<PoseStamped>::SharedPtr pub_pose_map;
#endif
};
