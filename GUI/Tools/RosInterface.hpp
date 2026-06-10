#pragma once

#if defined(ROS1)
    #include <ros/ros.h>
    #include <std_srvs/Trigger.h>
    #include <std_srvs/SetBool.h>
    #include <cob_srvs/SetInt.h>
    #include <cob_srvs/SetString.h>
    using namespace std_srvs;
    using namespace cob_srvs;
#elif defined(ROS2)
    #include <rclcpp/rclcpp.hpp>
    #include <std_srvs/srv/trigger.hpp>
    #include <std_srvs/srv/set_bool.hpp>
    #include <cob_srvs/srv/set_int.hpp>
    #include <cob_srvs/srv/set_string.hpp>
    using namespace std_srvs::srv;
    using namespace cob_srvs::srv;
#endif

// manually include some headers here, since they are missing from 'GUI.h'
#include <list>
#include "Core/Utils/Resolution.h"
#include "Core/Model/Model.h"
#include "GUI.h"

#include "Core/MultiMotionFusion.h"

class RosInterface {
public:
    RosInterface(GUI **gui, MultiMotionFusion **modelling);

#if defined(ROS1)
    bool on_reset(std_srvs::Trigger::Request &req, std_srvs::Trigger::Response &res);

    bool on_inhibit(std_srvs::SetBool::Request &req, std_srvs::SetBool::Response &res);

    bool on_pause(std_srvs::SetBool::Request &req, std_srvs::SetBool::Response &res);

    bool on_deactivate(cob_srvs::SetInt::Request &req, cob_srvs::SetInt::Response &res);

    bool on_set_odom_init(cob_srvs::SetString::Request &req, cob_srvs::SetString::Response &res);

    bool on_set_icp_refine(std_srvs::SetBool::Request &req, std_srvs::SetBool::Response &res);

    bool on_set_segm_mode(cob_srvs::SetString::Request &req, cob_srvs::SetString::Response &res);
#elif defined(ROS2)
    bool on_reset(Trigger::Request::ConstSharedPtr req, Trigger::Response::SharedPtr res);

    bool on_inhibit(SetBool::Request::ConstSharedPtr req, SetBool::Response::SharedPtr res);

    bool on_pause(SetBool::Request::ConstSharedPtr req, SetBool::Response::SharedPtr res);

    template<bool P>
    bool on_start_stop(Trigger::Request::ConstSharedPtr req, Trigger::Response::SharedPtr res);

    bool on_deactivate(SetInt::Request::ConstSharedPtr req, SetInt::Response::SharedPtr res);

    bool on_set_odom_init(SetString::Request::ConstSharedPtr req, SetString::Response::SharedPtr res);

    bool on_set_icp_refine(SetBool::Request::ConstSharedPtr req, SetBool::Response::SharedPtr res);

    bool on_set_segm_mode(SetString::Request::ConstSharedPtr req, SetString::Response::SharedPtr res);
#endif

#if defined(ROS2)
    rclcpp::Node::SharedPtr n;
#endif

private:
    // pointer to a pointer to the GUI, since the GUI will not have been
    // allocated yet when this class is constructed
    GUI **gui;

    MultiMotionFusion **modelling;

#if defined(ROS1)
    std::unique_ptr<ros::NodeHandle> n;
    ros::ServiceServer srv_reset;
    ros::ServiceServer srv_inhibit;
    ros::ServiceServer srv_pause;
    ros::ServiceServer srv_deactivate_model;
    ros::ServiceServer srv_set_odom_init;
    ros::ServiceServer srv_set_icp_refine;
    ros::ServiceServer srv_set_segm_mode;
#elif defined(ROS2)
    rclcpp::Service<Trigger>::SharedPtr srv_reset;
    rclcpp::Service<SetBool>::SharedPtr srv_inhibit;
    rclcpp::Service<SetBool>::SharedPtr srv_pause;
    rclcpp::Service<Trigger>::SharedPtr srv_start;
    rclcpp::Service<Trigger>::SharedPtr srv_stop;
    rclcpp::Service<SetInt>::SharedPtr srv_deactivate_model;
    rclcpp::Service<SetString>::SharedPtr srv_set_odom_init;
    rclcpp::Service<SetBool>::SharedPtr srv_set_icp_refine;
    rclcpp::Service<SetString>::SharedPtr srv_set_segm_mode;
#endif
};
