#pragma once

#include <ros/ros.h>
#include <std_srvs/Trigger.h>
#include <std_srvs/SetBool.h>
#include <cob_srvs/SetInt.h>
#include <cob_srvs/SetString.h>

// manually include some headers here, since they are missing from 'GUI.h'
#include <list>
#include "Core/Utils/Resolution.h"
#include "Core/Model/Model.h"
#include "GUI.h"

#include "Core/MultiMotionFusion.h"

class RosInterface {
public:
    RosInterface(GUI **gui, MultiMotionFusion **modelling);

    bool on_reset(std_srvs::Trigger::Request &req, std_srvs::Trigger::Response &res);

    bool on_inhibit(std_srvs::SetBool::Request &req, std_srvs::SetBool::Response &res);

    bool on_pause(std_srvs::SetBool::Request &req, std_srvs::SetBool::Response &res);

    bool on_deactivate(cob_srvs::SetInt::Request &req, cob_srvs::SetInt::Response &res);

    bool on_set_odom_init(cob_srvs::SetString::Request &req, cob_srvs::SetString::Response &res);

    bool on_set_icp_refine(std_srvs::SetBool::Request &req, std_srvs::SetBool::Response &res);

    bool on_set_segm_mode(cob_srvs::SetString::Request &req, cob_srvs::SetString::Response &res);

private:
    // pointer to a pointer to the GUI, since the GUI will not have been
    // allocated yet when this class is constructed
    GUI **gui;

    MultiMotionFusion **modelling;

    std::unique_ptr<ros::NodeHandle> n;
    ros::ServiceServer srv_reset;
    ros::ServiceServer srv_inhibit;
    ros::ServiceServer srv_pause;
    ros::ServiceServer srv_deactivate_model;
    ros::ServiceServer srv_set_odom_init;
    ros::ServiceServer srv_set_icp_refine;
    ros::ServiceServer srv_set_segm_mode;
};
