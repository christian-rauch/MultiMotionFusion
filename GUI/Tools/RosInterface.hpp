#pragma once

#include <ros/ros.h>
#include <std_srvs/Trigger.h>
#include <std_srvs/SetBool.h>

// manually include some headers here, since they are missing from 'GUI.h'
#include <list>
#include "Core/Utils/Resolution.h"
#include "Core/Model/Model.h"
#include "GUI.h"

class RosInterface {
public:
    RosInterface(GUI **gui);

    bool on_reset(std_srvs::Trigger::Request &req, std_srvs::Trigger::Response &res);

    bool on_inhibit(std_srvs::SetBool::Request &req, std_srvs::SetBool::Response &res);

    bool on_pause(std_srvs::SetBool::Request &req, std_srvs::SetBool::Response &res);

private:
    // pointer to a pointer to the GUI, since the GUI will not have been
    // allocated yet when this class is constructed
    GUI **gui;

    std::unique_ptr<ros::NodeHandle> n;
    ros::ServiceServer srv_reset;
    ros::ServiceServer srv_inhibit;
    ros::ServiceServer srv_pause;
};
