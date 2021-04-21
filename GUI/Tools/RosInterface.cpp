#ifdef ROSNODE

#include "RosInterface.hpp"

RosInterface::RosInterface(GUI **gui)
  : gui(gui)
{
  n = std::make_unique<ros::NodeHandle>("~");

  srv_reset = n->advertiseService("reset", &RosInterface::on_reset, this);
}

bool RosInterface::on_reset(std_srvs::Trigger::Request &req, std_srvs::Trigger::Response &res)
{
  (*gui)->reset->Ref()->Set(true);
  res.success = true;
  res.message = "reset map and models";
  return true;
}

#endif
