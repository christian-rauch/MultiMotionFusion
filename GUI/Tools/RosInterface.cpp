#ifdef ROSNODE

#include "RosInterface.hpp"

RosInterface::RosInterface(GUI **gui)
  : gui(gui)
{
  n = std::make_unique<ros::NodeHandle>("~");

  srv_reset = n->advertiseService("reset", &RosInterface::on_reset, this);

  srv_inhibit = n->advertiseService("inhibit", &RosInterface::on_inhibit, this);
}

bool RosInterface::on_reset(std_srvs::Trigger::Request &req, std_srvs::Trigger::Response &res)
{
  if (!*gui) {
    res.success = false;
    res.message = "GUI not initialised";
    return true;
  }

  (*gui)->reset->Ref()->Set(true);
  res.success = true;
  res.message = "reset map and models";
  return true;
}

bool RosInterface::on_inhibit(std_srvs::SetBool::Request &req, std_srvs::SetBool::Response &res)
{
  if (!*gui) {
    res.success = false;
    res.message = "GUI not initialised";
    return true;
  }

  const std::string action = req.data ? "inhibited" : "allowed";

  const bool apply_change = (*gui)->inhibitModels->Get() != req.data;

  if (!apply_change) {
    res.success = false;
    res.message = "inhibit settings not applied: spawning new models is already " + action;
  }
  else {
    (*gui)->inhibitModels->Ref()->Set(req.data);
    res.success = (*gui)->inhibitModels->Get() == req.data;
    if (res.success) {
      res.message = "spawning new models will be " + action;
    }
    else {
      res.message = "could not apply inhibit setting";
    }
  }

  return true;
}

#endif
