#ifdef ROSNODE

#include "RosInterface.hpp"

RosInterface::RosInterface(GUI **gui)
  : gui(gui)
{
  n = std::make_unique<ros::NodeHandle>("~");

  srv_reset = n->advertiseService("reset", &RosInterface::on_reset, this);

  srv_inhibit = n->advertiseService("inhibit", &RosInterface::on_inhibit, this);

  srv_pause = n->advertiseService("pause", &RosInterface::on_pause, this);
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

bool RosInterface::on_pause(std_srvs::SetBool::Request &req, std_srvs::SetBool::Response &res)
{
  if (!*gui) {
    res.success = false;
    res.message = "GUI not initialised";
    return true;
  }

  const std::string action = req.data ? "paused" : "running";

  const bool apply_change = (*gui)->pause->Get() != req.data;

  if (!apply_change) {
    res.success = false;
    res.message = "pause setting not applied: already " + action;
  }
  else {
    (*gui)->pause->Ref()->Set(req.data);
    res.success = (*gui)->pause->Get() == req.data;
    if (res.success) {
      res.message = "tracking and modelling is " + action;
    }
    else {
      res.message = "could not apply pause setting";
    }
  }

  return true;
}

bool RosInterface::on_deactivate(cob_srvs::SetInt::Request &req, cob_srvs::SetInt::Response &res)
{
  if (!*gui) {
    res.success = false;
    res.message = "GUI not initialised";
    return true;
  }
  const uint8_t id = req.data;

  std::cout << "deactivate model: " << int(id) << std::endl;

  if (id==0) {
    res.success = false;
    res.message = "cannot remove environment model (id: 0)";
    return true;
  }

  // search for model with id
  for (const ModelPointer &model : (*modelling)->getModels()) {
    if (model->getID()==id) {
      // schedule model for deactivation
      (*modelling)->scheduleDeactivation(model);
      // respond with success
      res.success = true;
      res.message = "removed model with id: "+std::to_string(id);
      return true;
    }
  }

  // model not found, it's either not active or does not exist at all
  res.success = false;
  res.message = "model "+std::to_string(id)+" does not exist";
  return true;
}

#endif
