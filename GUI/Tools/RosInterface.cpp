#ifdef ROSUI

#include "RosInterface.hpp"

RosInterface::RosInterface(GUI **gui, MultiMotionFusion **modelling)
  : gui(gui), modelling(modelling)
{
#if defined(ROS1)
  n = std::make_unique<ros::NodeHandle>("~");

  srv_reset = n->advertiseService("reset", &RosInterface::on_reset, this);

  srv_inhibit = n->advertiseService("inhibit", &RosInterface::on_inhibit, this);

  srv_pause = n->advertiseService("pause", &RosInterface::on_pause, this);

  srv_deactivate_model = n->advertiseService("deactivate_model", &RosInterface::on_deactivate, this);

  srv_set_odom_init = n->advertiseService("set_odom_init", &RosInterface::on_set_odom_init, this);

  srv_set_icp_refine = n->advertiseService("set_icp_refine", &RosInterface::on_set_icp_refine, this);

  srv_set_segm_mode = n->advertiseService("set_segm_mode", &RosInterface::on_set_segm_mode, this);
#elif defined(ROS2)
  using namespace std::placeholders;
  n = std::make_unique<rclcpp::Node>("mmf_ui");

  srv_reset = n->create_service<Trigger>("~/reset", std::bind(&RosInterface::on_reset, this, _1, _2));
  srv_inhibit = n->create_service<SetBool>("~/inhibit", std::bind(&RosInterface::on_inhibit, this, _1, _2));
  srv_pause = n->create_service<SetBool>("~/pause", std::bind(&RosInterface::on_pause, this, _1, _2));
  srv_start = n->create_service<Trigger>("~/start", std::bind(&RosInterface::on_start_stop<false>, this, _1, _2));
  srv_stop = n->create_service<Trigger>("~/stop", std::bind(&RosInterface::on_start_stop<true>, this, _1, _2));
  srv_deactivate_model = n->create_service<SetInt>("~/deactivate_model", std::bind(&RosInterface::on_deactivate, this, _1, _2));
  srv_set_odom_init = n->create_service<SetString>("~/set_odom_init", std::bind(&RosInterface::on_set_odom_init, this, _1, _2));
  srv_set_icp_refine = n->create_service<SetBool>("~/set_icp_refine", std::bind(&RosInterface::on_set_icp_refine, this, _1, _2));
  srv_set_segm_mode = n->create_service<SetString>("~/set_segm_mode", std::bind(&RosInterface::on_set_segm_mode, this, _1, _2));
#endif
}

#if defined(ROS1)
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

bool RosInterface::on_set_odom_init(cob_srvs::SetString::Request &req, cob_srvs::SetString::Response &res)
{
  if (!*modelling) {
    res.success = false;
    res.message = "modelling not initialised";
    return true;
  }

  static const std::unordered_set<std::string> valid = {{}, "kp", "tf"};

  res.success = valid.count(req.data);

  if (valid.count(req.data)) {
    (*modelling)->setOdomInit(req.data);
    res.message = "changed init to: " + req.data;
  }
  else {
    res.message = "invalid init mode: " + req.data;
  }

  return true;
}

bool RosInterface::on_set_icp_refine(std_srvs::SetBool::Request &req, std_srvs::SetBool::Response &res)
{
  if (!*modelling) {
    res.success = false;
    res.message = "modelling not initialised";
    return true;
  }

  (*modelling)->setOdomRefine(req.data);

  res.success = true;
  res.message = "ICP refinement changed to: " + std::to_string(req.data);

  return true;
}

bool RosInterface::on_set_segm_mode(cob_srvs::SetString::Request &req, cob_srvs::SetString::Response &res)
{
  if (!*modelling) {
    res.success = false;
    res.message = "modelling not initialised";
    return true;
  }

  static const std::unordered_set<std::string> valid = {{}, "flow_crf", };

  res.success = valid.count(req.data);

  if (valid.count(req.data)) {
    (*modelling)->setSegmMode(req.data);
    res.message = "changed segmentation mode to: " + req.data;
  }
  else {
    res.message = "invalid segmentation mode mode: " + req.data;
  }

  return true;
}
#elif defined(ROS2)
bool RosInterface::on_reset(Trigger::Request::ConstSharedPtr req, Trigger::Response::SharedPtr res)
{
    if (!*gui) {
        res->success = false;
        res->message = "GUI not initialised";
        return true;
    }

    (*gui)->reset->Ref()->Set(true);
    res->success = true;
    res->message = "reset map and models";
    return true;
}

bool RosInterface::on_inhibit(SetBool::Request::ConstSharedPtr req, SetBool::Response::SharedPtr res)
{
    if (!*gui) {
        res->success = false;
        res->message = "GUI not initialised";
        return true;
    }

    const std::string action = req->data ? "inhibited" : "allowed";

    const bool apply_change = (*gui)->inhibitModels->Get() != req->data;

    if (!apply_change) {
        res->success = false;
        res->message = "inhibit settings not applied: spawning new models is already " + action;
    }
    else {
        (*gui)->inhibitModels->Ref()->Set(req->data);
        res->success = (*gui)->inhibitModels->Get() == req->data;
        if (res->success) {
            res->message = "spawning new models will be " + action;
        }
        else {
            res->message = "could not apply inhibit setting";
        }
    }

    return true;
}

bool RosInterface::on_pause(SetBool::Request::ConstSharedPtr req, SetBool::Response::SharedPtr res)
{
    if (!*gui) {
        res->success = false;
        res->message = "GUI not initialised";
        return true;
    }

    const std::string action = req->data ? "paused" : "running";

    const bool apply_change = (*gui)->pause->Get() != req->data;

    if (!apply_change) {
        res->success = false;
        res->message = "pause setting not applied: already " + action;
    }
    else {
        (*gui)->pause->Ref()->Set(req->data);
        res->success = (*gui)->pause->Get() == req->data;
        if (res->success) {
            res->message = "tracking and modelling is " + action;
        }
        else {
            res->message = "could not apply pause setting";
        }
    }

    return true;
}

template<bool P>
bool RosInterface::on_start_stop(Trigger::Request::ConstSharedPtr req, Trigger::Response::SharedPtr res)
{
    if (!*gui) {
        res->success = false;
        res->message = "GUI not initialised";
        return true;
    }

    const std::string action = P ? "paused" : "running";

    const bool apply_change = (*gui)->pause->Get() != P;

    if (!apply_change) {
        res->success = false;
        res->message = "pause setting not applied: already " + action;
    }
    else {
        (*gui)->pause->Ref()->Set(P);
        res->success = (*gui)->pause->Get() == P;
        if (res->success) {
            res->message = "tracking and modelling is " + action;
        }
        else {
            res->message = "could not apply pause setting";
        }
    }

    return true;
}

bool RosInterface::on_deactivate(SetInt::Request::ConstSharedPtr req, SetInt::Response::SharedPtr res)
{
    if (!*gui) {
        res->success = false;
        res->message = "GUI not initialised";
        return true;
    }
    const uint8_t id = req->data;

    std::cout << "deactivate model: " << int(id) << std::endl;

    if (id==0) {
        res->success = false;
        res->message = "cannot remove environment model (id: 0)";
        return true;
    }

    // search for model with id
    for (const ModelPointer &model : (*modelling)->getModels()) {
        if (model->getID()==id) {
            // schedule model for deactivation
            (*modelling)->scheduleDeactivation(model);
            // respond with success
            res->success = true;
            res->message = "removed model with id: "+std::to_string(id);
            return true;
        }
    }

    // model not found, it's either not active or does not exist at all
    res->success = false;
    res->message = "model "+std::to_string(id)+" does not exist";
    return true;
}

bool RosInterface::on_set_odom_init(SetString::Request::ConstSharedPtr req, SetString::Response::SharedPtr res)
{
    if (!*modelling) {
        res->success = false;
        res->message = "modelling not initialised";
        return true;
    }

    static const std::unordered_set<std::string> valid = {{}, "kp", "tf"};

    res->success = valid.count(req->data);

    if (valid.count(req->data)) {
        (*modelling)->setOdomInit(req->data);
        res->message = "changed init to: " + req->data;
    }
    else {
        res->message = "invalid init mode: " + req->data;
    }

    return true;
}

bool RosInterface::on_set_icp_refine(SetBool::Request::ConstSharedPtr req, SetBool::Response::SharedPtr res)
{
    if (!*modelling) {
        res->success = false;
        res->message = "modelling not initialised";
        return true;
    }

    (*modelling)->setOdomRefine(req->data);

    res->success = true;
    res->message = "ICP refinement changed to: " + std::to_string(req->data);

    return true;
}

bool RosInterface::on_set_segm_mode(SetString::Request::ConstSharedPtr req, SetString::Response::SharedPtr res)
{
    if (!*modelling) {
        res->success = false;
        res->message = "modelling not initialised";
        return true;
    }

    static const std::unordered_set<std::string> valid = {{}, "flow_crf", };

    res->success = valid.count(req->data);

    if (valid.count(req->data)) {
        (*modelling)->setSegmMode(req->data);
        res->message = "changed segmentation mode to: " + req->data;
    }
    else {
        res->message = "invalid segmentation mode mode: " + req->data;
    }

    return true;
}
#endif

#endif
