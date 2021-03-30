#ifdef ROSNODE

#include "RosStatePublisher.hpp"
#include <sensor_msgs/Image.h>
#include <sensor_msgs/PointCloud.h>
#include <cv_bridge/cv_bridge.h>

RosStatePublisher::RosStatePublisher(const std::string &camera_frame) :
    camera_frame(camera_frame)
{
  n = std::make_unique<ros::NodeHandle>("~");

  it = std::make_unique<image_transport::ImageTransport>(*n);

  pub_segm = it->advertise("segmentation", 1);
}

void RosStatePublisher::pub_segmentation(const cv::Mat &segmentation, const int64_t timestamp_ns)
{
  std_msgs::Header hdr;
  hdr.frame_id = camera_frame;
  hdr.stamp.fromNSec(timestamp_ns);
  pub_segm.publish(cv_bridge::CvImage(hdr, "rgb8", segmentation).toImageMsg());
}

void RosStatePublisher::pub_models(const ModelList &models, const int64_t timestamp_ns)
{
  // all dense model point clouds are expressed in the camera frame
  std_msgs::Header hdr;
  hdr.frame_id = camera_frame;
  hdr.stamp.fromNSec(timestamp_ns);

  for (const ModelPointer &model : models) {
    const unsigned int id = model->getID();

    if (!pub_model_pc.count(id)) {
      // create new publisher for model
      pub_model_pc[id] = n->advertise<sensor_msgs::PointCloud>("model/dense/"+std::to_string(id), 1);
    }

    sensor_msgs::PointCloud point_cloud;
    point_cloud.header = hdr;

    const Model::SurfelMap surfelMap = model->downloadMap();

    const Eigen::Isometry3f T_0X(model->getPose());

    for (unsigned int i = 0; i < surfelMap.numPoints; i++) {
      Eigen::Vector4f pos4 = (*surfelMap.data)[(i * 3) + 0];
      const float conf = pos4[3];

      if (conf > model->getConfidenceThreshold()) {
        Eigen::Vector3f pos(pos4.x(), pos4.y(), pos4.z());
        pos = T_0X.inverse() * pos;
        geometry_msgs::Point32 p;
        p.x = pos.x();
        p.y = pos.y();
        p.z = pos.z();
        point_cloud.points.push_back(p);
      }
    }

    pub_model_pc[id].publish(point_cloud);
  }
}

#endif
