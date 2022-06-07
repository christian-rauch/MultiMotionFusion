#ifdef ROSNODE

#include "RosStatePublisher.hpp"
#include <sensor_msgs/Image.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/point_cloud2_iterator.h>
#include <std_msgs/String.h>
#include <cv_bridge/cv_bridge.h>
#include <eigen_conversions/eigen_msg.h>
#include <opencv2/imgcodecs.hpp>


struct surfel_t {
  // point
  Eigen::Vector3f point;  // 96 bit
  float confidence;       // 32 bit

  // colour
  float colour;       // 32 bit
  uint32_t unused;    // 32 bit
  float init_time;    // 32 bit
  float timestamp;    // 32 bit

  // normal
  Eigen::Vector3f normal; // 96 bit
  float radius;           // 32 bit
};

// a surfel should consume 3 x 4 x 32 = 384 bit = 48 byte
static_assert(sizeof(surfel_t) == 48, "struct surfel_t is misaligned");

static sensor_msgs::CameraInfo
get_camera_info()
{
  sensor_msgs::CameraInfo ci;
  ci.width = Resolution::getInstance().width();
  ci.height = Resolution::getInstance().height();
  ci.K[0] = ci.P[0] = Intrinsics::getInstance().fx();
  ci.K[2] = ci.P[2] = Intrinsics::getInstance().cx();
  ci.K[4] = ci.P[5] = Intrinsics::getInstance().fy();
  ci.K[5] = ci.P[6] = Intrinsics::getInstance().cy();
  ci.K[8] = ci.P[10] = 1;
  return ci;
}


RosStatePublisher::RosStatePublisher(const std::string &camera_frame) :
    camera_frame(camera_frame)
{
  n = std::make_unique<ros::NodeHandle>("~");

  it = std::make_unique<image_transport::ImageTransport>(*n);

  pub_segm = it->advertise("segmentation", 1);

  pub_camera_info = n->advertise<sensor_msgs::CameraInfo>("camera_info", 1);

  pub_status_message = n->advertise<std_msgs::String>("status", 1, true);
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

  // object poses (id>0) with respect to the global model (id=0)
  std::vector<geometry_msgs::TransformStamped> pose_objects;

  const Eigen::Isometry3f pose_global(models.front()->getPose());

  // reserve a single point cloud and its modifier
  sensor_msgs::PointCloud2 point_cloud;
  point_cloud.header = hdr;
  // unordered point cloud
  point_cloud.height = 1;
  point_cloud.is_dense = true;

  sensor_msgs::PointCloud2Modifier pc_modifier(point_cloud);
  pc_modifier.setPointCloud2FieldsByString(2, "xyz", "rgb");

  for (const ModelPointer &model : models) {
    const unsigned int id = model->getID();

    if (!pub_model_pc.count(id)) {
      // create new publisher for model
      pub_model_pc[id] = n->advertise<sensor_msgs::PointCloud2>("model/"+std::to_string(id)+"/dense", 1);
    }

    const Eigen::Isometry3f T_0X(model->getPose());

    if (id>0) {
      // object pose in camera frame
      const Eigen::Isometry3f T_cm = pose_global * T_0X.inverse();
      geometry_msgs::TransformStamped pose;
      tf::transformEigenToMsg(T_cm.cast<double>(), pose.transform);
      pose.header = hdr;
      pose.child_frame_id = "model/"+std::to_string(id);
      pose_objects.push_back(pose);
    }

    Model::SurfelMap surfel_map = model->downloadMap();
    // memory layout of 'surfel_map.data'
    // the surfel map is list of surfels consisting of point, colour, normal:
    // (p0, c0, n0) | (p1, c1, n1) | ... | (pN, cN, nN)

    const surfel_t *surfel = reinterpret_cast<const surfel_t *>(surfel_map.data->data());

    surfel_map.countValid(model->getConfidenceThreshold());

    point_cloud.width = surfel_map.numValid;
    pc_modifier.resize(surfel_map.numValid);

    // NOTE: iterators have to be created after resizing the point cloud
    // create a single iterator for a 3D float vector
    sensor_msgs::PointCloud2Iterator<Eigen::Vector3f>iter_xyz(point_cloud, "x");
    sensor_msgs::PointCloud2Iterator<uint8_t>iter_rgb(point_cloud, "rgb");

    for (unsigned int i = 0; i < surfel_map.numPoints; i++) {
      if (surfel[i].confidence > model->getConfidenceThreshold()) {
        *iter_xyz = T_0X.inverse() * surfel[i].point;
        ++iter_xyz;

        // NOTE: the 'float' colour is NOT the memory representation of an uint32,
        //       that float value has to be cast to an integere explicitely
        const uint32_t c = uint32_t(surfel[i].colour);

        std::memcpy(&(*iter_rgb), &c, 3 * sizeof(uint8_t));
        ++iter_rgb;
      }
    }

    pub_model_pc[id].publish(point_cloud);

    pc_modifier.clear();
  }

  sensor_msgs::CameraInfo ci = get_camera_info();
  ci.header = hdr;

  for (const ModelPointer &model : models) {
    const unsigned int id = model->getID();
    // get model projection
    const cv::Mat colour = model->getRGBProjection()->downloadTexture();
    const cv::Mat points = model->getVertexConfProjection()->downloadTexture();
    // convert metric points to depth in millimetre
    std::vector<cv::Mat> xyz;
    cv::split(points, xyz);
    cv::Mat depth_mm;
    xyz[2].convertTo(depth_mm, CV_16UC1, 1e3);

    if (!pub_model_proj_colour.count(id)) {
      pub_model_proj_colour[id] = n->advertise<sensor_msgs::CompressedImage>("model/"+std::to_string(id)+"/colour/compressed", 1);
    }
    if (!pub_model_proj_depth.count(id)) {
      pub_model_proj_depth[id] = n->advertise<sensor_msgs::CompressedImage>("model/"+std::to_string(id)+"/depth/compressed", 1);
      pub_camera_info_depth[id] = n->advertise<sensor_msgs::CameraInfo>("model/"+std::to_string(id)+"/camera_info", 1);
    }

    pub_model_proj_colour[id].publish(cv_bridge::CvImage(hdr, "rgb8", colour).toCompressedImageMsg());

    // we have to manually compress the 16 bit PNG image
    sensor_msgs::CompressedImage msg_depth;
    cv::imencode(".png", depth_mm, msg_depth.data);
    msg_depth.header = hdr;
    msg_depth.format = "16UC1; png compressed ";

    pub_model_proj_depth[id].publish(msg_depth);

    pub_camera_info_depth[id].publish(ci);
  }

  pub_camera_info.publish(ci);

  if (!pose_objects.empty()) {
    broadcaster.sendTransform(pose_objects);
  }
}

void RosStatePublisher::reset()
{
  // delete all model specific publisher
  // TODO: disconnecting and later reconnecting the subscribers will cause communication delays
#if 0
  pub_camera_info_depth.clear();
  pub_model_pc.clear();
  pub_model_proj_colour.clear();
  pub_model_proj_depth.clear();
#endif
}

void RosStatePublisher::send_status_message(const std::string &message)
{
  std_msgs::String msg;
  msg.data = message;
  pub_status_message.publish(msg);
}

#endif
