/*
 * This file is part of ElasticFusion.
 *
 * Copyright (C) 2015 Imperial College London
 *
 * The use of the code within this file and all code within files that
 * make up the software that is ElasticFusion is permitted for
 * non-commercial purposes only.  The full terms and conditions that
 * apply to the code within this file are detailed within the LICENSE.txt
 * file and at <http://www.imperial.ac.uk/dyson-robotics-lab/downloads/elastic-fusion/elastic-fusion-license/>
 * unless explicitly stated.  By downloading this file you agree to
 * comply with these terms.
 *
 * If you wish to use any of this code for commercial purposes then
 * please email researchcontracts.engineering@imperial.ac.uk.
 *
 */

#pragma once

#include "../Utils/RGBDOdometry.h"
#include "../Shaders/Shaders.h"
#include "../Shaders/Uniform.h"
#include "../Shaders/FillIn.h"
#include "../Shaders/FeedbackBuffer.h"
#include "../GPUTexture.h"
#include "../Utils/Resolution.h"
#include "../Utils/Stopwatch.h"
#include "../Utils/Intrinsics.h"
#include "../FrameData.h"
#include "ModelProjection.h"
#include <pangolin/gl/gl.h>
#include <memory>
#include <list>
#include <opencv2/imgproc/imgproc.hpp>
#include <Utils/PointTracker.hpp>
#include "Utils/RigidRANSAC.h"
#include <unordered_set>
#include <filesystem>

#include "Buffers.h"

namespace fs = std::filesystem;

struct OdometryConfig {
  // initialise ICP odometry
  // - (empty): do not initialise, effectively sets initial transformation to identity
  // - "kp": transformation between keypoint tracks
  // - "tf": transformation from log file
  std::string init;
  // frame name as "tf" initialisation source (default: colour optical frame)
  std::string init_frame;
  // pyramid level at which to initialise from keypoints
  int init_lvl = 0;

  // refine via ICP after initialisation (only considered if 'track_init' is set)
  bool icp_refine;

  // pyramid level at which to compute keypoint reprojection error
  int segm_lvl = 0;
};

class IModelMatcher;
class Model;
typedef std::shared_ptr<Model> ModelPointer;
typedef std::list<ModelPointer> ModelList;
typedef ModelList::iterator ModelListIterator;

struct ModelDetectionResult {
  // float prob.
  Eigen::Matrix4f pose;
  bool isGood;
};

class Model {
 public:
  // Shared data for each model
  struct GPUSetup {
    static GPUSetup& getInstance() {
      static GPUSetup instance;
      return instance;
    }

    // TODO: A lot of the attributes here should be either static or encapsulated elsewhere!
    std::shared_ptr<Shader> initProgram;
    std::shared_ptr<Shader> drawProgram;
    std::shared_ptr<Shader> drawSurfelProgram;

    // For supersample fusing
    std::shared_ptr<Shader> dataProgram;
    std::shared_ptr<Shader> updateProgram;
    std::shared_ptr<Shader> unstableProgram;
    std::shared_ptr<Shader> eraseProgram;
    pangolin::GlRenderBuffer renderBuffer;

    // NOTICE: The dimension of these textures suffice the VBO, not the sensor! See TEXTURE_DIMENSION
    pangolin::GlFramebuffer frameBuffer;  // Frame buffer, holding the following textures:
    GPUTexture updateMapVertsConfs;       // We render updated vertices vec3 + confidences to one texture
    GPUTexture updateMapColorsTime;       // We render updated colors vec3 + timestamps to another
    GPUTexture updateMapNormsRadii;       // We render updated normals vec3 + radii to another

    // Current depth / mask pyramid used for Odometry
    std::vector<DeviceArray2D<float>> depth_tmp;
    std::vector<DeviceArray2D<unsigned char>> mask_tmp;

    float outlierCoefficient = 0.9;

   private:
    GPUSetup();
  };

  enum class MatchingType { Drost };

  typedef Eigen::Matrix<cv::Point, Eigen::Dynamic, Eigen::Dynamic> MatrixXp2;
  typedef Eigen::Matrix<Eigen::Vector3d, Eigen::Dynamic, Eigen::Dynamic> MatrixXp3;

 public:
  static const int TEXTURE_DIMENSION;
  static const int MAX_VERTICES;
  static const int NODE_TEXTURE_DIMENSION;
  static const int MAX_NODES;

 private:
  // static std::list<unsigned char> availableIDs;

 public:
  Model(unsigned char id, float confidenceThresh, const OdometryConfig &odom_cfg, bool enableFillIn = true, bool enableErrorRecording = true,
        bool enablePoseLogging = false, MatchingType matchingType = MatchingType::Drost,
        float maxDepth = std::numeric_limits<float>::max());  // TODO: Default disable
  virtual ~Model();

  virtual bool load(const fs::path &model_path);

  // ----- Functions ----- //

  virtual unsigned int lastCount();
  static Eigen::Vector3f rodrigues2(const Eigen::Matrix3f& matrix);  // TODO mode to a nicer place

  // ----- Re-detection

  virtual void buildDescription();
  virtual ModelDetectionResult detectInRegion(const FrameData& frame, const cv::Rect& rect);

  // ----- Init

  virtual void initialise(const FeedbackBuffer& rawFeedback, const FeedbackBuffer& filteredFeedback);

  virtual void renderPointCloud(const Eigen::Matrix4f& vp, const bool drawUnstable, const bool drawPoints, const bool drawWindow,
                                const int colorType, const int time, const int timeDelta);

  static void generateCUDATextures(GPUTexture* depth, GPUTexture* mask);

  virtual void initICP(bool doFillIn, bool frameToFrameRGB, float depthCutoff, GPUTexture* rgb);

  // ----- Tracking and fusion

  virtual void performTracking(bool frameToFrameRGB, bool rgbOnly, float icpWeight, bool pyramid, bool fastOdom, bool so3,
                               float maxDepthProcessed, GPUTexture* rgb, int64_t logTimestamp, bool tryFillIn = false);

  // compute the projection error between keypoints on the trajectory for segmentation
  static std::tuple<Eigen::MatrixXd, Model::MatrixXp2, Model::MatrixXp3>
  computeTrackProjectionError(const tracker::Tracks &tracks);

  // project all 2D and 3D keypoints into the last frame of the model trajectory
  virtual tracker::Tracks computeTrackProjectionLastFrame(const tracker::Tracks& tracks, const size_t length = 0) const;

  // project all 3D keypoints into the first (initial) model frame
  virtual tracker::Tracks computeTrackProjectionFirstFrame() const;

  virtual tracker::Tracks computeTrackProjectionStartEnd(const tracker::Tracks& tracks, const size_t length);

  static cv::Mat drawLocalTracks2D(const tracker::Tracks &tracks, const cv::Mat &img, int msize = 10, bool mscale = false);

  // initialise the first set of tracks for the global model
  virtual void initGlobalTracks(const tracker::Tracks& tracks, const Eigen::Isometry3f &initial_pose = Eigen::Isometry3f::Identity(), const uint64_t &time = 0);

  // add/remove tracks
  virtual void updateTracks(const tracker::Tracks& tracks_add = {}, const tracker::Tracks &tracks_remove = {});

  virtual void removeLastTrackKeypoint();

  // re-estimate all model poses given the track subset
  virtual void refineTrackSubset(const tracker::Tracks& tracks, const ModelPointer &parent, const size_t &history = std::numeric_limits<size_t>::infinity());

  static RigidRANSAC::Result getLastTrackTransform(const tracker::Tracks &tracks, const RigidRANSAC::Config &config = {10, 0.03f, 0.6f});

  // get the transformation between the last two point sets on model tracks
  virtual RigidRANSAC::Result getLastTrackTransform() const;

  virtual RigidRANSAC::Result getBestMatch(const std::vector<tracker::KeypointPtr> &keypoints, const RigidRANSAC::Config &config) const;

  // Compute fusion-weight based on velocity
  virtual float computeFusionWeight(float weightMultiplier) const;

  // Assuming the indexMap is already computed, perform fusion. 1) associate data, 2) update model
  virtual void fuse(const int& time, GPUTexture* rgb, GPUTexture* mask, GPUTexture* depthRaw, GPUTexture* depthFiltered,
                    const float depthCutoff, const float weightMultiplier);

  // Always called after fuse. Copy unstable points to map.
  virtual void clean(const int& time, std::vector<float>& graph, const int timeDelta, const float depthCutoff, const bool isFern,
                     GPUTexture* depthFiltered, GPUTexture* mask);

  // ...
  virtual void eraseErrorGeometry(GPUTexture* depthFiltered);

  // ----- Prediction and fillin

  inline bool allowsFillIn() const { return fillIn ? true : false; }

  void performFillIn(GPUTexture* rawRGB, GPUTexture* rawDepth, bool frameToFrameRGB, bool lost);

  inline void combinedPredict(float depthCutoff, int time, int maxTime, int timeDelta, ModelProjection::Prediction predictionType) {
    indexMap.combinedPredict(getPose(), getModel(), depthCutoff, getConfidenceThreshold(), time, maxTime, timeDelta, predictionType);
  }

  inline void predictIndices(int time, float depthCutoff, int timeDelta) {
    indexMap.predictIndices(getPose(), time, getModel(), depthCutoff, timeDelta);
  }

  // ----- Getter ----- //

  inline float getConfidenceThreshold() const { return confidenceThreshold; }

  inline void setConfidenceThreshold(float confThresh) { confidenceThreshold = confThresh; }

  inline void setMaxDepth(float d) { maxDepth = d; }

  // Returns a vector of 4-float tuples: position0, color0, normal0, ..., positionN, colorN, normalN
  // Where position is (x,y,z,conf), color is (color encoded as a 24-bit integer, <unused>, initTime, timestamp), and normal
  // is (x,y,z,radius)
  struct SurfelMap {
    std::unique_ptr<std::vector<Eigen::Vector4f>> data;
    void countValid(const float& confThres) {
      numValid = 0;
      for (unsigned int i = 0; i < numPoints; i++) {
        const Eigen::Vector4f& pos = (*data)[(i * 3) + 0];
        if (pos[3] > confThres) numValid++;
      }
    }
    unsigned numPoints = 0;
    unsigned numValid = 0;
  };

  virtual SurfelMap downloadMap() const;

  // surfel memory representation
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

  // inline cv::Mat downloadUnaryConfTexture() {
  //    return indexMap.getUnaryConfTex()->downloadTexture();
  //}

  static void exportTracksPLY(const tracker::Tracks &tracks, const std::string &path, const Eigen::Isometry3f &pose = Eigen::Isometry3f::Identity(), bool with_descriptor = false, bool binary = true);

  void exportTracksPLY(const std::string &export_dir, const Eigen::Isometry3f &global_pose, bool binary = true) const;

  static void exportModelPLY(const SurfelMap &surfels, const float conf_threshold, const std::string &path, const Eigen::Isometry3f &pose = Eigen::Isometry3f::Identity());

  void exportModelPLY(const std::string &export_dir, const Eigen::Isometry3f &global_pose) const;

  inline cv::Mat downloadVertexConfTexture() { return indexMap.getSplatVertexConfTex()->downloadTexture(); }

  inline cv::Mat downloadICPErrorTexture() { return icpError->downloadTexture(); }
  inline cv::Mat downloadRGBErrorTexture() { return rgbError->downloadTexture(); }

  inline GPUTexture* getICPErrorTexture() { return icpError.get(); }
  inline GPUTexture* getRGBErrorTexture() { return rgbError.get(); }

  inline GPUTexture* getRGBProjection() { return indexMap.getSplatImageTex(); }
  inline GPUTexture* getVertexConfProjection() { return indexMap.getSplatVertexConfTex(); }
  inline GPUTexture* getNormalProjection() { return indexMap.getSplatNormalTex(); }
  inline GPUTexture* getTimeProjection() { return indexMap.getSplatTimeTex(); }
  // inline GPUTexture* getUnaryConfTexture() {
  //    return indexMap.getUnaryConfTex();
  //}
  inline GPUTexture* getFillInImageTexture() { return &(fillIn->imageTexture); }
  inline GPUTexture* getFillInNormalTexture() { return &(fillIn->normalTexture); }
  inline GPUTexture* getFillInVertexTexture() { return &(fillIn->vertexTexture); }

  virtual const OutputBuffer& getModel();

  inline const Eigen::Matrix4f& getPose() const { return pose; }
  inline const Eigen::Matrix4f& getLastPose() const { return lastPose; }
  inline void overridePose(const Eigen::Matrix4f& p) {
    pose = p;
    lastPose = p;
  }
  inline Eigen::Matrix4f getLastTransform() const { return getPose().inverse() * lastPose; }

  inline unsigned int getID() const { return id; }

  inline RGBDOdometry& getFrameOdometry() { return frameToModel; }
  inline ModelProjection& getIndexMap() { return indexMap; }

  const MatrixXp2& getTrackXY() const { return track_xy; };
  const MatrixXp3& getTrackPoint() const  { return track_p; };
  const Eigen::MatrixXd& getTrackProjError() const { return track_pe; };

  inline unsigned getUnseenCount() const { return unseenCount; }
  inline void resetUnseenCount() { unseenCount = 0; }
  inline int64_t incrementUnseenCount() {
    return std::min(++unseenCount, std::numeric_limits<int64_t>::max());
  }

  inline int64_t decrementUnseenCount(const int64_t &decrement = 1) {
    unseenCount -= decrement;
    return std::max(unseenCount, std::numeric_limits<int64_t>::min());
  }

  struct PoseLogItem {
    int64_t ts;
    Eigen::Matrix<float, 7, 1> p;  // x,y,z, qx,qy,qz,qw
  };
  inline bool isLoggingPoses() const { return poseLog.capacity() > 0; }
  inline std::vector<PoseLogItem>& getPoseLog() { return poseLog; }

  void appendPoses(const Eigen::Isometry3f& pose, const uint64_t &time) {
    timestamp_ns.push_back(time);
    poses.push_back(pose);
  };

  // ----- Save & Load ----- //

  void store(const fs::path &model_db_path, const Eigen::Isometry3f &pose, bool clear = true);

  void activate(const Eigen::Isometry3f &pose, const int64_t &timestamp);

 protected:
  // Current pose
  Eigen::Matrix4f pose;
  Eigen::Matrix4f lastPose;

  // poses and timestamps in nanoseconds
  std::vector<uint64_t> timestamp_ns;
  std::vector<Eigen::Isometry3f> poses;

  std::vector<PoseLogItem> poseLog;  // optional, for testing

  // Confidence Threshold (low in the beginning, increasing)
  float confidenceThreshold;
  float maxDepth;

  // surfel buffers (swapping in order to update)
  OutputBuffer vbos[2];  // Todo make dynamic buffer

  int target, renderSource;  // swapped after FUSE and CLEAN

  // MAX_VERTICES * Vertex::SIZE, currently 3072*3072 vertices
  static const int bufferSize;

  // Count surfels in model
  GLuint countQuery;
  unsigned int count;

  // 16 floats stored column-major yo'
  static GPUTexture deformationNodes;  // Todo outsource to derived class?

  OutputBuffer newUnstableBuffer;
  // Vbo, newUnstableFid;

  GLuint uvo;  // One element for each pixel (size=width*height), used as "layout (location = 0) in vec2 texcoord;" in data.vert
  int uvSize;
  unsigned int id;

  std::unique_ptr<GPUTexture> icpError;
  std::unique_ptr<GPUTexture> rgbError;

  // set of associated tracks in camera frame
  std::unordered_set<tracker::TrackPtr> tracks;

  // local projected tracks, stored when the model is deactivated
  // keypoints are synchronous with poses
  tracker::Tracks tracks_local;

  // track projection error
  Eigen::MatrixXd track_pe;
  MatrixXp2 track_xy;
  MatrixXp3 track_p;

  const GPUSetup& gpu;

  ModelProjection indexMap;
  RGBDOdometry frameToModel;

  int64_t unseenCount = 0;

  // Fill in holes in prediction (using raw data)
  std::unique_ptr<FillIn> fillIn;

  // Allows to detect inactive models in depth-map-region
  std::unique_ptr<IModelMatcher> modelMatcher;
};
