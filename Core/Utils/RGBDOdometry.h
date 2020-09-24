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

#ifndef RGBDODOMETRY_H_
#define RGBDODOMETRY_H_

#include "Stopwatch.h"
#include "../GPUTexture.h"
#include "../Cuda/cudafuncs.cuh"
#include "OdometryProvider.h"
#include "GPUConfig.h"

#include <vector>
#include <vector_types.h>
#include <queue>
#include <array>

struct OdometryConfig {
  // estimation mode:
  // - (empty): use default ICP, no keypoint transformation estimation
  // - "icp": ICP with keypoint correspondences
  // - "ls": RANSAC least-squares optimisation
  std::string mode_est;

  // motion source:
  // "est": use previous estimated transform
  // "ransac": independently use RANSAC on keypoints
  std::string segm_source;

  // segmentation mode:
  // "dense": reprojection of dense  depth (default)
  // "sparse": reprojection of sparse keypoints
  std::string segm_mode;

  size_t history;
};

class RGBDOdometry {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  RGBDOdometry(int width, int height, float cx, float cy, float fx, float fy, unsigned char maskID,
               const OdometryConfig &cfg,
               float distThresh = 0.10f,  // TODO Check, hardcoded scale?
               float angleThresh = sin(20.f * 3.14159254f / 180.f));

  virtual ~RGBDOdometry();

  const int2 &getPyramidDim(const int level);

  // Prepare current frame data for CUDA ICP execution
  // void initICP(GPUTexture * filteredDepth, const float depthCutoff, GPUTexture * mask); // frame to model
  void initICP(const std::vector<DeviceArray2D<float> >& depthPyramid, const std::vector<DeviceArray2D<unsigned char> >& maskPyramid,
               const float depthCutoff);                                                               // frame to model
  void initICP(GPUTexture* predictedVertices, GPUTexture* predictedNormals, const float depthCutoff);  // model to model

  // Prepare model data for CUDA ICP execution. Information from the last frame.
  void initICPModel(GPUTexture* predictedVertices, GPUTexture* predictedNormals, const float depthCutoff, const Eigen::Matrix4f& modelPose);

  void initRGB(GPUTexture* rgb);

  void initRGBModel(GPUTexture* rgb);

  void initRGBDFromPrevious(const Eigen::Matrix4f &pose);

  void initFirstRGB(GPUTexture* rgb);

  // Get relative transformation, executes optimisation
  void getIncrementalTransformation(Eigen::Vector3f& trans, Eigen::Matrix<float, 3, 3, Eigen::RowMajor>& rot, const bool& rgbOnly,
                                    const float& icpWeight, const bool& pyramid, const bool& fastOdom, const bool& so3,
                                    const cudaSurfaceObject_t& icpErrorSurface, const cudaSurfaceObject_t& rgbErrorSurface,
                                    const std::vector<std::unique_ptr<GPUTexture>> &projError);

  Eigen::MatrixXd getCovariance();

  void setNextKeypoints(const std::vector<Eigen::MatrixX2d> &kp_coordinates, const std::vector<Eigen::MatrixXd> &kp_descriptors);
  void setLastKeypointsFromPrevious();

  void setNextFeatureMap(const std::vector<cv::Mat> &feat);
  void setLastFeatureMapFromPrevious();

  void setLastSegmentation(const cv::Mat &segm);

  float lastICPError;
  float lastICPCount;
  float lastRGBError;
  float lastRGBCount;
  float lastSO3Error;
  float lastSO3Count;

  Eigen::Matrix<double, 6, 6, Eigen::RowMajor> lastA;
  Eigen::Matrix<double, 6, 1> lastb;

  static const int NUM_PYRS = 3;

 private:
  void populateRGBDData(GPUTexture* rgb, DeviceArray2D<float>* destDepths, DeviceArray2D<unsigned char>* destImages,
                        DeviceArray2D<unsigned char>* destMasks);

  DeviceArray<float> vmaps_tmp;
  DeviceArray<float> nmaps_tmp;

  // Prediction pyramid (projected)
  std::vector<DeviceArray2D<float> > vmaps_g_prev_;
  std::vector<DeviceArray2D<float> > nmaps_g_prev_;

  // Current frame pyramid
  std::vector<DeviceArray2D<float> > vmaps_curr_;
  std::vector<DeviceArray2D<float> > nmaps_curr_;

  CameraModel intr;

  DeviceArray<JtJJtrSE3> sumDataSE3;
  DeviceArray<JtJJtrSE3> outDataSE3;
  DeviceArray<float2> sumResidualRGB;

  DeviceArray<JtJJtrSO3> sumDataSO3;
  DeviceArray<JtJJtrSO3> outDataSO3;

  const int sobelSize;
  const float sobelScale;
  const float maxDepthDeltaRGB;
  const float maxDepthRGB;

  std::vector<int2> pyrDims;

  // Used during optimisation, rgb-related
  DeviceArray2D<short> nextdIdx[NUM_PYRS];
  DeviceArray2D<short> nextdIdy[NUM_PYRS];

  // Handle textures logic?
  DeviceArray2D<float> lastDepth[NUM_PYRS];
  DeviceArray2D<float> nextDepth[NUM_PYRS];

  DeviceArray2D<unsigned char> lastMask[NUM_PYRS];
  DeviceArray2D<unsigned char> nextMask[NUM_PYRS];

  DeviceArray2D<unsigned char> lastImage[NUM_PYRS];
  DeviceArray2D<unsigned char> nextImage[NUM_PYRS];
  DeviceArray2D<unsigned char> lastNextImage[NUM_PYRS];

  DeviceArray2D<DataTerm> corresImg[NUM_PYRS];

  DeviceArray2D<float3> pointClouds[NUM_PYRS];
  DeviceArray2D<float3> nextPointClouds[NUM_PYRS];

  std::vector<int> iterations;
  std::vector<float> minimumGradientMagnitudes;

  float distThres_;
  float angleThres_;

  Eigen::Matrix<double, 6, 6> lastCov;

  const int width;
  const int height;
  const float cx, cy, fx, fy;

  unsigned char maskID;

  const OdometryConfig cfg;

  // N x (2+D) matrix that stores N keypoints row-wise
  // with 2 normalised [0,1] coordinates (x,y) and a D feature vector
  DeviceArray2D<float> nextKeypoints[NUM_PYRS];
  DeviceArray2D<float> lastKeypoints[NUM_PYRS];
  DeviceArray2D<int> matchID[NUM_PYRS]; // N x 2: {(last, next)}
  DeviceArray2D<float> matchScores[NUM_PYRS];

  // W x H x D tensor
  DeviceArray2D<float> nextFeatureMaps[NUM_PYRS];
  DeviceArray2D<float> lastFeatureMaps[NUM_PYRS];

  // local host copies
  typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> mXXf;
  std::vector<mXXf> next_keypoints;
  std::vector<mXXf> last_keypoints;
  std::vector<std::vector<std::tuple<int, int, float>>> matches; // {(last, next, score)}
  std::vector<cv::Mat> next_features;
  std::vector<cv::Mat> last_features;
  cv::Mat next_segmentation;
  cv::Mat last_segmentation;

  // store list of previous correspondences and depth
  std::queue<std::array<cv::Mat_<uint8_t>, NUM_PYRS>> Nlast_image;
  std::queue<std::array<DeviceArray2D<float>, NUM_PYRS>> NlastDepth;
//  std::queue<std::array<DeviceArray2D<float>, NUM_PYRS>> NlastFeatureMaps;
  std::queue<std::array<mXXf, NUM_PYRS>> Nlast_keypoints;
  std::queue<Eigen::Isometry3f> Nlast_poses;
  std::queue<std::array<DeviceArray2D<unsigned char>, NUM_PYRS>> NlastMask;

  size_t iimg = 0;
};

#endif /* RGBDODOMETRY_H_ */
