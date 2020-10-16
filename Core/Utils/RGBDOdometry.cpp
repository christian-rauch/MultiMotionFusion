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

#include "RGBDOdometry.h"
#include "RigidRANSAC.h"

#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>


// download a 1-channel device array to an OpenCV matrix
template<typename T>
cv::Mat_<T> download(const DeviceArray2D<T> &array)
{
  cv::Mat_<T> img(array.rows(), array.cols());
  array.download(img.data, img.step);
  return img;
}

template<typename T>
void download(const DeviceArray2D<T> &array, cv::Mat_<T> &img)
{
  img.create(array.rows(), array.cols());
  array.download(img.data, img.step);
}

// upload 1D array to cv mat
template<typename T>
void upload(const cv::Mat_<T> &img, DeviceArray2D<T> &array)
{
  array.create(img.rows, img.cols);
  array.upload(img.data, img.step, img.rows, img.cols);
}

// upload ND array to cv mat
template<typename T>
void uploadND(const cv::Mat &img, DeviceArray2D<T> &array)
{
  array.create(img.rows, img.cols);
  array.upload(img.data, img.step, img.rows, img.cols*img.channels());
}

// upload row-major Eigen matrix to device array
template <typename T, int R, int C>
void upload_eigen(const Eigen::Matrix<T, R, C, Eigen::RowMajor> &matrix,
                  DeviceArray2D<T> &array)
{
  array.upload(matrix.data(), matrix.cols() * sizeof(T), matrix.rows(), matrix.cols());
}

std::vector<std::tuple<int, int, float>>
pairwise_matches(const Eigen::MatrixXf &last_keypoints,
                 const Eigen::MatrixXf &next_keypoints,
                 const cv::Mat_<bool> &last_mask = {})
{
  // remove 'last' keypoint coordinates outside of mask
  Eigen::MatrixXf last_keypoints_mask = Eigen::MatrixXf::Constant(last_keypoints.rows(), last_keypoints.cols(), std::numeric_limits<float>::signaling_NaN());
  std::vector<int> last_mask_id;
  if (!last_mask.empty()) {
    int kp_matches = 0;
    for(int i=0; i<last_keypoints.rows(); i++) {
        const Eigen::Array2f xy_norm = last_keypoints.leftCols(2).row(i);
        const cv::Point2i xy(xy_norm.x()*last_mask.cols, xy_norm.y()*last_mask.rows);
        if (last_mask.at<bool>(xy)) {
          // copy match over
          last_keypoints_mask.row(kp_matches) = last_keypoints.row(i);
          kp_matches++;
          // store original ID of last keypoint within valid segment
          last_mask_id.push_back(i);
        }
    }
    last_keypoints_mask.conservativeResize(kp_matches, Eigen::NoChange);
  }
  else {
    // use all keypoints
    last_keypoints_mask = last_keypoints;
  }

  // store correspondences (last_id, next_id)
  std::vector<std::tuple<int, int, float>> match_ids;

  if (last_keypoints_mask.rows()>0) {
    cv::Mat last_descr;
    cv::eigen2cv(Eigen::MatrixXf(last_keypoints_mask.rightCols(last_keypoints_mask.cols()-2)), last_descr);
    cv::Mat next_descr;
    cv::eigen2cv(Eigen::MatrixXf(next_keypoints.rightCols(next_keypoints.cols()-2)), next_descr);

    std::vector<cv::DMatch> matches;
    cv::BFMatcher(cv::NORM_L2, true).match(next_descr, last_descr, matches);

    for(const cv::DMatch &match : matches) {
      if (match.distance>0.7)
        continue;
      match_ids.push_back(std::make_tuple(last_mask.empty() ? match.trainIdx : last_mask_id[match.trainIdx],
                                          match.queryIdx,
                                          match.distance));
    }
  }

  return match_ids;
}

cv::Mat
draw_matches(const Eigen::MatrixXf &last_keypoints,
             const Eigen::MatrixXf &next_keypoints,
             const std::vector<std::tuple<int, int, float>> &correspondences,
             const cv::Mat &last_img, const cv::Mat &next_img = {})
{
  std::vector<cv::DMatch> matches;
  for(const auto &match : correspondences)
    matches.emplace_back(std::get<1>(match), std::get<0>(match), std::get<2>(match));

  std::vector<cv::KeyPoint> last_kp(last_keypoints.rows());
  for(size_t i=0; i<last_kp.size(); i++)
      last_kp[i].pt = cv::Point(last_keypoints.row(i)[0]*last_img.cols, last_keypoints.row(i)[1]*last_img.rows);

  std::vector<cv::KeyPoint> next_kp(next_keypoints.rows());
  for(size_t i=0; i<next_kp.size(); i++)
      next_kp[i].pt = cv::Point(next_keypoints.row(i)[0]*last_img.cols, next_keypoints.row(i)[1]*last_img.rows);

  cv::Mat img_matches;
  // use given current/next or empty image
  const cv::Mat current = next_img.empty() ? cv::Mat(last_img.size(), CV_8UC1, cv::Scalar(255)): next_img;
  cv::drawMatches(current, next_kp, last_img, last_kp, matches, img_matches,
                  cv::Scalar::all(-1), cv::Scalar::all(-1), std::vector<char>(),
                  cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
  return img_matches;
}

Eigen::Vector3f min_depth(const cv::Point &kp, const cv::Mat &pc, const int size) {
  // create mask
  cv::Mat_<bool> m(pc.size(), false);
  cv::circle(m, kp, size, cv::Scalar(true), -1);

  // extract depth
  std::vector<cv::Mat> xyz;
  cv::split(pc, xyz);
  const cv::Mat &d = xyz[2];
  m &= (d>0);

  Eigen::Vector3f v = Eigen::Vector3f::Zero();

  // find minimum depth location within masked area
  if( cv::countNonZero(m)>0) {
    cv::Point min_loc;
    cv::minMaxLoc(d, nullptr, nullptr, &min_loc, nullptr, m);
    cv::cv2eigen(pc.at<cv::Vec3f>(min_loc), v);
  }
  else {
    // mark invalid as NaN
    v *= std::numeric_limits<float>::quiet_NaN();
  }

  return v;
}

std::tuple<std::vector<cv::Point>, Eigen::VectorXf, Eigen::VectorXf>
inlier(const Eigen::Isometry3f &T_01,
       const DeviceArray2D<float3> &dpc0, const DeviceArray2D<float3> &dpc1,
       const Eigen::MatrixXf& kp0, const Eigen::MatrixXf &kp1, const cv::Mat &mask = {})
{
  // point clouds
  cv::Mat pc0(dpc0.rows(), dpc0.cols(), CV_32FC3);
  cv::Mat pc1(dpc1.rows(), dpc1.cols(), CV_32FC3);
  dpc0.download(pc0.data, dpc0.cols() * 3 * sizeof(float));
  dpc1.download(pc1.data, dpc1.cols() * 3 * sizeof(float));

  // correspondences
  const auto matches = pairwise_matches(kp0, kp1, mask);
  const int N = matches.size();

  std::vector<cv::Point> kp_next_valid;

  // some correspondences will have invalid depth and have to be removed
  int k = 0; // number of matches with valid depth
  Eigen::MatrixX3f p0 = Eigen::MatrixX3f::Zero(N, 3); // last (previous)
  Eigen::MatrixX3f p1 = Eigen::MatrixX3f::Zero(N, 3); // next (current)
  Eigen::VectorXf dist_feat = Eigen::VectorXf::Zero(N); // L2 distance of feature vectors
  for(int m=0; m<N; m++) {
    int ik0, ik1;
    float d;
    std::tie(ik0, ik1, d) = matches[m];
    Eigen::Vector3f v0, v1;
    // next
    const cv::Point p_next(kp1.row(ik1)[0] * pc1.size().width, kp1.row(ik1)[1] * pc1.size().height);
    cv::cv2eigen(pc1.at<cv::Vec3f>(p_next), v1);
    // last
    const cv::Point p_last(kp0.row(ik0)[0] * pc0.size().width, kp0.row(ik0)[1] * pc0.size().height);
    cv::cv2eigen(pc0.at<cv::Vec3f>(p_last), v0);

    if( !(std::isnan(v0.z()) || std::isnan(v1.z())) ) {
      p0.row(k) = v0;
      p1.row(k) = v1;
      dist_feat[k] = d;
      kp_next_valid.push_back(p_next);
      k++;
    }
  }

  // remove tail with empty correspondences
  p0.conservativeResize(k, Eigen::NoChange);
  p1.conservativeResize(k, Eigen::NoChange);
  dist_feat.conservativeResize(k, Eigen::NoChange);

  // distance of matched keypoints
  const Eigen::VectorXf dist_eucl = (p0 - (T_01 * p1.transpose()).transpose()).rowwise().norm();

  return {kp_next_valid, dist_eucl, dist_feat};
}

std::tuple<Eigen::Isometry3f, std::vector<cv::Point>>
ransac(const DeviceArray2D<float3> &dpc0, const DeviceArray2D<float3> &dpc1,
       const Eigen::MatrixXf& kp0, const Eigen::MatrixXf &kp1, const cv::Mat &mask,
       const float inlier_threshold)
{
  Eigen::Isometry3f kpT_nx = Eigen::Isometry3f::Identity();

  // weighted orthogonal procrustes on keypoint correspondences
  cv::Mat pc0(dpc0.rows(), dpc0.cols(), CV_32FC3);
  cv::Mat pc1(dpc1.rows(), dpc1.cols(), CV_32FC3);
  dpc0.download(pc0.data, dpc0.cols() * 3 * sizeof(float));
  dpc1.download(pc1.data, dpc1.cols() * 3 * sizeof(float));

  const auto matches = pairwise_matches(kp0, kp1, mask);

  std::vector<cv::Point> kp_next_valid;
  std::vector<cv::Point> kp_next_inlier;

  // maximum number of correspondences
  // some correspondences will have invalid depth and have to be removed
  const int N = matches.size();
  int k = 0; // number of matches with valid depth
  Eigen::MatrixX3f p0 = Eigen::MatrixX3f::Zero(N, 3); // last (previous)
  Eigen::MatrixX3f p1 = Eigen::MatrixX3f::Zero(N, 3); // next (current)
  Eigen::VectorXf d = Eigen::VectorXf::Zero(N);
  for(int m=0; m<N; m++) {
    int ik;
    // next
    Eigen::Vector3f v1;
    ik = std::get<1>(matches[m]);
    const cv::Point p_next(kp1.row(ik)[0] * pc1.size().width, kp1.row(ik)[1] * pc1.size().height);
    cv::cv2eigen(pc1.at<cv::Vec3f>(p_next), v1);
    // last
    Eigen::Vector3f v0;
    ik = std::get<0>(matches[m]);
    const cv::Point p_last(kp0.row(ik)[0] * pc0.size().width, kp0.row(ik)[1] * pc0.size().height);
    cv::cv2eigen(pc0.at<cv::Vec3f>(p_last), v0);

    if( !(std::isnan(v0.z()) || std::isnan(v1.z())) ) {
      p0.row(k) = v0;
      p1.row(k) = v1;
      d[k] = std::get<2>(matches[m]);
      kp_next_valid.push_back(p_next);
      k++;
    }
  }

  // remove tail with empty correspondences
  p0.conservativeResize(k, Eigen::NoChange);
  p1.conservativeResize(k, Eigen::NoChange);
  d.conservativeResize(k);

  // need at least 3 correspondences
  if (k>=3) {
    // model must have 10% of samples within 'inlier_threshold' error
    RigidRANSAC rrs(600, inlier_threshold, 0.1f);
    kpT_nx = rrs.estimate(p0,p1);
  }

  // find final inliers
  const Eigen::VectorXf dist = (p0 - (kpT_nx * p1.transpose()).transpose()).rowwise().norm();

  // get final inlier keypoints for segmentation seed
  for (int i=0; i<dist.size(); i++) {
    if (dist[i] < inlier_threshold) {
      kp_next_inlier.push_back(kp_next_valid[i]);
    }
  }

  return {kpT_nx, kp_next_inlier};
}

RGBDOdometry::RGBDOdometry(int width, int height, float cx, float cy, float fx, float fy, unsigned char mask, const OdometryConfig &cfg, float distThresh,
                           float angleThresh)
    : lastICPError(0),
      lastICPCount(width * height),
      lastRGBError(0),
      lastRGBCount(width * height),
      lastSO3Error(0),
      lastSO3Count(width * height),
      lastA(Eigen::Matrix<double, 6, 6, Eigen::RowMajor>::Zero()),
      lastb(Eigen::Matrix<double, 6, 1>::Zero()),
      sobelSize(3),
      sobelScale(1.0 / pow(2.0, sobelSize)),
      maxDepthDeltaRGB(0.07),
      maxDepthRGB(6.0),
      distThres_(distThresh),
      angleThres_(angleThresh),
      width(width),
      height(height),
      cx(cx),
      cy(cy),
      fx(fx),
      fy(fy),
      maskID(mask),
      cfg(cfg) {
  sumDataSE3.create(MAX_THREADS);
  outDataSE3.create(1);
  sumResidualRGB.create(MAX_THREADS);

  sumDataSO3.create(MAX_THREADS);
  outDataSO3.create(1);

  for (int i = 0; i < NUM_PYRS; i++) {
    int2 nextDim = {height >> i, width >> i};
    pyrDims.push_back(nextDim);
  }

  for (int i = 0; i < NUM_PYRS; i++) {
    lastDepth[i].create(pyrDims.at(i).x, pyrDims.at(i).y);
    lastImage[i].create(pyrDims.at(i).x, pyrDims.at(i).y);
    lastMask[i].create(pyrDims.at(i).x, pyrDims.at(i).y);

    nextDepth[i].create(pyrDims.at(i).x, pyrDims.at(i).y);
    nextImage[i].create(pyrDims.at(i).x, pyrDims.at(i).y);
    nextMask[i].create(pyrDims.at(i).x, pyrDims.at(i).y);

    lastNextImage[i].create(pyrDims.at(i).x, pyrDims.at(i).y);

    nextdIdx[i].create(pyrDims.at(i).x, pyrDims.at(i).y);
    nextdIdy[i].create(pyrDims.at(i).x, pyrDims.at(i).y);

    pointClouds[i].create(pyrDims.at(i).x, pyrDims.at(i).y);
    nextPointClouds[i].create(pyrDims.at(i).x, pyrDims.at(i).y);

    corresImg[i].create(pyrDims.at(i).x, pyrDims.at(i).y);
  }

  intr.cx = cx;
  intr.cy = cy;
  intr.fx = fx;
  intr.fy = fy;

  iterations.reserve(NUM_PYRS);

  vmaps_g_prev_.resize(NUM_PYRS);
  nmaps_g_prev_.resize(NUM_PYRS);

  vmaps_curr_.resize(NUM_PYRS);
  nmaps_curr_.resize(NUM_PYRS);

  next_keypoints.resize(NUM_PYRS);
  last_keypoints.resize(NUM_PYRS);
  matches.resize(NUM_PYRS);
  next_features.resize(NUM_PYRS);
  last_features.resize(NUM_PYRS);

  for (int i = 0; i < NUM_PYRS; ++i) {
    int pyr_rows = height >> i;
    int pyr_cols = width >> i;

    vmaps_g_prev_[i].create(pyr_rows * 3, pyr_cols);
    nmaps_g_prev_[i].create(pyr_rows * 3, pyr_cols);

    vmaps_curr_[i].create(pyr_rows * 3, pyr_cols);
    nmaps_curr_[i].create(pyr_rows * 3, pyr_cols);
  }

  vmaps_tmp.create(height * 4 * width);
  nmaps_tmp.create(height * 4 * width);

  minimumGradientMagnitudes.reserve(NUM_PYRS);
  minimumGradientMagnitudes[0] = 5;
  minimumGradientMagnitudes[1] = 3;
  minimumGradientMagnitudes[2] = 1;
}

RGBDOdometry::~RGBDOdometry() {}

const int2 &RGBDOdometry::getPyramidDim(const int level) {
  return pyrDims[level];
}

void RGBDOdometry::initICP(const std::vector<DeviceArray2D<float> >& depthPyramid,
                           const std::vector<DeviceArray2D<unsigned char> >& maskPyramid, const float depthCutoff) {
  for (int i = 0; i < RGBDOdometry::NUM_PYRS; ++i) {
    createVMap(intr(i), depthPyramid[i], maskPyramid[i], vmaps_curr_[i], depthCutoff, maskID);
    createNMap(vmaps_curr_[i], nmaps_curr_[i]);
  }

  cudaDeviceSynchronize();
}

void RGBDOdometry::initICP(GPUTexture* predictedVertices, GPUTexture* predictedNormals, const float depthCutoff) {
  cudaArray* textPtr;

  predictedVertices->cudaMap();
  textPtr = predictedVertices->getCudaArray();
  cudaMemcpyFromArray(vmaps_tmp.ptr(), textPtr, 0, 0, vmaps_tmp.sizeBytes(), cudaMemcpyDeviceToDevice);
  predictedVertices->cudaUnmap();

  predictedNormals->cudaMap();
  textPtr = predictedNormals->getCudaArray();
  cudaMemcpyFromArray(nmaps_tmp.ptr(), textPtr, 0, 0, nmaps_tmp.sizeBytes(), cudaMemcpyDeviceToDevice);
  predictedNormals->cudaUnmap();

  copyMaps(vmaps_tmp, nmaps_tmp, vmaps_curr_[0], nmaps_curr_[0]);

  for (int i = 1; i < NUM_PYRS; ++i) {
    resizeVMap(vmaps_curr_[i - 1], vmaps_curr_[i]);
    resizeNMap(nmaps_curr_[i - 1], nmaps_curr_[i]);
  }

  cudaDeviceSynchronize();
}

void RGBDOdometry::initICPModel(GPUTexture* predictedVertices, GPUTexture* predictedNormals, const float depthCutoff,
                                const Eigen::Matrix4f& modelPose) {
  cudaArray* textPtr;

  predictedVertices->cudaMap();
  textPtr = predictedVertices->getCudaArray();
  cudaMemcpyFromArray(vmaps_tmp.ptr(), textPtr, 0, 0, vmaps_tmp.sizeBytes(), cudaMemcpyDeviceToDevice);
  predictedVertices->cudaUnmap();

  predictedNormals->cudaMap();
  textPtr = predictedNormals->getCudaArray();
  cudaMemcpyFromArray(nmaps_tmp.ptr(), textPtr, 0, 0, nmaps_tmp.sizeBytes(), cudaMemcpyDeviceToDevice);
  predictedNormals->cudaUnmap();

  copyMaps(vmaps_tmp, nmaps_tmp, vmaps_g_prev_[0], nmaps_g_prev_[0]);

  for (int i = 1; i < NUM_PYRS; ++i) {
    resizeVMap(vmaps_g_prev_[i - 1], vmaps_g_prev_[i]);
    resizeNMap(nmaps_g_prev_[i - 1], nmaps_g_prev_[i]);
  }

  Eigen::Matrix<float, 3, 3, Eigen::RowMajor> Rcam = modelPose.topLeftCorner(3, 3);
  Eigen::Vector3f tcam = modelPose.topRightCorner(3, 1);

  mat33 device_Rcam = Rcam;
  float3 device_tcam = *reinterpret_cast<float3*>(tcam.data());

  for (int i = 0; i < NUM_PYRS; ++i) {
    tranformMaps(vmaps_g_prev_[i], nmaps_g_prev_[i], device_Rcam, device_tcam, vmaps_g_prev_[i], nmaps_g_prev_[i]);
  }

  cudaDeviceSynchronize();
}

void RGBDOdometry::populateRGBDData(GPUTexture* rgb, DeviceArray2D<float>* destDepths, DeviceArray2D<unsigned char>* destImages,
                                    DeviceArray2D<unsigned char>* destMasks) {
  verticesToDepth(vmaps_tmp, destDepths[0], maxDepthRGB);

  for (int i = 0; i + 1 < NUM_PYRS; i++) pyrDownGaussF(destDepths[i], destDepths[i + 1]);

  rgb->cudaMap();
  cudaArray* textPtr = rgb->getCudaArray();
  imageBGRToIntensity(textPtr, destImages[0]);
  rgb->cudaUnmap();

  for (int i = 0; i + 1 < NUM_PYRS; i++) {
    pyrDownUcharGauss(destImages[i], destImages[i + 1]);
    pyrDownUcharGauss(destMasks[i], destMasks[i + 1]);
  }

  cudaDeviceSynchronize();
}

void RGBDOdometry::initRGBModel(GPUTexture* rgb) {
  // NOTE: This depends on vmaps_tmp containing the corresponding depth from initICPModel
  populateRGBDData(rgb, &lastDepth[0], &lastImage[0], &lastMask[0]);
}

void RGBDOdometry::initRGBDFromPrevious(const Eigen::Matrix4f& pose) {
  // NOTE: This depends on vmaps_tmp containing the corresponding depth from initICPModel
  for (int i = 0; i < RGBDOdometry::NUM_PYRS; ++i) {
    nextImage[i].copyTo(lastImage[i]);
    nextDepth[i].copyTo(lastDepth[i]);
    vmaps_curr_[i].copyTo(vmaps_g_prev_[i]);
    nmaps_curr_[i].copyTo(nmaps_g_prev_[i]);
  }
  copyMaps2(vmaps_curr_[0], vmaps_tmp);
  copyMaps2(nmaps_curr_[0], nmaps_tmp);

  // add new depth image to end of queue
  // for the very first two images, there will be no valid 'nextDepth' yet
  if (iimg>1) {
    Nlast_image.emplace();
    for (int i = 0; i < RGBDOdometry::NUM_PYRS; ++i) {
      download(nextImage[i], Nlast_image.back()[i]);
    }

    NlastDepth.emplace();
    for (int i = 0; i < RGBDOdometry::NUM_PYRS; ++i) {
      nextDepth[i].copyTo(NlastDepth.back()[i]);
    }
  }

  // delete all but N last images
  while (Nlast_image.size()>cfg.history) {
    for (int i = 0; i < RGBDOdometry::NUM_PYRS; ++i) {
      Nlast_image.front()[i].release();
    }
    Nlast_image.pop();
  }

  while (NlastDepth.size()>cfg.history) {
    for (int i = 0; i < RGBDOdometry::NUM_PYRS; ++i) {
      NlastDepth.front()[i].release();
    }
    NlastDepth.pop();
  }

  // transform previous point cloud to initial camera pose at origin
  const mat33 device_Rcam = Eigen::Matrix<float, 3, 3, Eigen::RowMajor>(pose.topLeftCorner(3, 3));
  const float3 device_tcam = *reinterpret_cast<const float3*>(pose.topRightCorner(3, 1).data());
  for (int i = 0; i < NUM_PYRS; ++i) {
    tranformMaps(vmaps_g_prev_[i], nmaps_g_prev_[i], device_Rcam, device_tcam, vmaps_g_prev_[i], nmaps_g_prev_[i]);
  }
}

void RGBDOdometry::initRGB(GPUTexture* rgb) {
  // NOTE: This depends on vmaps_tmp containing the corresponding depth from initICP
  populateRGBDData(rgb, &nextDepth[0], &nextImage[0], &nextMask[0]);
}

void RGBDOdometry::initFirstRGB(GPUTexture* rgb) {
  rgb->cudaMap();
  cudaArray* textPtr = rgb->getCudaArray();
  imageBGRToIntensity(textPtr, lastNextImage[0]);
  rgb->cudaUnmap();

  for (int i = 0; i + 1 < NUM_PYRS; i++) {
    pyrDownUcharGauss(lastNextImage[i], lastNextImage[i + 1]);
  }
}

void RGBDOdometry::getIncrementalTransformation(Eigen::Vector3f& trans, Eigen::Matrix<float, 3, 3, Eigen::RowMajor>& rot,
                                                const bool& rgbOnly, const float& icpWeight, const bool& pyramid, const bool& fastOdom,
                                                const bool& so3, const cudaSurfaceObject_t& icpErrorSurface, const cudaSurfaceObject_t& rgbErrorSurface,
                                                const std::vector<std::unique_ptr<GPUTexture>> &projError,
                                                KpData *const kp_data) {
  bool icp = !rgbOnly && icpWeight > 0;
  bool rgb = rgbOnly || icpWeight < 100;

  const Eigen::Matrix<float, 3, 3, Eigen::RowMajor> Rprev = rot;
  const Eigen::Vector3f tprev = trans;

  Eigen::Matrix<float, 3, 3, Eigen::RowMajor> Rcurr = Rprev;
  Eigen::Vector3f tcurr = tprev;

//  std::cout << "tprev: " << std::endl << tprev.transpose() << " -> |" << tprev.norm() << "|" << std::endl;

  if (rgb) {
    for (int i = 0; i < NUM_PYRS; i++) {
      // sobelGaussian(nextImage[i], nextdIdx[i], nextdIdy[i]);
      computeDerivativeImages(nextImage[i], nextdIdx[i], nextdIdy[i]);
    }
  }

  // get keypoint correspondences
  // compute on CPU (matches), upload to GPU (matchID)
  for (int l = 0; l < NUM_PYRS; l++) {
    cv::Mat mask;
    cv::resize(last_segmentation==maskID, mask, cv::Size(lastDepth[l].cols(), lastDepth[l].rows()));
    if (next_keypoints[l].rows()>0) {
      matches[l] = pairwise_matches(last_keypoints[l], next_keypoints[l], mask);
//      cv::Mat img_matches = draw_matches(last_keypoints[l], next_keypoints[l], matches[l], mask);
//      cv::resize(img_matches, img_matches, cv::Size(2*lastDepth[0].cols()/2, lastDepth[0].rows()/2));
//      cv::imshow("matches "+std::to_string(maskID)+" L"+std::to_string(l), img_matches);
//      cv::waitKey(1);

      // upload indices
      if (!matches[l].empty()) {
        const int M = matches[l].size();
        Eigen::Matrix<int, Eigen::Dynamic, 2, Eigen::RowMajor> matches_norm(M, 2);
        for(int i=0; i<M; i++)
            matches_norm.row(i) = Eigen::Vector2i{std::get<0>(matches[l][i]), std::get<1>(matches[l][i])};
        upload_eigen(matches_norm, matchID[l]);
        Eigen::RowVectorXf scores = Eigen::RowVectorXf::Zero(M);
        for(int i=0; i<M; i++)
          scores[i] = std::get<2>(matches[l][i]);
        upload_eigen(scores, matchScores[l]);
      }
      else {
        // reset old buffers
        matchID[l].create(0,0);
        matchScores[l].create(0,0);
      }
    } // next_keypoints
  } // NUM_PYRS

  Eigen::Matrix<double, 3, 3, Eigen::RowMajor> resultR = Eigen::Matrix<double, 3, 3, Eigen::RowMajor>::Identity();

  if (so3) {
    int pyramidLevel = 2;

    Eigen::Matrix<float, 3, 3, Eigen::RowMajor> R_lr = Eigen::Matrix<float, 3, 3, Eigen::RowMajor>::Identity();

    Eigen::Matrix<double, 3, 3, Eigen::RowMajor> K = Eigen::Matrix<double, 3, 3, Eigen::RowMajor>::Zero();

    K(0, 0) = intr(pyramidLevel).fx;
    K(1, 1) = intr(pyramidLevel).fy;
    K(0, 2) = intr(pyramidLevel).cx;
    K(1, 2) = intr(pyramidLevel).cy;
    K(2, 2) = 1;

    float lastError = std::numeric_limits<float>::max() / 2;
    float lastCount = std::numeric_limits<float>::max() / 2;

    Eigen::Matrix<double, 3, 3, Eigen::RowMajor> lastResultR = Eigen::Matrix<double, 3, 3, Eigen::RowMajor>::Identity();

    for (int i = 0; i < 10; i++) {
      Eigen::Matrix<float, 3, 3, Eigen::RowMajor> jtj;
      Eigen::Matrix<float, 3, 1> jtr;

      Eigen::Matrix<double, 3, 3, Eigen::RowMajor> homography = K * resultR * K.inverse();

      mat33 imageBasis;
      memcpy(&imageBasis.data[0], homography.cast<float>().eval().data(), sizeof(mat33));

      Eigen::Matrix<double, 3, 3, Eigen::RowMajor> K_inv = K.inverse();
      mat33 kinv;
      memcpy(&kinv.data[0], K_inv.cast<float>().eval().data(), sizeof(mat33));

      Eigen::Matrix<double, 3, 3, Eigen::RowMajor> K_R_lr = K * resultR;
      mat33 krlr;
      memcpy(&krlr.data[0], K_R_lr.cast<float>().eval().data(), sizeof(mat33));

      float residual[2];

      TICK("so3Step");
      so3Step(lastNextImage[pyramidLevel], nextImage[pyramidLevel], imageBasis, kinv, krlr, sumDataSO3, outDataSO3, jtj.data(), jtr.data(),
              &residual[0], GPUConfig::getInstance().so3StepThreads, GPUConfig::getInstance().so3StepBlocks);
      TOCK("so3Step");

      lastSO3Error = sqrt(residual[0]) / residual[1];
      lastSO3Count = residual[1];

      // Converged
      if (lastSO3Error < lastError && fabs(lastError - lastSO3Count) < 0.001) {
        break;
      } else if (lastSO3Error > lastError + 0.001) {  // Diverging
        lastSO3Error = lastError;
        lastSO3Count = lastCount;
        resultR = lastResultR;
        break;
      }

      lastError = lastSO3Error;
      lastCount = lastSO3Count;
      lastResultR = resultR;

      Eigen::Vector3f delta = jtj.ldlt().solve(jtr);

      Eigen::Matrix<double, 3, 3, Eigen::RowMajor> rotUpdate = OdometryProvider::rodrigues(delta.cast<double>());

      R_lr = rotUpdate.cast<float>() * R_lr;

      for (int x = 0; x < 3; x++) {
        for (int y = 0; y < 3; y++) {
          resultR(x, y) = R_lr(x, y);
        }
      }
    }
  }

  iterations[0] = fastOdom ? 3 : 10;
  iterations[1] = pyramid ? 5 : 0;
  iterations[2] = pyramid ? 4 : 0;

  const Eigen::Matrix<float, 3, 3, Eigen::RowMajor> Rprev_inv = Rprev.inverse();
  const mat33 device_Rprev_inv = Rprev_inv;
  const float3 device_tprev = *reinterpret_cast<const float3*>(tprev.data());

  Eigen::Matrix<double, 4, 4, Eigen::RowMajor> resultRt = Eigen::Matrix<double, 4, 4, Eigen::RowMajor>::Identity();


  if (so3) {
    for (int x = 0; x < 3; x++) {
      for (int y = 0; y < 3; y++) {
        resultRt(x, y) = resultR(x, y);
      }
    }
  }

  // Per pyramid level
  for (int i = NUM_PYRS - 1; i >= 0; i--) {
    if (rgb) {
      projectToPointCloud(lastDepth[i], pointClouds[i], intr, i);
      projectToPointCloud(nextDepth[i], nextPointClouds[i], intr, i);
    }

//    std::cout << "py" << i << ": " << nextDepth[i].cols() << " x " << nextDepth[i].rows() << std::endl;

    Eigen::Matrix<double, 3, 3, Eigen::RowMajor> K = Eigen::Matrix<double, 3, 3, Eigen::RowMajor>::Zero();

    K(0, 0) = intr(i).fx;
    K(1, 1) = intr(i).fy;
    K(0, 2) = intr(i).cx;
    K(1, 2) = intr(i).cy;
    K(2, 2) = 1;

    lastRGBError = std::numeric_limits<float>::max();

    // do least-squares fitting with correspondences on CPU
    const bool kp_ls = cfg.mode_est == "ls" && !matches[i].empty();
    // do ICP update with correspondences on GPU
    const bool kp_icp = cfg.mode_est == "icp" && matchID[i].rows()>0;

    // transformation from previous to current frame
    Eigen::Isometry3f kpT = Eigen::Isometry3f::Identity();

    // least-squares optimisation of transformation via RANSAC on procrustes model
    if (kp_ls) {
      std::tie(kpT, std::ignore) = ransac(pointClouds[i], nextPointClouds[i],
                                          last_keypoints[i], next_keypoints[i],
                                          last_segmentation==maskID, 0.03f);
    }

    // Optimization iterations
    for (int j = 0; j < iterations[i]; j++) {
      Eigen::Matrix<double, 4, 4, Eigen::RowMajor> Rt = resultRt.inverse();

      Eigen::Matrix<double, 3, 3, Eigen::RowMajor> R = Rt.topLeftCorner(3, 3);

      Eigen::Matrix<double, 3, 3, Eigen::RowMajor> KRK_inv = K * R * K.inverse();
      mat33 krkInv;
      memcpy(&krkInv.data[0], KRK_inv.cast<float>().eval().data(), sizeof(mat33));

      Eigen::Vector3d Kt = Rt.topRightCorner(3, 1);
      Kt = K * Kt;
      float3 kt = {(float)Kt(0), (float)Kt(1), (float)Kt(2)};

      float sigma = 0;
      int rgbSize = 0;

      if (rgb && cfg.mode_est.empty()) {
        TICK("computeRgbResidual");
        computeRgbResidual(pow(minimumGradientMagnitudes[i], 2.0) / pow(sobelScale, 2.0), nextdIdx[i], nextdIdy[i], lastDepth[i],
                           nextDepth[i], lastImage[i], nextImage[i], lastMask[i], nextMask[i], corresImg[i], sumResidualRGB,
                           maxDepthDeltaRGB, kt, krkInv, sigma, rgbSize, GPUConfig::getInstance().rgbResThreads,
                           GPUConfig::getInstance().rgbResBlocks,
                           (i == 0 && j == iterations[i]-1) ? rgbErrorSurface : 0, maskID);
        TOCK("computeRgbResidual");
      }
      else if (kp_icp) {
        TICK("computeKPResidual");
        computeKPResidual(lastDepth[i], nextDepth[i],
                          lastKeypoints[i], nextKeypoints[i],
                          lastFeatureMaps[i], nextFeatureMaps[i],
                          matchID[i], matchScores[i], lastMask[0],
                          corresImg[i], sumResidualRGB,
                          sigma, rgbSize, GPUConfig::getInstance().rgbResThreads,
                          GPUConfig::getInstance().rgbResBlocks,
                          (i == 0 && j == iterations[i]-1) ? rgbErrorSurface : 0, maskID);
        TOCK("computeKPResidual");
      }
      else if (kp_ls) {
        // noop
      }
      else {
        // matches are only available from the second image on
        assert(iimg==0 && "no keypoint matches for ICP transformation estimation");
      }

      float tmpError = sqrt(sigma) / rgbSize;
      float sigmaVal = (tmpError < float(1e-6)) ? 1 : rgbSize;

      if (rgbOnly && tmpError > lastRGBError) {
        break;
      }

      lastRGBError = tmpError;
      lastRGBCount = rgbSize;

      if (rgbOnly) {
        sigmaVal = -1;  // Signals the internal optimisation to weight evenly
      }

      Eigen::Matrix<float, 6, 6, Eigen::RowMajor> A_icp;
      Eigen::Matrix<float, 6, 1> b_icp;

      const mat33 device_Rcurr = Rcurr;
      const float3 device_tcurr = *reinterpret_cast<float3*>(tcurr.data());

      // current frame data
      DeviceArray2D<float>& vmap_curr = vmaps_curr_[i];
      DeviceArray2D<float>& nmap_curr = nmaps_curr_[i];

      // model data
      DeviceArray2D<float>& vmap_g_prev = vmaps_g_prev_[i];
      DeviceArray2D<float>& nmap_g_prev = nmaps_g_prev_[i];

      float residual[2];

      // note: we always need to run the ICP step to access the reprojection error in 'icpErrorSurface'
      if (icp && cfg.mode_est.empty()) {
        TICK("icpStep");
        icpStep(device_Rcurr, device_tcurr, vmap_curr, nmap_curr, device_Rprev_inv, device_tprev, intr(i), vmap_g_prev, nmap_g_prev,
                distThres_, angleThres_, sumDataSE3, outDataSE3, A_icp.data(), b_icp.data(), &residual[0],
                GPUConfig::getInstance().icpStepThreads, GPUConfig::getInstance().icpStepBlocks,
                (i == 0 && j == iterations[i] - 1) ? icpErrorSurface : 0);
        TOCK("icpStep");
      }
      else if (kp_icp) {
        kpcStep(device_Rcurr, device_tcurr, vmap_curr, nmap_curr, device_Rprev_inv, device_tprev, intr(i), vmap_g_prev, nmap_g_prev,
                distThres_, angleThres_, corresImg[i], sumDataSE3, outDataSE3, A_icp.data(), b_icp.data(), &residual[0],
                GPUConfig::getInstance().icpStepThreads, GPUConfig::getInstance().icpStepBlocks,
                (i == 0 && j == iterations[i] - 1) ? icpErrorSurface : 0);
      }
      else if (kp_ls) {
        // noop
      }
      else {
        // matches are only available from the second image on
        assert(iimg==0 && "no keypoint matches for ICP transformation estimation");
      }

//      if (icp) {
//        std::cout << "icp A" << std::endl << A_icp << std::endl;
//        std::cout << "icp b" << std::endl << b_icp << std::endl;
//      }

      lastICPError = sqrt(residual[0]) / residual[1];
      lastICPCount = residual[1];

      Eigen::Matrix<float, 6, 6, Eigen::RowMajor> A_rgbd;
      Eigen::Matrix<float, 6, 1> b_rgbd;
      A_rgbd.setZero();
      b_rgbd.setZero();

      if (rgb && cfg.mode_est.empty()) {
        TICK("rgbStep");
        rgbStep(corresImg[i], sigmaVal, pointClouds[i], intr(i).fx, intr(i).fy, nextdIdx[i], nextdIdy[i], sobelScale, sumDataSE3,
                outDataSE3, A_rgbd.data(), b_rgbd.data(), GPUConfig::getInstance().rgbStepThreads, GPUConfig::getInstance().rgbStepBlocks);
        TOCK("rgbStep");
      }

//      if (rgb) {
//        std::cout << "rgb A" << std::endl << A_rgbd << std::endl;
//        std::cout << "rgb b" << std::endl << b_rgbd << std::endl;
//      }

      Eigen::Matrix<double, 6, 1> result;
      Eigen::Matrix<double, 6, 6, Eigen::RowMajor> dA_rgbd = A_rgbd.cast<double>();
      Eigen::Matrix<double, 6, 6, Eigen::RowMajor> dA_icp = A_icp.cast<double>();
      Eigen::Matrix<double, 6, 1> db_rgbd = b_rgbd.cast<double>();
      Eigen::Matrix<double, 6, 1> db_icp = b_icp.cast<double>();

      if (icp && rgb) {
        double w = icpWeight;
        lastA = dA_rgbd + w * w * dA_icp;
        lastb = db_rgbd + w * db_icp;
        result = lastA.ldlt().solve(lastb);
      } else if (icp) {
        lastA = dA_icp;
        lastb = db_icp;
        result = lastA.ldlt().solve(lastb);
      } else if (rgb) {
        lastA = dA_rgbd;
        lastb = db_rgbd;
        result = lastA.ldlt().solve(lastb);
      } else {
        assert(false && "Control shouldn't reach here");
      }

//      std::cout << "result " << std::endl << result << std::endl;

      Eigen::Isometry3f rgbOdom = Eigen::Isometry3f::Identity();

      if (cfg.mode_est.empty() || kp_icp) {
        OdometryProvider::computeUpdateSE3(resultRt, result, rgbOdom);
        assert(resultRt.cast<float>() == rgbOdom.matrix());
      }
      else if(kp_ls) {
        // apply the least-squares optimised transformation only once for the highest level / largest resolution
        if (i==0 && j==0)
          rgbOdom = kpT;
      }
      else {
        // matches are only available from the second image on
        assert(iimg==0 && "no keypoint matches for transformation update");
      }

//      std::cout << "odom update L" << i << std::endl << rgbOdom.matrix() << std::endl;

//      std::cout << "resultRt: " << std::endl << resultRt.inverse() << std::endl;
//      std::cout << "rgbOdom: " << std::endl << rgbOdom.matrix().inverse() << std::endl;

      Eigen::Isometry3f currentT;
      currentT.setIdentity();
      currentT.rotate(Rprev);
      currentT.translation() = tprev;

      currentT = currentT * rgbOdom.inverse();

//      std::cout << "currentT L" << i << std::endl << currentT.matrix() << std::endl;

      tcurr = currentT.translation();
      Rcurr = currentT.rotation();

//      std::cout << "tcurr: " << std::endl << tcurr.transpose() << " -> |" << tcurr.norm() << "|" << std::endl;
    } // iterations
  } // pyramid levels

  if (rgb && (tcurr - tprev).norm() > 0.3) {
    Rcurr = Rprev;
    tcurr = tprev;
  }

  if (so3) {
    for (int i = 0; i < NUM_PYRS; i++) {
      std::swap(lastNextImage[i], nextImage[i]);
    }
  }

  trans = tcurr;
  rot = Rcurr;

//  std::cout << "trans: " << std::endl << trans.transpose() << " -> |" << trans.norm() << "|" << std::endl;

  if (iimg>1) {
    // current pose in initial reference frame at t=0
    Eigen::Isometry3f T_0x = Eigen::Isometry3f::Identity();
    T_0x.translate(trans).rotate(rot);
  //  std::cout << "T: " << std::endl << T.matrix().inverse() << std::endl;

    Nlast_poses.push(T_0x);
    while (Nlast_poses.size()>cfg.history) {
      Nlast_poses.pop();
    }

    // In frame-to-frame mode, the reprojection error between two consecutive frames is too small
    // to create high enough errors for segmentation. We therefore have to compare to a reference
    // image further away in time.
    if (!NlastDepth.empty()) {
      const int i = 0;
      DeviceArray2D<float3> lastPointCloudsN;
      lastPointCloudsN.create(pyrDims.at(i).x, pyrDims.at(i).y);
      projectToPointCloud(NlastDepth.front()[i], lastPointCloudsN, intr, i);

      Eigen::Isometry3f T_nx;
      if(cfg.segm_source.empty() || cfg.segm_source=="est") {
        // transformation from previous estimation
        T_nx = Nlast_poses.front().inverse() * T_0x;
      }
      else if(cfg.segm_source=="ransac") {
        // transformation from keypoints
        std::tie(T_nx, std::ignore) = ransac(lastPointCloudsN, nextPointClouds[i],
                                             Nlast_keypoints.front()[i], next_keypoints[i],
                                             last_segmentation==maskID, 0.03f);
      }
      else {
        throw std::runtime_error("invalid segmentation mode: "+cfg.segm_mode);
      }

      const cv::Mat_<uint8_t> next_img = download(nextImage[i]);
      const cv::Mat_<uint8_t> last_img = Nlast_image.front()[i];
//      {
//        const auto matches = pairwise_matches(Nlast_keypoints.front()[i], next_keypoints[i]);
//        cv::Mat img_matches = draw_matches(Nlast_keypoints.front()[i], next_keypoints[i], matches, last_img, next_img);
//        cv::imshow("matches "+std::to_string(maskID), img_matches);
//        cv::waitKey(1);
//      }

      std::vector<cv::Point> kp_valid;
      Eigen::VectorXf distance;
      Eigen::VectorXf dist_feat;
      std::tie(kp_valid, distance, dist_feat) = inlier(T_nx, lastPointCloudsN, nextPointClouds[i],
                                                       Nlast_keypoints.front()[i], next_keypoints[i]);

      // reprojection error for motion segmentation, required for frame-to-frame
      const mat44 devT = Eigen::Matrix<float, 4, 4, Eigen::RowMajor>(T_nx.matrix());
      projectionError(devT, nextPointClouds[i], intr(i), lastPointCloudsN, NlastMask.front()[i], maskID, distThres_,
                      GPUConfig::getInstance().icpStepThreads, GPUConfig::getInstance().icpStepBlocks,
                      icpErrorSurface);

//      projectionFeatureDistance(devT, nextPointClouds[i], intr(i), lastPointCloudsN, nextFeatureMaps[i], NlastFeatureMaps.front()[i],
//                                NlastMask.front()[i], maskID, distThres_,
//                                GPUConfig::getInstance().icpStepThreads, GPUConfig::getInstance().icpStepBlocks,
//                                icpErrorSurface);

      kp_data->clear();
      if(cfg.segm_mode=="sparse")
      {
        for (size_t j=0; j<kp_valid.size(); j++) {
          kp_data->emplace_back(kp_valid[j], distance[int(j)]);
        }
        // vis
        cv::Mat img_inlier;
        cv::cvtColor(next_img, img_inlier, cv::COLOR_GRAY2BGR);
        const float err_max_vis = 0.05f;
        for (size_t j=0; j<kp_valid.size(); j++) {
          // interpolate between colours
          const float dist_norm = std::min(distance[j]/err_max_vis, 1.0f);
          const auto col = (1-dist_norm)*cv::viz::Color::blue() + dist_norm*cv::viz::Color::red();
          cv::circle(img_inlier, kp_valid[j], 5, col, -1);
        }
        cv::imshow("inlier eucl "+std::to_string(maskID), img_inlier);
//        cv::imwrite("/tmp/motsegm_m"+std::to_string(maskID)+"_i"+std::to_string(iimg)+".png", img_inlier);
      }

      cv::waitKey(1);
    }
  }

  iimg++;
}

Eigen::MatrixXd RGBDOdometry::getCovariance() { return lastA.cast<double>().lu().inverse(); }

void RGBDOdometry::setNextKeypoints(const std::vector<Eigen::MatrixX2d> &kp_coordinates, const std::vector<Eigen::MatrixXd> &kp_descriptors) {
  for(int i=0; i<NUM_PYRS; i++) {
    // storage order in DeviceArray2D is row-major
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> kp(kp_coordinates[i].rows(), kp_coordinates[i].cols()+kp_descriptors[i].cols());
    kp.leftCols(kp_coordinates[i].cols()) = kp_coordinates[i].cast<float>();
    kp.rightCols(kp_descriptors[i].cols()) = kp_descriptors[i].cast<float>();
    upload_eigen(kp, nextKeypoints[i]);
    next_keypoints[i] = kp;
  }
}

void RGBDOdometry::setLastKeypointsFromPrevious() {
  for(int i=0; i<NUM_PYRS; i++) {
    nextKeypoints[i].copyTo(lastKeypoints[i]);
  }
  last_keypoints = next_keypoints;

  // add new keypoints to end of queue
  if (iimg>1) {
    Nlast_keypoints.emplace();
    for(int i=0; i<NUM_PYRS; i++) {
      Nlast_keypoints.back()[i] = next_keypoints[i];
    }
  }

  // delete all but N last keypoints
  while (Nlast_keypoints.size()>cfg.history) {
    Nlast_keypoints.pop();
  }
}

void RGBDOdometry::setNextFeatureMap(const std::vector<cv::Mat> &feat) {
  for(int i=0; i<NUM_PYRS; i++) {
    assert(sizeof(float)*feat[i].channels()*feat[i].cols==feat[i].step);
    // upload feature map in native resolution of the inference
    // has to be upscaled on demand
    uploadND(feat[i], nextFeatureMaps[i]);

//    // scale to pyramid resolution
//    // upscale: linear interpolation of feature map
//    // downscale: nearest-neighbour
//    const cv::Size size_source = feat[i].size();
//    const cv::Size size_target(pyrDims[i].y, pyrDims[i].x);
//    const bool up = size_target.width>size_source.width || size_target.height>size_source.height;
//    const cv::InterpolationFlags scale_mode = up ? cv::INTER_LINEAR : cv::INTER_NEAREST;

//    cv::Mat feat_pyr;
//    cv::resize(feat[i], feat_pyr, size_target, 0, 0, scale_mode);
//    nextFeatureMaps[i].upload(feat_pyr.data, feat_pyr.step, feat_pyr.rows, feat_pyr.cols*feat_pyr.channels());
  }
  next_features = feat;
}

void RGBDOdometry::setLastFeatureMapFromPrevious() {
  for(int i=0; i<NUM_PYRS; i++) {
    nextFeatureMaps[i].copyTo(lastFeatureMaps[i]);
  }
  last_features = next_features;

  // TODO: do not store all last N feature maps for now to prevent "out of memory" issues
//  if (iimg>1) {
//    NlastFeatureMaps.emplace();
//    for (int i = 0; i < RGBDOdometry::NUM_PYRS; ++i) {
//      lastFeatureMaps[i].copyTo(NlastFeatureMaps.back()[i]);
//    }
//  }
}

void RGBDOdometry::setLastSegmentation(const cv::Mat &segm) {
    assert(sizeof(unsigned char)*segm.cols==segm.step);
    lastMask[0].upload(segm.data, segm.step, segm.rows, segm.cols);
    last_segmentation = segm;

    if (iimg>1) {
      NlastMask.emplace();
      for (int i = 0; i < RGBDOdometry::NUM_PYRS; ++i) {
        lastMask[i].copyTo(NlastMask.back()[i]);
      }
    }

    while (NlastMask.size()>cfg.history) {
      NlastMask.pop();
    }
}
