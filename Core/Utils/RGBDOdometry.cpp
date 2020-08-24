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

// upload row-major Eigen matrix to device array
template <typename T, int R, int C>
void upload_eigen(const Eigen::Matrix<T, R, C, Eigen::RowMajor> &matrix,
                  DeviceArray2D<T> &array)
{
  array.upload(matrix.data(), matrix.cols() * sizeof(T), matrix.rows(), matrix.cols());
}

RGBDOdometry::RGBDOdometry(int width, int height, float cx, float cy, float fx, float fy, unsigned char mask, float distThresh,
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
      maskID(mask) {
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

std::vector<std::tuple<int, int, float>>
pairwise_matches(const Eigen::MatrixXf &last_keypoints,
                 const Eigen::MatrixXf &next_keypoints,
                 const cv::Mat_<bool> &last_mask)
{
  // remove 'last' keypoint coordinates outside of mask
  Eigen::MatrixXf last_keypoints_mask = Eigen::MatrixXf::Constant(last_keypoints.rows(), last_keypoints.cols(), std::numeric_limits<float>::signaling_NaN());
  int kp_matches = 0;
  std::vector<int> last_mask_id;
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

  // store correspondences (last_id, next_id)
  std::vector<std::tuple<int, int, float>> match_ids;

  if (last_keypoints_mask.rows()>0) {
    cv::Mat last_descr;
    cv::eigen2cv(Eigen::MatrixXf(last_keypoints_mask.rightCols(last_keypoints_mask.cols()-2)), last_descr);
    cv::Mat next_descr;
    cv::eigen2cv(Eigen::MatrixXf(next_keypoints.rightCols(next_keypoints.cols()-2)), next_descr);

    std::vector<cv::DMatch> matches;
    cv::BFMatcher(cv::NORM_L2, true).match(next_descr, last_descr, matches);

    for(const cv::DMatch &match : matches)
      match_ids.push_back(std::make_tuple(last_mask_id[match.trainIdx], match.queryIdx, match.distance));
  }

  return match_ids;
}

cv::Mat
draw_matches(const Eigen::MatrixXf &last_keypoints,
             const Eigen::MatrixXf &next_keypoints,
             const std::vector<std::tuple<int, int, float>> &correspondences,
             const cv::Mat &mask)
{
  std::vector<cv::DMatch> matches;
  for(const auto &match : correspondences)
    matches.emplace_back(std::get<1>(match), std::get<0>(match), std::get<2>(match));

  std::vector<cv::KeyPoint> last_kp(last_keypoints.rows());
  for(size_t i=0; i<last_kp.size(); i++)
      last_kp[i].pt = cv::Point(last_keypoints.row(i)[0]*mask.cols, last_keypoints.row(i)[1]*mask.rows);

  std::vector<cv::KeyPoint> next_kp(next_keypoints.rows());
  for(size_t i=0; i<next_kp.size(); i++)
      next_kp[i].pt = cv::Point(next_keypoints.row(i)[0]*mask.cols, next_keypoints.row(i)[1]*mask.rows);

  cv::Mat img_matches;
  const cv::Mat empty(mask.size(), CV_8UC1, cv::Scalar(255)); // empty image
  cv::drawMatches(empty, next_kp, mask, last_kp, matches, img_matches);
  return img_matches;
}

void RGBDOdometry::getIncrementalTransformation(Eigen::Vector3f& trans, Eigen::Matrix<float, 3, 3, Eigen::RowMajor>& rot,
                                                const bool& rgbOnly, const float& icpWeight, const bool& pyramid, const bool& fastOdom,
                                                const bool& so3, const cudaSurfaceObject_t& icpErrorSurface, const cudaSurfaceObject_t& rgbErrorSurface,
                                                const std::vector<std::unique_ptr<GPUTexture>> &projError) {
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

  // dbg: show current and previous image
//  cv::imshow("next depth", download(nextDepth[0])/5);
//  cv::imshow("last depth", download(lastDepth[0])/5);
//  cv::imshow("next colour", download(nextImage[0]));
//  cv::imshow("last colour", download(lastImage[0]));
//  cv::waitKey(1);

  // keypoint correspondences
  for (int l = 0; l < NUM_PYRS; l++) {
    cv::Mat mask;
    cv::resize(last_segmentation==maskID, mask, cv::Size(lastDepth[l].cols(), lastDepth[l].rows()));
    if (next_keypoints[l].rows()>0) {
      matches[l] = pairwise_matches(last_keypoints[l], next_keypoints[l], mask);
      cv::Mat img_matches = draw_matches(last_keypoints[l], next_keypoints[l], matches[l], mask);
      cv::resize(img_matches, img_matches, cv::Size(2*lastDepth[0].cols()/2, lastDepth[0].rows()/2));
      cv::imshow("matches "+std::to_string(maskID)+" L"+std::to_string(l), img_matches);
      cv::waitKey(1);

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
    }
  }

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

    // transformation from previous to current frame
    Eigen::Isometry3f kpT = Eigen::Isometry3f::Identity();

    if (!matches[i].empty()) {
      // weighted orthogonal procrustes on keypoint correspondences
      cv::Mat pc0(pointClouds[i].rows(), pointClouds[i].cols(), CV_32FC3);
      cv::Mat pc1(nextPointClouds[i].rows(), nextPointClouds[i].cols(), CV_32FC3);
      pointClouds[i].download(pc0.data, pointClouds[i].cols() * 3 * sizeof(float));
      nextPointClouds[i].download(pc1.data, nextPointClouds[i].cols() * 3 * sizeof(float));
//        // dbg
//        const float max_depth_vis = 3;
//        std::array<cv::Mat,3> xyz0;
//        cv::split(pc0, xyz0);
//        cv::imshow("z0 - "+std::to_string(i), xyz0[2]/max_depth_vis);
//        std::array<cv::Mat,3> xyz1;
//        cv::split(pc1, xyz1);
//        cv::imshow("z1 - "+std::to_string(i), xyz1[2]/max_depth_vis);
//        cv::waitKey(1);

      // maximum number of correspondences
      // some correspondences will have invalid depth and have to be removed
      const int N = matches[i].size();
      int k = 0; // number of matches with valid depth
      Eigen::MatrixX3f p0 = Eigen::MatrixX3f::Zero(N, 3); // last (previous)
      Eigen::MatrixX3f p1 = Eigen::MatrixX3f::Zero(N, 3); // next (current)
      Eigen::VectorXf d = Eigen::VectorXf::Zero(N);
      for(int m=0; m<N; m++){
        int ik;
        int x,y;
        // next
        Eigen::Vector3f v1;
        ik = std::get<1>(matches[i][m]);
        x = next_keypoints[i].row(ik)[0] * pc1.size().width;
        y = next_keypoints[i].row(ik)[1] * pc1.size().height;
        cv::cv2eigen(pc1.at<cv::Vec3f>(y,x), v1);
        // last
        Eigen::Vector3f v0;
        ik = std::get<0>(matches[i][m]);
        x = last_keypoints[i].row(ik)[0] * pc0.size().width;
        y = last_keypoints[i].row(ik)[1] * pc0.size().height;
        cv::cv2eigen(pc0.at<cv::Vec3f>(y,x), v0);

        if( !(std::isnan(v0.z()) || std::isnan(v1.z()))) {
          p0.row(k) = v0;
          p1.row(k) = v1;
          d[k] = std::get<2>(matches[i][m]);
          k++;
        }
      }

      if (k>=3) {
        // remove tail with empty correspondences
        p0.conservativeResize(k, Eigen::NoChange);
        p1.conservativeResize(k, Eigen::NoChange);
        d.conservativeResize(k);

        // model must have 10% of samples within 3cm error
        RigidRANSAC rrs(600, 0.03f, 0.1f);
        kpT = rrs.estimate(p0,p1);

//          // convert L2 distances to weights with sum(w) = trace(W) = k
//          const auto we = (-d).array().exp();
//          const Eigen::VectorXf w = (we / we.sum()) * we.size();
//          const auto W = w.asDiagonal();

//          const Eigen::RowVector3f p0m = p0.colwise().mean();
//          const Eigen::RowVector3f p1m = p1.colwise().mean();

//          // weighted least-squares optimisation of rigid transformation
//          const auto A = (p1.rowwise()-p1m).transpose() * W * (p0.rowwise()-p0m);
//          Eigen::JacobiSVD<Eigen::MatrixXf> svd(A, Eigen::ComputeThinU | Eigen::ComputeThinV);

//          auto U = svd.matrixU();
//          auto V = svd.matrixV();
//          // guarantee that determinant of R is 1
//          auto S = Eigen::Vector3f(1, 1, U.determinant() * V.determinant()).asDiagonal();

//          const Eigen::Matrix3f R = U * S * V.transpose();
//          const Eigen::Vector3f t = p1m - (R * p0m.transpose()).transpose();
//          kpT.translate(t).rotate(R);

        const Eigen::VectorXf dist = (p0 - (kpT * p1.transpose()).transpose()).rowwise().norm();
        std::cout << "kp reproj err L" << i << " (" << dist.size() << "): " << dist.mean() << std::endl;
      } // at least k>=3 correspondences

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

      if (rgb && matchID[i].rows()==0) {
        TICK("computeRgbResidual");
        computeRgbResidual(pow(minimumGradientMagnitudes[i], 2.0) / pow(sobelScale, 2.0), nextdIdx[i], nextdIdy[i], lastDepth[i],
                           nextDepth[i], lastImage[i], nextImage[i], lastMask[i], nextMask[i], corresImg[i], sumResidualRGB,
                           maxDepthDeltaRGB, kt, krkInv, sigma, rgbSize, GPUConfig::getInstance().rgbResThreads,
                           GPUConfig::getInstance().rgbResBlocks,
                           (i == 0 && j == iterations[i]-1) ? rgbErrorSurface : 0, maskID);
        TOCK("computeRgbResidual");
      }
      else if (matchID[i].rows()>0) {
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
      if (icp && matchID[i].rows()==0) {
        TICK("icpStep");
        icpStep(device_Rcurr, device_tcurr, vmap_curr, nmap_curr, device_Rprev_inv, device_tprev, intr(i), vmap_g_prev, nmap_g_prev,
                distThres_, angleThres_, sumDataSE3, outDataSE3, A_icp.data(), b_icp.data(), &residual[0],
                GPUConfig::getInstance().icpStepThreads, GPUConfig::getInstance().icpStepBlocks,
                (i == 0 && j == iterations[i] - 1) ? icpErrorSurface : 0);
        TOCK("icpStep");
      }
      else if (matchID[i].rows()>0) {
        kpcStep(device_Rcurr, device_tcurr, vmap_curr, nmap_curr, device_Rprev_inv, device_tprev, intr(i), vmap_g_prev, nmap_g_prev,
                distThres_, angleThres_, corresImg[i], sumDataSE3, outDataSE3, A_icp.data(), b_icp.data(), &residual[0],
                GPUConfig::getInstance().icpStepThreads, GPUConfig::getInstance().icpStepBlocks,
                (i == 0 && j == iterations[i] - 1) ? icpErrorSurface : 0);
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

      if (rgb && matches[i].empty()) {
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

      if (matches[i].empty()) {
        OdometryProvider::computeUpdateSE3(resultRt, result, rgbOdom);
        assert(resultRt.cast<float>() == rgbOdom.matrix());
      }
      else {
        // apply the least-squares optimised transformation only once for the highest level / largest resolution
        if (i==0 && j==0)
          rgbOdom = kpT;
      }

//      std::cout << "odom update L" << i << std::endl << rgbOdom.matrix() << std::endl;

//      std::cout << "resultRt: " << std::endl << resultRt.inverse() << std::endl;
//      std::cout << "rgbOdom: " << std::endl << rgbOdom.matrix().inverse() << std::endl;

      // reprojection error for motion segmentation, required for frame-to-frame
//      const cudaSurfaceObject_t &rpesrf = (j == iterations[i] - 1) ? projError[i]->getCudaSurface() : 0;
      const cudaSurfaceObject_t &rpesrf = (i == 0 && j == iterations[i] - 1) ? icpErrorSurface : 0;
      if (rpesrf) {
        const mat44 devT = Eigen::Matrix<float, 4, 4, Eigen::RowMajor>(rgbOdom.matrix());
        projectionError(devT, nextPointClouds[i], intr(i), pointClouds[i], distThres_,
                        GPUConfig::getInstance().icpStepThreads, GPUConfig::getInstance().icpStepBlocks,
                        rpesrf);
      }

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

//  Eigen::Isometry3f T = Eigen::Isometry3f::Identity();
//  T.translate(trans).rotate(rot);
//  std::cout << "T: " << std::endl << T.matrix().inverse() << std::endl;
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
}

void RGBDOdometry::setNextFeatureMap(const std::vector<cv::Mat> &feat) {
  for(int i=0; i<NUM_PYRS; i++) {
    assert(sizeof(float)*feat[i].channels()*feat[i].cols==feat[i].step);
    nextFeatureMaps[i].upload(feat[i].data, feat[i].step, feat[i].rows, feat[i].cols);
  }
  next_features = feat;
}

void RGBDOdometry::setLastFeatureMapFromPrevious() {
  for(int i=0; i<NUM_PYRS; i++) {
    nextFeatureMaps[i].copyTo(lastFeatureMaps[i]);
  }
  last_features = next_features;
}

void RGBDOdometry::setLastSegmentation(const cv::Mat &segm) {
    assert(sizeof(unsigned char)*segm.cols==segm.step);
    lastMask[0].upload(segm.data, segm.step, segm.rows, segm.cols);
    last_segmentation = segm;
}
