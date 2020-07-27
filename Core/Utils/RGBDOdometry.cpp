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

void RGBDOdometry::initRGBDFromPrevious() {
  // NOTE: This depends on vmaps_tmp containing the corresponding depth from initICPModel
  for (int i = 0; i < RGBDOdometry::NUM_PYRS; ++i) {
    nextImage[i].copyTo(lastImage[i]);
    nextDepth[i].copyTo(lastDepth[i]);
    vmaps_curr_[i].copyTo(vmaps_g_prev_[i]);
    nmaps_curr_[i].copyTo(nmaps_g_prev_[i]);
  }
  copyMaps2(vmaps_curr_[0], vmaps_tmp);
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
  last_keypoints_mask = last_keypoints_mask.topRows(kp_matches);

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
                                                const bool& so3, const cudaSurfaceObject_t& icpErrorSurface,
                                                const cudaSurfaceObject_t& rgbErrorSurface) {
  bool icp = !rgbOnly && icpWeight > 0;
  bool rgb = rgbOnly || icpWeight < 100;

  Eigen::Matrix<float, 3, 3, Eigen::RowMajor> Rprev = rot;
  Eigen::Vector3f tprev = trans;

  Eigen::Matrix<float, 3, 3, Eigen::RowMajor> Rcurr = Rprev;
  Eigen::Vector3f tcurr = tprev;

  if (rgb) {
    for (int i = 0; i < NUM_PYRS; i++) {
      // sobelGaussian(nextImage[i], nextdIdx[i], nextdIdy[i]);
      computeDerivativeImages(nextImage[i], nextdIdx[i], nextdIdy[i]);
    }
  }

  // keypoint correspondences
  if (next_keypoints.rows()>0) {
    const auto matches = pairwise_matches(last_keypoints, next_keypoints, last_segmentation==maskID);
    const cv::Mat img_matches = draw_matches(last_keypoints, next_keypoints, matches, last_segmentation==maskID);
    cv::imshow("matches "+std::to_string(maskID), img_matches);
    cv::waitKey(1);

    // upload indices
    if (!matches.empty()) {
      Eigen::Matrix<int, Eigen::Dynamic, 2, Eigen::RowMajor> matches_norm(matches.size(), 2);
      for(int i=0; i<int(matches.size()); i++)
          matches_norm.row(i) = Eigen::Vector2i{std::get<0>(matches[i]), std::get<1>(matches[i])};
      upload_eigen(matches_norm, matchID);
      Eigen::RowVectorXf scores(matches.size());
      for(size_t i=0; i<matches.size(); i++)
        scores[int(i)] = std::get<2>(matches[i]);
      upload_eigen(scores, matchScores);
    }
    else {
      // reset old buffers
      matchID.create(0,0);
      matchScores.create(0,0);
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

  Eigen::Matrix<float, 3, 3, Eigen::RowMajor> Rprev_inv = Rprev.inverse();
  mat33 device_Rprev_inv = Rprev_inv;
  float3 device_tprev = *reinterpret_cast<float3*>(tprev.data());

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

    Eigen::Matrix<double, 3, 3, Eigen::RowMajor> K = Eigen::Matrix<double, 3, 3, Eigen::RowMajor>::Zero();

    K(0, 0) = intr(i).fx;
    K(1, 1) = intr(i).fy;
    K(0, 2) = intr(i).cx;
    K(1, 2) = intr(i).cy;
    K(2, 2) = 1;

    lastRGBError = std::numeric_limits<float>::max();

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

      if (rgb && nextKeypoints.rows()==0) {
        TICK("computeRgbResidual");
        computeRgbResidual(pow(minimumGradientMagnitudes[i], 2.0) / pow(sobelScale, 2.0), nextdIdx[i], nextdIdy[i], lastDepth[i],
                           nextDepth[i], lastImage[i], nextImage[i], lastMask[i], nextMask[i], corresImg[i], sumResidualRGB,
                           maxDepthDeltaRGB, kt, krkInv, sigma, rgbSize, GPUConfig::getInstance().rgbResThreads,
                           GPUConfig::getInstance().rgbResBlocks,
                           (i == 0 && j == iterations[i]-1) ? rgbErrorSurface : 0, maskID);
        TOCK("computeRgbResidual");
      }
      else if (nextKeypoints.rows()>0) {
        TICK("computeKPResidual");
//        std::cout << "next kp: " << nextKeypoints.rows() << std::endl;
//        std::cout << "last kp: " << lastKeypoints.rows() << std::endl;
//        std::cout << "matches: " << matchID.rows() << std::endl;
        // TODO: only apply keypoint search on single pyramid level and iteration
        computeKPResidual(lastDepth[i], nextDepth[i],
                          lastKeypoints, nextKeypoints,
                          lastFeatureMaps, nextFeatureMaps,
                          matchID, matchScores, lastMask[0],
                          corresImg[i], sumResidualRGB,
                          sigma, rgbSize, GPUConfig::getInstance().rgbResThreads,
                          GPUConfig::getInstance().rgbResBlocks,
                          (i == 0 && j == iterations[i]-1) ? rgbErrorSurface : 0, maskID);
        TOCK("computeKPResidual");
//        std::cout << "kp " << sigma << " / " << rgbSize << std::endl;
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

      mat33 device_Rcurr = Rcurr;
      float3 device_tcurr = *reinterpret_cast<float3*>(tcurr.data());

      // current frame data
      DeviceArray2D<float>& vmap_curr = vmaps_curr_[i];
      DeviceArray2D<float>& nmap_curr = nmaps_curr_[i];

      // model data
      DeviceArray2D<float>& vmap_g_prev = vmaps_g_prev_[i];
      DeviceArray2D<float>& nmap_g_prev = nmaps_g_prev_[i];

      float residual[2];

      // note: we always need to run the ICP step to access the reprojection error in 'icpErrorSurface'
      TICK("icpStep");
      icpStep(device_Rcurr, device_tcurr, vmap_curr, nmap_curr, device_Rprev_inv, device_tprev, intr(i), vmap_g_prev, nmap_g_prev,
              distThres_, angleThres_, sumDataSE3, outDataSE3, A_icp.data(), b_icp.data(), &residual[0],
              GPUConfig::getInstance().icpStepThreads, GPUConfig::getInstance().icpStepBlocks,
              (i == 0 && j == iterations[i] - 1) ? icpErrorSurface : 0);
      TOCK("icpStep");

//      std::cout << "icp A" << std::endl << A_icp << std::endl;
//      std::cout << "icp b" << std::endl << b_icp << std::endl;

      lastICPError = sqrt(residual[0]) / residual[1];
      lastICPCount = residual[1];

      Eigen::Matrix<float, 6, 6, Eigen::RowMajor> A_rgbd;
      Eigen::Matrix<float, 6, 1> b_rgbd;

      if (rgb && nextKeypoints.rows()==0) {
        TICK("rgbStep");
        rgbStep(corresImg[i], sigmaVal, pointClouds[i], intr(i).fx, intr(i).fy, nextdIdx[i], nextdIdy[i], sobelScale, sumDataSE3,
                outDataSE3, A_rgbd.data(), b_rgbd.data(), GPUConfig::getInstance().rgbStepThreads, GPUConfig::getInstance().rgbStepBlocks);
        TOCK("rgbStep");
      }
      else if (nextKeypoints.rows()>0) {
        // weighted orthogonal procrustes on keypoint correspondences
        cv::Mat pc0(nextPointClouds[i].rows(), nextPointClouds[i].cols(), CV_32FC3);
        cv::Mat pc1(pointClouds[i].rows(), pointClouds[i].cols(), CV_32FC3);
        pointClouds[i].download(pc1.data, pointClouds[i].cols() * 3 * sizeof(float));
        nextPointClouds[i].download(pc0.data, nextPointClouds[i].cols() * 3 * sizeof(float));
        // dbg
        std::array<cv::Mat,3> xyz0;
        cv::split(pc0, xyz0);
        cv::imshow("z0 - "+std::to_string(i), xyz0[2]);
        std::array<cv::Mat,3> xyz1;
        cv::split(pc1, xyz1);
        cv::imshow("z1 - "+std::to_string(i), xyz1[2]);
        cv::waitKey(1);
      }

//      std::cout << "rgb A" << std::endl << A_rgbd << std::endl;
//      std::cout << "rgb b" << std::endl << b_rgbd << std::endl;

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

      Eigen::Isometry3f rgbOdom;

      OdometryProvider::computeUpdateSE3(resultRt, result, rgbOdom);

      Eigen::Isometry3f currentT;
      currentT.setIdentity();
      currentT.rotate(Rprev);
      currentT.translation() = tprev;

      currentT = currentT * rgbOdom.inverse();

      tcurr = currentT.translation();
      Rcurr = currentT.rotation();
    }
  }

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
}

Eigen::MatrixXd RGBDOdometry::getCovariance() { return lastA.cast<double>().lu().inverse(); }

void RGBDOdometry::setNextKeypoints(const Eigen::MatrixX2f &kp_coordinates, const Eigen::MatrixXf &kp_descriptors) {
  // storage order in DeviceArray2D is row-major
  Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> kp(kp_coordinates.rows(), kp_coordinates.cols()+kp_descriptors.cols());
  kp.leftCols(kp_coordinates.cols()) = kp_coordinates;
  kp.rightCols(kp_descriptors.cols()) = kp_descriptors;
  upload_eigen(kp, nextKeypoints);
  next_keypoints = kp;
}

void RGBDOdometry::setLastKeypointsFromPrevious() {
    lastKeypoints = nextKeypoints;
    last_keypoints = next_keypoints;
}

void RGBDOdometry::setNextFeatureMap(const cv::Mat &feat) {
    assert(sizeof(float)*feat.channels()*feat.cols==feat.step);
    nextFeatureMaps.upload(feat.data, feat.step, feat.rows, feat.cols);
    next_features = feat;
}

void RGBDOdometry::setLastFeatureMapFromPrevious() {
    lastFeatureMaps = nextFeatureMaps;
    last_features = next_features;
}

void RGBDOdometry::setLastSegmentation(const cv::Mat &segm) {
    assert(sizeof(unsigned char)*segm.cols==segm.step);
    lastMask[0].upload(segm.data, segm.step, segm.rows, segm.cols);
    last_segmentation = segm;
}
