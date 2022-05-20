/*
 * This file is part of https://github.com/martinruenz/co-fusion
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>
 */

#include <cmath>
#include <list>
#include <tuple>

#include "Segmentation.h"
#include "ConnectedLabels.hpp"
#include "densecrf.h"
#include "../Model/Model.h"

#include <opencv2/video/tracking.hpp>

#ifdef SHOW_DEBUG_VISUALISATION
#include <iomanip>
#include "../Utils/Gnuplot.h"
#endif

#ifndef TICK
#define TICK(name)
#define TOCK(name)
#endif

#ifdef PRINT_CRF_TIMES
#include <chrono>
typedef std::chrono::high_resolution_clock Clock;
typedef std::chrono::system_clock::time_point TimePoint;
#endif

// show optical flow and dense reprojection probabilities
#define DBG_VIS_PROBS 0
// export images of the local keypoint reprojection and errors
#define DBG_EXP_ERRORS 0
// show keypoint reprojection and segmentation
#define DBG_VIS_SEGM_DEF 0

// DBG_EXP_ERRORS needs DBG_VIS_SEGM
#if DBG_EXP_ERRORS
#define DBG_VIS_SEGM 1
#else
#define DBG_VIS_SEGM DBG_VIS_SEGM_DEF
#endif

#if DBG_VIS_SEGM
#include <opencv2/viz/types.hpp>
#endif

SegmentationResult::ModelData::ModelData(unsigned t_id) : id(t_id) {}

SegmentationResult::ModelData::ModelData(unsigned t_id, ModelListIterator const& t_modelListIterator, cv::Mat const& t_lowICP,
                                         cv::Mat const& t_lowConf, unsigned t_superPixelCount, float t_avgConfidence)
    : id(t_id),
      modelListIterator(t_modelListIterator),
      lowICP(t_lowICP),
      lowConf(t_lowConf),
      superPixelCount(t_superPixelCount),
      avgConfidence(t_avgConfidence) {}

void Segmentation::init(int width, int height, METHOD method, const SegmentationConfiguration &cfg) {
  // TODO: Make customisable.
  slic = Slic(width, height, cfg.sp_size, gSLICr::RGB);
  this->method = method;
  this->cfg = cfg;
}

void Segmentation::setMode(const std::string &mode) {
  lock_cfg.lock();
  cfg.mode = mode;
  lock_cfg.unlock();
}

SegmentationResult Segmentation::performSegmentation(std::list<std::shared_ptr<Model>>& models, const FrameData& frame,
                                                     unsigned char nextModelID, bool allowNew, const tracker::Tracks &tracks) {
  if (frame.mask.total()) {
    assert(frame.mask.type() == CV_8UC1);
    assert(frame.mask.isContinuous());
    static std::vector<unsigned char> mapping(256, 0);  // FIXME

    SegmentationResult result;
    result.hasNewLabel = false;
    result.fullSegmentation = cv::Mat::zeros(frame.mask.rows, frame.mask.cols, CV_8UC1);

    unsigned char modelIdToIndex[256];
    unsigned char mIndex = 0;
    for (auto m : models) modelIdToIndex[m->getID()] = mIndex++;
    modelIdToIndex[nextModelID] = mIndex;

    std::vector<unsigned> outIdsArray(256, 0);  // Should be faster than using a set

    // Replace unseen with zeroes (except new label)
    for (unsigned i = 0; i < frame.mask.total(); i++) {
      unsigned char& vIn = frame.mask.data[i];
      if (vIn) {
        unsigned char& vOut = result.fullSegmentation.data[i];
        if (mapping[vIn] != 0) {
          vOut = mapping[vIn];
          outIdsArray[vOut]++;
          ;
        } else if (allowNew && !result.hasNewLabel) {
          vOut = nextModelID;
          mapping[vIn] = nextModelID;
          result.hasNewLabel = true;
          outIdsArray[vOut]++;
        }
      } else {
        outIdsArray[0]++;
      }
    }

    for (ModelListIterator m = models.begin(); m != models.end(); m++)
      result.modelData.push_back({(*m)->getID(), m, cv::Mat(), cv::Mat(), outIdsArray[(*m)->getID()] / (16 * 16), 0.4});
    if (result.hasNewLabel)
      result.modelData.push_back({nextModelID, ModelListIterator(), cv::Mat(), cv::Mat(),
                                  unsigned(std::max((float)(outIdsArray[nextModelID] / (16 * 16)), 1.0f)), 0.4});

    std::vector<unsigned> cnts(result.modelData.size(), 0);
    for (unsigned i = 0; i < frame.mask.total(); i++) {
      const size_t index = modelIdToIndex[result.fullSegmentation.data[i]];
      result.modelData[index].depthMean += ((const float*)frame.depth.data)[i];
      cnts[index]++;
    }
    for (size_t index = 0; index < result.modelData.size(); ++index) result.modelData[index].depthMean /= cnts[index] ? cnts[index] : 1;

    for (unsigned i = 0; i < frame.mask.total(); i++) {
      const size_t index = modelIdToIndex[result.fullSegmentation.data[i]];
      result.modelData[index].depthStd += std::abs(result.modelData[index].depthMean - ((const float*)frame.depth.data)[i]);
    }
    for (size_t iindex = 0; iindex < result.modelData.size(); ++iindex)
      result.modelData[iindex].depthStd /= cnts[iindex] ? cnts[iindex] : 1;

    return result;
  }

  lock_cfg.lock();
  const std::string mode = cfg.mode;
  lock_cfg.unlock();

  if (mode == "flow_crf") {
    return performSegmentationFlowCRF(models, frame, nextModelID, allowNew, tracks);
  }
  return performSegmentationCRF(models, frame, nextModelID, allowNew);
}

SegmentationResult Segmentation::performSegmentationCRF(std::list<std::shared_ptr<Model>>& models, const FrameData& frame,
                                                        unsigned char nextModelID, bool allowNew) {
  assert(models.size() < 256);

  static unsigned CFRAME = 0;
  CFRAME++;

  TICK("SLIC+SCALING");

  SegmentationResult result;

  const unsigned numExistingModels = models.size();
  const unsigned numLabels = allowNew ? numExistingModels + 1 : numExistingModels;

  slic.setInputImage(frame.rgb);
  slic.processFrame();

  // TODO Speedup! See notes!!
  result.lowRGB = slic.downsample();
  result.lowDepth = slic.downsampleThresholded<float>(frame.depth, 0.02);

  assert(result.lowDepth.total() == result.lowRGB.total());

  const unsigned lowWidth = result.lowRGB.cols;
  const unsigned lowHeight = result.lowRGB.rows;
  const unsigned lowTotal = result.lowRGB.total();
  const unsigned fullWidth = frame.rgb.cols;
  const unsigned fullHeight = frame.rgb.rows;

#ifdef SHOW_DEBUG_VISUALISATION
  struct LabelDebugImages {
    cv::Mat vertConfTex;
    cv::Mat icpFull;
    cv::Mat icpLow;
    cv::Mat crfImage;
  };
  std::vector<LabelDebugImages> labelDebugImages;
#endif

  // Compute depth range
  float depthMin = std::numeric_limits<float>::max();
  float depthMax = 0;
  for (unsigned i = 0; i < result.lowDepth.total(); i++) {
    float d = ((float*)result.lowDepth.data)[i];
    if (d > MAX_DEPTH || d < 0 || !std::isfinite(d)) {
      // assert(0);
      continue;
    }
    if (depthMax < d) depthMax = d;
    if (depthMin > d) depthMin = d;
  }
  result.depthRange = depthMax - depthMin;

  // Compute per model data (ICP texture..)
  unsigned char modelIdToIndex[256];
  unsigned char mIndex = 0;
  for (auto it = models.begin(); it != models.end(); it++) {
    auto& m = *it;

    cv::Mat vertConfTex = m->downloadVertexConfTexture();
    cv::Mat icpFull = m->downloadICPErrorTexture();
    cv::Mat icp = slic.downsample<float>(icpFull);
    cv::Mat conf = slic.downsample<float>(vertConfTex, 3);
    result.modelData.push_back({m->getID(), it, icp, conf});
    modelIdToIndex[m->getID()] = mIndex++;

    // Average confidence
    auto& modelData = result.modelData.back();
    float highestConf = 0;
    for (unsigned j = 0; j < lowTotal; j++) {
      float& c = ((float*)modelData.lowConf.data)[j];
      if (!std::isfinite(c)) {
        c = 0;
        continue;
      }
      if (highestConf < c) highestConf = c;
      modelData.avgConfidence += c;
    }
    modelData.avgConfidence /= lowTotal;

#ifdef SHOW_DEBUG_VISUALISATION
    labelDebugImages.push_back({vertConfTex, icpFull, icp});
#endif
  }
  if (allowNew) {
    modelIdToIndex[nextModelID] = mIndex;
    result.modelData.push_back({nextModelID});

#ifdef SHOW_DEBUG_VISUALISATION
    labelDebugImages.push_back({cv::Mat(), cv::Mat(), cv::Mat::zeros(lowHeight, lowWidth, CV_32FC1)});
#endif
  }

  TOCK("SLIC+SCALING");
  TICK("CRF-FULL");

  DenseCRF2D crf(lowWidth, lowHeight, numLabels);
  Eigen::MatrixXf unary(numLabels, lowTotal);
  cv::Mat unaryMaxLabel(lowHeight, lowWidth, CV_8UC1);
  unsigned char* pUnaryMaxLabel = unaryMaxLabel.data;

  // auto clamp = [](float val, float min, float max) -> float {
  //    return std::max(std::min(val,max), min);
  //};
  auto getError = [&result](unsigned char modelIndex, unsigned pixelIndex) -> float& {
    return ((float*)result.modelData[modelIndex].lowICP.data)[pixelIndex];
  };
  auto getDepth = [&result](unsigned pixelIndex) -> float& { return ((float*)result.lowDepth.data)[pixelIndex]; };
  auto getConf = [&result](unsigned char modelIndex, unsigned pixelIndex) -> float& {
    return ((float*)result.modelData[modelIndex].lowConf.data)[pixelIndex];
  };

  for (unsigned k = 0; k < lowTotal; k++) {
    // Special case: If no label supplies confidence, prefer background
    float confSum = 0;
    for (unsigned i = 0; i < models.size(); i++) confSum += getConf(i, k);

    if (getConf(0, k) < 0.3) getError(0, k) = result.depthRange * 0.01;
    for (unsigned i = 1; i < numExistingModels; i++) {
      float& error = getError(i, k);
      float& conf = getConf(i, k);
      assert(std::isfinite(error));

      if (conf <= 0.4) {
        error = result.depthRange * unaryKError;
      }
    }

    // Depth sorting & updating of confidences
    // const float maxConf = 50;
    std::multimap<float, unsigned> sortedLabels;
    for (unsigned i = 0; i < numExistingModels; i++) {
      float d = getDepth(k);
      if (d > MAX_DEPTH)
        d = MAX_DEPTH;
      else if (d < 0)
        d = 0;
      sortedLabels.emplace(d, i);
    }

    // Object probabilities
    unsigned i;
    float sum = 0;
    float lowestError = getError(0, k) / result.depthRange;  // std::numeric_limits<float>::max();
    for (auto& l : sortedLabels) {
      i = l.second;
      const SegmentationResult::ModelData& modelData = result.modelData[i];

      float error = ((float*)modelData.lowICP.data)[k];
      assert(std::isfinite(error) && error >= 0);
      error /= result.depthRange;
      if (error < lowestError) lowestError = error;

      unary(i, k) = unaryWeightError * error;
      sum += unary(i, k);
    }

    if (allowNew) {
      unary(models.size(), k) = std::max(unaryThresholdNew - unaryWeightError * lowestError, 0.01f);
      sum += unary(models.size(), k);
    }

    unsigned char maxIndex = 0;
    float maxVal = 0;
    for (unsigned char i = 0; i < numLabels; i++) {
      // Find max label
      if (maxVal < unary(i, k)) {
        maxVal = unary(i, k);
        maxIndex = i;
      }
    }
    pUnaryMaxLabel[k] = maxIndex;

  }  // unaries

  // Make borders uncertain
  if (false) {
    for (unsigned y = 1; y < lowHeight - 1; y++) {
      for (unsigned x = 1; x < lowWidth - 1; x++) {
        const unsigned idx = y * lowWidth + x;
        unsigned char l = pUnaryMaxLabel[idx];
        // if(pUnaryMaxLabel[idx] == 0){
        if (pUnaryMaxLabel[idx - 1] != l || pUnaryMaxLabel[idx + 1] != l || pUnaryMaxLabel[idx - lowWidth] != l ||
            pUnaryMaxLabel[idx + lowWidth] != l || pUnaryMaxLabel[idx - lowWidth - 1] != l || pUnaryMaxLabel[idx - lowWidth + 1] != l ||
            pUnaryMaxLabel[idx - lowWidth - 1] != l || pUnaryMaxLabel[idx - lowWidth + 1] != l) {
          assert(0);  // see -log
          for (unsigned char i = 0; i < numLabels; i++) unary(i, idx) = -log(1.0f / numLabels);
        }
      }
    }
  }

#ifdef SHOW_DEBUG_VISUALISATION  // Visualise unary potentials

  auto floatToUC3 = [](cv::Mat image, float min = 0, float max = 1) -> cv::Mat {
    float range = max - min;
    cv::Mat tmp(image.rows, image.cols, CV_8UC3);
    for (unsigned i = 0; i < image.total(); i++) {
      unsigned char v = (std::min(image.at<float>(i), max) - min) / range * 255;
      tmp.at<cv::Vec3b>(i) = cv::Vec3b(v, v, v);
      // tmp.at<unsigned char>(i) = (std::min(image.at<float>(i), max) - min) / range * 255;
    }
    return tmp;
  };

  // Convert CRF-Matrix to Mat, whilst color-coding values (0..1) using COLORMAP_HOT
  auto mapCRFToImage = [&](Eigen::MatrixXf& field, unsigned index, int valueScale = 0 /* 0 linear, 1 log, 2 exp*/) -> cv::Mat {

    cv::Mat r;

    std::function<float(float)> mapVal;
    switch (valueScale) {
      case 1:
        mapVal = [](float v) -> float { return -log(v + 1); };
        break;
      case 2:
        mapVal = [](float v) -> float { return exp(-v); };
        break;
      default:
        mapVal = [](float v) -> float { return v; };
        break;
    }

    // Create greyscale image
    cv::Mat greyscale(lowHeight, lowWidth, CV_8UC1);
    unsigned char* pGreyscale = greyscale.data;
    for (unsigned k = 0; k < lowTotal; k++) pGreyscale[k] = mapVal(field(index, k)) * 255;

    // Create heat-map
    cv::applyColorMap(greyscale, r, cv::COLORMAP_HOT);

    // Find invalid values and highlight
    cv::Vec3b* pResult = (cv::Vec3b*)r.data;
    for (unsigned k = 0; k < lowTotal; k++) {
      float v = mapVal(field(index, k));
      if (v < 0)
        pResult[k] = cv::Vec3b(255, 0, 0);
      else if (v > 1)
        pResult[k] = cv::Vec3b(0, 255, 0);
    }

    return r;
  };

  const unsigned char colors[31][3] = {
      {0, 0, 0},     {0, 0, 255},     {255, 0, 0},   {0, 255, 0},     {255, 26, 184},  {255, 211, 0},   {0, 131, 246},  {0, 140, 70},
      {167, 96, 61}, {79, 0, 105},    {0, 255, 246}, {61, 123, 140},  {237, 167, 255}, {211, 255, 149}, {184, 79, 255}, {228, 26, 87},
      {131, 131, 0}, {0, 255, 149},   {96, 0, 43},   {246, 131, 17},  {202, 255, 0},   {43, 61, 0},     {0, 52, 193},   {255, 202, 131},
      {0, 43, 96},   {158, 114, 140}, {79, 184, 17}, {158, 193, 255}, {149, 158, 123}, {255, 123, 175}, {158, 8, 0}};
  auto getColor = [&colors](unsigned index) -> cv::Vec3b {
    return (index == 255) ? cv::Vec3b(255, 255, 255) : (cv::Vec3b)colors[index % 31];
  };

  auto mapLabelToColorImage = [&getColor](cv::Mat input, bool white0 = false) -> cv::Mat {

    std::function<cv::Vec3b(unsigned)> getIndex;
    auto getColorWW = [&](unsigned index) -> cv::Vec3b { return (white0 && index == 0) ? cv::Vec3b(255, 255, 255) : getColor(index); };

    if (input.type() == CV_32SC1)
      getIndex = [&](unsigned i) -> cv::Vec3b { return getColorWW(input.at<int>(i)); };
    else if (input.type() == CV_8UC1)
      getIndex = [&](unsigned i) -> cv::Vec3b { return getColorWW(input.data[i]); };
    else
      assert(0);
    cv::Mat result(input.rows, input.cols, CV_8UC3);
    for (unsigned i = 0; i < result.total(); ++i) {
      ((cv::Vec3b*)result.data)[i] = getIndex(i);
    }
    return result;
  };

  auto showInputOverlay = [&](cv::Mat original, cv::Mat segmentation) -> cv::Mat {
    assert(original.type() == CV_8UC3);
    cv::Mat result(original.rows, original.cols, CV_8UC3);
    cv::Vec3b* pResult = ((cv::Vec3b*)result.data);
    cv::Vec3b* pOriginal = ((cv::Vec3b*)original.data);
    cv::Mat overlay = mapLabelToColorImage(segmentation, true);
    cv::Vec3b* pOverlay = ((cv::Vec3b*)overlay.data);
    for (unsigned i = 0; i < result.total(); i++) pResult[i] = 0.85 * pOverlay[i] + 0.15 * pOriginal[i];
    return result;
  };

  auto stackImagesHorizontally = [](std::vector<cv::Mat> images) -> cv::Mat {
    if (images.size() == 0) return cv::Mat();
    unsigned totalWidth = 0;
    unsigned currentCol = 0;
    for (cv::Mat& m : images) totalWidth += m.cols;
    cv::Mat result(images[0].rows, totalWidth, images[0].type());
    for (cv::Mat& m : images) {
      m.copyTo(result(cv::Rect(currentCol, 0, m.cols, m.rows)));
      currentCol += m.cols;
    }
    return result;
  };

  auto stackImagesVertically = [](std::vector<cv::Mat> images) -> cv::Mat {
    if (images.size() == 0) return cv::Mat();
    unsigned totalHeight = 0;
    unsigned currentRow = 0;
    for (cv::Mat& m : images) totalHeight += m.rows;
    cv::Mat result(totalHeight, images[0].cols, images[0].type());
    for (cv::Mat& m : images) {
      m.copyTo(result(cv::Rect(0, currentRow, m.cols, m.rows)));
      currentRow += m.rows;
    }
    return result;
  };

  for (unsigned i = 0; i < numLabels; ++i) labelDebugImages[i].crfImage = mapCRFToImage(unary, i, 2);
#endif

  crf.setUnaryEnergy(unary);
  crf.addPairwiseGaussian(2, 2, new PottsCompatibility(weightSmoothness));

  Eigen::MatrixXf feature(6, lowTotal);
  for (unsigned j = 0; j < lowHeight; j++)
    for (unsigned i = 0; i < lowWidth; i++) {
      unsigned index = j * lowWidth + i;
      feature(0, index) = i * scaleFeaturesPos;                              // sx
      feature(1, index) = j * scaleFeaturesPos;                              // sy
      feature(2, index) = frame.rgb.data[index * 3 + 0] * scaleFeaturesRGB;  // sr
      feature(3, index) = frame.rgb.data[index * 3 + 1] * scaleFeaturesRGB;  // sg
      feature(4, index) = frame.rgb.data[index * 3 + 2] * scaleFeaturesRGB;  // sb
      feature(5, index) = std::min(((float*)result.lowDepth.data)[index] * scaleFeaturesDepth, 100.0f);
      assert(feature(5, index) >= 0);
      assert(feature(5, index) <= 100);
    }
  crf.addPairwiseEnergy(feature, new PottsCompatibility(weightAppearance), DIAG_KERNEL, NORMALIZE_SYMMETRIC);  // addPairwiseBilateral

  // Run, very similar to crf.inference(crfIterations)
  Eigen::MatrixXf crfResult, tmp1, tmp2;

  assert(unary.cols() > 0 && unary.rows() > 0);
  for (int i = 0; i < unary.cols(); ++i)
    for (int j = 0; j < unary.rows(); ++j)
      if (unary(j, i) <= 1e-5) unary(j, i) = 1e-5;

  DenseCRF::expAndNormalize(crfResult, -unary);

  for (unsigned it = 0; it < crfIterations; it++) {
    tmp1 = -unary;
    for (unsigned int k = 0; k < crf.countPotentials(); k++) {
      crf.getPotential(k)->apply(tmp2, crfResult);
      tmp1 -= tmp2;
    }
    DenseCRF::expAndNormalize(crfResult, tmp1);
  }

  // Write segmentation (label with highest prob, map) to image
  // VectorXs map(lowTotal);
  cv::Mat map(lowHeight, lowWidth, CV_8UC1);
  int m;
  for (unsigned i = 0; i < lowTotal; i++) {
    crfResult.col(i).maxCoeff(&m);
    map.data[i] = result.modelData[m].id;
  }

  TOCK("CRF-FULL");

  std::vector<ComponentData> ccStats;
  cv::Mat connectedComponents = connectedLabels(map, &ccStats);
  // cv::imshow("Components", mapLabelToColorImage(connectedComponents));

  // Find mapping from labels to components
  std::map<int, std::list<int>> labelToComponents = mapLabelsToComponents(ccStats);

  const bool onlyKeepLargest = true;
  const bool checkNewModelSize = true;

  // Enforce connectivity (find regions of same label, only keep largest)
  // TODO: This is not always the best decision. Get smarter!
  if (onlyKeepLargest) {
    // skip background label
    for (auto l = std::next(labelToComponents.begin()); l != labelToComponents.end(); l++) {
      // for(auto& l : labelToComponents){
      std::list<int>& labelComponents = l->second;

      // Remove every component, except largest
      std::list<int>::iterator it = labelComponents.begin();
      std::list<int>::iterator s, it2;

      while ((it2 = std::next(it)) != labelComponents.end()) {
        if (ccStats[*it].size < ccStats[*it2].size) {
          s = it;
          it = it2;
        } else {
          s = it2;
        }
        ccStats[*s].label = 255;  // Remove all smaller components (here highlight instead of BG)
        labelComponents.erase(s);
      }
    }
  }

  // TODO Prevent "normal" labels from splitting up
  // Suppress new labels that are too big / too small
  if (allowNew && checkNewModelSize) {
    const int minSize = lowTotal * minRelSizeNew;
    const int maxSize = lowTotal * maxRelSizeNew;
    std::list<int>& l = labelToComponents[nextModelID];

    for (auto& cIndex : l) {
      int size = ccStats[cIndex].size;
      if (size < minSize || size > maxSize) ccStats[cIndex].label = 255;
    }
  }

  // Compute bounding box for each model
  for (SegmentationResult::ModelData& m : result.modelData) {
    for (const int& compIndex : labelToComponents[m.id]) {
      ComponentData& stats = ccStats[compIndex];
      if (stats.left < m.left) m.left = stats.left;
      if (stats.top < m.top) m.top = stats.top;
      if (stats.right > m.right) m.right = stats.right;
      if (stats.bottom > m.bottom) m.bottom = stats.bottom;
    }
    cv::Point2i p = slic.mapToHigh(m.left, m.top);
    m.left = p.x;
    m.top = p.y;
    p = slic.mapToHigh(m.right, m.bottom);
    m.right = p.x;
    m.bottom = p.y;
  }

  // Remove labels that are too close to border
  const unsigned borderSize = 20;
  for (SegmentationResult::ModelData& m : result.modelData) {
    if (m.id == 0) continue;

    if ((m.top < borderSize && m.bottom < borderSize) || (m.left < borderSize && m.right < borderSize) ||
        (m.top > fullHeight - borderSize && m.bottom > fullHeight - borderSize) ||
        (m.left > fullWidth - borderSize && m.right > fullWidth - borderSize)) {
      // TODO This is used more often. Create lambda!
      std::list<int>& l = labelToComponents[m.id];
      for (auto& cIndex : l) {
        ccStats[cIndex].label = 255;
      }
    }
  }

  // Update result (map)
  int* pComponents = (int*)connectedComponents.data;
  //#pragma omp parallel for
  for (unsigned i = 0; i < lowTotal; i++) map.data[i] = ccStats[pComponents[i]].label;

  if (true) {
    std::vector<float> sumsDepth(result.modelData.size(), 0);
    std::vector<float> sumsDeviation(result.modelData.size(), 0);
    std::vector<unsigned> cnts(result.modelData.size(), 0);
    for (unsigned i = 0; i < lowTotal; i++) {
      if (map.data[i] == 255) continue;
      const size_t index = modelIdToIndex[map.data[i]];
      const float d = getDepth(i);
      assert(d >= 0);
      if (!(index < sumsDepth.size())) {
        std::cout << "\n\nindex: " << index << " result.modelData.size() " << result.modelData.size()
                  << " map.data[i]: " << (int)map.data[i] << std::endl
                  << std::flush;
      }
      assert(index < sumsDepth.size());
      sumsDepth[index] += d;
      cnts[index]++;
    }
    for (size_t index = 0; index < result.modelData.size(); ++index)
      result.modelData[index].depthMean = cnts[index] ? sumsDepth[index] / cnts[index] : 0;

    for (unsigned i = 0; i < lowTotal; i++) {
      if (map.data[i] == 255) continue;
      const size_t index = modelIdToIndex[map.data[i]];
      const float d = getDepth(i);
      sumsDeviation[index] += std::abs(result.modelData[index].depthMean - d);
    }
    for (size_t index = 0; index < result.modelData.size(); ++index)
      result.modelData[index].depthStd = cnts[index] ? sumsDeviation[index] / cnts[index] : 0;

    for (unsigned i = 0; i < lowTotal; i++) {
      if (map.data[i] == 255) continue;
      const size_t index = modelIdToIndex[map.data[i]];
      if (index != 0) {
        const float d = getDepth(i);
        if (d > 1.1 * result.modelData[index].depthStd + result.modelData[index].depthMean) {
          // To update mean and std
          sumsDepth[index] -= d;
          sumsDeviation[index] -=
              (std::abs(result.modelData[index].depthMean - d));  // Todo this is only approximating the std, should be good enough
          cnts[index]--;
        }
      }
    }

    // Update mean and std
    for (size_t i = 0; i < result.modelData.size(); ++i) {
      // Careful, see todo comment above
      result.modelData[i].depthMean = cnts[i] ? sumsDepth[i] / cnts[i] : 0;
      result.modelData[i].depthStd = cnts[i] ? sumsDeviation[i] / cnts[i] : 0;
    }
  }

  // Count super-pixel per model (Check whether new model made it)
  for (unsigned k = 0; k < lowTotal; k++) {
    if (map.data[k] == 255) continue;
    result.modelData[modelIdToIndex[map.data[k]]].superPixelCount++;
  }

  // Suppress tiny labels
  // for(unsigned k = 0; k < lowTotal; k++){
  //    auto& l = map[k];
  //    if(result.labelSuperPixelCounts[l] < 5){
  //        result.labelSuperPixelCounts[l] = 0;
  //        if(k > 0) l = map[k-1];
  //        else l = 0;
  //    }
  //}

  if (allowNew) {
    if (result.modelData.back().superPixelCount > 0) {
      // result.numLabels++;
      result.hasNewLabel = true;
    } else {
      result.modelData.pop_back();
    }
  }

  // Upscale result and compute bounding boxes
  result.fullSegmentation = slic.upsample<unsigned char>(map);

#ifdef SHOW_DEBUG_VISUALISATION  // Visualise unary potentials
  const bool writeOverlay = false;
  const bool writeUnaries = false;
  const bool writeICP = false;
  const bool writeSLIC = false;

  const std::string outputPath = "/tmp/multimotionfusion";
  const int minWrite = 2;

  cv::Mat inputOverlay = showInputOverlay(frame.rgb, result.fullSegmentation);
  if (writeOverlay) cv::imwrite(outputPath + "overlay" + std::to_string(CFRAME) + ".png", inputOverlay);
  if (writeSLIC) cv::imwrite(outputPath + "superpixel" + std::to_string(CFRAME) + ".png", slic.drawSurfelBorders(true));

  unsigned i = 0;
  std::vector<cv::Mat> potentialViews;
  for (; i < numLabels; ++i) {
    cv::Mat imgVisLabel = mapCRFToImage(crfResult, i);
    cv::Mat unaryUpsampling = slic.upsample<cv::Vec3b>(labelDebugImages[i].crfImage);
    cv::Mat icpUpsampling = slic.upsample<cv::Vec3b>(floatToUC3(labelDebugImages[i].icpLow));
    cv::Mat potentialsBefore = slic.drawSurfelBorders(unaryUpsampling, false);
    cv::Mat potentialsAfter = slic.drawSurfelBorders(slic.upsample<cv::Vec3b>(imgVisLabel), false);
    potentialViews.push_back(stackImagesHorizontally({icpUpsampling, potentialsBefore, potentialsAfter}));
    if (writeUnaries) cv::imwrite(outputPath + "unaries" + std::to_string(i) + "-" + std::to_string(CFRAME) + ".png", potentialsBefore);
  }

  // Merge imgPotentialsBeforeAfter images:
  cv::Mat potentialView = stackImagesVertically(potentialViews);
  if (potentialView.rows > 940) {
    float s = 940.0f / potentialView.rows;
    cv::resize(potentialView, potentialView, cv::Size(potentialView.cols * s, potentialView.rows * s));
  }
  imshow("Potentials (before, after)", potentialView);

  if (writeUnaries)
    for (; i < minWrite; ++i)
      cv::imwrite(outputPath + "unaries" + std::to_string(i) + "-" + std::to_string(CFRAME) + ".png",
                  slic.drawSurfelBorders(cv::Mat::zeros(frame.rgb.rows, frame.rgb.cols, CV_8UC3), false));

  i = 0;
  for (; i < result.modelData.size(); i++) {
    SegmentationResult::ModelData& m = result.modelData[i];
    cv::Mat icp;
    m.lowICP.convertTo(icp, CV_8UC3, 255.0);
    if (writeICP) cv::imwrite(outputPath + "ICP" + std::to_string(i) + "-" + std::to_string(CFRAME) + ".png", icp);
  }
  for (; i < minWrite; ++i) {
    if (writeICP)
      cv::imwrite(outputPath + "ICP" + std::to_string(i) + "-" + std::to_string(CFRAME) + ".png",
                  cv::Mat::zeros(frame.rgb.rows, frame.rgb.cols, CV_8UC3));
  }

  cv::waitKey(1);
#endif

  return result;
}

SegmentationResult Segmentation::performSegmentationFlowCRF(std::list<std::shared_ptr<Model>>& models, const FrameData& frame,
                                                            unsigned char nextModelID, bool allowNew, const tracker::Tracks &tracks)
{
  static unsigned CFRAME = 0;
  CFRAME++;

  SegmentationResult result;
  result.fullSegmentation = cv::Mat(frame.rgb.size(), CV_8UC1, uint8_t(0));

  // number of active (currently tracked) models and potential new model
  const unsigned numLabels = unsigned(models.size()) + allowNew;

  // map from model ID to label ID
  // the labels in the current segment are continuous (0, 1, ...) while model list can be
  // unordered and contain missing model IDs after models have been removed
  // NOTE: this has to be an ordered map to iterate in order of model ids
  std::map<uint8_t, uint8_t> idx_map;

  cv::Mat next = frame.rgb;
  cv::Mat prev = prev_frame.rgb;

  auto point_inside = [](const cv::Point &point, const cv::Mat &img) -> bool {
    return point.inside({{}, img.size()});
  };

  // CRF scale
  constexpr double s = 0.25;
  // optical flow scale
  constexpr double flow_s = 0.25;
  const cv::Size src_size = next.size();
  const cv::Size crf_size = s * cv::Point(src_size);
  const cv::Size flw_size = flow_s * cv::Point(src_size);

  cv::Mat flow;
  cv::Mat gnext, gprev;
  typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajorBit> MatrixXf_r;
  MatrixXf_r magn_flow;
  if (!prev.empty()) {
    if (flow_s!=1) {
      cv::resize(next, next, flw_size, 0, 0, cv::INTER_AREA);
      cv::resize(prev, prev, flw_size, 0, 0, cv::INTER_AREA);
    }

    cv::cvtColor(next, gnext, cv::COLOR_BGR2GRAY);
    cv::cvtColor(prev, gprev, cv::COLOR_BGR2GRAY);

    TICK("segm/opt_flow");

    // empirically tested
    // prev, next, flow, pyr_scale, levels, winsize, iterations, poly_n, poly_sigma, flags
    cv::calcOpticalFlowFarneback(gprev, gnext, flow, 0.2, 1, 20, 2, 5, 15/50.0, 0);

    TOCK("segm/opt_flow");

    cv::resize(flow, flow, crf_size, cv::INTER_LINEAR);

    // show flow
    std::vector<cv::Mat> flow_x_y;
    cv::split(flow, flow_x_y);

    cv::Mat_<float> mag, ang;
    cv::cartToPolar(flow_x_y[0], flow_x_y[1], mag, ang);
    magn_flow = Eigen::Map<MatrixXf_r>((float*)mag.data, mag.rows, mag.cols);
#if DBG_VIS_SEGM
    std::vector<cv::Mat_<uint8_t>> hsv(3, {mag.size(), 0});
    hsv[0] = ang * 180./M_PI_2;
    cv::normalize(mag, hsv[2], 0, 255, cv::NORM_MINMAX);
    cv::Mat flow_vis;
    cv::merge(hsv, flow_vis);
    cv::cvtColor(flow_vis, flow_vis, cv::COLOR_HSV2BGR);
    cv::imshow("flow_vis", flow_vis);
    cv::Mat magn_scale;
    mag.convertTo(magn_scale, CV_8UC1, 50);
    cv::imshow("flow magn", magn_scale);
#endif
  } // prev

  // dense reprojection error
  std::vector<cv::Mat_<float>> proj_prob;
  cv::Mat_<float> expsum(crf_size, 0);
  constexpr float max_err = 0.03; // metre
  cv::Mat_<bool> invalid(frame.depth.size(), false);
  for (const ModelPointer &model : models) {
    const cv::Mat depth = model->getVertexConfProjection()->downloadTexture();
    std::vector<cv::Mat> xyz;
    cv::split(depth, xyz);
    // signed distance
    cv::Mat_<float> dist = cv::abs(frame.depth - xyz[2]);
#if DBG_VIS_PROBS
    cv::imshow("err m"+std::to_string(model->getID()), dist);
    cv::imwrite("/tmp/mmf/err_proj_m"+std::to_string(model->getID())+"_"+std::to_string(frame.timestamp)+".png", dist/0.1*256);
#endif
    // mark invalid depth
    invalid |= (frame.depth<1e-6) & (xyz[2]<1e-6);

    if (s!=1) {
      cv::resize(dist, dist, crf_size, 0, 0, cv::INTER_NEAREST);
    }

    // truncate distance
    cv::threshold(dist, dist, max_err, max_err, cv::THRESH_TRUNC);
#if DBG_VIS_PROBS
    cv::imshow("err trunc m"+std::to_string(model->getID()), dist/max_err);
#endif

    // turn high errors into low probabilities
    cv::exp(-1 * (dist/max_err), dist); // scale for better visualisation
    proj_prob.push_back(dist);

    expsum += dist;
  }

  cv::resize(invalid, invalid, expsum.size(), 0, 0, cv::INTER_NEAREST);
  // normalise exponential values to get probabilities
  for (cv::Mat_<float> &m : proj_prob) {
    m /= expsum;
    // equal probability for uncertain data
    m.setTo(1.0/proj_prob.size(), expsum==0);
    // 0 probability for invalid data
    m.setTo(0, invalid);
  }

#if DBG_VIS_PROBS
  // MAP
  {
  cv::Mat_<uint8_t> map(crf_size, 0);
  cv::Mat_<float> max_prob(crf_size, 0);
  for (size_t w=0; w<proj_prob.size(); w++) {
    map.setTo(w+1, proj_prob[w]>max_prob);
    proj_prob[w].copyTo(max_prob, proj_prob[w]>max_prob);
  }
  cv::Mat_<cv::Vec3b> map_rgb(crf_size, cv::Vec3b());
  std::uniform_real_distribution<double> unif(0,1);
  std::default_random_engine g;
  for (int u=0; u<crf_size.height; u++) {
    for (int v=0; v<crf_size.width; v++) {
      // unique colour per class ID
      g.seed(map.at<uint8_t>(u,v)+1);
      map_rgb.at<cv::Vec3b>(u,v) = cv::Vec3b(unif(g)*255, unif(g)*255, unif(g)*255);
    }
  }
  cv::imshow("map proj", map*50);
  cv::imshow("map proj colour", map_rgb);
  }
#endif

  if (!flow.empty()) {
    for (ModelListIterator m = models.begin(); m != models.end(); m++) {
      result.modelData.push_back({(*m)->getID(), m, {}, {}});
      idx_map[(*m)->getID()] = std::distance(models.begin(), m);
    }

    if (allowNew) {
      result.modelData.push_back({nextModelID});
      idx_map[nextModelID] = models.size();
    }

    TICK("segm/flowCRF");
    DenseCRF2D crf(crf_size.width, crf_size.height, int(numLabels));

    enum error_metric_t {METRE, PIXEL, METRE_S, PIXEL_S};

    constexpr error_metric_t metric = PIXEL_S;

    size_t minhist;
    double threshold;

    minhist = 2;

    switch (metric) {
    case METRE:
      threshold = 0.005;
      break;
    case METRE_S:
      threshold = 0.01; // 1cm/s
      break;
    case PIXEL:
      threshold = 3;
      break;
    case PIXEL_S:
      threshold = 20; // p/s
      break;
    }

#if DBG_VIS_SEGM
    // visualisation of local track projection
    int ms;
    bool scale;
    switch (metric) {
    case PIXEL_S:
      scale = true;
    case PIXEL:
      ms = threshold;
      break;
    default:
      scale = false;
      ms = 10;
    }
#endif

    // unary: Nmodels x Npixel
    TICK("segm/unary");
    Eigen::MatrixXf unary(numLabels, crf_size.area());
    // error of unkown association
    unary.fill(std::numeric_limits<float>::infinity());
    int label = 0;
    std::list<tracker::TrackPtr> outlier_set(tracks.begin(), tracks.end());
    for (const ModelPointer &model : models) {
      // test all global tracks
      const tracker::Tracks ltracks = model->computeTrackProjectionStartEnd(tracks, minhist);
#if DBG_VIS_SEGM
      const cv::Mat track_local_img = Model::drawLocalTracks2D(ltracks, frame.rgb, ms, scale);
      cv::imshow("model tracks (local) "+std::to_string(model->getID()), track_local_img);
#endif
#if DBG_EXP_ERRORS
      cv::imwrite("/tmp/mmf/track_local_m"+std::to_string(model->getID())+"_"+std::to_string(frame.timestamp)+".png", track_local_img);
#endif

#if DBG_VIS_SEGM
      cv::Mat track_err;
      cv::cvtColor(frame.rgb, track_err, cv::COLOR_RGB2GRAY);
      cv::cvtColor(track_err, track_err, cv::COLOR_GRAY2RGB);

#endif
#if DBG_VIS_SEGM
      cv::Mat track_vel = track_err.clone();
#endif

      for (size_t it=0; it<ltracks.size(); it++) {
        const auto kp0 = ltracks[it]->front();
        const auto kp1 = ltracks[it]->back();

        // skip invalid pairs
        if (kp0==nullptr || kp1==nullptr) { continue; }

        // skip keypoints without valid depth
        if (!kp0->coordinate.array().isFinite().all() ||
            !kp1->coordinate.array().isFinite().all() ||
            !point_inside(kp0->xy, frame.rgb) ||
            !point_inside(kp1->xy, frame.rgb))
        {
          outlier_set.remove(tracks[it]);
          continue;
        }

        const cv::Point &c1 = s * kp1->xy;

        // metric
        double v;
        switch (metric) {
        case METRE:
          v = (kp0->coordinate - kp1->coordinate).norm();
          break;
        case METRE_S:
          v = (kp1->coordinate - kp0->coordinate).norm() / ((kp1->timestamp - kp0->timestamp) * 1e-9);
          break;
        case PIXEL:
          v = cv::norm(cv::Point2f(kp0->xy) - cv::Point2f(kp1->xy));
          break;
        case PIXEL_S:
          v = cv::norm(cv::Point2f(kp1->xy) - cv::Point2f(kp0->xy)) / ((kp1->timestamp - kp0->timestamp) * 1e-9);
          break;
        }

        if (v > threshold) {
          // outlier
#if DBG_VIS_SEGM
          cv::circle(track_err, kp1->xy, 3, cv::Scalar(0, 0, 255), -1); // red
#endif
        }
        else {
          // inlier
          outlier_set.remove(tracks[it]);
#if DBG_VIS_SEGM
          cv::circle(track_err, kp1->xy, 3, cv::Scalar(255, 0, 0), -1); // blue
#endif
        }

#if DBG_VIS_SEGM
        // blue: low speed, red: high speed
        const double vn = v / (2*threshold);
        cv::circle(track_vel, kp1->xy, 3, cv::Scalar((1-vn) * 255, 0, vn * 255), -1);

#endif
        unary(label, c1.y*crf_size.width + c1.x) = v;
      }
      label++;
#if DBG_VIS_SEGM
      cv::imshow("track err "+std::to_string(model->getID()), track_err);
      cv::imshow("track vel "+std::to_string(model->getID()), track_vel);
#endif
#if DBG_EXP_ERRORS
      cv::imwrite("/tmp/mmf/track_err_m"+std::to_string(model->getID())+"_"+std::to_string(frame.timestamp)+".png", track_err);
#endif
    }

#if DBG_VIS_SEGM
    if (allowNew) {
      // outlier tracks are the outlier model's inlier tracks
      cv::Mat track_err;
      cv::cvtColor(frame.rgb, track_err, cv::COLOR_RGB2GRAY);
      cv::cvtColor(track_err, track_err, cv::COLOR_GRAY2RGB);
      for (const tracker::TrackPtr &track : outlier_set) {
        if (track->back()) {
          cv::circle(track_err, track->back()->xy, 3, cv::Scalar(0, 0, 255), -1);
        }
      }
      cv::imshow("outlier", track_err);
    }
#endif

    constexpr bool norm01 = true;

    if (norm01) {
      // scale error in [0,1], 0: match, 1: mis-match
      // current active models
      typedef Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic> MatrixXb;
      const Eigen::MatrixXf u_active = unary.topRows(models.size());
      const MatrixXb valid = u_active.array().isFinite();
      const Eigen::MatrixXf err_active = (u_active.array() > threshold).cast<float>();
      unary.topRows(models.size()) = valid.select(err_active, u_active);
      // outlier class, potential new model
      if (allowNew) {
        // assume a track matches the outlier model if it does not match any other active model
        const Eigen::RowVectorXf err_outlier = (u_active.array() < threshold).colwise().any().cast<float>();
        unary.row(numLabels-1) = valid.colwise().all().select(err_outlier, unary.row(numLabels-1));
      }
    }
    else {
      // use original metric error
      // and determine outlier error by transformation estimation on outlier tracks
      if (allowNew) {
        // get transformation estimate from outlier set
        const tracker::Tracks outlier_vec(outlier_set.begin(), outlier_set.end());
        const Eigen::Isometry3f Toutlier = Model::getLastTrackTransform(outlier_vec).transformation;

        // re-compute new projection error on this outlier tracks for outlier model
        for (size_t it=0; it<outlier_vec.size(); it++) {
          const auto kp0 = outlier_vec[it]->front();
          const auto kp1 = outlier_vec[it]->back();
          if (kp0==nullptr || kp1==nullptr) { continue; }

          double e;

          switch (metric) {
          case METRE:
            e = ((Toutlier.cast<double>().inverse() * kp0->coordinate.transpose()).transpose() - kp1->coordinate).norm();
            break;
          default:
            // TODO: 2D track projection
            throw std::runtime_error("unsupported");
          }

          if (std::isnan(e)) { continue; }

          const cv::Point &c1 = s * kp1->xy;
          unary(numLabels-1, c1.y*crf_size.width + c1.x) = e;
        }
      }
    }
    TOCK("segm/unary");

#if DBG_VIS_SEGM
    // DBG
    {
      std::vector<cv::Mat_<float>> errs(numLabels, {crf_size, 0});
      for (size_t l = 0; l < numLabels; ++l) {
        for (int u = 0; u < flow.rows; ++u) {
          for (int v = 0; v < flow.cols; ++v) {
            const int i = u * flow.cols + v;
              errs[l].at<float>(u,v) = unary(int(l), i);
          }
        }
        cv::imshow("errors "+std::to_string(l), errs[l]);
#if DBG_EXP_ERRORS
        cv::imwrite("/tmp/mmf/err_m"+std::to_string(l)+"_"+std::to_string(frame.timestamp)+".png", errs[l]*256);
#endif
      }
    }

#endif

    // turn errors to probabilities p(track | model) via softmax
    unary *= -1;
    for (int i = 0; i < unary.cols(); ++i) {
      const auto exp = unary.col(i).array().exp();
      if (exp.sum()>0) {
        // apply regular softmax
        unary.col(i) = exp / exp.sum();
      }
      else {
        // all infinite, assume equal probability for all models
        unary.col(i).fill(1 / float(numLabels));
      }
    }

    // log probability
    unary = -unary.array().log();

    crf.setUnaryEnergy(unary);

    crf.addPairwiseGaussian(3, 3, new PottsCompatibility(4*weightSmoothness));

    // feature optical flow: x, y, vx, vy
    Eigen::MatrixXf feature(4, crf_size.area());
    for (int u = 0; u < flow.rows; ++u) {
      for (int v = 0; v < flow.cols; ++v) {
        const int i = u * flow.cols + v;
        // coordinates
        feature.col(i).x() = v / 40;
        feature.col(i).y() = u / 40;
        // optical flow
        feature.col(i).z() = flow.at<cv::Point2f>(i).x * 10;
        feature.col(i).w() = flow.at<cv::Point2f>(i).y * 10;
      }
    }

    crf.addPairwiseEnergy(feature, new PottsCompatibility(weightAppearance));

    Eigen::MatrixXf prob_flow = crf.inference(crfIterations);

    Eigen::MatrixXf prob_proj(numLabels, crf_size.area());
    for (size_t i=0; i<proj_prob.size(); i++) {
      prob_proj.row(i) = Eigen::Map<Eigen::RowVectorXf>((float*)proj_prob[i].data, 1, crf_size.area());
    }

    // remove uncertain projection probabilities
    prob_proj = (prob_proj.array()<0.3).select(0, prob_proj);

#if DBG_VIS_PROBS
    {
      for (int i=0; i<prob_flow.rows(); i++) {
        Eigen::RowVectorXf v = prob_flow.row(i);
        cv::Mat_<float> prob_flow_cv(crf_size.height, crf_size.width, v.data());
        cv::imshow("prob flow m"+std::to_string(i), prob_flow_cv);
        cv::imwrite("/tmp/mmf/prob_flow_m"+std::to_string(i)+"_"+std::to_string(frame.timestamp)+".png", prob_flow_cv*256);
      }
      for (int i=0; i<prob_proj.rows(); i++) {
        Eigen::RowVectorXf v = prob_proj.row(i);
        cv::Mat_<float> prob_proj_cv(crf_size.height, crf_size.width, v.data());
        cv::imshow("prob proj m"+std::to_string(i), prob_proj_cv);
        cv::imwrite("/tmp/mmf/prob_proj_m"+std::to_string(i)+"_"+std::to_string(frame.timestamp)+".png", prob_proj_cv*256);
      }
    }

    {
      cv::Mat_<float> magn_flow_cv(crf_size.height, crf_size.width, magn_flow.data());
      cv::imwrite("/tmp/mmf/magn_flow_"+std::to_string(frame.timestamp)+".png", (magn_flow_cv/5.0)*256);
    }
#endif

    const Eigen::RowVectorXf unary_flow_magn = Eigen::Map<Eigen::RowVectorXf>(magn_flow.data(), crf_size.area());

    // map flow range 0.2 ... 5 to probabilities 0 ... 1
    const float flow_min = 0.2, flow_max = 5;
    Eigen::RowVectorXf prob_flow_magn = ((unary_flow_magn.array() - flow_min) / (flow_max - flow_min)).cwiseMax(0).cwiseMin(1);

#if DBG_VIS_PROBS
    {
      cv::Mat_<float> prob_flow_magn_cv(crf_size.height, crf_size.width, prob_flow_magn.data());
      cv::imshow("prob flow magn", prob_flow_magn_cv);
      cv::imwrite("/tmp/mmf/prob_flow_magn_"+std::to_string(frame.timestamp)+".png", prob_flow_magn_cv*256);
    }
#endif

    Eigen::MatrixXf prob_flow2 = Eigen::MatrixXf::Zero(prob_flow.rows(), prob_flow.cols());
    for (int l=0; l<int(prob_flow.rows()); l++) {
      prob_flow2.row(l) = prob_flow.row(l).array() * prob_flow_magn.array();
    }

#if DBG_VIS_PROBS
    {
      for (int i=0; i<prob_flow.rows(); i++) {
        Eigen::RowVectorXf v = prob_flow2.row(i);
        cv::Mat_<float> prob_flow_cv(crf_size.height, crf_size.width, v.data());
        cv::imshow("prob flow2 (*magn) m"+std::to_string(i), prob_flow_cv);
        cv::imwrite("/tmp/mmf/prob_flow2_m"+std::to_string(i)+"_"+std::to_string(frame.timestamp)+".png", prob_flow_cv*256);
      }
    }
#endif

    prob_flow = prob_flow2;

    const Eigen::MatrixXf prob = 1 - ((1 - prob_flow.array()) * (1 - prob_proj.array()));

#if DBG_VIS_PROBS
    // MAP (total)
    {
      cv::Mat_<uint8_t> map(crf_size, 0);
      cv::Mat_<float> max_prob(crf_size, 0);
      for (int w=0; w<prob.rows(); w++) {
        Eigen::RowVectorXf v = prob.row(w);
        cv::Mat_<float> prob_cv(crf_size.height, crf_size.width, v.data());
        cv::imshow("prob m"+std::to_string(w), prob_cv);
        cv::imwrite("/tmp/mmf/prob_m"+std::to_string(w)+"_"+std::to_string(frame.timestamp)+".png", prob_cv*256);

        map.setTo(w+1, prob_cv>max_prob);
        prob_cv.copyTo(max_prob, prob_cv>max_prob);
      }

      cv::imshow("map total", map*50);
    }
#endif

    const Eigen::VectorXi lbl = crf.currentMap(prob).cast<int>();
#if DBG_VIS_PROBS
    {
      Eigen::Matrix<uint8_t, 1, Eigen::Dynamic> lbl2 = lbl.cast<uint8_t>();
      cv::Mat_<uint8_t> lbl_cv(next.rows, next.cols, lbl2.data());
      cv::imshow("label", (lbl_cv+1)*50);
      cv::imwrite("/tmp/mmf/lbl_"+std::to_string(frame.timestamp)+".png", (lbl_cv+1)*50);
    }
#endif

    // create segmentation at CRF resolution
    // this will contain the model ID
    cv::Mat_<uint8_t> model_segm(crf_size, 0);
    for (int u = 0; u < flow.rows; ++u) {
      for (int v = 0; v < flow.cols; ++v) {
        const int i = u * flow.cols + v;
        const uint8_t &uid = uint8_t(result.modelData[size_t(lbl[i])].id);
        model_segm.at<uint8_t>(i) = uid;
      }
    }
#if DBG_VIS_PROBS
    cv::imwrite("/tmp/mmf/segm_"+std::to_string(frame.timestamp)+".png", 50*model_segm);
#endif

    // find largest blob
    cv::Mat_<uint8_t> model_segm_cont(model_segm.size(), 0);
    std::array<uint32_t, 256> segm_count;
    for (const auto &[m ,l] : idx_map) {
      std::vector<std::vector<cv::Point>> contours;
      cv::findContours(model_segm==m, contours, {}, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
      double max_a = 0;
      size_t max_i = 0;
      for (size_t j=0; j<contours.size(); j++) {
        const double a = cv::contourArea(contours[j]);
        if (a>max_a) {
          max_a = a;
          max_i = j;
        }
      }
      segm_count[m] = std::rint(max_a);
      cv::drawContours(model_segm_cont, contours, max_i, m, cv::FILLED);
    }

    model_segm = model_segm_cont;

#if DBG_VIS_PROBS
    cv::imshow("segm (model ID)", model_segm*50);
#endif

    // resize to original image dimension
    cv::resize(model_segm, result.fullSegmentation, frame.rgb.size(), 0, 0, cv::INTER_NEAREST);

    // scale size of segments according to image scale
    constexpr float scale_weight = 1.0/(s*s);

    // TODO: make configureable
    constexpr float new_model_size = 0.05f;

    // ModelData(t_id, t_modelListIterator, t_lowICP, t_lowConf, t_superPixelCount, t_avgConfidence);
    for (SegmentationResult::ModelData &mod : result.modelData) {
      mod.superPixelCount = uint(float(segm_count[mod.id]) * scale_weight);
      mod.avgConfidence = 0.4f;

      cv::Scalar dmean, dstddev;
      cv::meanStdDev(frame.depth, dmean, dstddev, result.fullSegmentation==mod.id);
      mod.depthMean = float(dmean[0]);
      mod.depthStd = float(dstddev[0]);
    }

    result.hasNewLabel = false;

    if (allowNew) {
      result.hasNewLabel = float(cv::countNonZero(model_segm==nextModelID)) / float(model_segm.size().area()) > new_model_size;

      if (!result.hasNewLabel) {
        // delete last, potentially new, model
        result.modelData.pop_back();
      }
    }

#if DBG_VIS_SEGM
    // DBG
    {
      cv::Mat lbls(crf_size, CV_8UC3);
      for (const auto &[m ,l] : idx_map) {
        // TODO: show all segments in unique colour
        const cv::Scalar c = (m==0) ? cv::viz::Color::blue() : cv::viz::Color::red();
        lbls.setTo(c, model_segm==m);
      }
      cv::resize(gprev, gprev, lbls.size());
      cv::cvtColor(gprev, gprev, cv::COLOR_GRAY2BGR);
      cv::addWeighted(gprev, 1, lbls, 0.5, 0, lbls);
      cv::imshow("segm", lbls);
    }
    cv::waitKey(1);
#endif
    TOCK("segm/flowCRF");
  } // uflow

  prev_frame = frame;

  return result;
}
