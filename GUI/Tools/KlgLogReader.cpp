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

#include "../Core/Utils/Macros.h"
#include "KlgLogReader.h"

KlgLogReader::KlgLogReader(std::string file, bool flipColors) : LogReader(file, flipColors) {
  assert(pangolin::FileExists(file.c_str()));

  fp = fopen(file.c_str(), "rb");

  currentFrame = 0;

  // TODO: get image dimensions from the encoded colour or depth image
  width = Resolution::getInstance().width();
  height = Resolution::getInstance().height();
  numPixels = width * height;

  if (!fread(&numFrames, sizeof(int32_t), 1, fp)) throw std::invalid_argument("Could not open log-file: " + file);

  depthBuffer = cv::Mat(height, width, CV_16UC1);
  rgbBuffer = cv::Mat(height, width, CV_8UC3);
  depthDecompressionBuffer = cv::Mat(height, width, CV_16UC1);
  rgbDecompressionBuffer = cv::Mat(height, width, CV_8UC3);

  std::cout << "Reading log file: " << file << " which has " << numFrames << " frames. " << std::endl;
}

KlgLogReader::~KlgLogReader() { fclose(fp); }

void KlgLogReader::getNext() {
  filePointers.push(ftell(fp));
  getCore();
}

void KlgLogReader::getPrevious() {
  assert(filePointers.size() > 0);
  fseek(fp, filePointers.top(), SEEK_SET);
  filePointers.pop();
  getCore();
}

void KlgLogReader::getCore() {
  CHECK_THROW(fread(&data.timestamp, sizeof(int64_t), 1, fp));
  CHECK_THROW(fread(&depthImageSize, sizeof(int32_t), 1, fp));
  CHECK_THROW(fread(&rgbImageSize, sizeof(int32_t), 1, fp));
  CHECK_THROW(fread(depthBuffer.data, depthImageSize, 1, fp));

  if (rgbImageSize > 0) {
    CHECK_THROW(fread(rgbBuffer.data, rgbImageSize, 1, fp));
  }

  if (depthImageSize != numPixels * 2) {
    unsigned long decompLength = numPixels * 2;
    uncompress(depthDecompressionBuffer.data, (unsigned long*)&decompLength, depthBuffer.data, depthImageSize);
    depthDecompressionBuffer.convertTo(data.depth, CV_32FC1, 0.001);
  } else {
    depthBuffer.convertTo(data.depth, CV_32FC1, 0.001);
  }

  data.rgb = rgbBuffer;
  if (rgbImageSize > 0) {
    if (rgbImageSize != numPixels * 3) {
      jpeg.readData(rgbBuffer.data, rgbImageSize, rgbDecompressionBuffer.data);
      data.rgb = rgbDecompressionBuffer;
    }
  } else {
    rgbBuffer.setTo(cv::Scalar(0, 0, 0));
  }

  if (flipColors) data.flipColors();

  currentFrame++;
}

void KlgLogReader::fastForward(int frame) {
  while (currentFrame < frame && hasMore()) {
    filePointers.push(ftell(fp));

    CHECK_THROW(fread(&data.timestamp, sizeof(int64_t), 1, fp));

    CHECK_THROW(fread(&depthImageSize, sizeof(int32_t), 1, fp));
    CHECK_THROW(fread(&rgbImageSize, sizeof(int32_t), 1, fp));

    CHECK_THROW(fread(depthBuffer.data, depthImageSize, 1, fp));

    if (rgbImageSize > 0) {
      CHECK_THROW(fread(rgbBuffer.data, rgbImageSize, 1, fp));
    }

    currentFrame++;
  }
}

int KlgLogReader::getNumFrames() { return numFrames; }

bool KlgLogReader::hasMore() { return currentFrame + 1 < numFrames; }

bool KlgLogReader::rewind() {
  if (filePointers.size() == 0) {
    fclose(fp);
    fp = fopen(file.c_str(), "rb");

    CHECK_THROW(fread(&numFrames, sizeof(int32_t), 1, fp));

    currentFrame = 0;

    return true;
  }

  return false;
}

const std::string KlgLogReader::getFile() { return file; }

FrameData KlgLogReader::getFrameData() { return data; }

void KlgLogReader::setAuto(bool value) {}
