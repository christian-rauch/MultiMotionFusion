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

#include "GPUTexture.h"
#include "Cuda/convenience.cuh"

#include "Utils/OpenGLErrorHandling.h"
#include <opencv2/highgui/highgui.hpp>

const std::string GPUTexture::RGB = "RGB";
const std::string GPUTexture::DEPTH_METRIC = "DEPTH_METRIC";
const std::string GPUTexture::DEPTH_METRIC_FILTERED = "DEPTH_METRIC_FILTERED";
const std::string GPUTexture::MASK = "MASKS";
const std::string GPUTexture::DEPTH_NORM = "DEPTH_NORM";
const std::string GPUTexture::MASK_COLOR = "MASKS_COLOR";

GPUTexture::GPUTexture(const int width, const int height, const GLenum internalFormat, const GLenum format, const GLenum dataType,
                       const bool draw, const bool cuda, unsigned int cudaFlags, std::string teststring)
    : myName(teststring),
      texture(new pangolin::GlTexture(width, height, internalFormat, draw, 0, format, dataType)),
      draw(draw),
      width(width),
      height(height),
      internalFormat(internalFormat),
      format(format),
      dataType(dataType),
      cudaRes(0),
      cudaSurfHandle(0) {
  if (cuda) {
    cudaGraphicsGLRegisterImage(&cudaRes, texture->tid, GL_TEXTURE_2D, cudaFlags);

    if (cudaFlags == cudaGraphicsRegisterFlagsSurfaceLoadStore) {
      assert(internalFormat == GL_R32F || internalFormat == GL_LUMINANCE32F_ARB);
      int numElements = width * height;
      float* data = new float[numElements];
      for (int i = 0; i < numElements; i++) data[i] = 0;
      texture->Upload(data, GL_RED, GL_FLOAT);
      delete[] data;

      cudaResourceDesc cudaResDesc;
      memset(&cudaResDesc, 0, sizeof(cudaResDesc));

      cudaMap();
      cudaArray* array = getCudaArray();

      cudaResDesc.resType = cudaResourceTypeArray;
      cudaResDesc.res.array.array = array;

      cudaSafeCall(cudaCreateSurfaceObject(&cudaSurfHandle, &cudaResDesc));
    }
  }
}

GPUTexture::~GPUTexture() {
  if (texture) {
    delete texture;
  }

  cudaUnmap();
  if (cudaRes) {
    cudaGraphicsUnregisterResource(cudaRes);
    cudaRes = 0;
  }
  cudaCheckError();
}

void GPUTexture::cudaMap() {
  if (!cudaIsMapped) {
    cudaGraphicsMapResources(1, &cudaRes);
    cudaIsMapped = true;
  }
}

void GPUTexture::cudaUnmap() {
  if (cudaIsMapped) {
    cudaGraphicsUnmapResources(1, &cudaRes);
    cudaIsMapped = false;
  }
}

cudaArray* GPUTexture::getCudaArray() {
  cudaArray* pArray;
  cudaGraphicsSubResourceGetMappedArray(&pArray, cudaRes, 0, 0);
  return pArray;
}

const cudaSurfaceObject_t& GPUTexture::getCudaSurface() { return cudaSurfHandle; }

cv::Mat GPUTexture::downloadTexture() {
  // Right now only 1-/4-channel float textures are supported
  // assert(format == GL_RED || format == GL_LUMINANCE || format == GL_RGBA);
  // assert(internalFormat == GL_RGBA32F || internalFormat == GL_LUMINANCE32F_ARB || internalFormat == GL_R32F);
  // assert(dataType == GL_FLOAT);

  int cvType = -1;
  if (dataType == GL_FLOAT) {
    switch (format) {
      case GL_RGBA:
        cvType = CV_32FC4;
        break;
      case GL_RGB:
        cvType = CV_32FC3;
        break;
      case GL_LUMINANCE:
        cvType = CV_32FC1;
        break;
      case GL_RED:
        cvType = CV_32FC1;
        break;
    }
  } else if (dataType == GL_UNSIGNED_BYTE) {
    switch (format) {
      case GL_RGBA:
        cvType = CV_8UC4;
        break;
      case GL_RGB:
        cvType = CV_8UC3;
        break;
      case GL_RED:
      case GL_RED_INTEGER:
        cvType = CV_8UC1;
        break;
    }
  }
  if (cvType == -1) throw std::invalid_argument("Unable to download texture.");

  // if(cvType == 29) {
  //    std::cout << "[HERE]";
  //}

  // const bool fourChannels = (internalFormat == GL_RGBA32F);
  cv::Mat result(height, width, cvType);

  bool cudaMapped = cudaIsMapped;
  if (cudaMapped) cudaUnmap();
  texture->Download(result.data, format, dataType);
  if (cudaMapped) cudaMap();

  checkGLErrors();

  return result;
}

void GPUTexture::save(const std::string& file) {
  // cudaDeviceSynchronize();
  if (cudaRes) cudaSafeCall(cudaGraphicsUnmapResources(1, &cudaRes));
  texture->Save(file);
  if (cudaRes) cudaSafeCall(cudaGraphicsMapResources(1, &cudaRes));
}
