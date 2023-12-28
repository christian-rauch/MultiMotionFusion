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

#include "Model.h"
#include "ModelMatching.h"

#include <opencv2/opencv.hpp>
#include <opencv2/viz/types.hpp>
#include <opencv2/core/eigen.hpp>

#include "Utils/RigidRANSAC.h"
#include "Utils/happly.h"

Model::GPUSetup::GPUSetup()
    : initProgram(loadProgramFromFile("init_unstable.vert")),
      drawProgram(loadProgramFromFile("draw_feedback.vert", "draw_feedback.frag")),
      drawSurfelProgram(loadProgramFromFile("draw_global_surface.vert", "draw_global_surface.frag", "draw_global_surface.geom")),
      dataProgram(loadProgramFromFile("data.vert", "data.frag", "data.geom")),
      updateProgram(loadProgramFromFile("update.vert")),
      unstableProgram(loadProgramGeomFromFile("copy_unstable.vert", "copy_unstable.geom")),
      renderBuffer(TEXTURE_DIMENSION, TEXTURE_DIMENSION),
      updateMapVertsConfs(TEXTURE_DIMENSION, TEXTURE_DIMENSION, GL_RGBA32F, GL_LUMINANCE, GL_FLOAT),
      updateMapColorsTime(TEXTURE_DIMENSION, TEXTURE_DIMENSION, GL_RGBA32F, GL_LUMINANCE, GL_FLOAT),
      updateMapNormsRadii(TEXTURE_DIMENSION, TEXTURE_DIMENSION, GL_RGBA32F, GL_LUMINANCE, GL_FLOAT) {
  frameBuffer.AttachColour(*updateMapVertsConfs.texture);
  frameBuffer.AttachColour(*updateMapColorsTime.texture);
  frameBuffer.AttachColour(*updateMapNormsRadii.texture);
  frameBuffer.AttachDepth(renderBuffer);

  updateProgram->Bind();
#ifdef NVIDIA_VARYINGS
  int locUpdate[3] = {
      glGetVaryingLocationNV(updateProgram->programId(), "vPosition0"), glGetVaryingLocationNV(updateProgram->programId(), "vColor0"),
      glGetVaryingLocationNV(updateProgram->programId(), "vNormRad0"),
  };
  glTransformFeedbackVaryingsNV(updateProgram->programId(), 3, locUpdate, GL_INTERLEAVED_ATTRIBS);
#else
  const GLchar* varyings[] = {"vPosition0", "vColor0", "vNormRad0"};
  glTransformFeedbackVaryings(updateProgram->programId(), 3, varyings, GL_INTERLEAVED_ATTRIBS);
#endif
  updateProgram->Unbind();

  dataProgram->Bind();
#ifdef NVIDIA_VARYINGS
  int dataUpdate[3] = {
      glGetVaryingLocationNV(dataProgram->programId(), "vPosition0"), glGetVaryingLocationNV(dataProgram->programId(), "vColor0"),
      glGetVaryingLocationNV(dataProgram->programId(), "vNormRad0"),
  };
  glTransformFeedbackVaryingsNV(dataProgram->programId(), 3, dataUpdate, GL_INTERLEAVED_ATTRIBS);
#else
  glTransformFeedbackVaryings(dataProgram->programId(), 3, varyings, GL_INTERLEAVED_ATTRIBS);
#endif
  dataProgram->Unbind();

  unstableProgram->Bind();
#ifdef NVIDIA_VARYINGS
  int unstableUpdate[3] = {
      glGetVaryingLocationNV(unstableProgram->programId(), "vPosition0"), glGetVaryingLocationNV(unstableProgram->programId(), "vColor0"),
      glGetVaryingLocationNV(unstableProgram->programId(), "vNormRad0"),
  };
  glTransformFeedbackVaryingsNV(unstableProgram->programId(), 3, unstableUpdate, GL_INTERLEAVED_ATTRIBS);
#else
  glTransformFeedbackVaryings(unstableProgram->programId(), 3, varyings, GL_INTERLEAVED_ATTRIBS);
#endif
  unstableProgram->Unbind();

  // eraseProgram->Bind();
  // int eraseUpdate[3] = {
  //    glGetVaryingLocationNV(eraseProgram->programId(), "vPosition0"),
  //    glGetVaryingLocationNV(eraseProgram->programId(), "vColor0"),
  //    glGetVaryingLocationNV(eraseProgram->programId(), "vNormRad0"),
  //};
  // glTransformFeedbackVaryingsNV(eraseProgram->programId(), 3, eraseUpdate, GL_INTERLEAVED_ATTRIBS);
  // eraseProgram->Unbind();

  initProgram->Bind();
#ifdef NVIDIA_VARYINGS
  int locInit[3] = {
      glGetVaryingLocationNV(initProgram->programId(), "vPosition0"), glGetVaryingLocationNV(initProgram->programId(), "vColor0"),
      glGetVaryingLocationNV(initProgram->programId(), "vNormRad0"),
  };

  glTransformFeedbackVaryingsNV(initProgram->programId(), 3, locInit, GL_INTERLEAVED_ATTRIBS);
#else
  glTransformFeedbackVaryings(initProgram->programId(), 3, varyings, GL_INTERLEAVED_ATTRIBS);
#endif
  initProgram->Unbind();

  depth_tmp.resize(RGBDOdometry::NUM_PYRS);
  mask_tmp.resize(RGBDOdometry::NUM_PYRS);

  for (int i = 0; i < RGBDOdometry::NUM_PYRS; ++i) {
    int pyr_rows = Resolution::getInstance().height() >> i;
    int pyr_cols = Resolution::getInstance().width() >> i;

    depth_tmp[i].create(pyr_rows, pyr_cols);
    mask_tmp[i].create(pyr_rows, pyr_cols);
  }
}

#ifdef MULTIMOTIONFUSION_NUM_SURFELS
const int Model::TEXTURE_DIMENSION = 32 * (int)(sqrt(MULTIMOTIONFUSION_NUM_SURFELS) / 32);
#else
const int Model::TEXTURE_DIMENSION = 1024;
#endif

const int Model::MAX_VERTICES = Model::TEXTURE_DIMENSION * Model::TEXTURE_DIMENSION;
const int Model::NODE_TEXTURE_DIMENSION = 16384;
const int Model::MAX_NODES = Model::NODE_TEXTURE_DIMENSION / 16;  // 16 floats per node

const int Model::bufferSize = Model::MAX_VERTICES * Vertex::SIZE;

GPUTexture Model::deformationNodes = GPUTexture(NODE_TEXTURE_DIMENSION, 1, GL_LUMINANCE32F_ARB, GL_LUMINANCE, GL_FLOAT);

tracker::KeypointPtr project_kp(const tracker::KeypointPtr &origin_kp, const Eigen::Isometry3d &T)
{
  if (origin_kp==nullptr)
    return nullptr;

  // project keypoint from origin frame (camera) to local model frame
  return  std::make_shared<tracker::Keypoint>(tracker::Keypoint{
                            origin_kp->timestamp,
                            origin_kp->xy,
                            T * origin_kp->coordinate.transpose(),
                            origin_kp->descriptor});
};

static bool point_inside(const cv::Point &point, const cv::Mat &img) {
  return point.inside({{}, img.size()});
}

Model::Model(unsigned char id, float confidenceThresh, const OdometryConfig &odom_cfg, bool enableFillIn, bool enableErrorRecording, bool enablePoseLogging,
             MatchingType matchingType, float maxDepthThesh)
    : pose(Eigen::Matrix4f::Identity()),
      lastPose(Eigen::Matrix4f::Identity()),
      confidenceThreshold(confidenceThresh),
      maxDepth(maxDepthThesh),
      target(0),
      renderSource(1),
      count(0),
      id(id),
      icpError(enableErrorRecording
                   ? std::make_unique<GPUTexture>(Resolution::getInstance().width(), Resolution::getInstance().height(), GL_R32F, GL_RED,
                                                  GL_FLOAT, true, true, cudaGraphicsRegisterFlagsSurfaceLoadStore, "ICP")
                   : nullptr),
      rgbError(enableErrorRecording ? std::make_unique<GPUTexture>(Resolution::getInstance().width(), Resolution::getInstance().height(), GL_R32F, GL_RED, GL_FLOAT, true, true, cudaGraphicsRegisterFlagsSurfaceLoadStore, "RGB") : nullptr),  // FIXME
      gpu(Model::GPUSetup::getInstance()),
      frameToModel(Resolution::getInstance().width(), Resolution::getInstance().height(), Intrinsics::getInstance().cx(),
                   Intrinsics::getInstance().cy(), Intrinsics::getInstance().fx(), Intrinsics::getInstance().fy(), id),
      fillIn(enableFillIn ? std::make_unique<FillIn>() : nullptr) {
  switch (matchingType) {
    case MatchingType::Drost:
      // removed
      break;
  }

  if (enablePoseLogging) poseLog.reserve(1000);

  float* vertices = new float[bufferSize];

  memset(&vertices[0], 0, bufferSize);

  glGenTransformFeedbacks(1, &vbos[0].stateObject);
  glGenBuffers(1, &vbos[0].dataBuffer);
  glBindBuffer(GL_ARRAY_BUFFER, vbos[0].dataBuffer);
  glBufferData(GL_ARRAY_BUFFER, bufferSize, &vertices[0], GL_STREAM_DRAW);
  glBindBuffer(GL_ARRAY_BUFFER, 0);

  glGenTransformFeedbacks(1, &vbos[1].stateObject);
  glGenBuffers(1, &vbos[1].dataBuffer);
  glBindBuffer(GL_ARRAY_BUFFER, vbos[1].dataBuffer);
  glBufferData(GL_ARRAY_BUFFER, bufferSize, &vertices[0], GL_STREAM_DRAW);
  glBindBuffer(GL_ARRAY_BUFFER, 0);

  delete[] vertices;

  vertices = new float[Resolution::getInstance().numPixels() * Vertex::SIZE];

  memset(&vertices[0], 0, Resolution::getInstance().numPixels() * Vertex::SIZE);

  glGenTransformFeedbacks(1, &newUnstableBuffer.stateObject);
  glGenBuffers(1, &newUnstableBuffer.dataBuffer);
  glBindBuffer(GL_ARRAY_BUFFER, newUnstableBuffer.dataBuffer);
  glBufferData(GL_ARRAY_BUFFER, Resolution::getInstance().numPixels() * Vertex::SIZE, &vertices[0], GL_STREAM_DRAW);
  glBindBuffer(GL_ARRAY_BUFFER, 0);

  delete[] vertices;

  std::vector<Eigen::Vector2f> uv;

  for (int i = 0; i < Resolution::getInstance().width(); i++)
    for (int j = 0; j < Resolution::getInstance().height(); j++)
      uv.push_back(
          Eigen::Vector2f(((float)i / (float)Resolution::getInstance().width()) + 1.0 / (2 * (float)Resolution::getInstance().width()),
                          ((float)j / (float)Resolution::getInstance().height()) + 1.0 / (2 * (float)Resolution::getInstance().height())));

  uvSize = uv.size();

  glGenBuffers(1, &uvo);
  glBindBuffer(GL_ARRAY_BUFFER, uvo);
  glBufferData(GL_ARRAY_BUFFER, uvSize * sizeof(Eigen::Vector2f), &uv[0], GL_STATIC_DRAW);
  glBindBuffer(GL_ARRAY_BUFFER, 0);

  gpu.initProgram->Bind();
  glGenQueries(1, &countQuery);

  // Empty both transform feedbacks
  glEnable(GL_RASTERIZER_DISCARD);

  glBindTransformFeedback(GL_TRANSFORM_FEEDBACK, vbos[0].stateObject);
  glBindBufferBase(GL_TRANSFORM_FEEDBACK_BUFFER, 0, vbos[0].dataBuffer);

  glBeginTransformFeedback(GL_POINTS);
  glDrawArrays(GL_POINTS, 0, 0);  // RUN GPU-PASS
  glEndTransformFeedback();

  glBindTransformFeedback(GL_TRANSFORM_FEEDBACK, 0);

  glBindTransformFeedback(GL_TRANSFORM_FEEDBACK, vbos[1].stateObject);
  glBindBufferBase(GL_TRANSFORM_FEEDBACK_BUFFER, 0, vbos[1].dataBuffer);

  glBeginTransformFeedback(GL_POINTS);

  glDrawArrays(GL_POINTS, 0, 0);  // RUN GPU-PASS

  glEndTransformFeedback();

  glDisable(GL_RASTERIZER_DISCARD);

  glBindTransformFeedback(GL_TRANSFORM_FEEDBACK, 0);

  gpu.initProgram->Unbind();

  std::cout << "Created model with max number of vertices: " << Model::MAX_VERTICES << std::endl;
}

Model::~Model() {
  glDeleteBuffers(1, &vbos[0].dataBuffer);
  glDeleteTransformFeedbacks(1, &vbos[0].stateObject);

  glDeleteBuffers(1, &vbos[1].dataBuffer);
  glDeleteTransformFeedbacks(1, &vbos[1].stateObject);

  glDeleteQueries(1, &countQuery);

  glDeleteBuffers(1, &uvo);

  glDeleteTransformFeedbacks(1, &newUnstableBuffer.stateObject);
  glDeleteBuffers(1, &newUnstableBuffer.dataBuffer);
}

void Model::initialise(const FeedbackBuffer& rawFeedback, const FeedbackBuffer& filteredFeedback) {
  gpu.initProgram->Bind();

  glBindBuffer(GL_ARRAY_BUFFER, rawFeedback.vbo);

  glEnableVertexAttribArray(0);
  glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, Vertex::SIZE, 0);

  glEnableVertexAttribArray(1);
  glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, Vertex::SIZE, reinterpret_cast<GLvoid*>(sizeof(Eigen::Vector4f)));

  glBindBuffer(GL_ARRAY_BUFFER, filteredFeedback.vbo);

  glEnableVertexAttribArray(2);
  glVertexAttribPointer(2, 4, GL_FLOAT, GL_FALSE, Vertex::SIZE, reinterpret_cast<GLvoid*>(sizeof(Eigen::Vector4f) * 2));

  glEnable(GL_RASTERIZER_DISCARD);

  glBindTransformFeedback(GL_TRANSFORM_FEEDBACK, vbos[target].stateObject);
  glBindBufferBase(GL_TRANSFORM_FEEDBACK_BUFFER, 0, vbos[target].dataBuffer);

  glBeginTransformFeedback(GL_POINTS);

  glBeginQuery(GL_TRANSFORM_FEEDBACK_PRIMITIVES_WRITTEN, countQuery);

  // It's ok to use either fid because both raw and filtered have the same amount of vertices
  glDrawTransformFeedback(GL_POINTS, rawFeedback.fid);  // RUN GPU-PASS

  glEndTransformFeedback();

  glEndQuery(GL_TRANSFORM_FEEDBACK_PRIMITIVES_WRITTEN);

  glGetQueryObjectuiv(countQuery, GL_QUERY_RESULT, &count);

  glDisable(GL_RASTERIZER_DISCARD);

  glDisableVertexAttribArray(0);
  glDisableVertexAttribArray(1);
  glDisableVertexAttribArray(2);
  glBindBuffer(GL_ARRAY_BUFFER, 0);
  glBindTransformFeedback(GL_TRANSFORM_FEEDBACK, 0);

  gpu.initProgram->Unbind();

  glFinish();
}

void Model::renderPointCloud(const Eigen::Matrix4f& mvp, const bool drawUnstable, const bool drawPoints, const bool drawWindow,
                             const int colorType, const int time, const int timeDelta) {
  std::shared_ptr<Shader> program = drawPoints ? gpu.drawProgram : gpu.drawSurfelProgram;

  program->Bind();

  // Eigen::Matrix4f mvp = vp;
  // if(id != 0) mvp =  mvp * pose.inverse();

  program->setUniform(Uniform("MVP", mvp));
  program->setUniform(Uniform("threshold", getConfidenceThreshold()));
  program->setUniform(Uniform("colorType", colorType));
  program->setUniform(Uniform("unstable", drawUnstable));
  program->setUniform(Uniform("drawWindow", drawWindow));
  program->setUniform(Uniform("time", time));
  program->setUniform(Uniform("timeDelta", timeDelta));
  program->setUniform(Uniform("maskID", id));

  // This is for the point shader
  program->setUniform(Uniform("pose", (Eigen::Matrix4f)Eigen::Matrix4f::Identity()));
  // TODO

  glBindBuffer(GL_ARRAY_BUFFER, vbos[target].dataBuffer);

  glEnableVertexAttribArray(0);
  glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, Vertex::SIZE, 0);

  glEnableVertexAttribArray(1);
  glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, Vertex::SIZE, reinterpret_cast<GLvoid*>(sizeof(Eigen::Vector4f) * 1));

  glEnableVertexAttribArray(2);
  glVertexAttribPointer(2, 4, GL_FLOAT, GL_FALSE, Vertex::SIZE, reinterpret_cast<GLvoid*>(sizeof(Eigen::Vector4f) * 2));

  glDrawTransformFeedback(GL_POINTS, vbos[target].stateObject);  // RUN GPU-PASS

  glDisableVertexAttribArray(0);
  glDisableVertexAttribArray(1);
  glDisableVertexAttribArray(2);
  glBindBuffer(GL_ARRAY_BUFFER, 0);

  program->Unbind();
}

const OutputBuffer& Model::getModel() { return vbos[target]; }

void Model::generateCUDATextures(GPUTexture* depth, GPUTexture* mask) {
  GPUSetup& gpu = GPUSetup::getInstance();
  std::vector<DeviceArray2D<float>>& depthPyr = gpu.depth_tmp;
  std::vector<DeviceArray2D<unsigned char>>& maskPyr = gpu.mask_tmp;

  cudaCheckError();

  depth->cudaMap();
  cudaArray* depthTexturePtr = depth->getCudaArray();
  cudaMemcpy2DFromArray(depthPyr[0].ptr(0), depthPyr[0].step(), depthTexturePtr, 0, 0, depthPyr[0].colsBytes(), depthPyr[0].rows(),
                        cudaMemcpyDeviceToDevice);
  depth->cudaUnmap();

  mask->cudaMap();
  cudaArray* maskTexturePtr = mask->getCudaArray();
  cudaMemcpy2DFromArray(maskPyr[0].ptr(0), maskPyr[0].step(), maskTexturePtr, 0, 0, maskPyr[0].colsBytes(), maskPyr[0].rows(),
                        cudaMemcpyDeviceToDevice);
  mask->cudaUnmap();

  cudaDeviceSynchronize();
  cudaCheckError();

  for (int i = 1; i < RGBDOdometry::NUM_PYRS; ++i) {
    pyrDownGaussF(depthPyr[i - 1], depthPyr[i]);
    pyrDownUcharGauss(maskPyr[i - 1], maskPyr[i]);  // FIXME Better filter
    // TODO Execute in parralel (two cuda streams)
  }
  cudaDeviceSynchronize();
  cudaCheckError();
}

void Model::initICP(bool doFillIn, bool frameToFrameRGB, float depthCutoff, GPUTexture* rgb) {
  TICK("odomInit - Model: " + std::to_string(id));

  // WARNING initICP* must be called before initRGB*
  if (doFillIn) {
    frameToModel.initICPModel(getFillInVertexTexture(), getFillInNormalTexture(), depthCutoff, getPose());
    frameToModel.initRGBModel(getFillInImageTexture());
  } else {
    frameToModel.initICPModel(getVertexConfProjection(), getNormalProjection(), depthCutoff, getPose());
    frameToModel.initRGBModel(frameToFrameRGB && allowsFillIn() ? getFillInImageTexture() : getRGBProjection());
  }

  // frameToModel.initICP(filteredDepth, depthCutoff, mask);
  frameToModel.initICP(gpu.depth_tmp, gpu.mask_tmp, depthCutoff);
  frameToModel.initRGB(rgb);

  TOCK("odomInit - Model: " + std::to_string(id));
}

void Model::performTracking(bool frameToFrameRGB, bool rgbOnly, float icpWeight, bool pyramid, bool fastOdom, bool so3,
                            float maxDepthProcessed, GPUTexture* rgb, int64_t logTimestamp, bool doFillIn) {
  assert(fillIn || !doFillIn);
  lastPose = pose;

  // TODO Allow fillIn again
  // move "old" next to last keypoints and images
  initICP(doFillIn, frameToFrameRGB, maxDepthProcessed, rgb);  // TODO: Don't copy RGB

  TICK("odom - Model: " + std::to_string(id));

  Eigen::Vector3f transObject = pose.topRightCorner(3, 1);
  Eigen::Matrix<float, 3, 3, Eigen::RowMajor> rotObject = pose.topLeftCorner(3, 3);

  getFrameOdometry().getIncrementalTransformation(transObject, rotObject, rgbOnly, icpWeight, pyramid, fastOdom, so3,
                                                  icpError->getCudaSurface(), rgbError->getCudaSurface());

  pose.topRightCorner(3, 1) = transObject;
  pose.topLeftCorner(3, 3) = rotObject;

  timestamp_ns.push_back(logTimestamp);
  poses.emplace_back(pose);

  TOCK("odom - Model: " + std::to_string(id));
}

std::tuple<Eigen::MatrixXd, Model::MatrixXp2, Model::MatrixXp3>
Model::computeTrackProjectionError(const tracker::Tracks &tracks) {
  // L2 distances of local points to previous point, Ntracks x Nimages
  Eigen::MatrixXd track_pe(int(tracks.size()), 0);
  MatrixXp2 track_xy(int(tracks.size()), 0);
  MatrixXp3 track_p(int(tracks.size()), 0);

  int it=0;
  for (const tracker::TrackPtr &track : tracks) {
    // resize the projection error matrix, assumes that all tracks have same length
    const int len_dist = int(track->size()-1);
    if (track_pe.cols() != len_dist) {
      track_pe.resize(Eigen::NoChange, len_dist);
    }

    for (int ik=0; ik<len_dist; ik++) {
      if ((*track)[size_t(ik)] != nullptr && (*track)[size_t(ik+1)] != nullptr) {
        track_pe(it,ik) = ((*track)[size_t(ik+1)]->coordinate - (*track)[size_t(ik)]->coordinate).norm();
      }
      else {
        track_pe(it,ik) = std::numeric_limits<double>::quiet_NaN();
      }
    }

    if (track_xy.cols() != int(track->size())) {
      track_xy.resize(Eigen::NoChange, int(track->size()));
    }
    if (track_p.cols() != int(track->size())) {
      track_p.resize(Eigen::NoChange, int(track->size()));
    }
    for (size_t ik=0; ik<track->size(); ik++) {
      track_xy(it,int(ik)) = ((*track)[ik] == nullptr) ? cv::Point(-1,-1) : (*track)[ik]->xy;
      track_p(it,int(ik)) = ((*track)[ik] == nullptr) ? Eigen::RowVector3d::Constant(std::numeric_limits<double>::quiet_NaN()) : (*track)[ik]->coordinate;
    }

    it++;
  } // tracks

  return {track_pe, track_xy, track_p};
}

tracker::Tracks Model::computeTrackProjectionLastFrame(const tracker::Tracks& tracks, const size_t length) const {
  assert(!poses.empty());
  tracker::Tracks ltracks(tracks.size()); // local tracks

  const size_t len_vis = (length==0) ? poses.size() : std::min(length, poses.size());

  // camera intrinsics
  const Eigen::Array2d c(Intrinsics::getInstance().cx(), Intrinsics::getInstance().cy());
  const Eigen::Array2d f(Intrinsics::getInstance().fx(), Intrinsics::getInstance().fy());

  for (size_t it=0; it<tracks.size(); it++) {
    ltracks[it] = std::make_shared<tracker::Track>();
    for (size_t ip=poses.size()-len_vis; ip<poses.size(); ip++) {
      const size_t id = tracks[it]->size() - poses.size() + ip;

      // project 3D keypoints from previous frames into current frame
      tracker::KeypointPtr kp = project_kp((*tracks[it])[id], (poses.at(ip)*Eigen::Isometry3f(pose.inverse())).cast<double>());

      if (kp) {
        // project onto 2D image plane
        const Eigen::Array3d p3 = kp->coordinate;
        const Eigen::Vector2i p2 = (c + (p3 / p3.z()).head<2>() * f).array().round().cast<int>();
        kp->xy = {p2.x(), p2.y()};
      }

      ltracks[it]->push_back(kp);
    }
  }

  return ltracks;
}

tracker::Tracks Model::computeTrackProjectionFirstFrame() const {
  assert(poses.size() == timestamp_ns.size());
  tracker::Tracks local_tracks;
  for (const tracker::TrackPtr &track : tracks) {
    if (track->empty())
      continue;
    local_tracks.push_back(std::make_shared<tracker::Track>(poses.size(), nullptr));
    assert(track->size() >= poses.size());
    const size_t offset = track->size() - poses.size();
    for (size_t ip=0; ip<poses.size(); ip++) {
      (*local_tracks.back())[ip] = project_kp((*track)[offset+ip], poses.at(ip).cast<double>());
    }
  }
  return local_tracks;
}

tracker::Tracks Model::computeTrackProjectionStartEnd(const tracker::Tracks& tracks, const size_t length) {
  assert(!poses.empty());

  const size_t len_vis = (length==0) ? poses.size() : std::min(length, poses.size());

  // camera intrinsics
  const Eigen::Array2d c(Intrinsics::getInstance().cx(), Intrinsics::getInstance().cy());
  const Eigen::Array2d f(Intrinsics::getInstance().fx(), Intrinsics::getInstance().fy());

  // start and end points in camera frame
  Eigen::MatrixX3d coordinates_start(tracks.size(), 3);
  Eigen::MatrixX3d coordinates_end(tracks.size(), 3);
  coordinates_start.setConstant(std::numeric_limits<double>::signaling_NaN());
  coordinates_end.setConstant(std::numeric_limits<double>::signaling_NaN());

  std::vector<uint64_t> ts_start(tracks.size(), 0);
  std::vector<uint64_t> ts_end(tracks.size(), 0);
  for (size_t i = 0; i < tracks.size(); ++i) {
    const tracker::KeypointPtr &kp0 = (*(tracks[i]->end()-len_vis));
    const tracker::KeypointPtr &kp1 = tracks[i]->back();
    if (kp0!=nullptr && kp0->coordinate.array().isFinite().all()) {
      coordinates_start.row(i) = kp0->coordinate;
      ts_start[i] = kp0->timestamp;
    }

    if (kp1!=nullptr && kp1->coordinate.array().isFinite().all()) {
      coordinates_end.row(i) = kp1->coordinate;
      ts_end[i] = kp1->timestamp;
    }
  }

  // transform to local model frames
  coordinates_start = (((*(poses.end()-len_vis)*Eigen::Isometry3f(pose.inverse())).cast<double>()) * coordinates_start.transpose()).transpose();
  coordinates_end = (((poses.back()*Eigen::Isometry3f(pose.inverse())).cast<double>()) * coordinates_end.transpose()).transpose();

  // project to image plane
  auto proj = [&c,&f](const Eigen::MatrixX3d &points) -> Eigen::MatrixX2d {
    return ((points.leftCols<2>().array().colwise() / points.rightCols<1>().array()).leftCols<2>().rowwise() * f.transpose()).rowwise() + c.transpose();
  };

  const Eigen::MatrixX2i x_start = proj(coordinates_start).array().round().cast<int>();
  const Eigen::MatrixX2i x_end = proj(coordinates_end).array().round().cast<int>();

  auto make_kp = [](const uint64_t timestamp, const Eigen::RowVector2i &xy, const Eigen::RowVector3d &coordinate) -> tracker::KeypointPtr {
    if (timestamp==0) { return nullptr; }
    return std::make_shared<tracker::Keypoint>(tracker::Keypoint{.timestamp = timestamp, .xy = {xy.x(), xy.y()}, .coordinate = coordinate});
  };

  tracker::Tracks ltracks(tracks.size()); // local tracks
  for (size_t i = 0; i < ltracks.size(); ++i) {
    ltracks[i] = std::make_shared<tracker::Track>();
    ltracks[i]->push_back(make_kp(ts_start[i], x_start.row(i), coordinates_start.row(i)));
    ltracks[i]->push_back(make_kp(ts_end[i], x_end.row(i), coordinates_end.row(i)));
  }

  return ltracks;
}

cv::Mat Model::drawLocalTracks2D(const tracker::Tracks &tracks, const cv::Mat &img, int msize, bool mscale) {
  cv::Mat img_tracks;
  cv::cvtColor(img, img_tracks, cv::COLOR_RGB2GRAY);
  cv::cvtColor(img_tracks, img_tracks, cv::COLOR_GRAY2RGB);
  std::uniform_real_distribution<double> u(0,1);
  std::default_random_engine g;
  size_t i = 0;
  for (const tracker::TrackPtr &track : tracks) {
    // unique colour
    g.seed(i++);
    const cv::viz::Color c(u(g)*255, u(g)*255, u(g)*255);
    tracker::KeypointPtr kp_start;
    tracker::KeypointPtr prev_kp;
    for (const tracker::KeypointPtr &kp : *track) {
      if (kp && point_inside(kp->xy, img)) {
        if (prev_kp && point_inside(prev_kp->xy, img)) {
          cv::line(img_tracks, kp->xy, prev_kp->xy, c, 2);
        }
        else {
          kp_start = kp;
        }
      }
      prev_kp = kp;
    }

    if (!(kp_start && prev_kp) || (kp_start == prev_kp))
      continue;

    // scale marker size if given in in pxl/s
    const float duration = mscale  ? abs((prev_kp->timestamp - kp_start->timestamp) * 1e-9) : 1;
    // start marker
    cv::circle(img_tracks, kp_start->xy, msize * duration, c, 1, cv::LINE_AA);
    // end marker
    cv::drawMarker(img_tracks, prev_kp->xy, c, cv::MARKER_TILTED_CROSS, msize * duration, 1, cv::LINE_AA);
  }
  return img_tracks;
}

void Model::initGlobalTracks(const tracker::Tracks& tracks, const Eigen::Isometry3f &initial_pose, const uint64_t &time) {
  assert(poses.size()==0);
  assert(this->tracks.size()==0);

  this->tracks = {tracks.begin(), tracks.end()};

  timestamp_ns.push_back(time);
  poses.emplace_back(initial_pose);
}

void Model::updateTracks(const tracker::Tracks& tracks_add, const tracker::Tracks& tracks_remove) {
  assert(!poses.empty());

  // add new inlier tracks with new pose estimates
  this->tracks.insert(tracks_add.begin(), tracks_add.end());

  // remove outlier tracks
  for (const tracker::TrackPtr &track : tracks_remove) {
    this->tracks.erase(track);
  }
}

void Model::removeLastTrackKeypoint() {
  for (const tracker::TrackPtr &track : tracks_local) {
    if (!track->empty())
      track->pop_back();
  }
}

void Model::refineTrackSubset(const tracker::Tracks& tracks, const ModelPointer &parent, const size_t &history) {
  if (tracks.empty()) { return; }

  // pose to 7D (px, py, pz, qx, qy, qz, qw)
  auto pose_7d = [](const Eigen::Isometry3f &pose) -> Eigen::Matrix<float, 7, 1> {
    Eigen::Matrix<float, 7, 1> p7d;
    // x, y, z
    p7d.head<3>() = pose.translation();
    // x, y, z, w
    p7d.tail<4>() << Eigen::Quaternionf(pose.rotation()).coeffs();
    return p7d;
  };

  // apply RANSAC on every set of track segments
  // model must have 60% of samples within 3cm error
  // this assumes that the 'tracks' are already associated to the model via segments
  RigidRANSAC rrs(10, 0.03f, 0.6f);

  const size_t len = std::min((*tracks.begin())->size(), history);
  // branch index
  const size_t end = parent->poses.size()-1;
  // point to which estimate the poses of new object
  const size_t start = end-len+1;
  timestamp_ns.resize(len);
  poses.resize(len);
  poses[0].setIdentity();
  if (isLoggingPoses()) {
    poseLog.push_back(parent->poseLog[start]);
    timestamp_ns[0] = parent->poseLog[start].ts;
  }

  const size_t ntracks = tracks.size();
  for (size_t ik=0, jk=1; jk < len; jk++) {
    Eigen::MatrixX3f p0s, p1s;
    p0s.resize(int(ntracks), Eigen::NoChange);
    p1s.resize(int(ntracks), Eigen::NoChange);

    int nvalid = 0;
    uint64_t t1 = 0; // timestamp
    for (const tracker::TrackPtr &track : tracks) {
      if ((*track)[start+ik] && (*track)[start+jk]) {
        t1 = track->at(start+jk)->timestamp;
        const Eigen::RowVector3d &p0 = track->at(start+ik)->coordinate;
        const Eigen::RowVector3d &p1 = track->at(start+jk)->coordinate;
        if (p0.array().isFinite().all() && p1.array().isFinite().all()) {
          p0s.row(nvalid) = p0.cast<float>();
          p1s.row(nvalid) = p1.cast<float>();
          nvalid++;
        }
      }
    }
    p0s.conservativeResize(nvalid, Eigen::NoChange);
    p1s.conservativeResize(nvalid, Eigen::NoChange);

    timestamp_ns[jk] = t1;

    // skip to the next frame if there are not enough correspondences
    if (nvalid<3) {
      poses[jk] = poses.at(ik);
      if (isLoggingPoses()) {
        const Eigen::Isometry3f Two = Eigen::Isometry3f(parent->getPose()) * poses[jk].inverse();
        poseLog.push_back({int64_t(t1), pose_7d(Two)});
      }
      continue;
    }

    // least squares estimate
    const Eigen::Isometry3f T_01 = rrs.estimate(p0s, p1s).transformation;
    assert(T_01.matrix().array().isFinite().all());
    poses[jk] = poses.at(ik) * T_01;
    if (isLoggingPoses()) {
      const Eigen::Isometry3f Two = Eigen::Isometry3f(parent->getPose()) * poses[jk].inverse();
      poseLog.push_back({int64_t(t1), pose_7d(Two)});
    }

    ik = jk;
  }

  // transform the pose trajectory such that the end (current time) is at origin
  for (Eigen::Isometry3f &p : poses) {
    p = poses.back().inverse() * p;
  }

  // the new 'initial' pose is the pose at the end of the track
  overridePose(poses.back().matrix()); // this should be close to identity

  // the 'poseLog' is extended later via 'pose', remove the last log again
  poseLog.pop_back();
}

RigidRANSAC::Result Model::getLastTrackTransform(const tracker::Tracks &tracks,
                                                 const RigidRANSAC::Config &config) {
  const size_t ntracks = tracks.size();
  Eigen::MatrixX3f p0s, p1s;
  p0s.resize(int(ntracks), Eigen::NoChange);
  p1s.resize(int(ntracks), Eigen::NoChange);

  int nvalid = 0;
  for (const tracker::TrackPtr &track : tracks) {
    if (track->empty())
      continue;
    tracker::KeypointPtr kp0 = track->end()[-2];
    tracker::KeypointPtr kp1 = track->end()[-1];
    if (kp0 && kp1) {
      const Eigen::RowVector3d &p0 = kp0->coordinate;
      const Eigen::RowVector3d &p1 = kp1->coordinate;
      if (p0.array().isFinite().all() && p1.array().isFinite().all()) {
        p0s.row(nvalid) = p0.cast<float>();
        p1s.row(nvalid) = p1.cast<float>();
        nvalid++;
      }
    }
  }
  p0s.conservativeResize(nvalid, Eigen::NoChange);
  p1s.conservativeResize(nvalid, Eigen::NoChange);

  // skip to the next frame if there are not enough correspondences
  if (nvalid<3) {
    return {.transformation = Eigen::Isometry3f::Identity()};
  }

  // least squares estimate
  RigidRANSAC rrs(config);
  const RigidRANSAC::Result res = rrs.estimate(p0s, p1s);
  assert(res.transformation.matrix().array().isFinite().all());
  return res;
}

RigidRANSAC::Result Model::getLastTrackTransform() const {
  return Model::getLastTrackTransform({this->tracks.begin(), this->tracks.end()});
}

RigidRANSAC::Result Model::getBestMatch(const std::vector<tracker::KeypointPtr> &keypoints, const RigidRANSAC::Config &config) const {
  // no previously stored model
  if (tracks_local.empty()) {
    std::cout << "model " << getID() << " has no stored tracks" << std::endl;
    return {};
  }

  const size_t nd = keypoints.front()->descriptor.size();

  cv::Mat_<float> query(keypoints.size(), nd);
  for (size_t i=0; i<keypoints.size(); i++) {
    cv::eigen2cv(keypoints[i]->descriptor, query.row(i));
  }

  std::vector<Eigen::MatrixXf> model_descriptors(tracks_local.front()->size());
  std::vector<Eigen::MatrixX3f> model_coordinates(tracks_local.front()->size());

  for (size_t i=0; i<tracks_local.front()->size(); i++) {
    size_t nkp_valid = 0;
    for (size_t j=0; j<tracks_local.size(); j++) {
      if (model_descriptors[i].size()==0) {
        model_descriptors[i].resize(tracks_local.size(), nd);
        model_coordinates[i].resize(tracks_local.size(), Eigen::NoChange);
      }

      const tracker::KeypointPtr &kpi = (*tracks_local[j])[i];
      if (kpi && kpi->coordinate.allFinite()) {
        model_descriptors[i].row(nkp_valid) = kpi->descriptor.cast<float>();
        model_coordinates[i].row(nkp_valid) = kpi->coordinate.cast<float>();
        nkp_valid++;
      }
    }

    // only keep valid points
    model_descriptors[i].conservativeResize(nkp_valid, Eigen::NoChange);
  }

  // convert Eigen to OpenCV matrix, skip time indices with empty data
  std::vector<cv::Mat_<float>> model_descriptors_cv;
  // map matches to time indices
  std::unordered_map<size_t, size_t> match_ids;
  size_t match_id = 0;
  for (size_t i=0; i<model_descriptors.size(); i++) {
    if (model_descriptors[i].rows()>0) {
      model_descriptors_cv.emplace_back();
      cv::eigen2cv(model_descriptors[i], model_descriptors_cv.back());
      match_ids[match_id] = i;
      match_id++;
    }
  }

  // pairwise matching of query discriptors from current model/segment
  // against all views of previous models
  std::unordered_map<size_t, std::list<std::tuple<size_t, size_t>>> matches_views;
  for (size_t i=0; i<model_descriptors_cv.size(); i++) {
    cv::BFMatcher matcher(cv::NORM_L2, true);
    std::vector<cv::DMatch> matches;
    matcher.match(query, model_descriptors_cv[i], matches);
    if (matches.size() >= 3) {
      for (const cv::DMatch &match : matches) {
        matches_views[i].push_back({match.queryIdx, match.trainIdx});
      }
    }
  }

  // RANSAC on all matches
  RigidRANSAC ransac(config);
  std::vector<RigidRANSAC::Result> estimates;
  for (const auto &[id_view, matches] : matches_views) {
    Eigen::MatrixX3f query(matches.size(), 3);
    Eigen::MatrixX3f train(matches.size(), 3);
    size_t imatch = 0;
    for (const auto &[id_query, id_train] : matches) {
      query.row(imatch) = keypoints[id_query]->coordinate.cast<float>();
      train.row(imatch) = model_coordinates[match_ids.at(id_view)].row(id_train);
      imatch++;
    }
    const RigidRANSAC::Result estimate = ransac.estimate(query, train);
    if (estimate.inlier.count() > 0) {
      estimates.push_back(estimate);
    }
  }

  if (estimates.empty()) {
    // no matching candidates
    return {};
  }

  // find estimate with smallest error
  auto min_ransac = [](const RigidRANSAC::Result &a, const RigidRANSAC::Result &b) {
    return a.error < b.error;
  };
  return *std::min_element(estimates.begin(), estimates.end(), min_ransac);
}

float Model::computeFusionWeight(float weightMultiplier) const {
  Eigen::Matrix4f diff = getLastTransform();
  Eigen::Vector3f diffTrans = diff.topRightCorner(3, 1);
  Eigen::Matrix3f diffRot = diff.topLeftCorner(3, 3);

  float weighting = std::max(diffTrans.norm(), rodrigues2(diffRot).norm());

  const float largest = 0.01;
  const float minWeight = 0.5;

  if (weighting > largest) weighting = largest;

  weighting = std::max(1.0f - (weighting / largest), minWeight) * weightMultiplier;

  return weighting;
}

void Model::fuse(const int& time, GPUTexture* rgb, GPUTexture* mask, GPUTexture* depthRaw, GPUTexture* depthFiltered,
                 const float depthCutoff, const float weightMultiplier) {
  TICK("Fuse::Data");
  // This first part does data association and computes the vertex to merge with, storing
  // in an array that sets which vertices to update by index
  gpu.frameBuffer.Bind();

  glPushAttrib(GL_VIEWPORT_BIT);

  glViewport(0, 0, gpu.renderBuffer.width, gpu.renderBuffer.height);

  glClearColor(0, 0, 0, 0);
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

  // PROGRAM1: Data association
  gpu.dataProgram->Bind();
  gpu.dataProgram->setUniform(Uniform("cSampler", 0));
  gpu.dataProgram->setUniform(Uniform("drSampler", 1));
  gpu.dataProgram->setUniform(Uniform("drfSampler", 2));
  gpu.dataProgram->setUniform(Uniform("indexSampler", 3));
  gpu.dataProgram->setUniform(Uniform("vertConfSampler", 4));
  gpu.dataProgram->setUniform(Uniform("colorTimeSampler", 5));
  gpu.dataProgram->setUniform(Uniform("normRadSampler", 6));
  gpu.dataProgram->setUniform(Uniform("maskSampler", 7));
  gpu.dataProgram->setUniform(Uniform("time", (float)time));
  gpu.dataProgram->setUniform(Uniform("weighting", computeFusionWeight(weightMultiplier)));
  gpu.dataProgram->setUniform(Uniform("maskID", id));

  gpu.dataProgram->setUniform(Uniform("cam", Eigen::Vector4f(Intrinsics::getInstance().cx(), Intrinsics::getInstance().cy(),
                                                             1.0 / Intrinsics::getInstance().fx(), 1.0 / Intrinsics::getInstance().fy())));
  gpu.dataProgram->setUniform(Uniform("cols", (float)Resolution::getInstance().cols()));
  gpu.dataProgram->setUniform(Uniform("rows", (float)Resolution::getInstance().rows()));
  gpu.dataProgram->setUniform(Uniform("scale", (float)ModelProjection::FACTOR));
  gpu.dataProgram->setUniform(Uniform("texDim", (float)TEXTURE_DIMENSION));
  gpu.dataProgram->setUniform(Uniform("pose", pose));
  gpu.dataProgram->setUniform(Uniform("maxDepth", std::min(depthCutoff, maxDepth)));

  glEnableVertexAttribArray(0);
  glBindBuffer(GL_ARRAY_BUFFER, uvo);
  glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, 0);

  glBindTransformFeedback(GL_TRANSFORM_FEEDBACK, newUnstableBuffer.stateObject);
  glBindBufferBase(GL_TRANSFORM_FEEDBACK_BUFFER, 0, newUnstableBuffer.dataBuffer);

  glActiveTexture(GL_TEXTURE0);
  glBindTexture(GL_TEXTURE_2D, rgb->texture->tid);

  glActiveTexture(GL_TEXTURE1);
  glBindTexture(GL_TEXTURE_2D, depthRaw->texture->tid);

  glActiveTexture(GL_TEXTURE2);
  glBindTexture(GL_TEXTURE_2D, depthFiltered->texture->tid);

  glActiveTexture(GL_TEXTURE3);
  glBindTexture(GL_TEXTURE_2D, indexMap.getSparseIndexTex()->texture->tid);

  glActiveTexture(GL_TEXTURE4);
  glBindTexture(GL_TEXTURE_2D, indexMap.getSparseVertConfTex()->texture->tid);

  glActiveTexture(GL_TEXTURE5);
  glBindTexture(GL_TEXTURE_2D, indexMap.getSparseColorTimeTex()->texture->tid);

  glActiveTexture(GL_TEXTURE6);
  glBindTexture(GL_TEXTURE_2D, indexMap.getSparseNormalRadTex()->texture->tid);

  glActiveTexture(GL_TEXTURE7);
  glBindTexture(GL_TEXTURE_2D, mask->texture->tid);

  glBeginTransformFeedback(GL_POINTS);

  glDrawArrays(GL_POINTS, 0, uvSize);  // RUN GPU-PASS

  glEndTransformFeedback();

  gpu.frameBuffer.Unbind();

  glBindTexture(GL_TEXTURE_2D, 0);

  glActiveTexture(GL_TEXTURE0);

  glDisableVertexAttribArray(0);
  glBindBuffer(GL_ARRAY_BUFFER, 0);
  glBindTransformFeedback(GL_TRANSFORM_FEEDBACK, 0);

  gpu.dataProgram->Unbind();

  glPopAttrib();

  glFinish();
  TOCK("Fuse::Data");

  TICK("Fuse::Update");
  // Next we update the vertices at the indexes stored in the update textures
  // Using a transform feedback conditional on a texture sample

  // PROGRAM2: Fusion
  gpu.updateProgram->Bind();

  gpu.updateProgram->setUniform(Uniform("vertSamp", 0));
  gpu.updateProgram->setUniform(Uniform("colorSamp", 1));
  gpu.updateProgram->setUniform(Uniform("normSamp", 2));
  gpu.updateProgram->setUniform(Uniform("texDim", (float)TEXTURE_DIMENSION));
  gpu.updateProgram->setUniform(Uniform("time", time));

  glBindBuffer(GL_ARRAY_BUFFER, vbos[target].dataBuffer);  // SELECT INPUT

  glEnableVertexAttribArray(0);
  glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, Vertex::SIZE, 0);

  glEnableVertexAttribArray(1);
  glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, Vertex::SIZE, reinterpret_cast<GLvoid*>(sizeof(Eigen::Vector4f)));

  glEnableVertexAttribArray(2);
  glVertexAttribPointer(2, 4, GL_FLOAT, GL_FALSE, Vertex::SIZE, reinterpret_cast<GLvoid*>(sizeof(Eigen::Vector4f) * 2));

  // SEE:
  // http://docs.nvidia.com/gameworks/content/gameworkslibrary/graphicssamples/opengl_samples/feedbackparticlessample.htm
  glEnable(GL_RASTERIZER_DISCARD);

  glBindTransformFeedback(GL_TRANSFORM_FEEDBACK, vbos[renderSource].stateObject);  // SELECT OUTPUT
  glBindBufferBase(GL_TRANSFORM_FEEDBACK_BUFFER, 0, vbos[renderSource].dataBuffer);

  // Enter transform feedback mode
  glBeginTransformFeedback(GL_POINTS);

  glActiveTexture(GL_TEXTURE0);
  glBindTexture(GL_TEXTURE_2D, gpu.updateMapVertsConfs.texture->tid);

  glActiveTexture(GL_TEXTURE1);
  glBindTexture(GL_TEXTURE_2D, gpu.updateMapColorsTime.texture->tid);

  glActiveTexture(GL_TEXTURE2);
  glBindTexture(GL_TEXTURE_2D, gpu.updateMapNormsRadii.texture->tid);

  glDrawTransformFeedback(GL_POINTS, vbos[target].stateObject);  // GPU-PASS (target=input)

  glEndTransformFeedback();

  glDisable(GL_RASTERIZER_DISCARD);

  glBindTexture(GL_TEXTURE_2D, 0);
  glActiveTexture(GL_TEXTURE0);

  glDisableVertexAttribArray(0);
  glDisableVertexAttribArray(1);
  glDisableVertexAttribArray(2);
  glBindBuffer(GL_ARRAY_BUFFER, 0);
  glBindTransformFeedback(GL_TRANSFORM_FEEDBACK, 0);

  gpu.updateProgram->Unbind();

  std::swap(target, renderSource);

  glFinish();
  TOCK("Fuse::Update");
}

void Model::clean(  // FIXME what happens with object models and ferns here?
    const int& time, std::vector<float>& graph, const int timeDelta, const float depthCutoff, const bool isFern, GPUTexture* depthFiltered,
    GPUTexture* mask) {
  assert(graph.size() / 16 < MAX_NODES);

  if (graph.size() > 0) {
    // Can be optimised by only uploading new nodes with offset
    glBindTexture(GL_TEXTURE_2D, deformationNodes.texture->tid);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, graph.size(), 1, GL_LUMINANCE, GL_FLOAT, graph.data());
  }

  TICK("Fuse::Copy");
  // Next we copy the new unstable vertices from the newUnstableFid transform feedback into the global map
  gpu.unstableProgram->Bind();
  gpu.unstableProgram->setUniform(Uniform("time", time));
  gpu.unstableProgram->setUniform(Uniform("confThreshold", getConfidenceThreshold()));
  gpu.unstableProgram->setUniform(Uniform("scale", (float)ModelProjection::FACTOR));
  gpu.unstableProgram->setUniform(Uniform("outlierCoeff", (float)gpu.outlierCoefficient));
  gpu.unstableProgram->setUniform(Uniform("indexSampler", 0));
  gpu.unstableProgram->setUniform(Uniform("vertConfSampler", 1));
  gpu.unstableProgram->setUniform(Uniform("colorTimeSampler", 2));
  gpu.unstableProgram->setUniform(Uniform("normRadSampler", 3));
  gpu.unstableProgram->setUniform(Uniform("nodeSampler", 4));
  gpu.unstableProgram->setUniform(Uniform("depthSamplerPrediction", 5));
  gpu.unstableProgram->setUniform(Uniform("depthSamplerInput", 6));
  gpu.unstableProgram->setUniform(Uniform("maskSampler", 7));
  gpu.unstableProgram->setUniform(Uniform("nodes", (float)(graph.size() / 16)));
  gpu.unstableProgram->setUniform(Uniform("nodeCols", (float)NODE_TEXTURE_DIMENSION));
  gpu.unstableProgram->setUniform(Uniform("timeDelta", timeDelta));
  gpu.unstableProgram->setUniform(Uniform("maxDepth", std::min(depthCutoff, maxDepth)));
  gpu.unstableProgram->setUniform(Uniform("isFern", (int)isFern));
  gpu.unstableProgram->setUniform(Uniform("maskID", id));

  Eigen::Matrix4f t_inv = pose.inverse();
  gpu.unstableProgram->setUniform(Uniform("t_inv", t_inv));

  gpu.unstableProgram->setUniform(Uniform("cam", Eigen::Vector4f(Intrinsics::getInstance().cx(), Intrinsics::getInstance().cy(),
                                                                 Intrinsics::getInstance().fx(), Intrinsics::getInstance().fy())));
  gpu.unstableProgram->setUniform(Uniform("cols", (float)Resolution::getInstance().cols()));
  gpu.unstableProgram->setUniform(Uniform("rows", (float)Resolution::getInstance().rows()));

#if 1
  glBindBuffer(GL_ARRAY_BUFFER, vbos[target].dataBuffer);
#endif

  glEnableVertexAttribArray(0);
  glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, Vertex::SIZE, 0);

  glEnableVertexAttribArray(1);
  glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, Vertex::SIZE, reinterpret_cast<GLvoid*>(sizeof(Eigen::Vector4f)));

  glEnableVertexAttribArray(2);
  glVertexAttribPointer(2, 4, GL_FLOAT, GL_FALSE, Vertex::SIZE, reinterpret_cast<GLvoid*>(sizeof(Eigen::Vector4f) * 2));

  glEnable(GL_RASTERIZER_DISCARD);

#if 1
  glBindTransformFeedback(GL_TRANSFORM_FEEDBACK, vbos[renderSource].stateObject);
  glBindBufferBase(GL_TRANSFORM_FEEDBACK_BUFFER, 0, vbos[renderSource].dataBuffer);
#endif

  glBeginTransformFeedback(GL_POINTS);

  glActiveTexture(GL_TEXTURE0);
  glBindTexture(GL_TEXTURE_2D, indexMap.getSparseIndexTex()->texture->tid);

  glActiveTexture(GL_TEXTURE1);
  glBindTexture(GL_TEXTURE_2D, indexMap.getSparseVertConfTex()->texture->tid);

  glActiveTexture(GL_TEXTURE2);
  glBindTexture(GL_TEXTURE_2D, indexMap.getSparseColorTimeTex()->texture->tid);

  glActiveTexture(GL_TEXTURE3);
  glBindTexture(GL_TEXTURE_2D, indexMap.getSparseNormalRadTex()->texture->tid);

  glActiveTexture(GL_TEXTURE4);
  glBindTexture(GL_TEXTURE_2D, deformationNodes.texture->tid);

  glActiveTexture(GL_TEXTURE5);
  glBindTexture(GL_TEXTURE_2D, indexMap.getDepthTex()->texture->tid);

  glActiveTexture(GL_TEXTURE6);
  glBindTexture(GL_TEXTURE_2D, depthFiltered->texture->tid);

  glActiveTexture(GL_TEXTURE7);
  glBindTexture(GL_TEXTURE_2D, mask->texture->tid);

  glBeginQuery(GL_TRANSFORM_FEEDBACK_PRIMITIVES_WRITTEN, countQuery);

  glDrawTransformFeedback(GL_POINTS, vbos[target].stateObject);  // RUN GPU-PASS

#if 0
    glBindTransformFeedback(GL_TRANSFORM_FEEDBACK, vbos[target].stateObject);
    glBindBufferBase(GL_TRANSFORM_FEEDBACK_BUFFER, 0, vbos[target].dataBuffer);
#endif

  glBindBuffer(GL_ARRAY_BUFFER, newUnstableBuffer.dataBuffer);

  glEnableVertexAttribArray(0);
  glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, Vertex::SIZE, 0);

  glEnableVertexAttribArray(1);
  glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, Vertex::SIZE, reinterpret_cast<GLvoid*>(sizeof(Eigen::Vector4f)));

  glEnableVertexAttribArray(2);
  glVertexAttribPointer(2, 4, GL_FLOAT, GL_FALSE, Vertex::SIZE, reinterpret_cast<GLvoid*>(sizeof(Eigen::Vector4f) * 2));

  glDrawTransformFeedback(GL_POINTS, newUnstableBuffer.stateObject);  // RUN GPU-PASS

  glEndQuery(GL_TRANSFORM_FEEDBACK_PRIMITIVES_WRITTEN);

  glGetQueryObjectuiv(countQuery, GL_QUERY_RESULT, &count);

  glEndTransformFeedback();

  glDisable(GL_RASTERIZER_DISCARD);

  glBindTexture(GL_TEXTURE_2D, 0);
  glActiveTexture(GL_TEXTURE0);

  glDisableVertexAttribArray(0);
  glDisableVertexAttribArray(1);
  glDisableVertexAttribArray(2);
  glBindBuffer(GL_ARRAY_BUFFER, 0);
  glBindTransformFeedback(GL_TRANSFORM_FEEDBACK, 0);

  gpu.unstableProgram->Unbind();

  std::swap(target, renderSource);

  glFinish();
  TOCK("Fuse::Copy");
}

void Model::eraseErrorGeometry(GPUTexture* depthFiltered) {
  TICK("Fuse::Erase");

  // Next we copy the new unstable vertices from the newUnstableFid transform feedback into the global map
  gpu.eraseProgram->Bind();
  gpu.eraseProgram->setUniform(Uniform("scale", (float)ModelProjection::FACTOR));
  gpu.eraseProgram->setUniform(Uniform("indexSampler", 0));
  gpu.eraseProgram->setUniform(Uniform("vertConfSampler", 1));
  gpu.eraseProgram->setUniform(Uniform("colorTimeSampler", 2));
  gpu.eraseProgram->setUniform(Uniform("normRadSampler", 3));
  gpu.eraseProgram->setUniform(Uniform("icpSampler", 4));
  gpu.eraseProgram->setUniform(Uniform("depthSamplerPrediction", 5));
  gpu.eraseProgram->setUniform(Uniform("depthSamplerInput", 6));
  // gpu.unstableProgram->setUniform(Uniform("maskSampler", 7));
  // gpu.unstableProgram->setUniform(Uniform("maxDepth", std::min(depthCutoff, maxDepth)));
  // gpu.unstableProgram->setUniform(Uniform("maskID", id));

  Eigen::Matrix4f t_inv = pose.inverse();
  gpu.eraseProgram->setUniform(Uniform("t_inv", t_inv));

  gpu.eraseProgram->setUniform(Uniform("cam", Eigen::Vector4f(Intrinsics::getInstance().cx(), Intrinsics::getInstance().cy(),
                                                              Intrinsics::getInstance().fx(), Intrinsics::getInstance().fy())));
  gpu.eraseProgram->setUniform(Uniform("cols", (float)Resolution::getInstance().cols()));
  gpu.eraseProgram->setUniform(Uniform("rows", (float)Resolution::getInstance().rows()));

#if 1
  glBindBuffer(GL_ARRAY_BUFFER, vbos[target].dataBuffer);
#endif

  glEnableVertexAttribArray(0);
  glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, Vertex::SIZE, 0);

  glEnableVertexAttribArray(1);
  glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, Vertex::SIZE, reinterpret_cast<GLvoid*>(sizeof(Eigen::Vector4f)));

  glEnableVertexAttribArray(2);
  glVertexAttribPointer(2, 4, GL_FLOAT, GL_FALSE, Vertex::SIZE, reinterpret_cast<GLvoid*>(sizeof(Eigen::Vector4f) * 2));

  glEnable(GL_RASTERIZER_DISCARD);

#if 1
  glBindTransformFeedback(GL_TRANSFORM_FEEDBACK, vbos[renderSource].stateObject);
  glBindBufferBase(GL_TRANSFORM_FEEDBACK_BUFFER, 0, vbos[renderSource].dataBuffer);
#endif

  glBeginTransformFeedback(GL_POINTS);

  glActiveTexture(GL_TEXTURE0);
  glBindTexture(GL_TEXTURE_2D, indexMap.getSparseIndexTex()->texture->tid);

  glActiveTexture(GL_TEXTURE1);
  glBindTexture(GL_TEXTURE_2D, indexMap.getSparseVertConfTex()->texture->tid);

  glActiveTexture(GL_TEXTURE2);
  glBindTexture(GL_TEXTURE_2D, indexMap.getSparseColorTimeTex()->texture->tid);

  glActiveTexture(GL_TEXTURE3);
  glBindTexture(GL_TEXTURE_2D, indexMap.getSparseNormalRadTex()->texture->tid);

  glActiveTexture(GL_TEXTURE4);
  glBindTexture(GL_TEXTURE_2D, icpError->texture->tid);

  glActiveTexture(GL_TEXTURE5);
  glBindTexture(GL_TEXTURE_2D, indexMap.getDepthTex()->texture->tid);

  glActiveTexture(GL_TEXTURE6);
  glBindTexture(GL_TEXTURE_2D, depthFiltered->texture->tid);

  // glBeginQuery(GL_TRANSFORM_FEEDBACK_PRIMITIVES_WRITTEN, countQuery);

  glDrawTransformFeedback(GL_POINTS, vbos[target].stateObject);  // RUN GPU-PASS

#if 0
    glBindTransformFeedback(GL_TRANSFORM_FEEDBACK, vbos[target].stateObject);
    glBindBufferBase(GL_TRANSFORM_FEEDBACK_BUFFER, 0, vbos[target].dataBuffer);
#endif

  // glBindBuffer(GL_ARRAY_BUFFER, newUnstableBuffer.dataBuffer);
  //
  // glEnableVertexAttribArray(0);
  // glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, Vertex::SIZE, 0);
  //
  // glEnableVertexAttribArray(1);
  // glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, Vertex::SIZE, reinterpret_cast<GLvoid*>(sizeof(Eigen::Vector4f)));
  //
  // glEnableVertexAttribArray(2);
  // glVertexAttribPointer(2, 4, GL_FLOAT, GL_FALSE, Vertex::SIZE, reinterpret_cast<GLvoid*>(sizeof(Eigen::Vector4f) * 2));
  //
  // glDrawTransformFeedback(GL_POINTS, newUnstableBuffer.stateObject); // RUN GPU-PASS

  // glEndQuery(GL_TRANSFORM_FEEDBACK_PRIMITIVES_WRITTEN);

  // glGetQueryObjectuiv(countQuery, GL_QUERY_RESULT, &count);

  glEndTransformFeedback();

  glDisable(GL_RASTERIZER_DISCARD);

  glBindTexture(GL_TEXTURE_2D, 0);
  glActiveTexture(GL_TEXTURE0);

  glDisableVertexAttribArray(0);
  glDisableVertexAttribArray(1);
  glDisableVertexAttribArray(2);
  glBindBuffer(GL_ARRAY_BUFFER, 0);
  glBindTransformFeedback(GL_TRANSFORM_FEEDBACK, 0);

  gpu.eraseProgram->Unbind();

  std::swap(target, renderSource);

  glFinish();
  TOCK("Fuse::Erase");
}

unsigned int Model::lastCount() { return count; }

Eigen::Vector3f Model::rodrigues2(const Eigen::Matrix3f& matrix) {
  Eigen::JacobiSVD<Eigen::Matrix3f> svd(matrix, Eigen::ComputeFullV | Eigen::ComputeFullU);
  Eigen::Matrix3f R = svd.matrixU() * svd.matrixV().transpose();

  double rx = R(2, 1) - R(1, 2);
  double ry = R(0, 2) - R(2, 0);
  double rz = R(1, 0) - R(0, 1);

  double s = sqrt((rx * rx + ry * ry + rz * rz) * 0.25);
  double c = (R.trace() - 1) * 0.5;
  c = c > 1. ? 1. : c < -1. ? -1. : c;

  double theta = acos(c);

  if (s < 1e-5) {
    double t;

    if (c > 0)
      rx = ry = rz = 0;
    else {
      t = (R(0, 0) + 1) * 0.5;
      rx = sqrt(std::max(t, 0.0));
      t = (R(1, 1) + 1) * 0.5;
      ry = sqrt(std::max(t, 0.0)) * (R(0, 1) < 0 ? -1.0 : 1.0);
      t = (R(2, 2) + 1) * 0.5;
      rz = sqrt(std::max(t, 0.0)) * (R(0, 2) < 0 ? -1.0 : 1.0);

      if (fabs(rx) < fabs(ry) && fabs(rx) < fabs(rz) && (R(1, 2) > 0) != (ry * rz > 0)) rz = -rz;
      theta /= sqrt(rx * rx + ry * ry + rz * rz);
      rx *= theta;
      ry *= theta;
      rz *= theta;
    }
  } else {
    double vth = 1 / (2 * s);
    vth *= theta;
    rx *= vth;
    ry *= vth;
    rz *= vth;
  }
  return Eigen::Vector3d(rx, ry, rz).cast<float>();
}

void Model::buildDescription() {
  if (modelMatcher) modelMatcher->buildModelDescription(this);
}

ModelDetectionResult Model::detectInRegion(const FrameData& frame, const cv::Rect& rect) {
  if (modelMatcher) return modelMatcher->detectInRegion(frame, rect);
  return ModelDetectionResult({Eigen::Matrix4f(), false});
}

Model::SurfelMap Model::downloadMap() const {
  SurfelMap result;
  result.numPoints = count;
  result.data = std::make_unique<std::vector<Eigen::Vector4f>>();
  result.data->resize(count * 3);  // The compiler should optimise this to be as fast as memset(&vertices[0], 0, count * Vertex::SIZE)

  glFinish();
  GLuint downloadVbo;

  // EFCHANGE Why was this done?
  // glGetBufferSubData(GL_ARRAY_BUFFER, 0, count * Vertex::SIZE, &(result.data->front()));

  glGenBuffers(1, &downloadVbo);
  glBindBuffer(GL_ARRAY_BUFFER, downloadVbo);
  glBufferData(GL_ARRAY_BUFFER, bufferSize, 0, GL_STREAM_DRAW);
  glBindBuffer(GL_ARRAY_BUFFER, 0);
  glBindBuffer(GL_ARRAY_BUFFER, 0);

  glBindBuffer(GL_COPY_READ_BUFFER, vbos[renderSource].dataBuffer);
  glBindBuffer(GL_COPY_WRITE_BUFFER, downloadVbo);

  glCopyBufferSubData(GL_COPY_READ_BUFFER, GL_COPY_WRITE_BUFFER, 0, 0, count * Vertex::SIZE);
  glGetBufferSubData(GL_COPY_WRITE_BUFFER, 0, count * Vertex::SIZE, &(result.data->front()));

  glBindBuffer(GL_COPY_READ_BUFFER, 0);
  glBindBuffer(GL_COPY_WRITE_BUFFER, 0);
  glDeleteBuffers(1, &downloadVbo);

  glFinish();

  return result;
}

void Model::exportTracksPLY(const tracker::Tracks &tracks, const std::string &path, const Eigen::Isometry3f &pose, bool with_descriptor, bool binary) {
  std::stringstream ss_hdr, ss_vrt, ss_edg, ss_trk;

  uint32_t vert_id = 0;
  uint32_t edge_id = 0;
  uint32_t trak_id = 0;
  std::queue<uint32_t> edge_ids;  // store consecutive keypoint neighbour connections
  std::vector<uint32_t> trak_ids; // store all keypoint IDs of a track
  static constexpr uint32_t uint32_max = std::numeric_limits<uint32_t>::max();
  for (const tracker::TrackPtr &track : tracks) {
    // clear FIFO buffer
    edge_ids = {};
    trak_ids = {};
    for (const tracker::KeypointPtr &kp : *track) {
      // add valid points
      if (kp!=nullptr && kp->coordinate.array().isFinite().all()) {
        const Eigen::RowVector3f vertices = pose * kp->coordinate.cast<float>().transpose();
        const Eigen::RowVectorXf descriptor = kp->descriptor.cast<float>();
        assert(descriptor.size() <= std::numeric_limits<uint16_t>::max());
        const uint16_t nd = descriptor.size();
        if (binary) {
          ss_vrt.write(reinterpret_cast<const char*>(vertices.data()), vertices.size() * sizeof(float));
          if (with_descriptor) {
            ss_vrt.write(reinterpret_cast<const char*>(&nd), sizeof(uint16_t));
            ss_vrt.write(reinterpret_cast<const char*>(descriptor.data()), nd * sizeof(float));
          }
        }
        else {
          ss_vrt << vertices;
          if (with_descriptor) {
            ss_vrt << " " << nd << " " << descriptor;
          }
          ss_vrt << std::endl;
        }

        edge_ids.push(vert_id);
        trak_ids.push_back(vert_id);
        vert_id++;
      }
      else {
        // if we encounter an invalid point, reset the track connection
        while (edge_ids.size()!=0) { edge_ids.pop(); }

        // mark an invalid keypoint by setting the maximum ID
        trak_ids.push_back(uint32_max);
      }

      // add track edge
      if (edge_ids.size()==2) {
        if (binary) {
          ss_edg.write(reinterpret_cast<const char*>(&edge_ids.front()), sizeof(uint32_t));
          ss_edg.write(reinterpret_cast<const char*>(&edge_ids.back()), sizeof(uint32_t));
        }
        else {
          ss_edg << edge_ids.front() << " " << edge_ids.back();
          ss_edg << std::endl;
        }
        edge_id++;
        edge_ids.pop();
      }
    }

    const uint32_t nk = trak_ids.size();
    if (binary) {
      ss_trk.write(reinterpret_cast<const char*>(&nk), sizeof(uint32_t));
      ss_trk.write(reinterpret_cast<const char*>(trak_ids.data()), nk * sizeof(uint32_t));
    }
    else {
      ss_trk << nk;
      for (size_t i = 0; i < trak_ids.size(); i++) {
        ss_trk << " " << trak_ids[i];
      }
      ss_trk << std::endl;
    }
    trak_id++;
  }

  // PLY header
  const std::string fmt = binary ? "binary_little_endian" : "ascii";
  ss_hdr << "ply" << std::endl << "format " << fmt << " 1.0" << std::endl;

  ss_hdr << "element vertex " << vert_id << std::endl;
  for(const char *c : {"x", "y", "z"}) {
    ss_hdr << "property float32 " << c << std::endl;
  }
  if (with_descriptor) {
    ss_hdr << "property list uint16 float32 descriptor"  << std::endl;
  }

  ss_hdr << "element edge " << edge_id << std::endl;
  ss_hdr << "property uint32 vertex1" << std::endl;
  ss_hdr << "property uint32 vertex2" << std::endl;

  ss_hdr << "element track " << trak_id << std::endl;
  ss_hdr << "property list uint32 uint32 vertex_index"  << std::endl;

  ss_hdr << "end_header" << std::endl;

  std::ofstream file;
  file.open(path);
  file << ss_hdr.rdbuf();
  file.close();

  std::ios::openmode mode = std::ios::app;
  if (binary)
    mode |= std::ios::binary;

  file.open(path, mode);
  file << ss_vrt.rdbuf();
  file << ss_edg.rdbuf();
  file << ss_trk.rdbuf();
  file.close();
}

void Model::exportTracksPLY(const std::string &export_dir, const Eigen::Isometry3f &global_pose, bool binary) const {
  // pre-multiply the exported tracks with the pose of the global model (id=0, static environment)
  // this transforms object tracks (id>0) from the start of their trajectory to the end
  const Eigen::Isometry3f Tp = global_pose * Eigen::Isometry3f(getPose()).inverse();

  const tracker::Tracks local_tracks = computeTrackProjectionFirstFrame();

  exportTracksPLY(local_tracks, export_dir+"/tracks-"+std::to_string(getID())+".ply", Tp, false, binary);
}

void Model::exportModelPLY(const SurfelMap &surfels, const float conf_threshold, const std::string &path, const Eigen::Isometry3f &pose) {
  // Open file
  std::ofstream fs;
  fs.open(path.c_str());

  // Write header
  fs << "ply";
  fs << "\nformat "
     << "binary_little_endian"
     << " 1.0";

  // Vertices
  fs << "\nelement vertex " << surfels.numValid;
  fs << "\nproperty float x"
        "\nproperty float y"
        "\nproperty float z";

  fs << "\nproperty uchar red"
        "\nproperty uchar green"
        "\nproperty uchar blue";

  fs << "\nproperty float nx"
        "\nproperty float ny"
        "\nproperty float nz";

  fs << "\nproperty float radius";

  fs << "\nend_header\n";

  // Close the file
  fs.close();

  // Open file in binary appendable
  std::ofstream fpout(path.c_str(), std::ios::app | std::ios::binary);

  Eigen::Matrix4f Tn = Tn.inverse().transpose();

  for (unsigned int i = 0; i < surfels.numPoints; i++) {
    Eigen::Vector4f pos = (*surfels.data)[(i * 3) + 0];
    float conf = pos[3];
    pos[3] = 1;

    if (conf > conf_threshold) {
      Eigen::Vector4f col = (*surfels.data)[(i * 3) + 1];
      Eigen::Vector4f nor = (*surfels.data)[(i * 3) + 2];
      pos = pose * pos;
      float radius = nor[3];
      nor[3] = 0;
      nor = Tn * nor;

      nor[0] *= -1;
      nor[1] *= -1;
      nor[2] *= -1;

      float value;
      memcpy(&value, &pos[0], sizeof(float));
      fpout.write(reinterpret_cast<const char*>(&value), sizeof(float));

      memcpy(&value, &pos[1], sizeof(float));
      fpout.write(reinterpret_cast<const char*>(&value), sizeof(float));

      memcpy(&value, &pos[2], sizeof(float));
      fpout.write(reinterpret_cast<const char*>(&value), sizeof(float));

      unsigned char r = int(col[0]) >> 16 & 0xFF;
      unsigned char g = int(col[0]) >> 8 & 0xFF;
      unsigned char b = int(col[0]) & 0xFF;

      fpout.write(reinterpret_cast<const char*>(&r), sizeof(unsigned char));
      fpout.write(reinterpret_cast<const char*>(&g), sizeof(unsigned char));
      fpout.write(reinterpret_cast<const char*>(&b), sizeof(unsigned char));

      memcpy(&value, &nor[0], sizeof(float));
      fpout.write(reinterpret_cast<const char*>(&value), sizeof(float));

      memcpy(&value, &nor[1], sizeof(float));
      fpout.write(reinterpret_cast<const char*>(&value), sizeof(float));

      memcpy(&value, &nor[2], sizeof(float));
      fpout.write(reinterpret_cast<const char*>(&value), sizeof(float));

      memcpy(&value, &radius, sizeof(float));
      fpout.write(reinterpret_cast<const char*>(&value), sizeof(float));
    }
  }

  // Close file
  fs.close();
}

void Model::exportModelPLY(const std::string &export_dir, const Eigen::Isometry3f &global_pose) const {
  SurfelMap surfelMap = downloadMap();
  surfelMap.countValid(getConfidenceThreshold());

  exportModelPLY(surfelMap, getConfidenceThreshold(), export_dir + "cloud-"+std::to_string(getID())+".ply", Eigen::Isometry3f(global_pose.matrix() * getPose().inverse()));
}

void Model::performFillIn(GPUTexture* rawRGB, GPUTexture* rawDepth, bool frameToFrameRGB, bool lost) {
  if (fillIn) {
    TICK("FillIn");
    fillIn->vertex(getVertexConfProjection(), rawDepth, lost);
    fillIn->normal(getNormalProjection(), rawDepth, lost);
    fillIn->image(getRGBProjection(), rawRGB, lost || frameToFrameRGB);
    TOCK("FillIn");
  }
}

void Model::store(const fs::path &model_db_path, const Eigen::Isometry3f &pose, bool clear) {
  if (!tracks_local.empty()) {
    // model has been stored before, skip
    return;
  }
  const fs::path model_dir = model_db_path / fs::path("model-"+std::to_string(getID()));
  if (!fs::exists(model_dir)) {
    fs::create_directories(model_dir);
  }

  // project camera tracks to local frames
  // this will only store the tracks (keypoint views) since the last model instantiation,
  // i.e. this will override the keypoint views from the previous "sessions",
  // so that only the latest keypoint views will be available for re-detection
  // TODO: append to previous local tracks
  tracks_local = computeTrackProjectionFirstFrame();

  // export dense and sparse representation to disk
  SurfelMap surfelMap = downloadMap();
  surfelMap.countValid(getConfidenceThreshold());
  exportModelPLY(surfelMap, getConfidenceThreshold(), model_dir / fs::path("cloud.ply"), pose);

  exportTracksPLY(tracks_local, model_dir / fs::path("tracks.ply"), pose, true, true);

  // clear camera tracks
  if (clear)
    tracks.clear();
}

void Model::activate(const Eigen::Isometry3f &pose, const int64_t& timestamp) {
  // restore tracks
  tracks.clear();
  tracks.insert(tracks_local.cbegin(), tracks_local.cend());
  // set new pose with current timestamp
  overridePose(pose.matrix());
  poses.clear();
  poses.push_back(pose);
  timestamp_ns.clear();
  timestamp_ns.push_back(timestamp);
}

bool Model::load(const fs::path &model_path) {
  // sparse data
  happly::PLYData sparse(model_path / fs::path("tracks.ply"));
  sparse.validate();

  // keypoint coordinates and descriptors
  const std::vector<std::array<double, 3>> coordinates = sparse.getVertexPositions();
  const std::vector<std::vector<float>> descriptors = sparse.getElement("vertex").getListProperty<float>("descriptor");
  const std::vector<std::vector<uint32_t>> tracks = sparse.getElement("track").getListProperty<uint32_t>("vertex_index");

  this->tracks.clear();
  static constexpr uint32_t uint32_max = std::numeric_limits<uint32_t>::max();
  for (const std::vector<uint32_t> &track_ids : tracks) {
    tracker::TrackPtr track = std::make_shared<tracker::Track>();
    for (const uint32_t &kpid : track_ids) {
      tracker::KeypointPtr kp;
      if (kpid == uint32_max) {
        // invalid
        kp = nullptr;
      }
      else {
        // valid
        kp = std::make_shared<tracker::Keypoint>();
        kp->coordinate = Eigen::Map<const Eigen::RowVector3d>(coordinates[kpid].data());
        kp->descriptor = Eigen::Map<const Eigen::RowVectorXf>(descriptors[kpid].data(), descriptors[kpid].size()).cast<double>();
      }
      track->push_back(kp);
    }

    this->tracks_local.push_back(track);
  }

  return true;
}
