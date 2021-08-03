#pragma once
#include <Eigen/Core>

class GroundTruthOdometryInterface {
public:
  virtual Eigen::Matrix4f getIncrementalTransformation(uint64_t timestamp) = 0;
};
