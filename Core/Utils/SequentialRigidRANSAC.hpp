#pragma once
#include "RigidRANSAC.h"
#include <vector>

class SequentialRigidRANSAC : public RigidRANSAC {
public:
  SequentialRigidRANSAC(int iterations, float inlier_threshold, float inlier_fraction);

  virtual std::vector<Result> estimate(const Eigen::MatrixX3f &p0, const Eigen::MatrixX3f &p1);
};
