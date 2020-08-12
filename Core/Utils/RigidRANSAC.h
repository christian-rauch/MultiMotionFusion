#pragma once
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <random>

class RigidRANSAC {
public:
  RigidRANSAC(int iterations, float inlier_threshold, float inlier_fraction);

  Eigen::Isometry3f estimate(const Eigen::MatrixX3f &p0, const Eigen::MatrixX3f &p1);

private:
  std::default_random_engine generator;

  int iterations;
  float inlier_threshold;
  float inlier_fraction;
};
