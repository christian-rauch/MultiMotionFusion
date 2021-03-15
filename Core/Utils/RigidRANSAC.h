#pragma once
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <random>

class RigidRANSAC {
public:
  typedef Eigen::Matrix<bool, Eigen::Dynamic, 1> VectorXb;

  struct Config {
    int iterations;
    float inlier_threshold;
    float inlier_fraction;
  };

  struct Result {
    Eigen::Isometry3f transformation;
    float error;
    VectorXb inlier;
  };

  RigidRANSAC(int iterations, float inlier_threshold, float inlier_fraction);

  RigidRANSAC(const Config &config);

  virtual Result estimate(const Eigen::MatrixX3f &p0, const Eigen::MatrixX3f &p1, const VectorXb &mask = {});

private:
  std::default_random_engine generator;

  const Config cfg;
};
