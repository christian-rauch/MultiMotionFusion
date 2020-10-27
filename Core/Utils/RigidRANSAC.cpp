#include "RigidRANSAC.h"
#include <limits>

#include <iostream>

// minimum number of data points to fit model (3D rigid transform)
static const int Nparams = 3;

RigidRANSAC::RigidRANSAC(int iterations, float inlier_threshold, float inlier_fraction)
  : iterations(iterations), inlier_threshold(inlier_threshold), inlier_fraction(inlier_fraction)
{

}

Eigen::Isometry3f
fit(const Eigen::MatrixX3f &p0, const Eigen::MatrixX3f &p1, const Eigen::VectorXf &weights = {})
{
  const Eigen::RowVector3f p0m = p0.colwise().mean();
  const Eigen::RowVector3f p1m = p1.colwise().mean();

  assert(weights.size() == 0 || weights.size() == p0.rows());

  const Eigen::MatrixXf W = ((weights.size()==0) ? Eigen::VectorXf::Ones(p0.rows()) : weights).asDiagonal();

  // least-squares optimisation of rigid transformation
  // find T_01 = (R_01,t_01) such that sum_i w_i * || (R_01 * p1_i - t_01) - p0_i ||^2 is minimised
  const Eigen::Matrix3f A = ((p1.rowwise()-p1m).transpose() * W * (p0.rowwise()-p0m)).transpose();

  assert(A.array().isFinite().all());

  Eigen::JacobiSVD<Eigen::Matrix3f> svd(A, Eigen::ComputeFullU | Eigen::ComputeFullV);

  const Eigen::Matrix3f U = svd.matrixU();
  const Eigen::Matrix3f V = svd.matrixV();
  // guarantee that determinant of R is 1
  const Eigen::Matrix3f S = Eigen::Vector3f(1, 1, U.determinant() * V.determinant()).asDiagonal();

  const Eigen::Isometry3f R(U * S * V.transpose());
  const Eigen::Translation3f t(p0m - (R * p1m.transpose()).transpose());

  return (t * R);
}

Eigen::VectorXf
apply(const Eigen::Isometry3f &T, const Eigen::MatrixX3f &p0, const Eigen::MatrixX3f &p1)
{
  return (p0 - (T * p1.transpose()).transpose()).rowwise().norm();
}

Eigen::Isometry3f
RigidRANSAC::estimate(const Eigen::MatrixX3f &p0, const Eigen::MatrixX3f &p1)
{
  assert(p0.rows() == p1.rows());

  // keep track of best model and its performance
  Eigen::Isometry3f bestT = fit(p0, p1);
  float bestE = std::numeric_limits<float>::max();

  const int N = p0.rows();

  std::uniform_int_distribution<int> distribution(0, N-1);

  for(int it=0; it<iterations; it++) {
    Eigen::VectorXf weights = Eigen::VectorXf::Zero(N);
    for(int p=0; p<Nparams; p++) {
      weights[distribution(generator)] = 1;
    }

    const Eigen::Isometry3f transform = fit(p0, p1, weights);
    const Eigen::VectorXf distance = apply(transform, p0, p1);

//    std::cout << "fit error (mean): " << distance.array().mean() << std::endl;
//    std::cout << "fit error (min): " << distance.array().minCoeff() << std::endl;
//    std::cout << "fit error (max): " << distance.array().maxCoeff() << std::endl;

    const auto inliers = distance.array() < inlier_threshold;
    const int Ninliers = inliers.cast<int>().sum();
//    std::cout << "inlier: " << Ninliers << std::endl;

    if(Ninliers > inlier_fraction*N) {
      // potential model
      const Eigen::Isometry3f Tall = fit(p0, p1, inliers.cast<float>());
      const float error = (apply(Tall, p0, p1).array() * inliers.cast<float>()).mean();
      if(error < bestE) {
        bestE = error;
        bestT = Tall;
      }
    }
  }

//  std::cout << "best fit: " << bestE << std::endl;

  return bestT;
}
