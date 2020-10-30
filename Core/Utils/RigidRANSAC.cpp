#include "RigidRANSAC.h"
#include <limits>
#include <algorithm>

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

RigidRANSAC::Result
RigidRANSAC::estimate(const Eigen::MatrixX3f &p0, const Eigen::MatrixX3f &p1, const VectorXb &mask)
{
  const int N = int(p0.rows());

  assert(p0.rows() == p1.rows());
  assert(N >= Nparams);
  assert(mask.size() == 0 || mask.size() == N);

  Result result;

  // keep track of best model and its performance
  result.transformation = fit(p0, p1, mask.cast<float>());
  result.error = std::numeric_limits<float>::max();

  for(int it=0; it<iterations; it++) {
    // random order of indices
    std::vector<Eigen::Index> idx;
    for (Eigen::Index i = 0; i < N; ++i) { idx.push_back(i); }
    std::shuffle(idx.begin(), idx.end(), generator);

    VectorXb weights = VectorXb::Zero(N);
    for (size_t i = 0; i < idx.size() && weights.count()<Nparams; ++i) {
      const Eigen::Index id = idx[i];
      weights[id] = (mask.size()>0) ? mask[id] : true;
    }

    assert(weights.count()==Nparams);

    const Eigen::Isometry3f transform = fit(p0, p1, weights.cast<float>());
    const Eigen::VectorXf distance = apply(transform, p0, p1);

    VectorXb inliers = (distance.array() < inlier_threshold);
    if (mask.size()>0) {
      inliers = inliers.array() && mask.array();
    }
    const Eigen::Index Ninliers = inliers.count();

    if(Ninliers > inlier_fraction*N) {
      // potential model
      const Eigen::Isometry3f Tall = fit(p0, p1, inliers.cast<float>());
      // mean error over inliers
      const float error = inliers.select(apply(Tall, p0, p1), 0).sum() / Ninliers;
      if(error < result.error) {
        result.error = error;
        result.transformation = Tall;
        result.inlier = inliers;
      }
    }
  }

  return result;
}
