#include "SequentialRigidRANSAC.hpp"

SequentialRigidRANSAC::SequentialRigidRANSAC(int iterations, float inlier_threshold, float inlier_fraction)
  : RigidRANSAC(iterations, inlier_threshold, inlier_fraction)
{

}

std::vector<RigidRANSAC::Result>
SequentialRigidRANSAC::estimate(const Eigen::MatrixX3f &p0, const Eigen::MatrixX3f &p1)
{
  assert(p0.rows() == p1.rows());

  std::vector<Result> results;

  VectorXb mask = VectorXb::Ones(p0.rows());

  while (mask.count()>=3) {
    const Result result = RigidRANSAC::estimate(p0, p1, mask);
    if (result.inlier.count()>0) {
      results.push_back(result);
      // remove inliers from set of points
      mask = result.inlier.select(false,mask);
    }
    else {
      break;
    }
  }

  return results;
}
