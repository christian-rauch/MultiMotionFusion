#include "../Utils/RigidRANSAC.h"
#include "../Utils/RigidRANSAC_priv.h"

#include <iostream>

int main(int argc, char *argv[])
{
  // random rotation and translation
  const Eigen::AngleAxisd R(Eigen::Matrix<double, 1, 1>::Random()[0]*M_PI, Eigen::Vector3d::Random().normalized());
  const Eigen::Translation3d t(Eigen::Vector3d::Random() * 10);
  const Eigen::Isometry3d T_01(t * R);

  constexpr size_t N = 1000;

  const Eigen::MatrixX3d p1 = Eigen::MatrixX3d::Random(N, Eigen::NoChange).rowwise() + Eigen::RowVector3d::Random();
  const Eigen::MatrixX3d p0 = (T_01 * p1.transpose()).transpose();

  // reference implementation by Shinji Umeyama, 1991
  // https://doi.org/10.1109/34.88573
  const Eigen::Matrix4d T_ref_umeyama = Eigen::umeyama(p1.transpose(), p0.transpose(), false);
  const bool match_umeyama = (T_ref_umeyama.inverse() * T_01).matrix().isIdentity();
  std::cout << "Umeyama: " << match_umeyama << std::endl;

  const Eigen::Isometry3f T_est_ls = fit(p0.cast<float>(), p1.cast<float>());
  const bool match_ls = (T_est_ls.inverse() * T_01.cast<float>()).matrix().isIdentity();
  std::cout << "least squares: " << match_ls << std::endl;

  RigidRANSAC ransac(100, 0.1, 0.1);
  // add noise to p1
  const Eigen::MatrixX3d p1n = p1 + Eigen::MatrixX3d::Random(N, Eigen::NoChange) * 0.01;
  const Eigen::Isometry3f T_est_ransac = ransac.estimate(p0.cast<float>(), p1n.cast<float>()).transformation;
  const bool match_ransac = (T_est_ransac.inverse() * T_01.cast<float>()).matrix().isIdentity(1e-3);
  std::cout << "RANSAC: " << match_ransac << std::endl;

  return match_umeyama & match_ls & match_ransac;
}
