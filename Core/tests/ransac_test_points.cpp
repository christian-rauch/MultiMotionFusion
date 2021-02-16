#include "../Utils/RigidRANSAC.h"
#include "../Utils/RigidRANSAC_priv.h"

#include <iostream>

int main(int argc, char *argv[])
{
  constexpr size_t N = 22;
  Eigen::MatrixX3f p0(N, 3);
  Eigen::MatrixX3f p1(N, 3);

  p0 <<
  0.183333, 0.0265833,     0.484,
  0.196862, 0.0290303,     0.479,
  0.188443, 0.0356761,     0.483,
  0.200557, 0.0379432,     0.477,
  0.203902, 0.0582576,     0.769,
  0.205636, 0.0398864,     0.468,
  0.213583, 0.0414811,     0.466,
  0.187097, 0.0491023,     0.447,
  0.193869, 0.0491023,     0.447,
  0.115909, 0.0448182,     0.408,
  0.153205, 0.0526894,     0.428,
   0.15825, 0.0543485,     0.422,
  0.119237, 0.0516439,     0.401,
     0.141,  0.052875,     0.423,
  0.127936, 0.0562917,     0.386,
  0.135081, 0.0461458,     0.443,
  0.219068, 0.0434659,     0.459,
  0.208987, 0.0490038,     0.761,
      0.21,      0.05,      0.44,
   0.28289, 0.0398788,     0.658,
  0.115606, 0.0470682,     0.436,
  0.138854, 0.0277708,     0.473;

  p1 <<
  0.179303, 0.0212576,     0.488,
    0.1925,   0.02475,     0.484,
   0.18447, 0.0313599,     0.487,
  0.196269, 0.0328636,     0.482,
  0.197193, 0.0500341,     0.777,
  0.201091, 0.0350114,     0.474,
  0.209631, 0.0365739,     0.471,
  0.182792, 0.0444167,     0.451,
  0.189625, 0.0444167,     0.451,
  0.111854,  0.039892,     0.413,
  0.147748,  0.048161,     0.431,
  0.152936, 0.0499053,     0.425,
  0.114854, 0.0477917,     0.407,
  0.136182, 0.0478258,     0.428,
  0.124091, 0.0509659,      0.39,
  0.128659, 0.0420455,     0.444,
  0.215303, 0.0377879,     0.464,
  0.201919, 0.0377689,     0.767,
  0.206023, 0.0454091,     0.444,
  0.201398, 0.0244943,     0.479,
  0.111085, 0.0459375,     0.441,
  0.135795, 0.0226326,     0.478;

  std::cout << ">> distance:" << std::endl << (p0-p1).rowwise().norm() << std::endl << std::endl;

  // least-squares fit on all data
  const Eigen::Isometry3f T_est_ls = fit(p0, p1);
  const Eigen::VectorXf ls_dist = apply(T_est_ls, p0, p1);
  std::cout << ">> least-squares (" << ls_dist.mean() << "):" << std::endl << ls_dist << std::endl << std::endl;

  // least-squares fit on manually selected inlier
  const RigidRANSAC::VectorXb w = (p0-p1).rowwise().norm().array()<0.10;
  const Eigen::Isometry3f T_est_lsw = fit(p0, p1, w);
  const Eigen::VectorXf lsw_dist = apply(T_est_lsw, p0, p1);
  std::cout << ">> least-squares/w (" << lsw_dist.mean() << "):" << std::endl << lsw_dist << std::endl << std::endl;

  // RANSAC with automatic selection of inlier
  // 10 iterations, 60% of samples below 3cm distance
  RigidRANSAC ransac(10, 0.03, 0.6);
  const Eigen::Isometry3f T_01 = ransac.estimate(p0, p1).transformation;
  const Eigen::VectorXf ransac_dist = apply(T_01, p0, p1);
  std::cout << ">> RANSAC (" << ransac_dist.mean() << "):" << std::endl << ransac_dist << std::endl << std::endl;
}
