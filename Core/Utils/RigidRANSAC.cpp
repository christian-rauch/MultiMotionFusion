#include "RigidRANSAC.h"
#include <limits>
#include <algorithm>

// minimum number of data points to fit model (3D rigid transform)
static const int Nparams = 3;

// hash function for 3D point correspondence
// https://wjngkoh.wordpress.com/2015/03/04/c-hash-function-for-eigen-matrix-and-vector/
template<>
struct std::hash<Eigen::RowVector3f>
{
    std::size_t operator()(Eigen::RowVector3f const& v) const noexcept
    {
      std::size_t seed = 0;
      for (int i = 0; i < v.size(); ++i)
        seed ^= std::hash<float>()(v[i]) + 0xBADEAFFE + (seed << 6) + (seed >> 2);
      return seed;
    }
};

template<>
struct std::hash<std::pair<Eigen::RowVector3f, Eigen::RowVector3f>>
{
    std::size_t operator()(std::pair<Eigen::RowVector3f, Eigen::RowVector3f> const& v) const noexcept
    {
      std::size_t seed = 0;
      seed ^= std::hash<Eigen::RowVector3f>()(v.first) + 0xCAFED00D + (seed << 6) + (seed >> 2);
      seed ^= std::hash<Eigen::RowVector3f>()(v.second) + 0xCAFED00D + (seed << 6) + (seed >> 2);
      return seed;
    }
};

std::tuple<Eigen::MatrixX3f,Eigen::MatrixX3f>
sort(const Eigen::MatrixX3f &p0, const Eigen::MatrixX3f &p1)
{
  assert(p0.rows() == p1.rows());

  // store hash per point correspondence with index
  std::vector<std::pair<std::size_t, std::size_t> > hash(p0.rows());
  for (std::size_t i = 0; i < hash.size(); i++)
    hash[i] = {std::hash<std::pair<Eigen::RowVector3f, Eigen::RowVector3f>>()({p0.row(i), p1.row(i)}), i};

  // sort original indices by hash
  std::sort(hash.begin(), hash.end());

  Eigen::MatrixX3f p0s(p0.rows(), 3);
  Eigen::MatrixX3f p1s(p1.rows(), 3);

  for (std::size_t i = 0; i < hash.size(); i++) {
    p0s.row(i) = p0.row(hash[i].second);
    p1s.row(i) = p1.row(hash[i].second);
  }

  return {p0s, p1s};
}

RigidRANSAC::RigidRANSAC(int iterations, float inlier_threshold, float inlier_fraction) :
  cfg({iterations, inlier_threshold, inlier_fraction})
{
  //
}

RigidRANSAC::RigidRANSAC(const Config &config) :
  cfg(config)
{
  //
}

Eigen::Isometry3f
fit(const Eigen::MatrixX3f &p0, const Eigen::MatrixX3f &p1, const RigidRANSAC::VectorXb &mask = {})
{
  assert(mask.size() == 0 || mask.size() == p0.rows());

  Eigen::MatrixX3f p0sel;
  Eigen::MatrixX3f p1sel;

  if(mask.size()==0) {
    p0sel = p0;
    p1sel = p1;
  }
  else {
    // copy selected rows
    const size_t nsel = mask.count();
    p0sel.resize(nsel, Eigen::NoChange);
    p1sel.resize(nsel, Eigen::NoChange);
    for(size_t i=0, j=0; i<size_t(mask.size()); i++) {
      if(mask[i]) {
        p0sel.row(j) = p0.row(i);
        p1sel.row(j) = p1.row(i);
        j++;
      }
    }
  }

  const Eigen::RowVector3f p0m = p0sel.colwise().mean();
  const Eigen::RowVector3f p1m = p1sel.colwise().mean();

  // least-squares optimisation of rigid transformation
  // find T_01 = (R_01,t_01) such that sum_i w_i * || (R_01 * p1_i - t_01) - p0_i ||^2 is minimised
  const Eigen::Matrix3f A = ((p1sel.rowwise()-p1m).transpose() * (p0sel.rowwise()-p0m)).transpose();

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

  // sort (p0,p1)-correspondences by hash
  Eigen::MatrixX3f p0s, p1s;
  std::tie(p0s,p1s) = sort(p0,p1);

  // keep track of best model and its performance
  result.transformation = fit(p0s, p1s, mask);
  result.error = std::numeric_limits<float>::infinity();

  for(int it=0; it<cfg.iterations; it++) {
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

    const Eigen::Isometry3f transform = fit(p0s, p1s, weights);
    const Eigen::VectorXf distance = apply(transform, p0s, p1s);

    VectorXb inliers = (distance.array() < cfg.inlier_threshold);
    if (mask.size()>0) {
      inliers = inliers.array() && mask.array();
    }
    const Eigen::Index Ninliers = inliers.count();

    if(Ninliers > std::max<int>(std::rint(cfg.inlier_fraction*N), Nparams)) {
      // potential model
      const Eigen::Isometry3f Tall = fit(p0s, p1s, inliers);
      // mean error over inliers
      const float error = inliers.select(apply(Tall, p0s, p1s), 0).sum() / Ninliers;
      if(error < result.error) {
        result.error = error;
        result.transformation = Tall;
        result.inlier = inliers;
      }
    }
  }

  return result;
}
