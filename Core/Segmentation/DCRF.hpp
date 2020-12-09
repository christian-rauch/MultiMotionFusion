#pragma once

#include "densecrf_base.h"
#include <memory>
#include <Eigen/Core>

class DCRF
{
public:
  typedef Eigen::Matrix<short, Eigen::Dynamic, 1> VectorXs;

  DCRF(const int W, const int H, const int M);

  ~DCRF();

  void setUnaryEnergy( const Eigen::MatrixXf & unary );

  void addPairwiseGaussian( float sx, float sy/*, LabelCompatibility * function=NULL, KernelType kernel_type=DIAG_KERNEL, NormalizationType normalization_type=NORMALIZE_SYMMETRIC */);

  void addPairwiseEnergy( const Eigen::MatrixXf & features/*, LabelCompatibility * function, KernelType kernel_type=DIAG_KERNEL, NormalizationType normalization_type=NORMALIZE_SYMMETRIC */);

  Eigen::MatrixXf inference( int n_iterations ) const;

  VectorXs map( int n_iterations ) const;

private:
  std::unique_ptr<dcrf_cuda::DenseCRF> crf;

  int W, H;
  int M;

  float *unary_gpu;
};
