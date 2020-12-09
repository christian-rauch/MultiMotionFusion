#include "DCRF.hpp"
#include "densecrf_gpu.cuh"
#include <exception>

#define INSTANTIATE(M) case M: return std::make_unique<dcrf_cuda::DenseCRFGPU<M>>(N);

std::unique_ptr<dcrf_cuda::DenseCRF> InstantiateDCRF(const int N, const int M) {
  switch (M) {
  INSTANTIATE(1); // warning: division by zero
  INSTANTIATE(2);
  INSTANTIATE(3);
  INSTANTIATE(4);
  INSTANTIATE(5);
  INSTANTIATE(6);
  INSTANTIATE(7);
  INSTANTIATE(8);
  INSTANTIATE(9);
  INSTANTIATE(10);
  }
  throw std::runtime_error("invalid M");
}

DCRF::DCRF(const int W, const int H, const int M) :
  crf(InstantiateDCRF(W*H, M)),
  W(W), H(H), M(M),
  unary_gpu(nullptr)
{
  //
}

DCRF::~DCRF()
{
  cudaFree(unary_gpu);
}

void DCRF::setUnaryEnergy( const Eigen::MatrixXf & unary )
{
  // TODO: use pointer to already allocated GPU matrix
  if (unary_gpu==nullptr) {
    cudaMalloc((void**)&unary_gpu, sizeof(float) * W * H * M);
  }
  cudaMemcpy(unary_gpu, unary.data(), sizeof(short) * W * H, cudaMemcpyHostToDevice);

  crf->setUnaryEnergy(unary_gpu);
}

void DCRF::addPairwiseGaussian( float sx, float sy )
{
  //
}

void DCRF::addPairwiseEnergy( const Eigen::MatrixXf & features )
{
  //
}

Eigen::MatrixXf DCRF::inference( int n_iterations ) const
{
  crf->inference(n_iterations, false);
  const float *prob_gpu = crf->getProbability();

  Eigen::MatrixXf prob_cpu(W*H, H);
  cudaMemcpy(prob_cpu.data(), prob_gpu, sizeof(float) * W * H * M, cudaMemcpyDeviceToHost);

  return prob_cpu;
}

DCRF::VectorXs DCRF::map( int n_iterations ) const
{
  crf->inference(n_iterations, true);
  const short *map_gpu = crf->getMap();

  VectorXs map_cpu(W*H);
  cudaMemcpy(map_cpu.data(), map_gpu, sizeof(short) * W * H, cudaMemcpyDeviceToHost);

  return map_cpu;
}
