/*
 * This file is part of ElasticFusion.
 *
 * Copyright (C) 2015 Imperial College London
 * 
 * The use of the code within this file and all code within files that 
 * make up the software that is ElasticFusion is permitted for 
 * non-commercial purposes only.  The full terms and conditions that 
 * apply to the code within this file are detailed within the LICENSE.txt 
 * file and at <http://www.imperial.ac.uk/dyson-robotics-lab/downloads/elastic-fusion/elastic-fusion-license/> 
 * unless explicitly stated.  By downloading this file you agree to 
 * comply with these terms.
 *
 * If you wish to use any of this code for commercial purposes then 
 * please email researchcontracts.engineering@imperial.ac.uk.
 *
 * Software License Agreement (BSD License)
 *
 *  Copyright (c) 2011, Willow Garage, Inc.
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   * Neither the name of Willow Garage, Inc. nor the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 *
 *  Author: Anatoly Baskeheev, Itseez Ltd, (myname.mysurname@mycompany.com)
 */

#include "cudafuncs.cuh"
#include "convenience.cuh"
#include "operators.cuh"
#include <math_constants.h>

//#include <Eigen/Eigen>

#if __CUDA_ARCH__ < 350
template<typename T>
__device__ __forceinline__ T __ldg(const T* ptr)
{
    return *ptr;
}
#endif

__inline__  __device__ JtJJtrSE3 warpReduceSum(JtJJtrSE3 val)
{
    for(int offset = warpSize / 2; offset > 0; offset /= 2)
    {
        val.aa += __shfl_down_sync(0xFFFFFFFF, val.aa, offset);
        val.ab += __shfl_down_sync(0xFFFFFFFF, val.ab, offset);
        val.ac += __shfl_down_sync(0xFFFFFFFF, val.ac, offset);
        val.ad += __shfl_down_sync(0xFFFFFFFF, val.ad, offset);
        val.ae += __shfl_down_sync(0xFFFFFFFF, val.ae, offset);
        val.af += __shfl_down_sync(0xFFFFFFFF, val.af, offset);
        val.ag += __shfl_down_sync(0xFFFFFFFF, val.ag, offset);

        val.bb += __shfl_down_sync(0xFFFFFFFF, val.bb, offset);
        val.bc += __shfl_down_sync(0xFFFFFFFF, val.bc, offset);
        val.bd += __shfl_down_sync(0xFFFFFFFF, val.bd, offset);
        val.be += __shfl_down_sync(0xFFFFFFFF, val.be, offset);
        val.bf += __shfl_down_sync(0xFFFFFFFF, val.bf, offset);
        val.bg += __shfl_down_sync(0xFFFFFFFF, val.bg, offset);

        val.cc += __shfl_down_sync(0xFFFFFFFF, val.cc, offset);
        val.cd += __shfl_down_sync(0xFFFFFFFF, val.cd, offset);
        val.ce += __shfl_down_sync(0xFFFFFFFF, val.ce, offset);
        val.cf += __shfl_down_sync(0xFFFFFFFF, val.cf, offset);
        val.cg += __shfl_down_sync(0xFFFFFFFF, val.cg, offset);

        val.dd += __shfl_down_sync(0xFFFFFFFF, val.dd, offset);
        val.de += __shfl_down_sync(0xFFFFFFFF, val.de, offset);
        val.df += __shfl_down_sync(0xFFFFFFFF, val.df, offset);
        val.dg += __shfl_down_sync(0xFFFFFFFF, val.dg, offset);

        val.ee += __shfl_down_sync(0xFFFFFFFF, val.ee, offset);
        val.ef += __shfl_down_sync(0xFFFFFFFF, val.ef, offset);
        val.eg += __shfl_down_sync(0xFFFFFFFF, val.eg, offset);

        val.ff += __shfl_down_sync(0xFFFFFFFF, val.ff, offset);
        val.fg += __shfl_down_sync(0xFFFFFFFF, val.fg, offset);

        val.residual += __shfl_down_sync(0xFFFFFFFF, val.residual, offset);
        val.inliers += __shfl_down_sync(0xFFFFFFFF, val.inliers, offset);
    }

    return val;
}

__inline__  __device__ JtJJtrSE3 blockReduceSum(JtJJtrSE3 val)
{
    static __shared__ JtJJtrSE3 shared[32];

    int lane = threadIdx.x % warpSize;

    int wid = threadIdx.x / warpSize;

    val = warpReduceSum(val);

    //write reduced value to shared memory
    if(lane == 0)
    {
        shared[wid] = val;
    }
    __syncthreads();

    const JtJJtrSE3 zero = {0, 0, 0, 0, 0, 0, 0, 0,
                            0, 0, 0, 0, 0, 0, 0, 0,
                            0, 0, 0, 0, 0, 0, 0, 0,
                            0, 0, 0, 0, 0};

    //ensure we only grab a value from shared memory if that warp existed
    val = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : zero;

    if(wid == 0)
    {
        val = warpReduceSum(val);
    }

    return val;
}

__global__ void reduceSum(JtJJtrSE3 * in, JtJJtrSE3 * out, int N)
{
    JtJJtrSE3 sum = {0, 0, 0, 0, 0, 0, 0, 0,
                     0, 0, 0, 0, 0, 0, 0, 0,
                     0, 0, 0, 0, 0, 0, 0, 0,
                     0, 0, 0, 0, 0};

    for(int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x)
    {
        sum.add(in[i]);
    }

    sum = blockReduceSum(sum);

    if(threadIdx.x == 0)
    {
        out[blockIdx.x] = sum;
    }
}

__inline__  __device__ JtJJtrSO3 warpReduceSum(JtJJtrSO3 val)
{
    for(int offset = warpSize / 2; offset > 0; offset /= 2)
    {
        val.aa += __shfl_down_sync(0xFFFFFFFF, val.aa, offset);
        val.ab += __shfl_down_sync(0xFFFFFFFF, val.ab, offset);
        val.ac += __shfl_down_sync(0xFFFFFFFF, val.ac, offset);
        val.ad += __shfl_down_sync(0xFFFFFFFF, val.ad, offset);

        val.bb += __shfl_down_sync(0xFFFFFFFF, val.bb, offset);
        val.bc += __shfl_down_sync(0xFFFFFFFF, val.bc, offset);
        val.bd += __shfl_down_sync(0xFFFFFFFF, val.bd, offset);

        val.cc += __shfl_down_sync(0xFFFFFFFF, val.cc, offset);
        val.cd += __shfl_down_sync(0xFFFFFFFF, val.cd, offset);

        val.residual += __shfl_down_sync(0xFFFFFFFF, val.residual, offset);
        val.inliers += __shfl_down_sync(0xFFFFFFFF, val.inliers, offset);
    }

    return val;
}

__inline__  __device__ JtJJtrSO3 blockReduceSum(JtJJtrSO3 val)
{
    static __shared__ JtJJtrSO3 shared[32];

    int lane = threadIdx.x % warpSize;

    int wid = threadIdx.x / warpSize;

    val = warpReduceSum(val);

    //write reduced value to shared memory
    if(lane == 0)
    {
        shared[wid] = val;
    }
    __syncthreads();

    const JtJJtrSO3 zero = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

    //ensure we only grab a value from shared memory if that warp existed
    val = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : zero;

    if(wid == 0)
    {
        val = warpReduceSum(val);
    }

    return val;
}

__global__ void reduceSum(JtJJtrSO3 * in, JtJJtrSO3 * out, int N)
{
    JtJJtrSO3 sum = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

    for(int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x)
    {
        sum.add(in[i]);
    }

    sum = blockReduceSum(sum);

    if(threadIdx.x == 0)
    {
        out[blockIdx.x] = sum;
    }
}

struct ICPReduction
{
    mat33 Rcurr;
    float3 tcurr;

    PtrStep<float> vmap_curr;
    PtrStep<float> nmap_curr;

    mat33 Rprev_inv;
    float3 tprev;

    CameraModel intr;

    PtrStep<float> vmap_g_prev;
    PtrStep<float> nmap_g_prev;

    float distThres;
    float angleThres;

    int cols;
    int rows;
    int N;

    JtJJtrSE3 * out;
    cudaSurfaceObject_t outErrorSurface;

    __device__ __forceinline__ bool
    search (int & x, int & y, float3& n, float3& d, float3& s) const
    {
        float3 vcurr;
        vcurr.x = vmap_curr.ptr (y       )[x];
        vcurr.y = vmap_curr.ptr (y + rows)[x];
        vcurr.z = vmap_curr.ptr (y + 2 * rows)[x];

        float3 vcurr_g = Rcurr * vcurr + tcurr;
        float3 vcurr_cp = Rprev_inv * (vcurr_g - tprev);

        int2 ukr;
        ukr.x = __float2int_rn (vcurr_cp.x * intr.fx / vcurr_cp.z + intr.cx);
        ukr.y = __float2int_rn (vcurr_cp.y * intr.fy / vcurr_cp.z + intr.cy);

        if(ukr.x < 0 || ukr.y < 0 || ukr.x >= cols || ukr.y >= rows || vcurr_cp.z < 0){
            // This magic number is picked to be small, so that during super-pixel downsampling it gets
            // either overwritten by larger values, or allows to check ICP<0 => has outlier (ICP==0.0001 => only outlier)
            if(outErrorSurface) surf2Dwrite(0.0f, outErrorSurface, x*sizeof(float), y);
            return false;
        }

        float3 vprev_g;
        vprev_g.x = __ldg(&vmap_g_prev.ptr (ukr.y       )[ukr.x]);
        vprev_g.y = __ldg(&vmap_g_prev.ptr (ukr.y + rows)[ukr.x]);
        vprev_g.z = __ldg(&vmap_g_prev.ptr (ukr.y + 2 * rows)[ukr.x]);

        float3 ncurr;
        ncurr.x = nmap_curr.ptr (y)[x];
        ncurr.y = nmap_curr.ptr (y + rows)[x];
        ncurr.z = nmap_curr.ptr (y + 2 * rows)[x];

        float3 ncurr_g = Rcurr * ncurr;

        float3 nprev_g;
        nprev_g.x =  __ldg(&nmap_g_prev.ptr (ukr.y)[ukr.x]);
        nprev_g.y = __ldg(&nmap_g_prev.ptr (ukr.y + rows)[ukr.x]);
        nprev_g.z = __ldg(&nmap_g_prev.ptr (ukr.y + 2 * rows)[ukr.x]);

        float dist = norm (vprev_g - vcurr_g);
        float sine = norm (cross (ncurr_g, nprev_g));

        if(outErrorSurface) surf2Dwrite(isfinite(dist) ? dist : 0.0f, outErrorSurface, x*sizeof(float), y);

        n = nprev_g;
        d = vprev_g;
        s = vcurr_g;

        return (sine < angleThres && dist <= distThres && !isnan (ncurr.x) && !isnan (nprev_g.x));
    }

    __device__ __forceinline__ JtJJtrSE3
    getProducts(int & i) const
    {
        int y = i / cols;
        int x = i - (y * cols);

        float3 n_cp, d_cp, s_cp;

        bool found_coresp = search (x, y, n_cp, d_cp, s_cp);

        float row[7] = {0, 0, 0, 0, 0, 0, 0};

        if(found_coresp)
        {
            s_cp = Rprev_inv * (s_cp - tprev);
            d_cp = Rprev_inv * (d_cp - tprev);
            n_cp = Rprev_inv * (n_cp);

            *(float3*)&row[0] = n_cp;
            *(float3*)&row[3] = cross (s_cp, n_cp);
            row[6] = dot (n_cp, s_cp - d_cp);
        }

        JtJJtrSE3 values = {row[0] * row[0],
                            row[0] * row[1],
                            row[0] * row[2],
                            row[0] * row[3],
                            row[0] * row[4],
                            row[0] * row[5],
                            row[0] * row[6],

                            row[1] * row[1],
                            row[1] * row[2],
                            row[1] * row[3],
                            row[1] * row[4],
                            row[1] * row[5],
                            row[1] * row[6],

                            row[2] * row[2],
                            row[2] * row[3],
                            row[2] * row[4],
                            row[2] * row[5],
                            row[2] * row[6],

                            row[3] * row[3],
                            row[3] * row[4],
                            row[3] * row[5],
                            row[3] * row[6],

                            row[4] * row[4],
                            row[4] * row[5],
                            row[4] * row[6],

                            row[5] * row[5],
                            row[5] * row[6],

                            row[6] * row[6],
                            float(found_coresp)};

        return values;
    }

    __device__ __forceinline__ void
    operator () () const
    {
        JtJJtrSE3 sum = {0, 0, 0, 0, 0, 0, 0, 0,
                         0, 0, 0, 0, 0, 0, 0, 0,
                         0, 0, 0, 0, 0, 0, 0, 0,
                         0, 0, 0, 0, 0};

        for(int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x)
        {
            JtJJtrSE3 val = getProducts(i);

            sum.add(val);
        }

        sum = blockReduceSum(sum);

        if(threadIdx.x == 0)
        {
            out[blockIdx.x] = sum;
        }
    }
};

__global__ void icpKernel(const ICPReduction icp)
{
    icp();
}

void icpStep(const mat33& Rcurr,
             const float3& tcurr,
             const DeviceArray2D<float>& vmap_curr,
             const DeviceArray2D<float>& nmap_curr,
             const mat33& Rprev_inv,
             const float3& tprev,
             const CameraModel& intr,
             const DeviceArray2D<float>& vmap_g_prev,
             const DeviceArray2D<float>& nmap_g_prev,
             float distThres,
             float angleThres,
             DeviceArray<JtJJtrSE3> & sum,
             DeviceArray<JtJJtrSE3> & out,
             float * matrixA_host,
             float * vectorB_host,
             float * residual_host,
             int threads,
             int blocks,
             const cudaSurfaceObject_t& icpErrorSurface)
{
    int cols = vmap_curr.cols ();
    int rows = vmap_curr.rows () / 3;

    ICPReduction icp;

    icp.Rcurr = Rcurr;
    icp.tcurr = tcurr;

    icp.vmap_curr = vmap_curr;
    icp.nmap_curr = nmap_curr;

    icp.Rprev_inv = Rprev_inv;
    icp.tprev = tprev;

    icp.intr = intr;

    icp.vmap_g_prev = vmap_g_prev;
    icp.nmap_g_prev = nmap_g_prev;

    icp.distThres = distThres;
    icp.angleThres = angleThres;

    icp.cols = cols;
    icp.rows = rows;

    icp.N = cols * rows;
    icp.out = sum;
    icp.outErrorSurface = icpErrorSurface;

    icpKernel<<<blocks, threads>>>(icp);

    reduceSum<<<1, MAX_THREADS>>>(sum, out, blocks);

    cudaSafeCall(cudaGetLastError());
    cudaSafeCall(cudaDeviceSynchronize());

    float host_data[32];
    out.download((JtJJtrSE3 *)&host_data[0]);

    int shift = 0;
    for (int i = 0; i < 6; ++i)
    {
        for (int j = i; j < 7; ++j)
        {
            float value = host_data[shift++];
            if (j == 6)
                vectorB_host[i] = value;
            else
                matrixA_host[j * 6 + i] = matrixA_host[i * 6 + j] = value;
        }
    }

    residual_host[0] = host_data[27];
    residual_host[1] = host_data[28];
}

struct ProjectionError
{
    // transformation from last to current frame
    mat44 Tcurr;

    PtrStep<float> vmap_curr;

    CameraModel intr;

    PtrStep<float> vmap_g_prev;

    float distThres;

    int cols;
    int rows;

    cudaSurfaceObject_t outErrorSurface;

    __device__ __forceinline__ bool
    distance(const int &x, const int &y) const
    {
        // 3D coordinate at (x,y)
        const float3 vcurr {
          .x = vmap_curr.ptr (y       )[x],
          .y = vmap_curr.ptr (y + rows)[x],
          .z = vmap_curr.ptr (y + 2 * rows)[x],
        };

        // transform to previous camera frame
        const float4 a = Tcurr * hom34(vcurr);
        const float3 vcurr_cp = {a.x, a.y, a.z};

        // project to previous image plane
        int2 ukr;
        ukr.x = __float2int_rn (vcurr_cp.x * intr.fx / vcurr_cp.z + intr.cx);
        ukr.y = __float2int_rn (vcurr_cp.y * intr.fy / vcurr_cp.z + intr.cy);

        if(ukr.x < 0 || ukr.y < 0 || ukr.x >= cols || ukr.y >= rows || vcurr_cp.z < 0){
            // This magic number is picked to be small, so that during super-pixel downsampling it gets
            // either overwritten by larger values, or allows to check ICP<0 => has outlier (ICP==0.0001 => only outlier)
            if(outErrorSurface) surf2Dwrite(0.0f, outErrorSurface, x*sizeof(float), y);
            return false;
        }

        // point at projected coordinate in previous camera
        float3 vprev_g;
        vprev_g.x = __ldg(&vmap_g_prev.ptr (ukr.y       )[ukr.x]);
        vprev_g.y = __ldg(&vmap_g_prev.ptr (ukr.y + rows)[ukr.x]);
        vprev_g.z = __ldg(&vmap_g_prev.ptr (ukr.y + 2 * rows)[ukr.x]);

        const float dist = norm(vprev_g - vcurr_cp);

        if(outErrorSurface) surf2Dwrite(isfinite(dist) ? dist : 0.0f, outErrorSurface, x*sizeof(float), y);

        return dist <= distThres;
    }

    __device__ __forceinline__ void
    operator () () const
    {
        for(int i = blockIdx.x * blockDim.x + threadIdx.x; i < rows * cols; i += blockDim.x * gridDim.x)
        {
            const int y = i / cols;
            const int x = i - (y * cols);
            distance(x, y);
        }
    }
};

__global__ void rpeKernel(const ProjectionError rp)
{
    rp();
}

void projectionError(const mat44& Tcurr,
                     const DeviceArray2D<float>& vmap_curr,
                     const CameraModel& intr,
                     const DeviceArray2D<float>& vmap_g_prev,
                     float distThres,
                     int threads,
                     int blocks,
                     const cudaSurfaceObject_t& rpeSurface)
{
    int cols = vmap_curr.cols ();
    int rows = vmap_curr.rows () / 3;

    ProjectionError rpe;

    rpe.Tcurr = Tcurr;

    rpe.vmap_curr = vmap_curr;

    rpe.intr = intr;

    rpe.vmap_g_prev = vmap_g_prev;

    rpe.distThres = distThres;

    rpe.cols = cols;
    rpe.rows = rows;

    rpe.outErrorSurface = rpeSurface;

    rpeKernel<<<blocks, threads>>>(rpe);

    cudaSafeCall(cudaGetLastError());
    cudaSafeCall(cudaDeviceSynchronize());
}

#define FLT_EPSILON ((float)1.19209290E-07F)

struct RGBReduction
{
    PtrStepSz<DataTerm> corresImg;

    float sigma;
    PtrStepSz<float3> cloud;
    float fx;
    float fy;
    PtrStepSz<short> dIdx;
    PtrStepSz<short> dIdy;
    float sobelScale;

    int cols;
    int rows;
    int N;

    JtJJtrSE3 * out;

    __device__ __forceinline__ JtJJtrSE3
    getProducts(int & i) const
    {
        const DataTerm & corresp = corresImg.data[i];

        bool found_coresp = corresp.valid && !isnan(cloud.ptr(corresp.zero.y)[corresp.zero.x].z);

        float row[7];

        if(found_coresp)
        {
            float w = sigma + std::abs(corresp.diff);

            w = w > FLT_EPSILON ? 1.0f / w : 1.0f;

            //Signals RGB only tracking, so we should only
            if(sigma == -1)
            {
                w = 1;
            }

            row[6] = -w * corresp.diff;

            float3 cloudPoint = {cloud.ptr(corresp.zero.y)[corresp.zero.x].x,
                                 cloud.ptr(corresp.zero.y)[corresp.zero.x].y,
                                 cloud.ptr(corresp.zero.y)[corresp.zero.x].z};

            float invz = 1.0 / cloudPoint.z;
            float dI_dx_val = w * sobelScale * dIdx.ptr(corresp.one.y)[corresp.one.x];
            float dI_dy_val = w * sobelScale * dIdy.ptr(corresp.one.y)[corresp.one.x];
            float v0 = dI_dx_val * fx * invz;
            float v1 = dI_dy_val * fy * invz;
            float v2 = -(v0 * cloudPoint.x + v1 * cloudPoint.y) * invz;

            row[0] = v0;
            row[1] = v1;
            row[2] = v2;
            row[3] = -cloudPoint.z * v1 + cloudPoint.y * v2;
            row[4] =  cloudPoint.z * v0 - cloudPoint.x * v2;
            row[5] = -cloudPoint.y * v0 + cloudPoint.x * v1;
        }
        else
        {
            row[0] = row[1] = row[2] = row[3] = row[4] = row[5] = row[6] = 0.f;
        }

        JtJJtrSE3 values = {row[0] * row[0],
                            row[0] * row[1],
                            row[0] * row[2],
                            row[0] * row[3],
                            row[0] * row[4],
                            row[0] * row[5],
                            row[0] * row[6],

                            row[1] * row[1],
                            row[1] * row[2],
                            row[1] * row[3],
                            row[1] * row[4],
                            row[1] * row[5],
                            row[1] * row[6],

                            row[2] * row[2],
                            row[2] * row[3],
                            row[2] * row[4],
                            row[2] * row[5],
                            row[2] * row[6],

                            row[3] * row[3],
                            row[3] * row[4],
                            row[3] * row[5],
                            row[3] * row[6],

                            row[4] * row[4],
                            row[4] * row[5],
                            row[4] * row[6],

                            row[5] * row[5],
                            row[5] * row[6],

                            row[6] * row[6],
                            float(found_coresp)};

        return values;
    }

    __device__ __forceinline__ void
    operator () () const
    {
        JtJJtrSE3 sum = {0, 0, 0, 0, 0, 0, 0, 0,
                         0, 0, 0, 0, 0, 0, 0, 0,
                         0, 0, 0, 0, 0, 0, 0, 0,
                         0, 0, 0, 0, 0};

        for(int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x)
        {
            JtJJtrSE3 val = getProducts(i);

            sum.add(val);
        }

        sum = blockReduceSum(sum);

        if(threadIdx.x == 0)
        {
            out[blockIdx.x] = sum;
        }
    }
};

__global__ void rgbKernel (const RGBReduction rgb)
{
    rgb();
}

void rgbStep(const DeviceArray2D<DataTerm> & corresImg,
             const float & sigma,
             const DeviceArray2D<float3> & cloud,
             const float & fx,
             const float & fy,
             const DeviceArray2D<short> & dIdx,
             const DeviceArray2D<short> & dIdy,
             const float & sobelScale,
             DeviceArray<JtJJtrSE3> & sum,
             DeviceArray<JtJJtrSE3> & out,
             float * matrixA_host,
             float * vectorB_host,
             int threads,
             int blocks)
{
    RGBReduction rgb;

    rgb.corresImg = corresImg;
    rgb.cols = corresImg.cols();
    rgb.rows = corresImg.rows();
    rgb.sigma = sigma;
    rgb.cloud = cloud;
    rgb.fx = fx;
    rgb.fy = fy;
    rgb.dIdx = dIdx;
    rgb.dIdy = dIdy;
    rgb.sobelScale = sobelScale;
    rgb.N = rgb.cols * rgb.rows;
    rgb.out = sum;

    rgbKernel<<<blocks, threads>>>(rgb);

    reduceSum<<<1, MAX_THREADS>>>(sum, out, blocks);

    cudaSafeCall(cudaGetLastError());
    cudaSafeCall(cudaDeviceSynchronize());

    float host_data[32];
    out.download((JtJJtrSE3 *)&host_data[0]);

    int shift = 0;
    for (int i = 0; i < 6; ++i)
    {
        for (int j = i; j < 7; ++j)
        {
            float value = host_data[shift++];
            if (j == 6)
                vectorB_host[i] = value;
            else
                matrixA_host[j * 6 + i] = matrixA_host[i * 6 + j] = value;
        }
    }
}

__inline__  __device__ int2 warpReduceSum(int2 val)
{
    for(int offset = warpSize / 2; offset > 0; offset /= 2)
    {
        val.x += __shfl_down_sync(0xFFFFFFFF, val.x, offset);
        val.y += __shfl_down_sync(0xFFFFFFFF, val.y, offset);
    }

    return val;
}

__inline__  __device__ float2 warpReduceSumF2(float2 val)
{
    for(int offset = warpSize / 2; offset > 0; offset /= 2)
    {
        val.x += __shfl_down(val.x, offset);
        val.y += __shfl_down(val.y, offset);
    }

    return val;
}

__inline__  __device__ int2 blockReduceSum(int2 val)
{
    static __shared__ int2 shared[32];

    int lane = threadIdx.x % warpSize;

    int wid = threadIdx.x / warpSize;

    val = warpReduceSum(val);

    //write reduced value to shared memory
    if(lane == 0)
    {
        shared[wid] = val;
    }
    __syncthreads();

    const int2 zero = {0, 0};

    //ensure we only grab a value from shared memory if that warp existed
    val = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : zero;

    if(wid == 0)
    {
        val = warpReduceSum(val);
    }

    return val;
}

__inline__  __device__ float2 blockReduceSumF2(float2 val)
{
    static __shared__ float2 shared[32];

    int lane = threadIdx.x % warpSize;

    int wid = threadIdx.x / warpSize;

    val = warpReduceSumF2(val);

    //write reduced value to shared memory
    if(lane == 0)
    {
        shared[wid] = val;
    }
    __syncthreads();

    const float2 zero = {0, 0};

    //ensure we only grab a value from shared memory if that warp existed
    val = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : zero;

    if(wid == 0)
    {
        val = warpReduceSumF2(val);
    }

    return val;
}

__global__ void reduceSum(int2 * in, int2 * out, int N)
{
    int2 sum = {0, 0};

    for(int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x)
    {
        sum.x += in[i].x;
        sum.y += in[i].y;
    }

    sum = blockReduceSum(sum);

    if(threadIdx.x == 0)
    {
        out[blockIdx.x] = sum;
    }
}

__global__ void reduceSumF2(float2 * in, float2 * out, int N)
{
    float2 sum = {0, 0};

    for(int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x)
    {
        sum.x += in[i].x;
        sum.y += in[i].y;
    }

    sum = blockReduceSumF2(sum);

    if(threadIdx.x == 0)
    {
        out[blockIdx.x] = sum;
    }
}

struct RGBResidual
{
    float minScale;

    PtrStepSz<short> dIdx;
    PtrStepSz<short> dIdy;

    PtrStepSz<float> lastDepth;
    PtrStepSz<float> nextDepth;

    PtrStepSz<unsigned char> lastImage;
    PtrStepSz<unsigned char> nextImage;

    PtrStepSz<unsigned char> lastMask;
    PtrStepSz<unsigned char> nextMask;

    mutable PtrStepSz<DataTerm> corresImg;

    float maxDepthDelta;

    float3 kt;
    mat33 krkinv;

    int cols;
    int rows;
    int N;

#ifdef MASK_RGB_RESIDUAL
    unsigned char maskID;
#endif

    int pitch;
    int imgPitch;

    float2 * out;
    cudaSurfaceObject_t outErrorSurface;

    __device__ __forceinline__ float2
    getProducts(int k) const
    {
        int i = k / cols;
        int j0 = k - (i * cols);

        float2 value = {0, 0};

        DataTerm corres;

        corres.valid = false;

        if(i >= 0 && i < rows && j0 >= 0 && j0 < cols)
        {
            if(j0 < cols - 5 && i < rows - 1)
            {
                bool valid = true;

                for(int u = max(i - 2, 0); u < min(i + 2, rows); u++)
                {
                    for(int v = max(j0 - 2, 0); v < min(j0 + 2, cols); v++)
                    {

                        valid = valid && (nextImage.ptr(u)[v] > 0)
#ifdef MASK_RGB_RESIDUAL
                                && (nextMask.ptr(u)[v] == maskID)
#endif
                                ;
                    }
                }

                if(valid)
                {
                    short * ptr_input_x = (short*) ((unsigned char*) dIdx.data + i * pitch);
                    short * ptr_input_y = (short*) ((unsigned char*) dIdy.data + i * pitch);

                    short valx = ptr_input_x[j0];
                    short valy = ptr_input_y[j0];
                    float mTwo = (valx * valx) + (valy * valy);

                    if(mTwo >= minScale)
                    {
                        int y = i;
                        int x = j0;

                        float d1 = nextDepth.ptr(y)[x];

                        if(!isnan(d1))
                        {
                            float transformed_d1 = (float)(d1 * (krkinv.data[2].x * x + krkinv.data[2].y * y + krkinv.data[2].z) + kt.z);
                            int u0 = __float2int_rn((d1 * (krkinv.data[0].x * x + krkinv.data[0].y * y + krkinv.data[0].z) + kt.x) / transformed_d1);
                            int v0 = __float2int_rn((d1 * (krkinv.data[1].x * x + krkinv.data[1].y * y + krkinv.data[1].z) + kt.y) / transformed_d1);

                            if(u0 >= 0 && v0 >= 0 && u0 < lastDepth.cols && v0 < lastDepth.rows)
                            {
                                float d0 = lastDepth.ptr(v0)[u0];

                                if(d0 > 0 && std::abs(transformed_d1 - d0) <= maxDepthDelta && lastImage.ptr(v0)[u0] != 0)
                                {
                                    corres.zero.x = u0;
                                    corres.zero.y = v0;
                                    corres.one.x = x;
                                    corres.one.y = y;
                                    corres.diff = static_cast<float>(nextImage.ptr(y)[x]) - static_cast<float>(lastImage.ptr(v0)[u0]);
                                    corres.valid = true;
                                    value.x = 1;
                                    value.y = corres.diff * corres.diff;
                                    if(outErrorSurface) surf2Dwrite(0.001f * value.y, outErrorSurface, x*sizeof(float), y);
                                }
                            }
                        }
                    }
                }
            }
        }

        if(!corres.valid && outErrorSurface) surf2Dwrite(0.0f, outErrorSurface, j0*sizeof(float), i);
        corresImg.data[k] = corres;

        return value;
    }

    __device__ __forceinline__ void
    operator () () const
    {
        float2 sum = {0, 0};

        for(int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x)
        {
            float2 val = getProducts(i);
            sum.x += val.x;
            sum.y += val.y;
        }

        sum = blockReduceSumF2(sum);

        if(threadIdx.x == 0)
        {
            out[blockIdx.x] = sum;
        }
    }
};

__global__ void residualKernel (const RGBResidual rgb)
{
    rgb();
}

void computeRgbResidual(const float & minScale,
                        const DeviceArray2D<short> & dIdx,
                        const DeviceArray2D<short> & dIdy,
                        const DeviceArray2D<float> & lastDepth,
                        const DeviceArray2D<float> & nextDepth,
                        const DeviceArray2D<unsigned char> & lastImage,
                        const DeviceArray2D<unsigned char> & nextImage,
                        const DeviceArray2D<unsigned char> & lastMask,
                        const DeviceArray2D<unsigned char> & nextMask,
                        DeviceArray2D<DataTerm> & corresImg,
                        DeviceArray<float2> & sumResidual,
                        const float maxDepthDelta,
                        const float3 & kt,
                        const mat33 & krkinv,
                        float & sigmaSum,
                        int & count,
                        int threads,
                        int blocks,
                        const cudaSurfaceObject_t& rgbErrorSurface,
                        unsigned char maskID)
{
    int cols = nextImage.cols ();
    int rows = nextImage.rows ();

    RGBResidual rgb;

    rgb.minScale = minScale;

    rgb.dIdx = dIdx;
    rgb.dIdy = dIdy;

    rgb.lastDepth = lastDepth;
    rgb.nextDepth = nextDepth;

    rgb.lastImage = lastImage;
    rgb.nextImage = nextImage;

    rgb.lastMask = lastMask;
    rgb.nextMask = nextMask;

    rgb.corresImg = corresImg;

    rgb.maxDepthDelta = maxDepthDelta;

    rgb.kt = kt;
    rgb.krkinv = krkinv;

    rgb.cols = cols;
    rgb.rows = rows;
    rgb.pitch = dIdx.step();
    rgb.imgPitch = nextImage.step();

    rgb.N = cols * rows;
    rgb.out = sumResidual;
    rgb.outErrorSurface = rgbErrorSurface;

#ifdef MASK_RGB_RESIDUAL
    rgb.maskID = maskID;
#endif

    residualKernel<<<blocks, threads>>>(rgb);

    float2 out_host = {0, 0};
    float2 * out;

    cudaMalloc(&out, sizeof(int2));
    cudaMemcpy(out, &out_host, sizeof(int2), cudaMemcpyHostToDevice);

    reduceSumF2<<<1, MAX_THREADS>>>(sumResidual, out, blocks);

    cudaSafeCall(cudaGetLastError());
    cudaSafeCall(cudaDeviceSynchronize());

    cudaMemcpy(&out_host, out, sizeof(int2), cudaMemcpyDeviceToHost);
    cudaFree(out);

    count = out_host.x;
    sigmaSum = out_host.y;
}

struct KPResidual
{
    PtrStepSz<float> lastKeypoints;
    PtrStepSz<float> nextKeypoints;

    PtrStepSz<float> lastFeatureMaps;
    PtrStepSz<float> nextFeatureMaps;

    PtrStepSz<int> matchID;
    PtrStepSz<float> matchScore;

    PtrStepSz<unsigned char> lastMask;

    mutable PtrStepSz<DataTerm> corresImg;

    int cols;
    int rows;
    int Nlast;
    int Nnext;
    int Nmatches;
    int feat_dim;

    unsigned char maskID;

    float2 * out;
    cudaSurfaceObject_t outErrorSurface;

//    mutable PtrStepSz<float> outAdjMat;

//    __device__ __forceinline__ void
//    getDistances(const int k) const
//    {
//        // next and last keypoint ID for indices of row and column of adjacency matrix
//        const int ikn = k / Nlast;          // next, row
//        const int ikl = k - (ikn * Nlast);  // last, column

//        const int2 kl = {int(lastKeypoints.ptr(ikl)[0] * cols), int(lastKeypoints.ptr(ikl)[1] * rows)};
//        const int2 kn = {int(nextKeypoints.ptr(ikn)[0] * cols), int(nextKeypoints.ptr(ikn)[1] * rows)};

////        if(outErrorSurface) surf2Dwrite(200.0f, outErrorSurface, kl.x*sizeof(float), kl.y);
////        if(outErrorSurface) surf2Dwrite(200.0f, outErrorSurface, kn.x*sizeof(float), kl.y);

//        // ignore keypoints outside of masked area
//        if(lastMask.ptr(kl.y)[kl.x] != maskID) {
//            outAdjMat.ptr(ikn)[ikl] = CUDART_INF_F;
//        }
//        else {
//            // L2-norm distance between keypoint descriptors
//            outAdjMat.ptr(ikn)[ikl] = 0;
//            for(int i=0; i<feat_dim; i++) {
//                const float d = lastKeypoints.ptr(ikl)[2+i] - nextKeypoints.ptr(ikl)[2+i];
//                outAdjMat.ptr(ikn)[ikl] += d*d;
//            }
//            outAdjMat.ptr(ikn)[ikl] = sqrt(outAdjMat.ptr(ikn)[ikl]);
//        }
////        printf("(%i,%i) %f\n", ikn, ikl, outAdjMat.ptr(ikn)[ikl]);
//    }

//    __device__ __forceinline__ int2
//    findMatches(const int ikl) const
//    {
//        const short2 kl = {short(lastKeypoints.ptr(ikl)[0] * cols), short(lastKeypoints.ptr(ikl)[1] * rows)};

//        int2 value = {0, 0};

////        if(outErrorSurface) surf2Dwrite(200.0f, outErrorSurface, kl.x*sizeof(float), kl.y);

//        if(lastMask.ptr(kl.y)[kl.x] != maskID) {
//            return value;
//        }

//        // check for best match in list->next direction
//        float min_val = CUDART_INF_F;
//        int min_id;
//        for(int ikn=0; ikn<Nnext; ikn++) {
//            const float dist_ln = outAdjMat.ptr(ikn)[ikl];
//            if(!isinf(dist_ln) && dist_ln < min_val) {
//                min_val = dist_ln;
//                min_id = ikn;
//            }
//        }

//        if (!isinf(min_val)) {
//            const short2 min_kn = {short(nextKeypoints.ptr(min_id)[0] * cols), short(nextKeypoints.ptr(min_id)[1] * rows)};
//            DataTerm corres;
//            corres.valid = true;
//            corres.zero = min_kn;
//            corres.one = kl;
//            corres.diff = min_val;
//            corresImg.ptr(min_kn.y)[min_kn.x] = corres;
//            value.x = 1;
//            value.y = min_val*min_val;
////            printf("(%i,%i) %f\n", min_kn.x, min_kn.y, min_val);
////            if(outErrorSurface) surf2Dwrite(200.0f, outErrorSurface, min_kn.x*sizeof(float), min_kn.y);
//        }
////        printf("(%i,%i) %f\n", value.x, value.y, min_val);

//        return value;
//    }

    __device__ __forceinline__ float2
    getProducts(int k) const
    {
//        int i = k / cols;
//        int j0 = k - (i * cols);

//        int y = i;
//        int x = j0;

        // next and last keypoint ID for indices of row and column of adjacency matrix
//        const int ikn = k / Nlast;          // next, row
//        const int ikl = k - (ikn * Nlast);  // last, column

//        outAdjMat.ptr(ikn)[ikl] = 0;

        float2 value = {0, 0};

        DataTerm corres;

//        corres.valid = false;

        const int ikl = matchID.ptr(k)[0];
        const int ikn = matchID.ptr(k)[1];

//        float kx = lastKeypoints.ptr(k)[0];
//        float ky = lastKeypoints.ptr(k)[1];
//        printf("last kp %i (%f, %f)\n", k, kx, ky);

//        const int klx = lastKeypoints.ptr(kl)[0] * cols;
//        const int kly = lastKeypoints.ptr(kl)[1] * rows;

        const short2 kl = {short(lastKeypoints.ptr(ikl)[0] * cols), short(lastKeypoints.ptr(ikl)[1] * rows)};
        const short2 kn = {short(nextKeypoints.ptr(ikn)[0] * cols), short(nextKeypoints.ptr(ikn)[1] * rows)};

//        const int knx = lastKeypoints.ptr(kn)[0] * cols;
//        const int kny = lastKeypoints.ptr(kn)[1] * rows;

//        // ignore keypoints outside of masked area
//        if(lastMask.ptr(kl.y)[kl.x] != maskID) {
//            outAdjMat.ptr(ikn)[ikl] = CUDART_INF_F;
//            return value;
//        }

//        printf("last kp %i (%i, %i)\n", ikl, kl.x, kl.y);

//        printf("last kp %i (%f, %f, %f, %f, %f)\n", ikl, lastKeypoints.ptr(ikl)[2+0],
//                                                         lastKeypoints.ptr(ikl)[2+1],
//                                                         lastKeypoints.ptr(ikl)[2+2],
//                                                         lastKeypoints.ptr(ikl)[2+3],
//                                                         lastKeypoints.ptr(ikl)[2+4]);

//        outAdjMat.ptr(ikn)[ikl] = 0;
//        for(int i=0; i<feat_dim; i++) {
//            outAdjMat.ptr(ikn)[ikl] = lastKeypoints.ptr(ikl)[2+i] * nextKeypoints.ptr(ikl)[2+i];
//        }
//        outAdjMat.ptr(ikn)[ikl] = sqrt(outAdjMat.ptr(ikn)[ikl]);

//        float nkx = nextKeypoints.ptr(0)[0];
//        float nky = nextKeypoints.ptr(0)[1];
//        printf("next kp %i (%f, %f)\n", 0, nkx, nky);

//        printf("score r,c %i,%i\n", matchScore.rows, matchScore.cols);
//        printf("score %i %f\n", k, matchScore.ptr(0)[k]);

        corres.valid = true;
        corres.zero = kn;
        corres.one = kl;
        corres.diff = matchScore.ptr(0)[k]; // L2 feature distance

        value.x = 1;
        value.y = corres.diff*corres.diff;

        if(outErrorSurface) surf2Dwrite(corres.diff*150.0f, outErrorSurface, kn.x*sizeof(float), kn.y);

        corresImg.ptr(kn.y)[kn.x] = corres;

        return value;
    }

//    __device__ __forceinline__ void
//    getAdjMat() const
//    {
//      for(int i = blockIdx.x * blockDim.x + threadIdx.x; i < Nlast*Nnext; i += blockDim.x * gridDim.x)
//      {
//          getDistances(i);
//      }
//    }

//    __device__ __forceinline__ void
//    getMatchDist() const
//    {
//      float2 sum = {0, 0};

//      for(int i = blockIdx.x * blockDim.x + threadIdx.x; i < Nlast; i += blockDim.x * gridDim.x)
//      {
//          int2 val = findMatches(i);
//          sum.x += val.x;
//          sum.y += val.y;
////          printf("sum (%i,%i) %f\n", sum.x, sum.y, sum);
//      }
////      printf("sum (%i,%i) %f\n", sum.x, sum.y, sum);

//      sum = blockReduceSumF2(sum);

//      if(threadIdx.x == 0)
//      {
//          out[blockIdx.x] = sum;
//      }
//    }

    __device__ __forceinline__ void
    reset() const
    {
      for(int k = blockIdx.x * blockDim.x + threadIdx.x; k < rows*cols; k += blockDim.x * gridDim.x)
      {
          int i = k / cols;
          int j0 = k - (i * cols);

          DataTerm corres;
          corres.valid = false;

          if(outErrorSurface) surf2Dwrite(0.0f, outErrorSurface, j0*sizeof(float), i);
          corresImg.data[k] = corres;
      }
    }

    __device__ __forceinline__ void
    operator () () const
    {
        float2 sum = {0, 0};

//        for(int i = 0; i < 10; i++) {
//            float nkx = nextKeypoints.ptr(i)[0];
//            float nky = nextKeypoints.ptr(i)[1];
//            printf("next kp %i (%f, %f)\n", i, nkx, nky);
//        }

//        // keypoint distances
//        for(int i = blockIdx.x * blockDim.x + threadIdx.x; i < Nlast*Nnext; i += blockDim.x * gridDim.x)
//        {
//            getDistances(i);
//        }

        for(int i = blockIdx.x * blockDim.x + threadIdx.x; i < Nmatches; i += blockDim.x * gridDim.x)
        {
            float2 val = getProducts(i);
            sum.x += val.x;
            sum.y += val.y;
        }

        sum = blockReduceSumF2(sum);

        if(threadIdx.x == 0)
        {
            out[blockIdx.x] = sum;
        }
    }
};

__global__ void residualKernel (const KPResidual kp)
{
    kp();
}

__global__ void kernl_kp_reset (const KPResidual kp)
{
    kp.reset();
}

//__global__ void kern_kp_dist (const KPResidual kp)
//{
//  kp.getAdjMat();
//}

//__global__ void kern_kp_match_dist (const KPResidual kp)
//{
//  kp.getMatchDist();
//}

void computeKPResidual(const DeviceArray2D<float> & lastDepth,
                        const DeviceArray2D<float> & nextDepth,
                        const DeviceArray2D<float> & lastKeypoints,
                        const DeviceArray2D<float> & nextKeypoints,
                        const DeviceArray2D<float> & lastFeatureMaps,
                        const DeviceArray2D<float> & nextFeatureMaps,
                        const DeviceArray2D<int> & matchID,
                        const DeviceArray2D<float> & matchScore,
                        const DeviceArray2D<unsigned char> & lastMask,
                        DeviceArray2D<DataTerm> & corresImg,
                        DeviceArray<float2> & sumResidual,
                        float & sigmaSum,
                        int & count,
                        int threads,
                        int blocks,
                        const cudaSurfaceObject_t& rgbErrorSurface,
                        unsigned char maskID)
{
    int cols = nextDepth.cols ();
    int rows = nextDepth.rows ();

    KPResidual kpr;

    kpr.lastKeypoints = lastKeypoints;
    kpr.nextKeypoints = nextKeypoints;

    kpr.lastFeatureMaps = lastFeatureMaps;
    kpr.nextFeatureMaps = nextFeatureMaps;

    kpr.matchID = matchID;
    kpr.matchScore = matchScore;

    kpr.lastMask = lastMask;

    kpr.corresImg = corresImg;

    kpr.cols = cols;
    kpr.rows = rows;

    // keypoint rows: 2 coordinates + D features
    kpr.feat_dim = nextKeypoints.cols()-2;

    kpr.Nlast = lastKeypoints.rows();
    kpr.Nnext = nextKeypoints.rows();
    kpr.Nmatches = matchID.rows();

    kpr.out = sumResidual;
    kpr.outErrorSurface = rgbErrorSurface;

//    DeviceArray2D<float> adj_mat(nextKeypoints.rows(), lastKeypoints.rows());
//    kpr.outAdjMat = adj_mat;

    kpr.maskID = maskID;

    // reset the correspondence and residual image
    kernl_kp_reset<<<blocks, threads>>>(kpr);

    residualKernel<<<blocks, threads>>>(kpr);

    // construct adjacency matrix
//    kern_kp_dist<<<blocks, threads>>>(rgb);

    // find pairwise matches in adjacency matrix
//    kern_kp_match_dist<<<blocks, threads>>>(rgb);

//    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> am(nextKeypoints.rows(), lastKeypoints.rows());
//    adj_mat.download(am.data(), am.cols()*sizeof(float));
//    std::cout << am << std::endl;

//    for(int i=0; i<kpr.outAdjMat.rows; i++) {
//        for(int j=0; j<kpr.outAdjMat.cols; j++) {
//          std::cout << kpr.outAdjMat.ptr(i)[j] << " ";
//        }
//        std::cout << std::endl;
//    }

    float2 out_host = {0, 0};
    float2 * out;

    cudaMalloc(&out, sizeof(int2));
    cudaMemcpy(out, &out_host, sizeof(int2), cudaMemcpyHostToDevice);

    reduceSumF2<<<1, MAX_THREADS>>>(sumResidual, out, blocks);

//    std::cout << out_host.x << ", " << out_host.y << std::endl;

    cudaSafeCall(cudaGetLastError());
    cudaSafeCall(cudaDeviceSynchronize());

    cudaMemcpy(&out_host, out, sizeof(int2), cudaMemcpyDeviceToHost);
    cudaFree(out);

    count = int(out_host.x);
    sigmaSum = out_host.y;
}

struct SO3Reduction
{
    PtrStepSz<unsigned char> lastImage;
    PtrStepSz<unsigned char> nextImage;

    mat33 imageBasis;
    mat33 kinv;
    mat33 krlr;
    bool gradCheck;

    int cols;
    int rows;
    int N;

    JtJJtrSO3 * out;

    __device__ __forceinline__ float2
    getGradient(const PtrStepSz<unsigned char> img, int x, int y) const
    {
        float2 gradient;

        float actu = static_cast<float>(img.ptr(y)[x]);

        float back = static_cast<float>(img.ptr(y)[x - 1]);
        float fore = static_cast<float>(img.ptr(y)[x + 1]);
        gradient.x = ((back + actu) / 2.0f) - ((fore + actu) / 2.0f);

        back = static_cast<float>(img.ptr(y - 1)[x]);
        fore = static_cast<float>(img.ptr(y + 1)[x]);
        gradient.y = ((back + actu) / 2.0f) - ((fore + actu) / 2.0f);

        return gradient;
    }

    __device__ __forceinline__ JtJJtrSO3
    getProducts(int k) const
    {
        int y = k / cols;
        int x = k - (y * cols);

        bool found_coresp = false;

        float3 unwarpedReferencePoint = {float(x), float(y), 1.0f};

        float3 warpedReferencePoint = imageBasis * unwarpedReferencePoint;

        int2 warpedReferencePixel = {__float2int_rn(warpedReferencePoint.x / warpedReferencePoint.z),
                                     __float2int_rn(warpedReferencePoint.y / warpedReferencePoint.z)};

        if(warpedReferencePixel.x >= 1 &&
           warpedReferencePixel.x < cols - 1 &&
           warpedReferencePixel.y >= 1 &&
           warpedReferencePixel.y < rows - 1 &&
           x >= 1 &&
           x < cols - 1 &&
           y >= 1 &&
           y < rows - 1)
        {
            found_coresp = true;
        }

        float row[4];
        row[0] = row[1] = row[2] = row[3] = 0.f;

        if(found_coresp)
        {
            float2 gradNext = getGradient(nextImage, warpedReferencePixel.x, warpedReferencePixel.y);
            float2 gradLast = getGradient(lastImage, x, y);

            float gx = (gradNext.x + gradLast.x) / 2.0f;
            float gy = (gradNext.y + gradLast.y) / 2.0f;

            float3 point = kinv * unwarpedReferencePoint;

            float z2 = point.z * point.z;

            float a = krlr.data[0].x;
            float b = krlr.data[0].y;
            float c = krlr.data[0].z;

            float d = krlr.data[1].x;
            float e = krlr.data[1].y;
            float f = krlr.data[1].z;

            float g = krlr.data[2].x;
            float h = krlr.data[2].y;
            float i = krlr.data[2].z;

            //Aren't jacobians great fun
            float3 leftProduct = {((point.z * (d * gy + a * gx)) - (gy * g * y) - (gx * g * x)) / z2,
                                  ((point.z * (e * gy + b * gx)) - (gy * h * y) - (gx * h * x)) / z2,
                                  ((point.z * (f * gy + c * gx)) - (gy * i * y) - (gx * i * x)) / z2};

            float3 jacRow = cross(leftProduct, point);

            row[0] = jacRow.x;
            row[1] = jacRow.y;
            row[2] = jacRow.z;
            row[3] = -(static_cast<float>(nextImage.ptr(warpedReferencePixel.y)[warpedReferencePixel.x]) - static_cast<float>(lastImage.ptr(y)[x]));
        }

        JtJJtrSO3 values = {row[0] * row[0],
                            row[0] * row[1],
                            row[0] * row[2],
                            row[0] * row[3],

                            row[1] * row[1],
                            row[1] * row[2],
                            row[1] * row[3],

                            row[2] * row[2],
                            row[2] * row[3],

                            row[3] * row[3],
                            float(found_coresp)};

        return values;
    }

    __device__ __forceinline__ void
    operator () () const
    {
        JtJJtrSO3 sum = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

        for(int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x)
        {
            JtJJtrSO3 val = getProducts(i);

            sum.add(val);
        }

        sum = blockReduceSum(sum);

        if(threadIdx.x == 0)
        {
            out[blockIdx.x] = sum;
        }
    }
};

__global__ void so3Kernel (const SO3Reduction so3)
{
    so3();
}

void so3Step(const DeviceArray2D<unsigned char> & lastImage,
             const DeviceArray2D<unsigned char> & nextImage,
             const mat33 & imageBasis,
             const mat33 & kinv,
             const mat33 & krlr,
             DeviceArray<JtJJtrSO3> & sum,
             DeviceArray<JtJJtrSO3> & out,
             float * matrixA_host,
             float * vectorB_host,
             float * residual_host,
             int threads,
             int blocks)
{
    int cols = nextImage.cols();
    int rows = nextImage.rows();

    SO3Reduction so3;

    so3.lastImage = lastImage;

    so3.nextImage = nextImage;

    so3.imageBasis = imageBasis;
    so3.kinv = kinv;
    so3.krlr = krlr;

    so3.cols = cols;
    so3.rows = rows;

    so3.N = cols * rows;

    so3.out = sum;

    so3Kernel<<<blocks, threads>>>(so3);

    reduceSum<<<1, MAX_THREADS>>>(sum, out, blocks);

    cudaSafeCall(cudaGetLastError());
    cudaSafeCall(cudaDeviceSynchronize());

    float host_data[11];
    out.download((JtJJtrSO3 *)&host_data[0]);

    int shift = 0;
    for (int i = 0; i < 3; ++i)
    {
        for (int j = i; j < 4; ++j)
        {
            float value = host_data[shift++];
            if (j == 3)
                vectorB_host[i] = value;
            else
                matrixA_host[j * 3 + i] = matrixA_host[i * 3 + j] = value;
        }
    }

    residual_host[0] = host_data[9];
    residual_host[1] = host_data[10];
}
