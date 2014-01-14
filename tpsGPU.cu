
/*
 * Copyright 1993-2013 NVIDIA Corporation.  All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property and
 * proprietary rights in and to this software and related documentation.
 * Any use, reproduction, disclosure, or distribution of this software
 * and related documentation without an express license agreement from
 * NVIDIA Corporation is strictly prohibited.
 *
 * Please refer to the applicable NVIDIA end user license agreement (EULA)
 * associated with this source code for terms and conditions that govern
 * your use of this NVIDIA software.
 *
 */

#include "common.h"
#include "core/cuda_devptrs.hpp"

// include kernels
//#include "downscaleKernel.cuh"
//#include "upscaleKernel.cuh"
//#include "warpingKernel.cuh"
//#include "derivativesKernel.cuh"
//#include "solverKernel.cuh"
//#include "addKernel.cuh"
#include "basisKernel.cuh"

__global__
void KstarFuncKernel(cv::gpu::PtrStepSzf dev_c_pos,
		cv::gpu::PtrStepSzf  K_cc,
		int w, int h, int s, int c_num);

///////////////////////////////////////////////////////////////////////////////
/// \brief method logic
///
/// handles memory allocations, control flow
/// \param[in]  p_value      		pixel value of source image
/// \param[in]  c_value      		value of control points
/// \param[in]  c_pos        		position of control points
/// \param[in]  width        		images width
/// \param[in]  height       		images height
/// \param[in]  stride       		images stride
/// \param[in]  c_num	     		number of control points 
/// \param[out] M_tps_value_cp    	tps result of computeflowgold
///////////////////////////////////////////////////////////////////////////////
void ComputeTPSGPU(const float *p_value, 
		     const float *c_value,
		     const float *c_pos, 
		     int width, int height, int stride, int c_num,
                     cv::gpu::PtrStepSzf M_tps_value_cp, float *K_cc)
{
    // device memory pointers
    float *dev_c_pos;

    //const int dataSize = stride * height * sizeof(float);
    const int cposSize = c_num * 2  * sizeof(float);

    checkCudaErrors(cudaMalloc((void**)&dev_c_pos, cposSize));

    checkCudaErrors(cudaMemcpy((void *)dev_c_pos, c_pos, cposSize, cudaMemcpyHostToDevice));
    
    TpsBasisFunction(dev_c_pos, width, height, stride, c_num, M_tps_value_cp);

    
    // cleanup
    checkCudaErrors(cudaFree(dev_c_pos));
}


///////////////////////////////////////////////////////////////////////////////
/// \brief calculate tps basis funciton: u*u*log(u*u), CUDA kernel.
///
/// \param[in]  *dev_c_pos		position of control points
/// \param[in]  c_num		number of control points
/// \param[in]  w       	width
/// \param[in]  h       	height
/// \param[in]  s       	stride
/// \param[out] *dev_K_cc 		calculate K* (to be send to CPU and invert) 
///////////////////////////////////////////////////////////////////////////////
void K_star_KFunction(cv::gpu::PtrStepSzf dev_c_pos, int w, int h, int s, int c_num, cv::gpu::PtrStepSzf dev_K_cc)
{
	// CTA size
	dim3 threads(32, 6);
	// grid size
	dim3 blocks(iDivUp(c_num + 3, threads.x), iDivUp(c_num + 3, threads.y));
	//int *p_pos = 0;
	KstarFuncKernel<<<blocks, threads>>>(dev_c_pos, dev_K_cc, w, h, s, c_num);

}


///////////////////////////////////////////////////////////////////////////////
/// \brief calculate tps basis funciton: u*u*log(u*u), CUDA kernel.
///
/// \param[in]  *c_pos          	position of control points
/// \param[in]  w       		width
/// \param[in]  h       		height
/// \param[in]  s       		stride
/// \param[in]  c_num             	number of control points	
/// \param[out] *M_tps_value_cp 	result of basis function 
///////////////////////////////////////////////////////////////////////////////
__global__
void KstarFuncKernel(cv::gpu::PtrStepSzf dev_c_pos,
		cv::gpu::PtrStepSzf  K_cc,
		int w, int h, int s, int c_num)
{
	const int ix = threadIdx.x + blockIdx.x * blockDim.x;
	const int iy = threadIdx.y + blockIdx.y * blockDim.y;

	// position within global memory array
	//int pos = min(ix, c_num + 2) + min(iy, c_num + 2) * s;

	if (ix >= (c_num + 3) || iy >= (c_num + 3)) return;
  
	int c_tmp_w_x = 0; 
	int c_tmp_h_x = 0;
	int c_tmp_w_y = 0; 
	int c_tmp_h_y = 0;
	int pixel_pos_w = 0;
	int pixel_pos_h = 0;
	int dif_pos = 0;
	// calculate K*  basis function
	double cc_pos_norm_pow2 = 0;

	if ((ix == iy) || ((ix >= c_num) && (iy >= c_num))) 
		K_cc(ix, iy) = 0;
	else if ((ix <= (c_num - 1)) && (iy <= (c_num - 1)))
	{
		c_tmp_w_x = dev_c_pos(ix, 0);
		c_tmp_h_x = dev_c_pos(ix, 1); 

		c_tmp_w_y = dev_c_pos(iy, 0);
		c_tmp_h_y = dev_c_pos(iy, 1); 

		cc_pos_norm_pow2 = (c_tmp_w_x - c_tmp_w_y) * (c_tmp_w_x - c_tmp_w_y) + (c_tmp_h_x - c_tmp_h_y) * (c_tmp_h_x - c_tmp_h_y);
		K_cc(ix, iy)= cc_pos_norm_pow2 *  log(cc_pos_norm_pow2); 
	}
	else
	{
		if ( ix <= (c_num - 1))
		{
			pixel_pos_w = dev_c_pos(ix, 0);
			pixel_pos_h = dev_c_pos(ix, 1);
			// get rid of one if , WAW, but the same value
			dif_pos = iy - c_num;
			if (dif_pos == 0)
				K_cc(ix, iy) = 1;
			else if (dif_pos == 1)
				K_cc(ix, iy) = pixel_pos_w;
			else
				K_cc(ix, iy) = pixel_pos_h;
		}

		if ( iy <= (c_num - 1))
		{
			pixel_pos_w = dev_c_pos(iy, 0);
			pixel_pos_h = dev_c_pos(iy, 1);
			// get rid of one if , WAW, but the same value
			dif_pos = ix - c_num;
			if (dif_pos == 0)
				K_cc(ix, iy) = 1;
			else if (dif_pos == 1)
				K_cc(ix, iy) = pixel_pos_w;
			else
				K_cc(ix, iy) = pixel_pos_h;
		}
	}
}

