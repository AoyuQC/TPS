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
#include "cv.h"
#include <iostream>
#include "opencv.hpp"
#include "gpu/gpu.hpp"
#include "core/cuda_devptrs.hpp"

#include "tpsGPU.h"

//private define
//#define DEBUG_K_STAR_GPU
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
void ComputeTPSCVGPU(const float *p_value, 
		     const float *c_value,
		     const float *c_pos, 
		     int width, int height, int stride, int c_num,
                     cv::Mat I, float *K_cc)
{
 printf("Computing tps map on GPU...\n");
 
 // M (mx(n+3)) : reshape M = [M_U M_P] M_P: m pixel points
 cv::gpu::GpuMat gpu_M;
 cv::Mat M;
 gpu_M.create(width * stride - c_num, c_num + 3, CV_32FC1);

 ComputeTPSGPU(p_value, c_value, c_pos, width, height, stride, c_num, gpu_M, K_cc); 

 gpu_M.download(M);

 // K* ((n+3)xn) : 
 cv::Mat temp_c_pos;
 temp_c_pos.create(c_num, 2, CV_32FC1);
 int i = 0;
 for (; i < (2 * c_num - 1);)
 {
 	temp_c_pos.at<float>(i/2, 0) = c_pos[i];	
 	temp_c_pos.at<float>(i/2, 1) = c_pos[i + 1];	
 	i = i + 2;
 }
 cv::gpu::GpuMat op_c_pos(temp_c_pos);
 cv::gpu::GpuMat op_K_cc;
 op_K_cc.create((c_num + 3), (c_num + 3), CV_32FC1); 
 // compute K for onpencv to invert
 K_star_KFunction(op_c_pos, width, height, stride, c_num, op_K_cc);
 
 cv::Mat gpu_host_inverted(c_num + 3,c_num + 3,CV_32FC1);
 cv::Mat gpu_host_K(c_num + 3,c_num + 3,CV_32FC1);
 cv::Mat gpu_K_star(c_num + 3,c_num,CV_32FC1);

 op_K_cc.download(gpu_host_K);
 invert(gpu_host_K, gpu_host_inverted, cv::DECOMP_SVD);
 gpu_K_star = gpu_host_inverted.colRange(0, c_num).rowRange(0, c_num + 3);

 #ifdef DEBUG_K_STAR_GPU
	 std::cout << "gpu k: "<< gpu_host_K << std::endl;
	 std::cout << "inverted: " << gpu_host_inverted << std::endl;
	 std::cout << "k*: " << gpu_K_star << std::endl;
 #endif

 //construct I = M * K* * Y
 cv::Mat Y = cv::Mat::zeros(c_num, 1, CV_32FC1);
 int c_count = 0;
 for (;c_count < c_num; c_count++)
 {
	Y.at<float>(c_count, 0) = c_value[c_count];
 }
 
 I = M * gpu_K_star * Y;  
}

