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

#ifndef FLOW_CUDA_H
#define FLOW_CUDA_H


void ComputeTPSCVGPU(const float *p_valueI, // source frame
                     const float *c_value, // control points 
		     const float *c_pos, //position of control points
                     int width,       // frame width
                     int height,      // frame height
                     int stride,      // row access stride
                     int c_num,     // smoothness coefficient
                     float *tps_value,       // output vertical flow
                     float *K_cc);       //k* to be inverted 

#endif
