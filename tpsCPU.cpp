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
#include "tpsCPU.h"
#include "cv.h"
#include <iostream>

void TpsBasisFunc_CP(float *c_pos, int width, int height, int stride, int c_num, float *M_tps_value_cp, cv::Mat cv_M);
void K_star_tps(float *c_pos, int c_num, cv::Mat K_star);
///////////////////////////////////////////////////////////////////////////////
/// \brief method logic
///
/// handles memory allocation and control flow
/// \param[in]  p_value      pixel value of source image
/// \param[in]  c_value      value of control points
/// \param[in]  c_pos        position of control points
/// \param[in]  width        images width
/// \param[in]  height       images height
/// \param[in]  stride       images stride
/// \param[in]  c_num	     number of control points 
/// \param[out] tps_value    tps result of computeflowgold
///////////////////////////////////////////////////////////////////////////////
void ComputeTPSCPU(float *p_value, float *c_value, float *c_pos,
                     int width, int height, int stride, int c_num,
                     float *tps_value)
{
 printf("Computing tps on CPU...\n");
 
 const int pixelCountAligned = height * stride;
 const int pixelAvailable = height * stride - c_num;
 float *p_value_cpu = new float [pixelCountAligned];
 float *c_value_cpu = new float [c_num];
 float *c_pos_cpu = new float [c_num * 2];
 p_value_cpu = p_value;
 c_value_cpu = c_value;
 c_pos_cpu = c_pos; 

 //float *M_U = new float [pixelAvailable * c_num];
 float *M = new float [pixelAvailable * (c_num + 3)];
 //float *K_star = new float [(c_num + 3) * c_num];
 cv::Mat cv_M = cv::Mat::zeros(pixelAvailable, c_num + 3, CV_32FC1);
 cv::Mat K_star = cv::Mat::zeros(c_num + 3, c_num, CV_32FC1);
 cv::Mat I = cv::Mat::zeros(pixelAvailable, 1, CV_32FC1);

 // M (mx(n+3)) : reshape M = [M_U M_P] M_P: m pixel points
 TpsBasisFunc_CP(c_pos_cpu, width, height, stride, c_num, M, cv_M);
 
 //printf("finish tpsbasisfunc \n");
 // K* ((n+3)xn) : 
 K_star_tps(c_pos_cpu, c_num, K_star); // copy
 
 //std::cout << "K*: " << K_star << std::endl;
 //int i = 0;
 //int j = 0;
 //int dif_count  = 0;
 //for (; i < pixelAvailable; i++)
 //{
	 //for (; j < c_num + 3; j++)
	 //{
		 //if (cv_M.at<float>(i,j) != M[j * pixelAvailable + i])
			 //dif_count++;
	 //}
 //}
 //printf("dif_count is  %d \n", dif_count);
 //construct I = M * K* * Y
 cv::Mat Y = cv::Mat::zeros(c_num, 1, CV_32FC1);
 int c_count = 0;
 for (;c_count < c_num; c_count++)
 {
	Y.at<float>(c_count, 0) = c_value_cpu[c_count];
 }

 I = cv_M * K_star * Y;
 
 memcpy(tps_value, M, pixelAvailable * (c_num + 3) * sizeof(float));
 
 // cleanup
 delete [] M;
}

///////////////////////////////////////////////////////////////////////////////
/// \brief tps basis function: control points and pixel points 
///
/// compute tps basis function between control points and pixel points 
/// \param[in]  c_pos 	     position of control points
/// \param[in]  width        images width
/// \param[in]  height       images height
/// \param[in]  stride       images stride
/// \param[in]  c_num        number of control points 
/// \param[out] M_tps_value_cp tps basis function between control points and pixel points
//////////////////////////////////////////////////////////////////////////////
void TpsBasisFunc_CP(float *c_pos, int width, int height, int stride, int c_num, float *M_tps_value_cp, cv::Mat cv_M)
{ 
 //param for M_U 
 // M_U (mxn) : compute basis function: n control points v.s m pixel points
 int c_count = 0;
 int tps_count = 0;
 int value_status = 1;
 int c_pos_cur = 0; 
 //param for M_pixel	
 int assign_count = 0;
 int pixelAvailable = width * height - c_num;
 int ones_pos = pixelAvailable * c_num;
 int x_pos = pixelAvailable * (c_num + 1);
 int y_pos = pixelAvailable * (c_num + 2);
 
 for (; c_count < c_num * 2; )
 {
 int c_tmp_w = c_pos[c_count];
 int c_tmp_h = c_pos[c_count + 1];
 for (int i = 0; i < height; ++i)
 {
  for (int j = 0; j < width; ++j)
  {
   int pos = j + i * stride;
   for (int k = 0; k < c_num * 2;)
   {
    int c_pos_cur = c_pos[k] + c_pos[k + 1] * stride;
    if (pos == c_pos_cur)
    {
     value_status = 0;
    }
    k = k + 2;
   }
   if (value_status == 1)
   {
    double cp_pos_norm_pow2 = pow(abs(j - c_tmp_w),2) + pow(abs(i - c_tmp_h),2); 
    M_tps_value_cp[tps_count] = cp_pos_norm_pow2 * log(cp_pos_norm_pow2);
    // temp for cv_M
    int cv_c_pos = (tps_count - tps_count % pixelAvailable) / pixelAvailable;  
    int cv_p_pos = tps_count % pixelAvailable;
    cv_M.at<float>(cv_p_pos, cv_c_pos) = M_tps_value_cp[tps_count];
    if (assign_count < pixelAvailable)
    {
     M_tps_value_cp[tps_count + ones_pos] = 1;
     M_tps_value_cp[tps_count + x_pos] = j;
     M_tps_value_cp[tps_count + y_pos] = i;
     cv_M.at<float>(cv_p_pos, c_num) = 1;
     cv_M.at<float>(cv_p_pos, c_num + 1) = j;
     cv_M.at<float>(cv_p_pos, c_num + 2) = i;
     assign_count ++;
    }	
    tps_count++;
   }
   value_status = 1;
  }
 }
 c_count = c_count + 2;
 }
} 
///////////////////////////////////////////////////////////////////////////////
/// \brief calculate K_start(k*)  function: control points 
/// K_star ((n+3)*n):
/// K = [K_star_L , K_star_pixel ; K_star_pixel(t) , 0]
/// K -> (inverse) K(-1)
/// K(-1) -> get ((n+3)*n) K_star
/// \param[in]  c_pos 	     position of control points
/// \param[in]  c_num 	     number of control points 
/// \param[out] M_tps_value_cp tps basis function between control points and pixel points
//////////////////////////////////////////////////////////////////////////////
void K_star_tps(float *c_pos, int c_num, cv::Mat K_star)
{
 int K_pos = 0;
 int K_star_pos = 0;
 //param for K_star_L	
 int count_ori = 0;
 int count_tmp = 0;
 int c_ori_w = 0;
 int c_ori_h = 0;
 int c_tmp_w = 0;
 int c_tmp_h = 0;
 int i,j = 0;
 double cc_pos_norm_pow2 = 0;
 //param for K_star_pixel
 int assign_count = 0;

 cv::Mat K = cv::Mat::zeros(c_num + 3,c_num + 3,CV_32FC1);
 float *K_star_L = new float [c_num * c_num];
 float *K_star_pixel = new float [c_num * 3];

 //calculate K_star_L
 for (i = 0; i < c_num * 2;)
 {
  c_ori_w = c_pos[i];
  c_ori_h = c_pos[i + 1];

  for (j = 0; j < c_num * 2;)
  {
   K_star_pos = count_ori + count_tmp * c_num;
   K_pos = count_ori + count_tmp * (c_num + 3);
   c_tmp_w = c_pos[j];
   c_tmp_h = c_pos[j + 1];

   if (i == j)
   {
    K_star_L[K_star_pos] = 0;
    K.at<float>(count_ori, count_tmp) = 0;
   } else 				
   {
    cc_pos_norm_pow2 = pow(abs(c_ori_w - c_tmp_w),2) + pow(abs(c_ori_h - c_tmp_h),2); 
    K_star_L[K_star_pos] = cc_pos_norm_pow2 * log(cc_pos_norm_pow2);
    K.at<float>(count_ori, count_tmp) =  K_star_L[K_star_pos];
   }
   
   //create K_star_pixel and its transpose
   if (assign_count < c_num)
   {
    K_star_pixel[count_tmp] = 1;
    K_star_pixel[count_tmp + c_num] = c_tmp_w;
    K_star_pixel[count_tmp + c_num * 2] = c_tmp_h;
    //assign K_star_pixel and its transpose to K
    //assign K_star_pixel
    K.at<float>(c_num, count_tmp) =  1;
    K.at<float>(c_num + 1, count_tmp) =  c_tmp_w;
    K.at<float>(c_num + 2, count_tmp) =  c_tmp_h;
    //assign tramspose 
    K.at<float>(count_tmp, c_num) =  1;
    K.at<float>(count_tmp, c_num + 1) =  c_tmp_w;
    K.at<float>(count_tmp, c_num + 2) =  c_tmp_h;

    assign_count++;
   }

   count_tmp++;
   j = j + 2;
  }

  count_tmp = 0;
  count_ori++;
  i = i + 2;
 }
 
 //printf("start fill 0 \n");
 ////fiil the rest of K with 0
 //for (i = 0; i < 3; i++) 
 //{
  //K[c_num + (c_num + i) * (c_num + 3)] = 0;
  //K[(c_num + 1) + (c_num + i) * (c_num + 3)] = 0;
  //K[(c_num + 2) + (c_num + i) * (c_num + 3)] = 0;
 //}

     
 //printf("start show K \n");
 //for (i = 0; i < (c_num + 3); i++)
 //{
  //for (j = 0; j < (c_num + 3); j++)
  //{
   //printf("%f \t",K[i + j * (c_num + 3)]);
   //if (j == (c_num + 2))
   //{
    //printf("\n");
   //}
  //}
 //}
 ////test cv invert
 //cv::Mat A(c_num + 3,c_num + 3,CV_32FC1);

 //for (unsigned int i = 0; i < c_num + 3; i++)
 //{
  //for (unsigned int j = 0; j < c_num + 3; j++)
  //{
   //A.at<float>(i,j) = K[i + j * (c_num + 3)];
  //}
 //}

 cv::Mat inverted(c_num + 3,c_num + 3,CV_32FC1);
 invert(K, inverted, cv::DECOMP_SVD);
 
 std::cout << "k: "<< K << std::endl;
 std::cout << "inverted: " << inverted << std::endl;

 
 printf("finish show K \n");

 printf("start fill result \n");
 //fill results to K_star
 for (i = 0; i < (c_num + 3); i++)
 {
  for (j = 0; j < c_num; j++)
  {
	  //K_star_pos = i + j * (c_num + 3);
   K_star.at<float>(i,j) = inverted.at<float>(i,j);
  }
 }


 delete [] K_star_L;
 delete [] K_star_pixel;
}
   









