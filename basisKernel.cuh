#include "common.h"
__global__ void BasisFuncKernel(const float *c_pos, cv::gpu::PtrStepSzf tps_basis, int w, int h, int s, int c_num);
//__global__ void KstarFuncKernel(const float *c_pos, float *K_cc, int w, int h, int s, int c_num);

///////////////////////////////////////////////////////////////////////////////
/// \brief calculate tps basis funciton: u*u*log(u*u), CUDA kernel.
///
/// \param[in]  *c_pos		position of control points
/// \param[in]  c_num		number of control points
/// \param[in]  w       	width
/// \param[in]  h       	height
/// \param[in]  s       	stride
/// \param[out] *tps_basis 	result of basis function 
///////////////////////////////////////////////////////////////////////////////
static
void TpsBasisFunction(const float *c_pos,
                    int w, int h, int s, int c_num,
                    cv::gpu::PtrStepSzf M_tps_value_cp)
{
	// CTA size
	dim3 threads(32, 6);
	// grid size
	dim3 blocks(iDivUp(w, threads.x), iDivUp(h, threads.y));
	//int *p_pos = 0;
	BasisFuncKernel<<<blocks, threads>>>(c_pos, M_tps_value_cp, w, h, s, c_num);

	//int cudaerror = 0;	
	//cudaerror = cudaDeviceSynchronize();
	//printf(" cudaerror is %d \n",cudaerror);
	//printf(" pixel pos is %d \n",*p_pos);
}


///////////////////////////////////////////////////////////////////////////////
/// \brief calculate tps basis funciton: u*u*log(u*u), CUDA kernel.
///
/// \param[in]  *c_pos		position of control points
/// \param[in]  c_num		number of control points
/// \param[in]  w       	width
/// \param[in]  h       	height
/// \param[in]  s       	stride
/// \param[out] *K_cc 		calculate K* (to be send to CPU and invert) 
///////////////////////////////////////////////////////////////////////////////
//static
//void K_star_KFunction(const float *c_pos, 
		     //int w, int h, int s, int c_num,
		     //float *K_cc)
//{
	//// CTA size
	//dim3 threads(32, 6);
	//// grid size
	//dim3 blocks(iDivUp(c_num + 3, threads.x), iDivUp(c_num + 3, threads.y));
	////int *p_pos = 0;
	//KstarFuncKernel<<<blocks, threads>>>(c_pos, K_cc, w, h, s, c_num);

//}


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
//__global__
//void KstarFuncKernel(const float *c_pos,
		//float *K_cc,
		//int w, int h, int s, int c_num)
//{
	//const int ix = threadIdx.x + blockIdx.x * blockDim.x;
	//const int iy = threadIdx.y + blockIdx.y * blockDim.y;

	//// position within global memory array
	//int pos = min(ix, c_num + 2) + min(iy, c_num + 2) * s;

	//if (ix >= (c_num + 3) || iy >= (c_num + 3)) return;
  
	//int c_tmp_w_x = 0; 
	//int c_tmp_h_x = 0;
	//int c_tmp_w_y = 0; 
	//int c_tmp_h_y = 0;
	//int pixel_pos_w = 0;
	//int pixel_pos_h = 0;
	//int dif_pos = 0;
	//// calculate K*  basis function
	//double cc_pos_norm_pow2 = 0;

	//if ((ix == iy) || ((ix >= c_num) && (iy >= c_num))) 
		//K_cc[pos] = 0;
	//else if ((ix <= (c_num - 1)) && (iy <= (c_num - 1)))
	//{
		//c_tmp_w_x = c_pos[ix];
		//c_tmp_h_x = c_pos[ix + 1]; 

		//c_tmp_w_y = c_pos[iy];
		//c_tmp_h_y = c_pos[iy + 1]; 

		//cc_pos_norm_pow2 = (c_tmp_w_x - c_tmp_w_y) * (c_tmp_w_x - c_tmp_w_y) + (c_tmp_h_x - c_tmp_h_y) * (c_tmp_h_x - c_tmp_h_y);
		//K_cc[pos]= cc_pos_norm_pow2 *  log(cc_pos_norm_pow2); 
	//}
	//else
	//{
		//if ( ix <= (c_num - 1))
		//{
			//pixel_pos_w = c_pos[ix];
			//pixel_pos_h = c_pos[ix + 1];
			//// get rid of one if , WAW, but the same value
			//dif_pos = iy - c_num;
			//if (dif_pos == 0)
				//K_cc[pos] = 1;
			//else if (dif_pos == 1)
				//K_cc[pos] = pixel_pos_w;
			//else
				//K_cc[pos] = pixel_pos_h;
		//}

		//if ( iy <= (c_num - 1))
		//{
			//pixel_pos_w = c_pos[iy];
			//pixel_pos_h = c_pos[iy + 1];
			//// get rid of one if , WAW, but the same value
			//dif_pos = ix - c_num;
			//if (dif_pos == 0)
				//K_cc[pos] = 1;
			//else if (dif_pos == 1)
				//K_cc[pos] = pixel_pos_w;
			//else
				//K_cc[pos] = pixel_pos_h;
		//}
	//}
//}




/////////////////////////////////////////////////////////////////////////////////
///// \brief calculate tps basis funciton: u*u*log(u*u), CUDA kernel.
/////
///// \param[in]  *c_pos          	position of control points
///// \param[in]  w       		width
///// \param[in]  h       		height
///// \param[in]  s       		stride
///// \param[in]  c_num             	number of control points	
///// \param[out] *M_tps_value_cp 	result of basis function 
/////////////////////////////////////////////////////////////////////////////////
__global__
void BasisFuncKernel(const float *c_pos,
		cv::gpu::PtrStepSzf M_tps_value_cp,
		int w, int h, int s, int c_num)
{
	const int ix = threadIdx.x + blockIdx.x * blockDim.x;
	const int iy = threadIdx.y + blockIdx.y * blockDim.y;

	// position within global memory array
	int pos = min(ix, w - 1) + min(iy, h - 1) * s;

	if (ix >= w || iy >= h) return;

	int num = 0;
	int c_count = c_num;
	int temp_ix = ix;
	int temp_iy = iy;
	int c_tmp_w = 0; 
	int c_tmp_h = 0;
	int c_pos_pixel = 0;
	int tps_pos = pos;
	//int wrong_index = 0; 
	for (;num < c_num * 2;)
	{
		c_tmp_w = c_pos[c_num * 2 - num - 2];
		c_tmp_h = c_pos[c_num * 2 - num - 1];
		c_pos_pixel = c_tmp_w + c_tmp_h * s;
		if (pos > c_pos_pixel)
		{
			tps_pos = pos - c_count;
			//if (temp_ix > (num + 1))
			//{
			//temp_ix = temp_ix - (num + 1);
			//} else 
			//{
			//temp_iy = temp_iy - 1;
			//temp_ix = w - 1 - (num + 1);
			//}
			//if (temp_ix < 0 || temp_iy < 0)
			//printf("\n index error temp_ix = %d and temp_iy = %d \n", temp_ix, temp_iy);;
			//wrong_index = pos;	
			break;
		} else
		{
			if (pos == c_pos_pixel)
				return;
		}
		c_count = c_count - 1;	
		num = num + 2;
	}
	// calculate tps basis function
	double cp_pos_norm_pow2 = 0;
	num = 0;
	c_count = 0;
	for(;num < c_num * 2;)
	{
		c_tmp_w = c_pos[num];
		c_tmp_h = c_pos[num + 1];
		cp_pos_norm_pow2 = (temp_ix - c_tmp_w) * (temp_ix -c_tmp_w) + (temp_iy - c_tmp_h) * (temp_iy - c_tmp_h);
		//M_tps_value_cp[tps_pos + c_count * (h * w - c_num)]= cp_pos_norm_pow2 *  log(cp_pos_norm_pow2); 
		M_tps_value_cp(tps_pos,  c_count)= cp_pos_norm_pow2 *  log(cp_pos_norm_pow2); 
		//tps_basis[tps_pos + c_count * (h * w - c_num)]= cp_pos_norm_pow2; 
		if (c_count == 0)
		{
			M_tps_value_cp(tps_pos,  c_num) = 1; 
			M_tps_value_cp(tps_pos,  c_num + 1) = temp_ix; 
			M_tps_value_cp(tps_pos,  c_num + 2) = temp_iy; 
		}	
		c_count = c_count + 1; 
		num = num + 2;	
	}	
}

