/*
    This file is part of ParTI!.

    ParTI! is free software: you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License as
    published by the Free Software Foundation, either version 3 of
    the License, or (at your option) any later version.

    ParTI! is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU Lesser General Public
    License along with ParTI!.
    If not, see <http://www.gnu.org/licenses/>.
*/
#include <ParTI.h>
#include "sptensor.h"
#include <cuda_runtime.h>


template <typename T>
__device__ static void print_array(const T array[], sptNnzIndex length, T start_index) {
    if(length == 0) {
        return;
    }
    printf("%d", (int) (array[0] + start_index));
    sptNnzIndex i;
    for(i = 1; i < length; ++i) {
        printf(", %d", (int) (array[i] + start_index));
    }
    printf("\n");
}


__device__ static void print_array(const sptValue array[], sptNnzIndex length, sptNnzIndex start_index) {
    if(length == 0) {
        return;
    }
    printf("%.2f", array[0] + start_index);
    sptNnzIndex i;
    for(i = 1; i < length; ++i) {
        printf(", %.2f", array[i] + start_index);
    }
    printf("\n");
}


/* impl_num = 15 */
__global__ void spt_MTTKRPKernelRankSplitNnz3DOneKernel(
    const sptIndex mode,
    const sptIndex nmodes,
    const sptNnzIndex nnz,
    const sptIndex R,
    const sptIndex stride,
    const sptIndex * Xndims,
    sptIndex ** const Xinds,
    const sptValue * Xvals,
    const sptIndex * dev_mats_order,
    sptValue ** dev_mats)
{
    sptNnzIndex num_loops_nnz = 1;
    sptNnzIndex const nnz_per_loop = gridDim.x * blockDim.y;
    if(nnz > nnz_per_loop) {
        num_loops_nnz = (nnz + nnz_per_loop - 1) / nnz_per_loop;
    }

    const sptNnzIndex tidx = threadIdx.x;  // index rank
    const sptNnzIndex tidy = threadIdx.y;  // index nnz
    sptNnzIndex x;
    const sptIndex num_loops_r = R / blockDim.x;
    const sptIndex rest_loop = R - num_loops_r * blockDim.x;


    sptIndex const * const mode_ind = Xinds[mode];
    sptValue * const mvals = (sptValue*)dev_mats[nmodes];
    sptIndex times_mat_index = dev_mats_order[1];
    sptValue * times_mat = dev_mats[times_mat_index];
    sptIndex * times_inds = Xinds[times_mat_index];
    sptIndex times_mat_index_2 = dev_mats_order[2];
    sptValue * times_mat_2 = dev_mats[times_mat_index_2];
    sptIndex * times_inds_2 = Xinds[times_mat_index_2];

    for(sptNnzIndex nl=0; nl<num_loops_nnz; ++nl) {
        x = blockIdx.x * blockDim.y + tidy + nl * nnz_per_loop;
        if(x < nnz) {
            sptIndex const mode_i = mode_ind[x];
            sptIndex tmp_i = times_inds[x];
            sptValue const entry = Xvals[x];
            sptIndex tmp_i_2 = times_inds_2[x];
            sptValue tmp_val = 0;
            sptIndex r;

            for(sptIndex l=0; l<num_loops_r; ++l) {
                r = tidx + l * blockDim.x;
                tmp_val = entry * times_mat[tmp_i * stride + r] * times_mat_2[tmp_i_2 * stride + r];
                atomicAdd(&(mvals[mode_i * stride + r]), tmp_val);
            }

            if(rest_loop > 0 && tidx < rest_loop) {
                r = tidx + num_loops_r * blockDim.x;
                tmp_val = entry * times_mat[tmp_i * stride + r] * times_mat_2[tmp_i_2 * stride + r];
                atomicAdd(&(mvals[mode_i * stride + r]), tmp_val);
            }
        }
   
    }

}


/* impl_num = 16 */
__global__ void spt_MTTKRPKernelRankSplitNnzRB3DOneKernel(
    const sptIndex mode,
    const sptIndex nmodes,
    const sptNnzIndex nnz,
    const sptIndex R,
    const sptIndex stride,
    const sptIndex * Xndims,
    sptIndex ** const Xinds,
    const sptValue * Xvals,
    const sptIndex * dev_mats_order,
    sptValue ** dev_mats)
{
    sptNnzIndex num_loops_nnz = 1;
    sptNnzIndex const nnz_per_loop = gridDim.x * blockDim.y;
    if(nnz > nnz_per_loop) {
        num_loops_nnz = (nnz + nnz_per_loop - 1) / nnz_per_loop;
    }

    const sptNnzIndex tidx = threadIdx.x;  // index rank
    const sptNnzIndex tidy = threadIdx.y;  // index nnz
    sptNnzIndex x;
    const sptIndex num_loops_r = R / blockDim.x;
    const sptIndex rest_loop = R - num_loops_r * blockDim.x;
    sptIndex r;


    sptIndex const * const mode_ind = Xinds[mode];
    sptValue * const mvals = (sptValue*)dev_mats[nmodes];
    sptIndex times_mat_index = dev_mats_order[1];
    sptValue * times_mat = dev_mats[times_mat_index];
    sptIndex * times_inds = Xinds[times_mat_index];
    sptIndex times_mat_index_2 = dev_mats_order[2];
    sptValue * times_mat_2 = dev_mats[times_mat_index_2];
    sptIndex * times_inds_2 = Xinds[times_mat_index_2];


    for(sptIndex l=0; l<num_loops_r; ++l) {
        r = tidx + l * blockDim.x;

        for(sptNnzIndex nl=0; nl<num_loops_nnz; ++nl) {
            x = blockIdx.x * blockDim.y + tidy + nl * nnz_per_loop;
            if(x < nnz) {
                sptIndex const mode_i = mode_ind[x];
                sptIndex tmp_i = times_inds[x];
                sptValue const entry = Xvals[x];
                sptIndex tmp_i_2 = times_inds_2[x];
                sptValue tmp_val = 0;

                tmp_val = entry * times_mat[tmp_i * stride + r] * times_mat_2[tmp_i_2 * stride + r];
                atomicAdd(&(mvals[mode_i * stride + r]), tmp_val);
            }
        }
    }  // End for l: num_loops_r

    if(rest_loop > 0 && tidx < rest_loop) {
        r = tidx + num_loops_r * blockDim.x;

        for(sptNnzIndex nl=0; nl<num_loops_nnz; ++nl) {
            x = blockIdx.x * blockDim.y + tidy + nl * nnz_per_loop;
            if(x < nnz) {
                sptIndex const mode_i = mode_ind[x];
                sptIndex tmp_i = times_inds[x];
                sptValue const entry = Xvals[x];
                sptIndex tmp_i_2 = times_inds_2[x];
                sptValue tmp_val = 0;

                tmp_val = entry * times_mat[tmp_i * stride + r] * times_mat_2[tmp_i_2 * stride + r];
                atomicAdd(&(mvals[mode_i * stride + r]), tmp_val);
            }
        }
    }   // End if rest_loop

}


/* impl_num = 09 */
__global__ void spt_MTTKRPKernelScratch(
    const sptIndex mode,
    const sptIndex nmodes,
    const sptNnzIndex nnz,
    const sptIndex R,
    const sptIndex stride,
    const sptIndex * Xndims,
    sptIndex ** const Xinds,
    const sptValue * Xvals,
    const sptIndex * dev_mats_order,
    sptValue ** dev_mats,
    sptValue * dev_scratch,
    sptNnzIndex block_offset) 
{
    const sptNnzIndex tidx = threadIdx.x;
    const sptNnzIndex x = (blockIdx.x + block_offset) * blockDim.x + tidx;

    sptIndex const * const mode_ind = Xinds[mode];
    /* The 64-bit floating-point version of atomicAdd() is only supported by devices of compute capability 6.x and higher. */
    sptValue * const mvals = (sptValue*)dev_mats[nmodes];

    if(x < nnz) {
      sptIndex times_mat_index = dev_mats_order[1];
      sptValue * times_mat = dev_mats[times_mat_index];
      sptIndex * times_inds = Xinds[times_mat_index];
      sptIndex tmp_i = times_inds[x];
      sptValue const entry = Xvals[x];
      for(sptIndex r=0; r<R; ++r) {
        dev_scratch[x * stride + r] = entry * times_mat[tmp_i * stride + r];
      }

      for(sptIndex i=2; i<nmodes; ++i) {
        times_mat_index = dev_mats_order[i];
        times_mat = dev_mats[times_mat_index];
        times_inds = Xinds[times_mat_index];
        tmp_i = times_inds[x];
        for(sptIndex r=0; r<R; ++r) {
          dev_scratch[x * stride + r] *= times_mat[tmp_i * stride + r];
        }
      }

    }
   __syncthreads();

    if(x < nnz) {
      sptIndex const mode_i = mode_ind[x];
      // printf("x: %lu, mode_i: %lu\n", x, mode_i);
      for(sptIndex r=0; r<R; ++r) {
        atomicAdd(&(mvals[mode_i * stride + r]), dev_scratch[x * stride + r]);
      }
    }
   __syncthreads();

}




