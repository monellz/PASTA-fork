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
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include "sptensor.h"


/* impl_num = 14 */
__global__ void spt_TTMRankRBNnzKernel(
    sptValue *Y_val, 
    sptIndex Y_stride, 
    sptNnzIndex Y_nnz,
    const sptValue * __restrict__ X_val, 
    sptNnzIndex X_nnz, 
    const sptIndex * __restrict__ X_inds_m,
    const sptNnzIndex * __restrict__ fiberidx_val, 
    sptNnzIndex fiberidx_len,
    const sptValue * __restrict__ U_val, 
    sptIndex U_nrows, 
    sptIndex U_ncols, 
    sptIndex U_stride)
{
    sptNnzIndex num_loops_nnz = 1;
    sptNnzIndex const nnz_per_loop = gridDim.x * blockDim.y;
    if(Y_nnz > nnz_per_loop) {
        num_loops_nnz = (Y_nnz + nnz_per_loop - 1) / nnz_per_loop;
    }

    const sptNnzIndex tidx = threadIdx.x;    // Index rank
    const sptNnzIndex tidy = threadIdx.y;    // Index nnz
    sptNnzIndex x;
    const sptIndex num_loops_r = U_ncols / blockDim.x;
    const sptIndex rest_loop = U_ncols - num_loops_r * blockDim.x;
    sptIndex r;

    for(sptIndex l=0; l<num_loops_r; ++l) {
        r = tidx + l * blockDim.x;

        for(sptNnzIndex nl=0; nl<num_loops_nnz; ++nl) {
            x = blockIdx.x * blockDim.y + tidy + nl * nnz_per_loop;
            if(x < Y_nnz) {
                const sptNnzIndex inz_begin = fiberidx_val[x];
                const sptNnzIndex inz_end = fiberidx_val[x+1];

                for(sptNnzIndex i = inz_begin; i < inz_end; ++i) {
                    const sptIndex row = X_inds_m[i];
                    Y_val[x*Y_stride + r] += X_val[i] * U_val[row*U_stride + r];
                    __syncthreads();
                }
            }
        }
    }

    if(rest_loop > 0 && tidx < rest_loop) {
        r = tidx + num_loops_r * blockDim.x;

        for(sptNnzIndex nl=0; nl<num_loops_nnz; ++nl) {
            x = blockIdx.x * blockDim.y + tidy + nl * nnz_per_loop;
            if(x < Y_nnz) {
                const sptNnzIndex inz_begin = fiberidx_val[x];
                const sptNnzIndex inz_end = fiberidx_val[x+1];

                for(sptNnzIndex i = inz_begin; i < inz_end; ++i) {
                    const sptIndex row = X_inds_m[i];
                    Y_val[x*Y_stride + r] += X_val[i] * U_val[row*U_stride + r];
                    __syncthreads();
                }
            }
        }
    }

}


/* impl_num = 15 */
__global__ void spt_TTMRankRBNnzKernelSM(
    sptValue *Y_val, 
    sptIndex Y_stride, sptNnzIndex Y_nnz,
    const sptValue * __restrict__ X_val, 
    sptNnzIndex X_nnz, 
    const sptIndex * __restrict__ X_inds_m,
    const sptNnzIndex * __restrict__ fiberidx_val, 
    sptNnzIndex fiberidx_len,
    const sptValue * __restrict__ U_val, 
    sptIndex U_nrows, 
    sptIndex U_ncols, 
    sptIndex U_stride) 
{
    extern __shared__ sptValue mem_pool[];
    sptValue * const Y_shr = (sptValue *) mem_pool; // size U_ncols

    sptNnzIndex num_loops_nnz = 1;
    sptNnzIndex const nnz_per_loop = gridDim.x * blockDim.y;
    if(Y_nnz > nnz_per_loop) {
        num_loops_nnz = (Y_nnz + nnz_per_loop - 1) / nnz_per_loop;
    }
    
    const sptNnzIndex tidx = threadIdx.x;
    const sptNnzIndex tidy = threadIdx.y;
    sptNnzIndex x;
    const sptIndex num_loops_r = U_ncols / blockDim.x;
    const sptIndex rest_loop = U_ncols - num_loops_r * blockDim.x;
    sptIndex r;


    for(sptIndex l=0; l<num_loops_r; ++l) {
        r = tidx + l * blockDim.x;
        for(sptNnzIndex nl=0; nl<num_loops_nnz; ++nl) {
            x = blockIdx.x * blockDim.y + tidy + nl * nnz_per_loop;

            Y_shr[tidy * Y_stride + tidx] = 0;
            __syncthreads();

            if(x < Y_nnz) {
                const sptNnzIndex inz_begin = fiberidx_val[x];
                const sptNnzIndex inz_end = fiberidx_val[x+1];
                for(sptNnzIndex i = inz_begin; i < inz_end; ++i) {
                    const sptIndex row = X_inds_m[i];
                    Y_shr[tidy*Y_stride + tidx] += X_val[i] * U_val[row*U_stride + r]; 
                }
                __syncthreads();

                Y_val[x*Y_stride + r] = Y_shr[tidy*Y_stride + tidx];
                __syncthreads();
            }
        }
    }


    if(rest_loop > 0 && tidx < rest_loop) {
        r = tidx + num_loops_r * blockDim.x;

        for(sptNnzIndex nl=0; nl<num_loops_nnz; ++nl) {
            x = blockIdx.x * blockDim.y + tidy + nl * nnz_per_loop;

            Y_shr[tidy * Y_stride + tidx] = 0;
            __syncthreads();

            if(x < Y_nnz) {
                const sptNnzIndex inz_begin = fiberidx_val[x];
                const sptNnzIndex inz_end = fiberidx_val[x+1];
                for(sptNnzIndex i = inz_begin; i < inz_end; ++i) {
                    const sptIndex row = X_inds_m[i];
                    Y_shr[tidy*Y_stride + tidx] += X_val[i] * U_val[row*U_stride + r]; 
                }
                __syncthreads();

                Y_val[x*Y_stride + r] = Y_shr[tidy*Y_stride + tidx];
                __syncthreads();
            }
        }
    }

}