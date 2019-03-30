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

#include <pasta.h>
#include "mttkrp_cuda_kernels.h"
#include <inttypes.h>

int sptMTTKRPKernelHiCOO(
    const sptIndex mode,
    const sptIndex nmodes,
    const sptNnzIndex nnz,
    const sptNnzIndex max_nnzb,
    const sptIndex R,
    const sptIndex stride,
    const sptElementIndex sb_bits,
    const sptIndex blength,
    const int impl_num,
    sptIndex * const dev_ndims,
    sptNnzIndex * const dev_bptr,
    sptBlockIndex ** const dev_binds,
    sptElementIndex ** const dev_einds,
    sptValue * const dev_values,
    sptIndex * const dev_mats_order,
    sptValue ** const dev_mats)
{
    int result = 0;

    /* Maximum settings */
    sptIndex max_nthreads_per_block = 256;
    sptIndex max_nblocks = 32768;
    sptIndex max_R = 4;

    sptIndex nthreadsx = 0;
    sptIndex nthreadsy = 0;
    sptIndex nblocks = 0;
    // sptIndex shr_size = 0;
    sptNnzIndex all_nblocks = blength;

    switch(nmodes) {
    case 3: /* 3-D tensors */
    case 4: /* 4-D tensors */
        switch(impl_num) {
        /* Matrix blocked implementations */
        case 14:
            nthreadsx = R;
            if(R <= max_R)
                nthreadsx = R;
            else
                nthreadsx = max_R;
            if(max_nnzb <= max_nthreads_per_block / max_R)
                nthreadsy = max_nnzb;
            else
                nthreadsy = max_nthreads_per_block / max_R;
            if(all_nblocks < max_nblocks) {
                nblocks = all_nblocks;
            } else {
                nblocks = max_nblocks;
            }
            // shr_size = 2 * nmodes * sizeof(sptIndex);
            break;

        default:
            printf("Provide correct impl_num.\n");
        }
        break;

    default:
        printf("Do not support >4 nmodes.\n");
    }   // End switch nmodes

    dim3 dimBlock(nthreadsx, nthreadsy);

    switch(nmodes) {
    case 3: /* 3-D tensors */
        switch(impl_num) {
        /* Matrix blocked implementations */
        case 14:
            printf("\nExecute spt_MTTKRPKernelRankSplitHiCOORB_3D_MatrixBlocked (%u, %u, %u)\n", nblocks, nthreadsx, nthreadsy);
            spt_MTTKRPKernelRankSplitHiCOORB_3D_MatrixBlocked<<<nblocks, dimBlock>>>(
                mode,
                nmodes,
                nnz,
                R,
                stride,
                sb_bits,
                blength,
                dev_ndims,
                dev_bptr,
                dev_binds,
                dev_einds,
                dev_values,
                dev_mats_order,
                dev_mats);
            break;
        default:
            printf("Provide correct impl_num.\n");
        }
        break;

    case 4: /* 4-D tensors */
        switch(impl_num) {
        /* Matrix blocked implementations */
        case 14:
            printf("\nExecute spt_MTTKRPKernelRankSplitHiCOORB_4D_MatrixBlocked (%u, %u, %u)\n", nblocks, nthreadsx, nthreadsy);
            spt_MTTKRPKernelRankSplitHiCOORB_4D_MatrixBlocked<<<nblocks, dimBlock>>>(
                mode,
                nmodes,
                nnz,
                R,
                stride,
                sb_bits,
                blength,
                dev_ndims,
                dev_bptr,
                dev_binds,
                dev_einds,
                dev_values,
                dev_mats_order,
                dev_mats);
            break;
        default:
            printf("Provide correct impl_num.\n");
        }
        break;

    default:
        printf("Do not support >4 nmodes.\n");
    }   // End switch nmodes

    result = cudaThreadSynchronize();
    spt_CheckCudaError(result != 0, "Cuda HiSpTns MTTKRP");

    return 0;
}



/* impl_num = 14  Matrix Blocked, 2-D, with rank blocking.
 * Limitation: max_R * blockDim.y (max_nnz) <= 1024.
 */
__global__ void spt_MTTKRPKernelRankSplitHiCOORB_3D_MatrixBlocked(
    const sptIndex mode,
    const sptIndex nmodes,
    const sptNnzIndex nnz,
    const sptIndex R,
    const sptIndex stride,
    const sptElementIndex sb_bits,
    const sptNnzIndex blength,
    sptIndex * const dev_ndims,
    sptNnzIndex * const dev_bptr,
    sptBlockIndex ** const dev_binds,
    sptElementIndex ** const dev_einds,
    sptValue * const dev_values,
    sptIndex * const dev_mats_order,
    sptValue ** const dev_mats)
{
    sptNnzIndex const all_nblocks = blength;
    const sptIndex tidx = threadIdx.x;
    const sptIndex tidy = threadIdx.y;
    sptNnzIndex z;
    const sptIndex num_loops_r = R / blockDim.x;
    const sptIndex rest_loop = R - num_loops_r * blockDim.x;
    sptNnzIndex num_loops_nnz;

    sptValue * const mvals = dev_mats[nmodes];
    sptIndex const times_mat_index_1 = dev_mats_order[1];
    sptValue * const times_mat_1 = dev_mats[times_mat_index_1];
    sptIndex const times_mat_index_2 = dev_mats_order[2];
    sptValue * const times_mat_2 = dev_mats[times_mat_index_2];

    sptNnzIndex num_loops_blocks = 1;
    if(all_nblocks > gridDim.x) {
        num_loops_blocks = (all_nblocks + gridDim.x - 1) / gridDim.x;
    }

    for(sptNnzIndex nb=0; nb<num_loops_blocks; ++nb) {
        /* Block level */
        sptNnzIndex b = blockIdx.x + nb * gridDim.x;
        if(b < blength) {
            /* TODO: duplicated in registers */
            sptValue * blocked_mvals = mvals + (dev_binds[mode][b] << sb_bits) * stride;
            sptValue * blocked_times_mat_1 = times_mat_1 + (dev_binds[times_mat_index_1][b] << sb_bits) * stride;
            sptValue * blocked_times_mat_2 = times_mat_2 + (dev_binds[times_mat_index_2][b] << sb_bits) * stride;

            sptNnzIndex const bptr_begin = dev_bptr[b];
            sptNnzIndex const bptr_end = dev_bptr[b+1];
            sptNnzIndex const nnzb = bptr_end - bptr_begin;
            num_loops_nnz = (nnzb + blockDim.y - 1) / blockDim.y;

            /* Thread level */
            for(sptNnzIndex zl = 0; zl < num_loops_nnz; ++ zl) {
                z = tidy + zl * blockDim.y + bptr_begin;
                if(z < bptr_end) {
                    sptValue const entry = dev_values[z];
                    sptElementIndex const mode_i = dev_einds[mode][z];
                    sptElementIndex const tmp_i_1 = dev_einds[times_mat_index_1][z];
                    sptElementIndex const tmp_i_2 = dev_einds[times_mat_index_2][z];

                    sptValue * const bmvals_row = blocked_mvals + mode_i * stride;

                    sptIndex r;
                    sptValue tmp_val = 0;
                    for(sptIndex l=0; l<num_loops_r; ++l) {
                        r = tidx + l * blockDim.x;
                        tmp_val = entry * blocked_times_mat_1[tmp_i_1 * stride + r] * blocked_times_mat_2[tmp_i_2 * stride + r];
                        atomicAdd(&(bmvals_row[r]), tmp_val);
                    }

                    if(rest_loop > 0 && tidx < rest_loop) {
                        r = tidx + num_loops_r * blockDim.x;
                        tmp_val = entry * blocked_times_mat_1[tmp_i_1 * stride + r] * blocked_times_mat_2[tmp_i_2 * stride + r];
                        atomicAdd(&(bmvals_row[r]), tmp_val);
                    }

                }
            }   // End loop entries
        }
    }   // End loop blocks

}


/* impl_num = 14  Matrix Blocked, 2-D, with rank blocking.
 * Limitation: max_R * blockDim.y (max_nnz) <= 1024.
 */
__global__ void spt_MTTKRPKernelRankSplitHiCOORB_4D_MatrixBlocked(
    const sptIndex mode,
    const sptIndex nmodes,
    const sptNnzIndex nnz,
    const sptIndex R,
    const sptIndex stride,
    const sptElementIndex sb_bits,
    const sptNnzIndex blength,
    sptIndex * const dev_ndims,
    sptNnzIndex * const dev_bptr,
    sptBlockIndex ** const dev_binds,
    sptElementIndex ** const dev_einds,
    sptValue * const dev_values,
    sptIndex * const dev_mats_order,
    sptValue ** const dev_mats)
{
    sptNnzIndex const all_nblocks = blength;
    const sptIndex tidx = threadIdx.x;
    const sptIndex tidy = threadIdx.y;
    sptNnzIndex z;
    const sptIndex num_loops_r = R / blockDim.x;
    const sptIndex rest_loop = R - num_loops_r * blockDim.x;
    sptNnzIndex num_loops_nnz;

    sptValue * const mvals = dev_mats[nmodes];
    sptIndex const times_mat_index_1 = dev_mats_order[1];
    sptValue * const times_mat_1 = dev_mats[times_mat_index_1];
    sptIndex const times_mat_index_2 = dev_mats_order[2];
    sptValue * const times_mat_2 = dev_mats[times_mat_index_2];
    sptIndex const times_mat_index_3 = dev_mats_order[3];
    sptValue * const times_mat_3 = dev_mats[times_mat_index_3];

    sptNnzIndex num_loops_blocks = 1;
    if(all_nblocks > gridDim.x) {
        num_loops_blocks = (all_nblocks + gridDim.x - 1) / gridDim.x;
    }

    for(sptNnzIndex nb=0; nb<num_loops_blocks; ++nb) {
        /* Block level */
        sptNnzIndex b = blockIdx.x + nb * gridDim.x;
        if(b < blength) {
            /* TODO: duplicated in registers */
            sptValue * blocked_mvals = mvals + (dev_binds[mode][b] << sb_bits) * stride;
            sptValue * blocked_times_mat_1 = times_mat_1 + (dev_binds[times_mat_index_1][b] << sb_bits) * stride;
            sptValue * blocked_times_mat_2 = times_mat_2 + (dev_binds[times_mat_index_2][b] << sb_bits) * stride;
            sptValue * blocked_times_mat_3 = times_mat_3 + (dev_binds[times_mat_index_3][b] << sb_bits) * stride;

            sptNnzIndex const bptr_begin = dev_bptr[b];
            sptNnzIndex const bptr_end = dev_bptr[b+1];
            sptNnzIndex const nnzb = bptr_end - bptr_begin;
            num_loops_nnz = (nnzb + blockDim.y - 1) / blockDim.y;

            /* Thread level */
            for(sptNnzIndex zl = 0; zl < num_loops_nnz; ++ zl) {
                z = tidy + zl * blockDim.y + bptr_begin;
                if(z < bptr_end) {
                    sptValue const entry = dev_values[z];
                    sptElementIndex const mode_i = dev_einds[mode][z];
                    sptElementIndex const tmp_i_1 = dev_einds[times_mat_index_1][z];
                    sptElementIndex const tmp_i_2 = dev_einds[times_mat_index_2][z];
                    sptElementIndex const tmp_i_3 = dev_einds[times_mat_index_3][z];

                    sptValue * const bmvals_row = blocked_mvals + mode_i * stride;

                    sptIndex r;
                    sptValue tmp_val = 0;
                    for(sptIndex l=0; l<num_loops_r; ++l) {
                        r = tidx + l * blockDim.x;
                        tmp_val = entry 
                                    * blocked_times_mat_1[tmp_i_1 * stride + r] 
                                    * blocked_times_mat_2[tmp_i_2 * stride + r] 
                                    * blocked_times_mat_3[tmp_i_3 * stride + r];
                        atomicAdd(&(bmvals_row[r]), tmp_val);
                    }

                    if(rest_loop > 0 && tidx < rest_loop) {
                        r = tidx + num_loops_r * blockDim.x;
                        tmp_val = entry 
                                    * blocked_times_mat_1[tmp_i_1 * stride + r] 
                                    * blocked_times_mat_2[tmp_i_2 * stride + r]
                                    * blocked_times_mat_3[tmp_i_3 * stride + r];
                        atomicAdd(&(bmvals_row[r]), tmp_val);
                    }

                }
            }   // End loop entries
        }
    }   // End loop blocks

}


