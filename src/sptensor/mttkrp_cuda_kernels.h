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

#ifndef PASTA_MTTKRP_KERNELS_H
#define PASTA_MTTKRP_KERNELS_H

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
    sptValue ** dev_mats);

/* impl_num = 15 */
__global__ void spt_MTTKRPKernelRankSplitNnz4DOneKernel(
    const sptIndex mode,
    const sptIndex nmodes,
    const sptNnzIndex nnz,
    const sptIndex R,
    const sptIndex stride,
    const sptIndex * Xndims,
    sptIndex ** const Xinds,
    const sptValue * Xvals,
    const sptIndex * dev_mats_order,
    sptValue ** dev_mats);


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
    sptValue ** dev_mats);


/* for arbitrary nmodes. Scratch is necessary for tensors with arbitrary modes. */
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
    sptNnzIndex block_offset);


#endif