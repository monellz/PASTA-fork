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

#ifndef PARTI_MMUL_KERNELS_H
#define PARTI_MMUL_KERNELS_H


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
    sptIndex U_stride);

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
    sptIndex U_stride);

#endif