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

#ifndef PARTI_VMUL_KERNELS_H
#define PARTI_VMUL_KERNELS_H

/* impl_num = 11 */
__global__ void spt_TTVNnzKernel(
    sptValue *Y_val, 
    sptNnzIndex Y_nnz,
    const sptValue * __restrict__ X_val, 
    sptNnzIndex X_nnz, 
    const sptIndex * __restrict__ X_inds_m,
    const sptNnzIndex * __restrict__ fiberidx_val, 
    sptNnzIndex fiberidx_len,
    const sptValue * __restrict__ V_val, 
    sptIndex V_nrows);


#endif