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
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include "sptensor.h"



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
    sptIndex V_nrows)
{
    sptNnzIndex num_loops_nnz = 1;
    sptNnzIndex const nnz_per_loop = gridDim.x * blockDim.x;
    if(Y_nnz > nnz_per_loop) {
        num_loops_nnz = (Y_nnz + nnz_per_loop - 1) / nnz_per_loop;
    }

    const sptNnzIndex tidx = threadIdx.x;
    sptNnzIndex x;

    for(sptNnzIndex nl=0; nl<num_loops_nnz; ++nl) {
        x = blockIdx.x * blockDim.x + tidx + nl * nnz_per_loop;
        if(x < Y_nnz) {
            const sptNnzIndex inz_begin = fiberidx_val[x];
            const sptNnzIndex inz_end = fiberidx_val[x+1];

            for(sptNnzIndex i = inz_begin; i < inz_end; ++i) {
                const sptIndex row = X_inds_m[i];
                Y_val[x] += X_val[i] * V_val[row];
            }
        }
        __syncthreads();
    }

}
