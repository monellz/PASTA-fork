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
#include "sptensor.h"


__global__ void spt_dMulKernel(
    sptValue *Z_val, 
    const sptValue * __restrict__ X_val, 
    const sptValue * __restrict__ Y_val, 
    sptNnzIndex nnz)
{
    sptNnzIndex num_loops_nnz = 1;
    sptNnzIndex const nnz_per_loop = gridDim.x * blockDim.x;
    if(nnz > nnz_per_loop) {
        num_loops_nnz = (nnz + nnz_per_loop - 1) / nnz_per_loop;
    }

    const sptNnzIndex tidx = threadIdx.x;
    sptNnzIndex x;

    for(sptNnzIndex nl=0; nl<num_loops_nnz; ++nl) {
        x = blockIdx.x * blockDim.x + tidx + nl * nnz_per_loop;
        if(x < nnz) {
            Z_val[x] = X_val[x] * Y_val[x];
        }
    }

}


/**
 * Multiply a sparse tensors with a scalar.
 * @param[out] Z the result of a*X, should be uninitialized
 * @param[in]  a the input scalar
 * @param[in]  X the input X
 */
int sptCudaSparseTensorDotMulEq(sptSparseTensor *Z, const sptSparseTensor *X, const sptSparseTensor *Y, int collectZero)
{
   sptNnzIndex i;
    /* Ensure X and Y are in same shape */
    if(Y->nmodes != X->nmodes) {
        spt_CheckError(SPTERR_SHAPE_MISMATCH, "Cuda SpTns DotMul", "shape mismatch");
    }
    for(i = 0; i < X->nmodes; ++i) {
        if(Y->ndims[i] != X->ndims[i]) {
            spt_CheckError(SPTERR_SHAPE_MISMATCH, "Cuda SpTns DotMul", "shape mismatch");
        }
    }
    /* Ensure X and Y have exactly the same nonzero distribution */
    if(Y->nnz != X->nnz) {
        spt_CheckError(SPTERR_SHAPE_MISMATCH, "Cuda SpTns DotMul", "nonzero distribution mismatch");
    }
    int result;

    sptTimer timer;
    sptNewTimer(&timer, 0);
    double copy_time_cpu, copy_time_gpu, comp_time, collect_time, total_time;

    /* Allocate space */
    sptCopySparseTensorAllocateOnly(Z, X);

    /* Copy indices on CPU */
    sptStartTimer(timer);
    sptCopySparseTensorCopyOnly(Z, X);
    sptStopTimer(timer);
    copy_time_cpu = sptPrintElapsedTime(timer, "sptCopySparseTensor");

    /* Device memory allocation */
    sptValue *Z_val = NULL;
    result = cudaMalloc((void **) &Z_val, Z->nnz * sizeof (sptValue));
    spt_CheckCudaError(result != 0, "Cuda SpTns DotMul");
    sptValue *X_val = NULL;
    result = cudaMalloc((void **) &X_val, X->nnz * sizeof (sptValue));
    spt_CheckCudaError(result != 0, "Cuda SpTns DotMul");
    sptValue *Y_val = NULL;
    result = cudaMalloc((void **) &Y_val, Y->nnz * sizeof (sptValue));
    spt_CheckCudaError(result != 0, "Cuda SpTns DotMul");

    /* Device memory copy */
    sptStartTimer(timer);
    cudaMemcpy(X_val, X->values.data, X->nnz * sizeof (sptValue), cudaMemcpyHostToDevice);
    cudaMemcpy(Y_val, Y->values.data, Y->nnz * sizeof (sptValue), cudaMemcpyHostToDevice);
    sptStopTimer(timer);
    copy_time_gpu = sptPrintElapsedTime(timer, "Device copy");


    sptStartTimer(timer);

    const sptNnzIndex max_nblocks = 32768;
    const sptNnzIndex max_nthreads_per_block = 256;

    sptNnzIndex nthreadsx = 1;
    sptNnzIndex all_nblocks = 0;
    sptNnzIndex nblocks = 0;

    if(X->nnz < max_nthreads_per_block) {
        nthreadsx = X->nnz;
        nblocks = 1;
    } else {
        nthreadsx = max_nthreads_per_block;
        all_nblocks = (X->nnz + nthreadsx -1) / nthreadsx;
        if(all_nblocks < max_nblocks) {
            nblocks = all_nblocks;
        } else {
            nblocks = max_nblocks;
        }
    }
    dim3 dimBlock(nthreadsx);
    printf("all_nblocks: %lu, nthreadsx: %lu\n", all_nblocks, nthreadsx);

    printf("[Cuda SpTns DotMul] spt_sAddKernel<<<%lu, (%lu)>>>\n", nblocks, nthreadsx);
    spt_dMulKernel<<<nblocks, dimBlock>>>(Z_val, X_val, Y_val, X->nnz);
    result = cudaThreadSynchronize();
    spt_CheckCudaError(result != 0, "Cuda SpTns DotMul kernel");

    sptStopTimer(timer);
    comp_time = sptPrintElapsedTime(timer, "Cuda SpTns DotMul");

    /* Check whether elements become zero after adding.
       If so, fill the gap with the [nnz-1]'th element.
    */
    sptStartTimer(timer);
    if(collectZero == 1) {
        sptSparseTensorCollectZeros(Z);
    }
    sptStopTimer(timer);
    collect_time = sptPrintElapsedTime(timer, "sptSparseTensorCollectZeros");

    /* Copy back to CPU */
    sptStartTimer(timer);
    cudaMemcpy(Z->values.data, Z_val, Z->nnz * sizeof (sptValue), cudaMemcpyDeviceToHost);
    sptStopTimer(timer);
    copy_time_gpu += sptPrintElapsedTime(timer, "Device copy back");
    
    sptFreeTimer(timer);
    result = cudaFree(X_val);
    spt_CheckCudaError(result != 0, "Cuda SpTns DotMul");
    result = cudaFree(Y_val);
    spt_CheckCudaError(result != 0, "Cuda SpTns DotMul");
    result = cudaFree(Z_val);
    spt_CheckCudaError(result != 0, "Cuda SpTns DotMul");

    total_time = copy_time_cpu + copy_time_gpu + comp_time + collect_time;
    printf("[Total time]: %lf\n", total_time);
    printf("\n");
    
    return 0;
}
