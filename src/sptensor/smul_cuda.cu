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


__global__ void spt_sMulKernel(
    sptValue *Z_val, 
    const sptValue * __restrict__ X_val, 
    sptNnzIndex X_nnz,
    sptValue a)
{
    sptNnzIndex num_loops_nnz = 1;
    sptNnzIndex const nnz_per_loop = gridDim.x * blockDim.x;
    if(X_nnz > nnz_per_loop) {
        num_loops_nnz = (X_nnz + nnz_per_loop - 1) / nnz_per_loop;
    }

    const sptNnzIndex tidx = threadIdx.x;
    sptNnzIndex x;

    for(sptNnzIndex nl=0; nl<num_loops_nnz; ++nl) {
        x = blockIdx.x * blockDim.x + tidx + nl * nnz_per_loop;
        if(x < X_nnz) {
            Z_val[x] *= a;
        }
        __syncthreads();
    }

}


/**
 * Multiply a sparse tensors with a scalar.
 * @param[out] Z the result of a*X, should be uninitialized
 * @param[in]  a the input scalar
 * @param[in]  X the input X
 */
int sptCudaSparseTensorMulScalar(sptSparseTensor *Z, sptSparseTensor *X, sptValue a)
{
    sptAssert(a != 0.0);
    int result;

    sptTimer timer;
    sptNewTimer(&timer, 0);

    sptStartTimer(timer);
    sptCopySparseTensor(Z, X, 1);
    sptStopTimer(timer);
    sptPrintElapsedTime(timer, "sptCopySparseTensor");

    sptStartTimer(timer);
    sptValue *Z_val = NULL;
    result = cudaMalloc((void **) &Z_val, Z->nnz * sizeof (sptValue));
    spt_CheckCudaError(result != 0, "CUDA SpTns MulScalar");
    cudaMemcpy(Z_val, Z->values.data, Z->nnz * sizeof (sptValue), cudaMemcpyHostToDevice);
    sptValue *X_val = NULL;
    result = cudaMalloc((void **) &X_val, X->nnz * sizeof (sptValue));
    spt_CheckCudaError(result != 0, "CUDA SpTns MulScalar");
    cudaMemcpy(X_val, X->values.data, X->nnz * sizeof (sptValue), cudaMemcpyHostToDevice);
    sptStopTimer(timer);
    sptPrintElapsedTime(timer, "Device malloc and copy");

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

    printf("[CUDA SpTns MulScalar] spt_sMulKernel<<<%lu, (%lu)>>>\n", nblocks, nthreadsx);
    spt_sMulKernel<<<nblocks, dimBlock>>>(Z_val, X_val, X->nnz, a);
    result = cudaThreadSynchronize();
    spt_CheckCudaError(result != 0, "CUDA SpTns MulScalar kernel");

    sptStopTimer(timer);
    sptPrintElapsedTime(timer, "Cpu SpTns MulScalar");
    sptFreeTimer(timer);
    printf("\n");

    cudaMemcpy(Z->values.data, Z_val, Z->nnz * sizeof (sptValue), cudaMemcpyDeviceToHost);
    result = cudaFree(X_val);
    spt_CheckCudaError(result != 0, "CUDA SpTns MulScalar");
    result = cudaFree(Z_val);
    spt_CheckCudaError(result != 0, "CUDA SpTns MulScalar");

    return 0;
}
