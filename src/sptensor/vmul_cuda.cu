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
#include "vmul_cuda_kernels.h"


int sptCudaSparseTensorMulVector(
    sptSparseTensor *Y,
    sptSparseTensor *X,
    const sptValueVector *V,
    sptIndex const mode,
    sptIndex const impl_num,
    sptNnzIndex const smen_size) 
{

    if(mode >= X->nmodes) {
        spt_CheckError(SPTERR_SHAPE_MISMATCH, "CUDA SpTns * Vec", "shape mismatch");
    }
    if(X->ndims[mode] != V->len) {
        spt_CheckError(SPTERR_SHAPE_MISMATCH, "CUDA SpTns * Vec", "shape mismatch");
    }

    int result;
    sptIndex *ind_buf;
    sptNnzIndexVector fiberidx;
    sptTimer timer;
    sptNewTimer(&timer, 0);

    sptStartTimer(timer);
    sptSparseTensorSortIndexAtMode(X, mode, 0);
    sptStopTimer(timer);
    sptPrintElapsedTime(timer, "sptSparseTensorSortIndexAtMode");

    sptStartTimer(timer);
    ind_buf = new sptIndex[X->nmodes * sizeof *ind_buf];
    spt_CheckOSError(!ind_buf, "CUDA  SpTns * Vec");
    for(sptIndex m = 0; m < X->nmodes; ++m) {
        if(m < mode)
            ind_buf[m] = X->ndims[m];
        else if(m > mode)
            ind_buf[m - 1] = X->ndims[m];
    }

    result = sptNewSparseTensor(Y, X->nmodes - 1, ind_buf);
    free(ind_buf);
    spt_CheckError(result, "CUDA SpTns * Vec", NULL);
    sptSparseTensorSetIndices(Y, &fiberidx, mode, X);
    sptStopTimer(timer);
    sptPrintElapsedTime(timer, "Allocate output tensor");

    sptValue *Y_val = NULL;
    result = cudaMalloc((void **) &Y_val, Y->nnz * sizeof (sptValue));
    spt_CheckCudaError(result != 0, "CUDA SpTns * Mtx");
    // jli: Add memset to Y.
    cudaMemset(Y_val, 0, Y->nnz * sizeof (sptValue));
    sptValue *X_val = NULL;
    result = cudaMalloc((void **) &X_val, X->nnz * sizeof (sptValue));
    spt_CheckCudaError(result != 0, "CUDA SpTns * Mtx");
    cudaMemcpy(X_val, X->values.data, X->nnz * sizeof (sptValue), cudaMemcpyHostToDevice);
    sptIndex *X_inds_m = NULL;
    result = cudaMalloc((void **) &X_inds_m, X->nnz * sizeof (sptIndex));
    spt_CheckCudaError(result != 0, "CUDA SpTns * Mtx");
    cudaMemcpy(X_inds_m, X->inds[mode].data, X->nnz * sizeof (sptIndex), cudaMemcpyHostToDevice);
    sptValue *V_val = NULL;
    result = cudaMalloc((void **) &V_val, V->len * sizeof (sptValue));
    spt_CheckCudaError(result != 0, "CUDA SpTns * Mtx");
    cudaMemcpy(V_val, V->data, V->len * sizeof (sptValue), cudaMemcpyHostToDevice);
    sptNnzIndex *fiberidx_val = NULL;
    result = cudaMalloc((void **) &fiberidx_val, fiberidx.len * sizeof (sptNnzIndex));
    spt_CheckCudaError(result != 0, "CUDA SpTns * Mtx");
    cudaMemcpy(fiberidx_val, fiberidx.data, fiberidx.len * sizeof (sptNnzIndex), cudaMemcpyHostToDevice);

    const sptNnzIndex max_nblocks = 32768;
    const sptNnzIndex max_nthreads_per_block = 256;
    sptNnzIndex max_nthreadsy = 16;

    sptNnzIndex nthreadsx = 1;
    sptNnzIndex nthreadsy = 1;
    sptNnzIndex all_nblocks = 0;
    sptNnzIndex nblocks = 0;

    const char *env_PARTI_TTV_NTHREADS = getenv("PARTI_TTV_NTHREADS");

    switch(impl_num) {
    // case 1:
    case 11: // Naive, 1D
        if(Y->nnz < max_nthreads_per_block) {
            nthreadsx = Y->nnz;
            nblocks = 1;
        } else {
            nthreadsx = max_nthreads_per_block;
            all_nblocks = (Y->nnz + nthreadsx -1) / nthreadsx;
            if(all_nblocks < max_nblocks) {
                nblocks = all_nblocks;
            } else {
                nblocks = max_nblocks;
            }
        }
        break;
    }
    dim3 dimBlock(nthreadsx, nthreadsy);
    printf("all_nblocks: %lu, nthreadsx: %lu, nthreadsy: %lu\n", all_nblocks, nthreadsx, nthreadsy);

    sptStartTimer(timer);


    switch(impl_num) {
    // case 1:
    case 11: // Naive
        printf("[CUDA SpTns * Mtx] spt_TTVNnzKernel<<<%lu, (%lu, %lu)>>>\n", nblocks, nthreadsx, nthreadsy);
        spt_TTVNnzKernel<<<nblocks, dimBlock>>>(
            Y_val, Y->nnz,
            X_val, X->nnz, X_inds_m,
            fiberidx_val, fiberidx.len,
            V_val, V->len);
        break;
    }
    result = cudaThreadSynchronize();
    spt_CheckCudaError(result != 0, "CUDA SpTns * Vec kernel");

    sptStopTimer(timer);
    sptPrintElapsedTime(timer, "CUDA SpTns * Vec");
    sptFreeTimer(timer);

    cudaMemcpy(Y->values.data, Y_val, Y->nnz * sizeof (sptValue), cudaMemcpyDeviceToHost);
    result = cudaFree(fiberidx_val);
    spt_CheckCudaError(result != 0, "CUDA SpTns * Vec");
    result = cudaFree(V_val);
    spt_CheckCudaError(result != 0, "CUDA SpTns * Vec");
    result = cudaFree(X_inds_m);
    spt_CheckCudaError(result != 0, "CUDA SpTns * Vec");
    result = cudaFree(X_val);
    spt_CheckCudaError(result != 0, "CUDA SpTns * Vec");
    result = cudaFree(Y_val);
    spt_CheckCudaError(result != 0, "CUDA SpTns * Vec");
    sptFreeNnzIndexVector(&fiberidx);


    return 0;
}
