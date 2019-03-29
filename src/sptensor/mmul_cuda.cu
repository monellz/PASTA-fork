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
#include "mmul_cuda_kernels.h"


int sptCudaSparseTensorMulMatrix(
    sptSemiSparseTensor *Y,
    sptSparseTensor *X,
    const sptMatrix *U,
    sptIndex const mode,
    sptIndex const impl_num,
    sptNnzIndex const smem_size) 
{
    int result;
    sptIndex *ind_buf;
    sptIndex m;
    sptNnzIndexVector fiberidx;
    if(mode >= X->nmodes) {
        spt_CheckError(SPTERR_SHAPE_MISMATCH, "Cuda SpTns * Mtx", "shape mismatch");
    }
    if(X->ndims[mode] != U->nrows) {
        spt_CheckError(SPTERR_SHAPE_MISMATCH, "Cuda SpTns * Mtx", "shape mismatch");
    }

    sptSparseTensorSortIndexAtMode(X, mode, 0);
    ind_buf = new sptIndex[X->nmodes * sizeof *ind_buf];
    for(m = 0; m < X->nmodes; ++m) {
        ind_buf[m] = X->ndims[m];
    }
    ind_buf[mode] = U->ncols;
    result = sptNewSemiSparseTensor(Y, X->nmodes, mode, ind_buf);
    delete[] ind_buf;
    spt_CheckError(result, "Cuda SpTns * Mtx", NULL);
    sptSemiSparseTensorSetIndices(Y, &fiberidx, X);
    if(Y->values.stride != U->stride) {
        spt_CheckError(SPTERR_SHAPE_MISMATCH, "Cuda SpTns * Mtx", "shape mismatch");
    }
    sptIndex stride = U->stride;

    double flen = (double)X->nnz / fiberidx.len;
    sptNnzIndex tmp_flen = (fiberidx.data[1] - fiberidx.data[0]) - flen;
    double fvar = tmp_flen * tmp_flen;
    for(sptNnzIndex i=1; i<fiberidx.len - 1; ++i) {
        tmp_flen = (fiberidx.data[i+1] - fiberidx.data[i]) - flen;
        fvar += tmp_flen * tmp_flen;
    }
    tmp_flen = (X->nnz - fiberidx.data[fiberidx.len - 1]) - flen;
    fvar += tmp_flen * tmp_flen;
    fvar = sqrt(fvar);
    printf("nfibs: %zu, flen: %.2f, fvar: %.2f\n", fiberidx.len, flen, fvar);

    sptValue *Y_val = NULL;
    result = cudaMalloc((void **) &Y_val, Y->nnz * stride * sizeof (sptValue));
    spt_CheckCudaError(result != 0, "Cuda SpTns * Mtx");
    // jli: Add memset to Y.
    cudaMemset(Y_val, 0, Y->nnz * stride * sizeof (sptValue));
    sptValue *X_val = NULL;
    result = cudaMalloc((void **) &X_val, X->nnz * sizeof (sptValue));
    spt_CheckCudaError(result != 0, "Cuda SpTns * Mtx");
    cudaMemcpy(X_val, X->values.data, X->nnz * sizeof (sptValue), cudaMemcpyHostToDevice);
    sptIndex *X_inds_m = NULL;
    result = cudaMalloc((void **) &X_inds_m, X->nnz * sizeof (sptIndex));
    spt_CheckCudaError(result != 0, "Cuda SpTns * Mtx");
    cudaMemcpy(X_inds_m, X->inds[mode].data, X->nnz * sizeof (sptIndex), cudaMemcpyHostToDevice);
    sptValue *U_val = NULL;
    result = cudaMalloc((void **) &U_val, U->nrows * stride * sizeof (sptValue));
    spt_CheckCudaError(result != 0, "Cuda SpTns * Mtx");
    cudaMemcpy(U_val, U->values, U->nrows * stride * sizeof (sptValue), cudaMemcpyHostToDevice);
    sptNnzIndex *fiberidx_val = NULL;
    result = cudaMalloc((void **) &fiberidx_val, fiberidx.len * sizeof (sptNnzIndex));
    spt_CheckCudaError(result != 0, "Cuda SpTns * Mtx");
    cudaMemcpy(fiberidx_val, fiberidx.data, fiberidx.len * sizeof (sptNnzIndex), cudaMemcpyHostToDevice);

    const sptNnzIndex max_nblocks = 32768;
    const sptNnzIndex max_nthreads_per_block = 256;
    sptNnzIndex max_nthreadsy = 16;

    sptNnzIndex nthreadsx = 1;
    sptNnzIndex nthreadsy = 1;
    sptNnzIndex all_nblocks = 0;
    sptNnzIndex nblocks = 0;

    const char *env_PASTA_TTM_NTHREADS = getenv("PASTA_TTM_NTHREADS");

    switch(impl_num) {
    case 14:
        if(U->ncols <= max_nthreadsy)
            nthreadsx = U->ncols;
        else
            nthreadsx = max_nthreadsy;
        nthreadsy = max_nthreads_per_block / nthreadsx;

        if(Y->nnz < nthreadsy) {
            nthreadsy = Y->nnz;
            nblocks = 1;
        } else {
            all_nblocks = (Y->nnz + nthreadsy -1) / nthreadsy;
            if(all_nblocks < max_nblocks) {
                nblocks = all_nblocks;
            } else {
                nblocks = max_nblocks;
            }
        }
        break;
    case 15:
        if(U->ncols <= max_nthreadsy)
            nthreadsx = U->ncols;
        else
            nthreadsx = max_nthreadsy;
        nthreadsy = max_nthreads_per_block / nthreadsx;

        if(Y->nnz < nthreadsy) {
            nthreadsy = Y->nnz;
            nblocks = 1;
        } else {
            all_nblocks = (Y->nnz + nthreadsy -1) / nthreadsy;
            if(all_nblocks < max_nblocks) {
                nblocks = all_nblocks;
            } else {
                nblocks = max_nblocks;
            }
        }
        sptAssert(smem_size >= nthreadsx * nthreadsy * sizeof (sptValue));
        break;
    }
    dim3 dimBlock(nthreadsx, nthreadsy);
    printf("all_nblocks: %lu, nthreadsx: %lu, nthreadsy: %lu\n", all_nblocks, nthreadsx, nthreadsy);

    sptTimer timer;
    sptNewTimer(&timer, 0);
    sptStartTimer(timer);


    switch(impl_num) { 
    case 14:  
        printf("[Cuda SpTns * Mtx] spt_TTMRankRBNnzKernel<<<%lu, (%lu, %lu)>>>\n", nblocks, nthreadsx, nthreadsy);
        spt_TTMRankRBNnzKernel<<<nblocks, dimBlock>>>(
            Y_val, stride, Y->nnz,
            X_val, X->nnz, X_inds_m,
            fiberidx_val, fiberidx.len,
            U_val, U->nrows, U->ncols, stride);
        break; 
    case 15:  
        printf("[Cuda SpTns * Mtx] spt_TTMRankRBNnzKernelSM<<<%lu, (%lu, %lu), %lu>>>\n", nblocks, nthreadsx, nthreadsy, smem_size);
        spt_TTMRankRBNnzKernelSM<<<nblocks, dimBlock, smem_size>>>(
            Y_val, stride, Y->nnz,
            X_val, X->nnz, X_inds_m,
            fiberidx_val, fiberidx.len,
            U_val, U->nrows, U->ncols, stride);
        break; 
    }
    result = cudaThreadSynchronize();
    spt_CheckCudaError(result != 0, "Cuda SpTns * Mtx kernel");

    sptStopTimer(timer);
    sptPrintElapsedTime(timer, "Cuda SpTns * Mtx");
    sptFreeTimer(timer);

    cudaMemcpy(Y->values.values, Y_val, Y->nnz * stride * sizeof (sptValue), cudaMemcpyDeviceToHost);
    result = cudaFree(fiberidx_val);
    spt_CheckCudaError(result != 0, "Cuda SpTns * Mtx");
    result = cudaFree(U_val);
    spt_CheckCudaError(result != 0, "Cuda SpTns * Mtx");
    result = cudaFree(X_inds_m);
    spt_CheckCudaError(result != 0, "Cuda SpTns * Mtx");
    result = cudaFree(X_val);
    spt_CheckCudaError(result != 0, "Cuda SpTns * Mtx");
    result = cudaFree(Y_val);
    spt_CheckCudaError(result != 0, "Cuda SpTns * Mtx");
    sptFreeNnzIndexVector(&fiberidx);

    return 0;
}
