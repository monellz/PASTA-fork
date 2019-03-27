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
#include "../mmul_cuda_kernels.h"


int sptCudaSparseTensorMulMatrixHiCOO(
    sptSemiSparseTensorHiCOO *hiY,
    sptSparseTensorHiCOOGeneral *hiX,
    const sptMatrix *U,
    sptIndex const mode,
    sptIndex const impl_num,
    sptNnzIndex const smen_size) 
{
    if(mode >= hiX->nmodes) {
        spt_CheckError(SPTERR_SHAPE_MISMATCH, "CUDA HiSpTns * Mtx", "shape mismatch");
    }
    if(hiX->ndims[mode] != U->nrows) {
        spt_CheckError(SPTERR_SHAPE_MISMATCH, "CUDA HiSpTns * Mtx", "shape mismatch");
    }
    if(hiX->nmodes != hiX->ncmodes + 1) {
        spt_CheckError(SPTERR_SHAPE_MISMATCH, "CPU  HiSpTns * Mtx", "shape mismatch");
    }

    sptIndex stride = U->stride;
    int result;
    sptIndex *ind_buf;
    sptIndex m;
    sptNnzIndexVector fiberidx;

    ind_buf = new sptIndex[hiX->nmodes * sizeof *ind_buf];
    for(m = 0; m < hiX->nmodes; ++m) {
        ind_buf[m] = hiX->ndims[m];
    }
    ind_buf[mode] = U->ncols;
    result = sptNewSemiSparseTensorHiCOO(hiY, hiX->nmodes, ind_buf, mode, hiX->sb_bits);
    delete[] ind_buf;
    spt_CheckError(result, "CUDA HiSpTns * Mtx", NULL);
    if(hiY->values.stride != stride) {
        spt_CheckError(SPTERR_SHAPE_MISMATCH, "CPU  HiSpTns * Mtx", "shape mismatch");
    }

    sptSemiSparseTensorSetIndicesHiCOO(hiY, &fiberidx, hiX);

    double flen = (double)hiX->nnz / fiberidx.len;
    sptNnzIndex tmp_flen = (fiberidx.data[1] - fiberidx.data[0]) - flen;
    double fvar = tmp_flen * tmp_flen;
    for(sptNnzIndex i=1; i<fiberidx.len - 1; ++i) {
        tmp_flen = (fiberidx.data[i+1] - fiberidx.data[i]) - flen;
        fvar += tmp_flen * tmp_flen;
    }
    tmp_flen = (hiX->nnz - fiberidx.data[fiberidx.len - 1]) - flen;
    fvar += tmp_flen * tmp_flen;
    fvar = sqrt(fvar);
    printf("nfibs: %zu, flen: %.2f, fvar: %.2f\n", fiberidx.len, flen, fvar);

    sptValue *Y_val = NULL;
    result = cudaMalloc((void **) &Y_val, hiY->nnz * stride * sizeof (sptValue));
    spt_CheckCudaError(result != 0, "CUDA HiSpTns * Mtx");
    // jli: Add memset to Y.
    cudaMemset(Y_val, 0, hiY->nnz * stride * sizeof (sptValue));
    sptValue *X_val = NULL;
    result = cudaMalloc((void **) &X_val, hiX->nnz * sizeof (sptValue));
    spt_CheckCudaError(result != 0, "CUDA HiSpTns * Mtx");
    cudaMemcpy(X_val, hiX->values.data, hiX->nnz * sizeof (sptValue), cudaMemcpyHostToDevice);
    sptIndex *X_inds_m = NULL;
    result = cudaMalloc((void **) &X_inds_m, hiX->nnz * sizeof (sptIndex));
    spt_CheckCudaError(result != 0, "CUDA HiSpTns * Mtx");
    cudaMemcpy(X_inds_m, hiX->inds[0].data, hiX->nnz * sizeof (sptIndex), cudaMemcpyHostToDevice);
    sptValue *U_val = NULL;
    result = cudaMalloc((void **) &U_val, U->nrows * stride * sizeof (sptValue));
    spt_CheckCudaError(result != 0, "CUDA HiSpTns * Mtx");
    cudaMemcpy(U_val, U->values, U->nrows * stride * sizeof (sptValue), cudaMemcpyHostToDevice);
    sptNnzIndex *fiberidx_val = NULL;
    result = cudaMalloc((void **) &fiberidx_val, fiberidx.len * sizeof (sptNnzIndex));
    spt_CheckCudaError(result != 0, "CUDA HiSpTns * Mtx");
    cudaMemcpy(fiberidx_val, fiberidx.data, fiberidx.len * sizeof (sptNnzIndex), cudaMemcpyHostToDevice);

    const sptNnzIndex max_nblocks = 32768;
    const sptNnzIndex max_nthreads_per_block = 256;
    sptNnzIndex max_nthreadsy = 16;

    sptNnzIndex nthreadsx = 1;
    sptNnzIndex nthreadsy = 1;
    sptNnzIndex all_nblocks = 0;
    sptNnzIndex nblocks = 0;

    const char *env_PARTI_TTM_NTHREADS = getenv("PARTI_TTM_NTHREADS");

    switch(impl_num) {
    // case 1:
    case 11: // Naive, 1D
        if(hiY->nnz < max_nthreads_per_block) {
            nthreadsx = hiY->nnz;
            nblocks = 1;
        } else {
            nthreadsx = max_nthreads_per_block;
            all_nblocks = (hiY->nnz + nthreadsx -1) / nthreadsx;
            if(all_nblocks < max_nblocks) {
                nblocks = all_nblocks;
            } else {
                nblocks = max_nblocks;
            }
        }
        break;
    case 12:
        if(U->ncols <= max_nthreadsy)
            nthreadsy = U->ncols;
        else
            nthreadsy = max_nthreadsy;
        nthreadsx = max_nthreads_per_block / nthreadsy;

        if(hiY->nnz < nthreadsx) {
            nthreadsx = hiY->nnz;
            nblocks = 1;
        } else {
            all_nblocks = (hiY->nnz + nthreadsx -1) / nthreadsx;
            if(all_nblocks < max_nblocks) {
                nblocks = all_nblocks;
            } else {
                nblocks = max_nblocks;
            }
        }
        break;
    case 13:
    case 14:
        if(U->ncols <= max_nthreadsy)
            nthreadsx = U->ncols;
        else
            nthreadsx = max_nthreadsy;
        nthreadsy = max_nthreads_per_block / nthreadsx;

        if(hiY->nnz < nthreadsy) {
            nthreadsy = hiY->nnz;
            nblocks = 1;
        } else {
            all_nblocks = (hiY->nnz + nthreadsy -1) / nthreadsy;
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

        if(hiY->nnz < nthreadsy) {
            nthreadsy = hiY->nnz;
            nblocks = 1;
        } else {
            all_nblocks = (hiY->nnz + nthreadsy -1) / nthreadsy;
            if(all_nblocks < max_nblocks) {
                nblocks = all_nblocks;
            } else {
                nblocks = max_nblocks;
            }
        }
        sptAssert(smen_size >= nthreadsx * nthreadsy * sizeof (sptValue));
        break;
    }
    dim3 dimBlock(nthreadsx, nthreadsy);
    printf("all_nblocks: %lu, nthreadsx: %lu, nthreadsy: %lu\n", all_nblocks, nthreadsx, nthreadsy);

    sptTimer timer;
    sptNewTimer(&timer, 0);
    sptStartTimer(timer);


    switch(impl_num) {
    // case 1:
    case 11: // Naive
        printf("[CUDA HiSpTns * Mtx] spt_TTMNnzKernel<<<%lu, (%lu, %lu)>>>\n", nblocks, nthreadsx, nthreadsy);
        spt_TTMNnzKernel<<<nblocks, dimBlock>>>(
            Y_val, stride, hiY->nnz,
            X_val, hiX->nnz, X_inds_m,
            fiberidx_val, fiberidx.len,
            U_val, U->nrows, U->ncols, stride);
        break;
    case 12:  
        printf("[CUDA HiSpTns * Mtx] spt_TTMNnzRankKernel<<<%lu, (%lu, %lu)>>>\n", nblocks, nthreadsx, nthreadsy);
        spt_TTMNnzRankKernel<<<nblocks, dimBlock>>>(
            Y_val, stride, hiY->nnz,
            X_val, hiX->nnz, X_inds_m,
            fiberidx_val, fiberidx.len,
            U_val, U->nrows, U->ncols, stride);
        break; 
    case 13:  
        printf("[CUDA HiSpTns * Mtx] spt_TTMRankNnzKernel<<<%lu, (%lu, %lu)>>>\n", nblocks, nthreadsx, nthreadsy);
        spt_TTMRankNnzKernel<<<nblocks, dimBlock>>>(
            Y_val, stride, hiY->nnz,
            X_val, hiX->nnz, X_inds_m,
            fiberidx_val, fiberidx.len,
            U_val, U->nrows, U->ncols, stride);
        break; 
    case 14:  
        printf("[CUDA HiSpTns * Mtx] spt_TTMRankRBNnzKernel<<<%lu, (%lu, %lu)>>>\n", nblocks, nthreadsx, nthreadsy);
        spt_TTMRankRBNnzKernel<<<nblocks, dimBlock>>>(
            Y_val, stride, hiY->nnz,
            X_val, hiX->nnz, X_inds_m,
            fiberidx_val, fiberidx.len,
            U_val, U->nrows, U->ncols, stride);
        break; 
    case 15:  
        printf("[CUDA HiSpTns * Mtx] spt_TTMRankRBNnzKernelSM<<<%lu, (%lu, %lu), %lu>>>\n", nblocks, nthreadsx, nthreadsy, smen_size);
        spt_TTMRankRBNnzKernelSM<<<nblocks, dimBlock, smen_size>>>(
            Y_val, stride, hiY->nnz,
            X_val, hiX->nnz, X_inds_m,
            fiberidx_val, fiberidx.len,
            U_val, U->nrows, U->ncols, stride);
        break; 
    }
    result = cudaThreadSynchronize();
    spt_CheckCudaError(result != 0, "CUDA HiSpTns * Mtx kernel");

    sptStopTimer(timer);
    sptPrintElapsedTime(timer, "CUDA HiSpTns * Mtx");
    sptFreeTimer(timer);

    cudaMemcpy(hiY->values.values, Y_val, hiY->nnz * stride * sizeof (sptValue), cudaMemcpyDeviceToHost);
    result = cudaFree(fiberidx_val);
    spt_CheckCudaError(result != 0, "CUDA HiSpTns * Mtx");
    result = cudaFree(U_val);
    spt_CheckCudaError(result != 0, "CUDA HiSpTns * Mtx");
    result = cudaFree(X_inds_m);
    spt_CheckCudaError(result != 0, "CUDA HiSpTns * Mtx");
    result = cudaFree(X_val);
    spt_CheckCudaError(result != 0, "CUDA HiSpTns * Mtx");
    result = cudaFree(Y_val);
    spt_CheckCudaError(result != 0, "CUDA HiSpTns * Mtx");
    sptFreeNnzIndexVector(&fiberidx);

    return 0;
}
