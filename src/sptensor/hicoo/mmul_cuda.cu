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
#include "../mmul_cuda_kernels.h"


int sptCudaSparseTensorMulMatrixHiCOO(
    sptSemiSparseTensorHiCOO *hiY,
    sptSparseTensorHiCOOGeneral *hiX,
    const sptMatrix *U,
    sptIndex const mode,
    sptIndex const impl_num,
    sptNnzIndex const smem_size) 
{
    if(mode >= hiX->nmodes) {
        spt_CheckError(SPTERR_SHAPE_MISMATCH, "Cuda HiSpTns * Mtx", "shape mismatch");
    }
    if(hiX->ndims[mode] != U->nrows) {
        spt_CheckError(SPTERR_SHAPE_MISMATCH, "Cuda HiSpTns * Mtx", "shape mismatch");
    }
    if(hiX->nmodes != hiX->ncmodes + 1) {
        spt_CheckError(SPTERR_SHAPE_MISMATCH, "Cuda HiSpTns * Mtx", "shape mismatch");
    }

    sptIndex stride = U->stride;
    int result;
    sptIndex *ind_buf;
    sptIndex m;
    sptNnzIndexVector fiberidx;
    sptNnzIndexVector bptr; // Do NOT free it
    sptTimer timer;
    sptNewTimer(&timer, 0);
    double setfiber_time, allocate_time, preprocess_time, copy_time_cpu, copy_time_gpu, comp_time, total_time;

    /* Set fibers */
    sptStartTimer(timer);
    sptSemiSparseTensorSetFibersHiCOO(&bptr, &fiberidx, hiX);
    sptStopTimer(timer);
    setfiber_time = sptPrintElapsedTime(timer, "sptSemiSparseTensorSetFibersHiCOO");

    /* Allocate output Y */
    sptStartTimer(timer);
    ind_buf = new sptIndex[hiX->nmodes * sizeof *ind_buf];
    for(m = 0; m < hiX->nmodes; ++m) {
        ind_buf[m] = hiX->ndims[m];
    }
    ind_buf[mode] = U->ncols;
    result = sptNewSemiSparseTensorHiCOOWithBptr(hiY, hiX->nmodes, ind_buf, fiberidx.len - 1, mode, hiX->sb_bits, &bptr);
    spt_CheckError(result, "Cuda HiSpTns * Mtx", NULL);
    delete[] ind_buf;
    if(hiY->values.stride != stride) {
        spt_CheckError(SPTERR_SHAPE_MISMATCH, "Cuda HiSpTns * Mtx", "shape mismatch");
    }
    sptStopTimer(timer);
    allocate_time = sptPrintElapsedTime(timer, "sptNewSemiSparseTensorHiCOOWithBptr");

    preprocess_time = setfiber_time + allocate_time;
    printf("[Total preprocess time]: %lf\n", preprocess_time);

    /* Set indices */
    sptStartTimer(timer);
    sptSemiSparseTensorSetIndicesHiCOO(hiY, &fiberidx, hiX);
    sptStopTimer(timer);
    copy_time_cpu = sptPrintElapsedTime(timer, "Copy indices");

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
    spt_CheckCudaError(result != 0, "Cuda HiSpTns * Mtx");
    sptValue *X_val = NULL;
    result = cudaMalloc((void **) &X_val, hiX->nnz * sizeof (sptValue));
    spt_CheckCudaError(result != 0, "Cuda HiSpTns * Mtx");
    sptIndex *X_inds_m = NULL;
    result = cudaMalloc((void **) &X_inds_m, hiX->nnz * sizeof (sptIndex));
    spt_CheckCudaError(result != 0, "Cuda HiSpTns * Mtx");
    sptValue *U_val = NULL;
    result = cudaMalloc((void **) &U_val, U->nrows * stride * sizeof (sptValue));
    spt_CheckCudaError(result != 0, "Cuda HiSpTns * Mtx");
    sptNnzIndex *fiberidx_val = NULL;
    result = cudaMalloc((void **) &fiberidx_val, fiberidx.len * sizeof (sptNnzIndex));
    spt_CheckCudaError(result != 0, "Cuda HiSpTns * Mtx");

    /* Copy data to GPU */
    sptStartTimer(timer);
    cudaMemset(Y_val, 0, hiY->nnz * stride * sizeof (sptValue));
    cudaMemcpy(X_val, hiX->values.data, hiX->nnz * sizeof (sptValue), cudaMemcpyHostToDevice);
    cudaMemcpy(X_inds_m, hiX->inds[0].data, hiX->nnz * sizeof (sptIndex), cudaMemcpyHostToDevice);
    cudaMemcpy(U_val, U->values, U->nrows * stride * sizeof (sptValue), cudaMemcpyHostToDevice);
    cudaMemcpy(fiberidx_val, fiberidx.data, fiberidx.len * sizeof (sptNnzIndex), cudaMemcpyHostToDevice);
    sptStopTimer(timer);
    copy_time_gpu = sptPrintElapsedTime(timer, "Device copy");

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
        sptAssert(smem_size >= nthreadsx * nthreadsy * sizeof (sptValue));
        break;
    }
    dim3 dimBlock(nthreadsx, nthreadsy);
    printf("all_nblocks: %lu, nthreadsx: %lu, nthreadsy: %lu\n", all_nblocks, nthreadsx, nthreadsy);

    /* Computation */
    sptStartTimer(timer);

    switch(impl_num) {
    case 14:  
        printf("[Cuda HiSpTns * Mtx] spt_TTMRankRBNnzKernel<<<%lu, (%lu, %lu)>>>\n", nblocks, nthreadsx, nthreadsy);
        spt_TTMRankRBNnzKernel<<<nblocks, dimBlock>>>(
            Y_val, stride, hiY->nnz,
            X_val, hiX->nnz, X_inds_m,
            fiberidx_val, fiberidx.len,
            U_val, U->nrows, U->ncols, stride);
        break; 
    case 15:  
        printf("[Cuda HiSpTns * Mtx] spt_TTMRankRBNnzKernelSM<<<%lu, (%lu, %lu), %lu>>>\n", nblocks, nthreadsx, nthreadsy, smem_size);
        spt_TTMRankRBNnzKernelSM<<<nblocks, dimBlock, smem_size>>>(
            Y_val, stride, hiY->nnz,
            X_val, hiX->nnz, X_inds_m,
            fiberidx_val, fiberidx.len,
            U_val, U->nrows, U->ncols, stride);
        break; 
    }
    result = cudaThreadSynchronize();
    spt_CheckCudaError(result != 0, "Cuda HiSpTns * Mtx kernel");

    sptStopTimer(timer);
    comp_time = sptPrintElapsedTime(timer, "Cuda HiSpTns * Mtx");

    /* Copy back to CPU */
    sptStartTimer(timer);
    cudaMemcpy(hiY->values.values, Y_val, hiY->nnz * stride * sizeof (sptValue), cudaMemcpyDeviceToHost);
    sptStopTimer(timer);
    copy_time_gpu += sptPrintElapsedTime(timer, "Device copy back");

    sptFreeTimer(timer);
    result = cudaFree(fiberidx_val);
    spt_CheckCudaError(result != 0, "Cuda HiSpTns * Mtx");
    result = cudaFree(U_val);
    spt_CheckCudaError(result != 0, "Cuda HiSpTns * Mtx");
    result = cudaFree(X_inds_m);
    spt_CheckCudaError(result != 0, "Cuda HiSpTns * Mtx");
    result = cudaFree(X_val);
    spt_CheckCudaError(result != 0, "Cuda HiSpTns * Mtx");
    result = cudaFree(Y_val);
    spt_CheckCudaError(result != 0, "Cuda HiSpTns * Mtx");
    sptFreeNnzIndexVector(&fiberidx);

    total_time = copy_time_cpu + copy_time_gpu + comp_time;
    printf("[Total time]: %lf\n", total_time);
    printf("\n");

    return 0;
}
