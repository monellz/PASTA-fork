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
#include "../vmul_cuda_kernels.h"


int sptCudaSparseTensorMulVectorHiCOO(
    sptSparseTensorHiCOO *hiY,
    sptSparseTensorHiCOOGeneral *hiX,
    const sptValueVector *V,
    sptIndex const mode,
    sptIndex const impl_num,
    sptNnzIndex const smem_size) 
{

    if(mode >= hiX->nmodes) {
        spt_CheckError(SPTERR_SHAPE_MISMATCH, "Cuda HiSpTns * Vec", "shape mismatch");
    }
    if(hiX->ndims[mode] != V->len) {
        spt_CheckError(SPTERR_SHAPE_MISMATCH, "Cuda HiSpTns * Vec", "shape mismatch");
    }

    int result;
    sptIndex *ind_buf;
    sptNnzIndexVector fiberidx;
    sptNnzIndexVector bptr; // Do NOT free it
    sptTimer timer;
    sptNewTimer(&timer, 0);
    double setfiber_time, allocate_time, preprocess_time, copy_time_cpu, copy_time_gpu, comp_time, total_time;

    /* Set fibers */
    sptStartTimer(timer);
    sptSparseTensorSetFibersHiCOO(&bptr, &fiberidx, hiX);
    sptStopTimer(timer);
    setfiber_time = sptPrintElapsedTime(timer, "sptSparseTensorSetFibersHiCOO");

    /* Allocate output Y */
    sptStartTimer(timer);
    ind_buf = new sptIndex[hiX->nmodes * sizeof *ind_buf];
    spt_CheckOSError(!ind_buf, "Cuda HiSpTns * Vec");
    for(sptIndex m = 0; m < hiX->nmodes; ++m) {
        if(m < mode)
            ind_buf[m] = hiX->ndims[m];
        else if(m > mode)
            ind_buf[m - 1] = hiX->ndims[m];
    }

    result = sptNewSparseTensorHiCOOWithBptr(hiY, hiX->nmodes - 1, ind_buf, fiberidx.len - 1, hiX->sb_bits, &bptr);
    spt_CheckError(result, "Cuda HiSpTns * Vec", NULL);
    free(ind_buf);
    sptStopTimer(timer);
    allocate_time = sptPrintElapsedTime(timer, "sptNewSparseTensorHiCOOWithBptr");

    preprocess_time = setfiber_time + allocate_time;
    printf("[Total preprocess time]: %lf\n", preprocess_time);

    /* Set indices */
    sptStartTimer(timer);
    sptSparseTensorSetIndicesHiCOO(hiY, &fiberidx, hiX);
    sptStopTimer(timer);
    copy_time_cpu = sptPrintElapsedTime(timer, "Copy indices");

    sptValue *Y_val = NULL;
    result = cudaMalloc((void **) &Y_val, hiY->nnz * sizeof (sptValue));
    spt_CheckCudaError(result != 0, "Cuda HiSpTns * Vec");
    sptValue *X_val = NULL;
    result = cudaMalloc((void **) &X_val, hiX->nnz * sizeof (sptValue));
    spt_CheckCudaError(result != 0, "Cuda HiSpTns * Vec");
    sptIndex *X_inds_m = NULL;
    result = cudaMalloc((void **) &X_inds_m, hiX->nnz * sizeof (sptIndex));
    spt_CheckCudaError(result != 0, "Cuda HiSpTns * Vec");
    sptValue *V_val = NULL;
    result = cudaMalloc((void **) &V_val, V->len * sizeof (sptValue));
    spt_CheckCudaError(result != 0, "Cuda HiSpTns * Vec");
    sptNnzIndex *fiberidx_val = NULL;
    result = cudaMalloc((void **) &fiberidx_val, fiberidx.len * sizeof (sptNnzIndex));
    spt_CheckCudaError(result != 0, "Cuda HiSpTns * Vec");
    
    /* Copy data to GPU */
    sptStartTimer(timer);
    cudaMemset(Y_val, 0, hiY->nnz * sizeof (sptValue));
    cudaMemcpy(X_val, hiX->values.data, hiX->nnz * sizeof (sptValue), cudaMemcpyHostToDevice);
    cudaMemcpy(X_inds_m, hiX->inds[0].data, hiX->nnz * sizeof (sptIndex), cudaMemcpyHostToDevice);
    cudaMemcpy(V_val, V->data, V->len * sizeof (sptValue), cudaMemcpyHostToDevice);
    cudaMemcpy(fiberidx_val, fiberidx.data, fiberidx.len * sizeof (sptNnzIndex), cudaMemcpyHostToDevice);
    sptStopTimer(timer);
    copy_time_gpu = sptPrintElapsedTime(timer, "Device copy");

    const sptNnzIndex max_nblocks = 32768;
    const sptNnzIndex max_nthreads_per_block = 256;

    sptNnzIndex nthreadsx = 1;
    sptNnzIndex nthreadsy = 1;
    sptNnzIndex all_nblocks = 0;
    sptNnzIndex nblocks = 0;

    const char *env_PASTA_TTV_NTHREADS = getenv("PASTA_TTV_NTHREADS");

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
    }
    dim3 dimBlock(nthreadsx, nthreadsy);
    printf("all_nblocks: %lu, nthreadsx: %lu, nthreadsy: %lu\n", all_nblocks, nthreadsx, nthreadsy);

    /* Computation */
    sptStartTimer(timer);

    switch(impl_num) {
    // case 1:
    case 11: // Naive
        printf("[Cuda HiSpTns * Vec] spt_TTVNnzKernel<<<%lu, (%lu, %lu)>>>\n", nblocks, nthreadsx, nthreadsy);
        spt_TTVNnzKernel<<<nblocks, dimBlock>>>(
            Y_val, hiY->nnz,
            X_val, hiX->nnz, X_inds_m, 
            fiberidx_val, fiberidx.len,
            V_val, V->len);
        break;
    }
    result = cudaThreadSynchronize();
    spt_CheckCudaError(result != 0, "Cuda HiSpTns * Vec kernel");

    sptStopTimer(timer);
    comp_time = sptPrintElapsedTime(timer, "Cuda HiSpTns * Vec");
    
    /* Copy back to CPU */
    sptStartTimer(timer);
    cudaMemcpy(hiY->values.data, Y_val, hiY->nnz * sizeof (sptValue), cudaMemcpyDeviceToHost);
    sptStopTimer(timer);
    copy_time_gpu += sptPrintElapsedTime(timer, "Device copy back");

    sptFreeTimer(timer);
    result = cudaFree(fiberidx_val);
    spt_CheckCudaError(result != 0, "Cuda HiSpTns * Vec");
    result = cudaFree(V_val);
    spt_CheckCudaError(result != 0, "Cuda HiSpTns * Vec");
    result = cudaFree(X_inds_m);
    spt_CheckCudaError(result != 0, "Cuda HiSpTns * Vec");
    result = cudaFree(X_val);
    spt_CheckCudaError(result != 0, "Cuda HiSpTns * Vec");
    result = cudaFree(Y_val);
    spt_CheckCudaError(result != 0, "Cuda HiSpTns * Vec");
    sptFreeNnzIndexVector(&fiberidx);

    total_time = copy_time_cpu + copy_time_gpu + comp_time;
    printf("[Total time]: %lf\n", total_time);
    printf("\n");

    return 0;
}
