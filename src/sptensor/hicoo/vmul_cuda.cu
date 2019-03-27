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
    sptNnzIndex const smen_size) 
{

    if(mode >= hiX->nmodes) {
        spt_CheckError(SPTERR_SHAPE_MISMATCH, "CUDA HiSpTns * Vec", "shape mismatch");
    }
    if(hiX->ndims[mode] != V->len) {
        spt_CheckError(SPTERR_SHAPE_MISMATCH, "CUDA HiSpTns * Vec", "shape mismatch");
    }

    int result;
    sptIndex *ind_buf;
    sptNnzIndexVector fiberidx;
    sptTimer timer;
    sptNewTimer(&timer, 0);

    sptStartTimer(timer);
    ind_buf = new sptIndex[hiX->nmodes * sizeof *ind_buf];
    spt_CheckOSError(!ind_buf, "CUDA  SpTns * Vec");
    for(sptIndex m = 0; m < hiX->nmodes; ++m) {
        if(m < mode)
            ind_buf[m] = hiX->ndims[m];
        else if(m > mode)
            ind_buf[m - 1] = hiX->ndims[m];
    }

    result = sptNewSparseTensorHiCOO(hiY, hiX->nmodes - 1, ind_buf, 0, hiX->sb_bits);
    free(ind_buf);
    spt_CheckError(result, "CUDA HiSpTns * Vec", NULL);
    sptSparseTensorSetIndicesHiCOO(hiY, &fiberidx, hiX);
    sptStopTimer(timer);
    sptPrintElapsedTime(timer, "Allocate output tensor");

    sptValue *Y_val = NULL;
    result = cudaMalloc((void **) &Y_val, hiY->nnz * sizeof (sptValue));
    spt_CheckCudaError(result != 0, "CUDA HiSpTns * Vec");
    // jli: Add memset to Y.
    cudaMemset(Y_val, 0, hiY->nnz * sizeof (sptValue));
    sptValue *X_val = NULL;
    result = cudaMalloc((void **) &X_val, hiX->nnz * sizeof (sptValue));
    spt_CheckCudaError(result != 0, "CUDA HiSpTns * Vec");
    cudaMemcpy(X_val, hiX->values.data, hiX->nnz * sizeof (sptValue), cudaMemcpyHostToDevice);
    sptIndex *X_inds_m = NULL;
    result = cudaMalloc((void **) &X_inds_m, hiX->nnz * sizeof (sptIndex));
    spt_CheckCudaError(result != 0, "CUDA HiSpTns * Vec");
    cudaMemcpy(X_inds_m, hiX->inds[0].data, hiX->nnz * sizeof (sptIndex), cudaMemcpyHostToDevice);
    sptValue *V_val = NULL;
    result = cudaMalloc((void **) &V_val, V->len * sizeof (sptValue));
    spt_CheckCudaError(result != 0, "CUDA HiSpTns * Vec");
    cudaMemcpy(V_val, V->data, V->len * sizeof (sptValue), cudaMemcpyHostToDevice);
    sptNnzIndex *fiberidx_val = NULL;
    result = cudaMalloc((void **) &fiberidx_val, fiberidx.len * sizeof (sptNnzIndex));
    spt_CheckCudaError(result != 0, "CUDA HiSpTns * Vec");
    cudaMemcpy(fiberidx_val, fiberidx.data, fiberidx.len * sizeof (sptNnzIndex), cudaMemcpyHostToDevice);

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

    sptStartTimer(timer);


    switch(impl_num) {
    // case 1:
    case 11: // Naive
        printf("[CUDA HiSpTns * Vec] spt_TTVNnzKernel<<<%lu, (%lu, %lu)>>>\n", nblocks, nthreadsx, nthreadsy);
        spt_TTVNnzKernel<<<nblocks, dimBlock>>>(
            Y_val, hiY->nnz,
            X_val, hiX->nnz, X_inds_m, 
            fiberidx_val, fiberidx.len,
            V_val, V->len);
        break;
    }
    result = cudaThreadSynchronize();
    spt_CheckCudaError(result != 0, "CUDA HiSpTns * Vec kernel");

    sptStopTimer(timer);
    sptPrintElapsedTime(timer, "CUDA HiSpTns * Vec");
    sptFreeTimer(timer);

    cudaMemcpy(hiY->values.data, Y_val, hiY->nnz * sizeof (sptValue), cudaMemcpyDeviceToHost);
    result = cudaFree(fiberidx_val);
    spt_CheckCudaError(result != 0, "CUDA HiSpTns * Vec");
    result = cudaFree(V_val);
    spt_CheckCudaError(result != 0, "CUDA HiSpTns * Vec");
    result = cudaFree(X_inds_m);
    spt_CheckCudaError(result != 0, "CUDA HiSpTns * Vec");
    result = cudaFree(X_val);
    spt_CheckCudaError(result != 0, "CUDA HiSpTns * Vec");
    result = cudaFree(Y_val);
    spt_CheckCudaError(result != 0, "CUDA HiSpTns * Vec");
    sptFreeNnzIndexVector(&fiberidx);


    return 0;
}
