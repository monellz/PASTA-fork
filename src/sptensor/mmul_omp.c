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
#ifdef PASTA_USE_OPENMP

#include <pasta.h>
#include <stdlib.h>
#include "sptensor.h"

int sptOmpSparseTensorMulMatrix(sptSemiSparseTensor *Y, sptSparseTensor *X, const sptMatrix *U, sptIndex const mode) 
{
    if(mode >= X->nmodes) {
        spt_CheckError(SPTERR_SHAPE_MISMATCH, "Omp SpTns * Mtx", "shape mismatch");
    }
    if(X->ndims[mode] != U->nrows) {
        spt_CheckError(SPTERR_SHAPE_MISMATCH, "Omp SpTns * Mtx", "shape mismatch");
    }

    sptIndex stride = U->stride;
    int result;
    sptIndex *ind_buf;
    sptIndex m;
    sptNnzIndexVector fiberidx;
    sptTimer timer;
    sptNewTimer(&timer, 0);
    double sort_time, setfiber_time, allocate_time, preprocess_time, copy_time, comp_time, total_time;

    /* Sort tensor except mode */
    sptStartTimer(timer);
    sptSparseTensorSortIndexAtMode(X, mode, 0);
    sptStopTimer(timer);
    sort_time = sptPrintElapsedTime(timer, "sptSparseTensorSortIndexAtMode");
    
    /* Set fibers */
    sptStartTimer(timer);
    sptSemiSparseTensorSetFibers(&fiberidx, X, mode);
    sptStopTimer(timer);
    setfiber_time = sptPrintElapsedTime(timer, "sptSparseTensorSetFibers");

    /* Allocate output Y */
    sptStartTimer(timer);
    ind_buf = malloc(X->nmodes * sizeof *ind_buf);
    spt_CheckOSError(!ind_buf, "Omp SpTns * Mtx");
    for(m = 0; m < X->nmodes; ++m) {
        ind_buf[m] = X->ndims[m];
    }
    ind_buf[mode] = U->ncols;
    result = sptNewSemiSparseTensorWithNnz(Y, X->nmodes, mode, ind_buf, fiberidx.len - 1);
    spt_CheckError(result, "Omp SpTns * Mtx", NULL);
    free(ind_buf);
    if(Y->values.stride != stride) {
        spt_CheckError(SPTERR_SHAPE_MISMATCH, "Omp SpTns * Mtx", "shape mismatch");
    }
    sptStopTimer(timer);
    allocate_time = sptPrintElapsedTime(timer, "sptNewSemiSparseTensorWithNnz");

    preprocess_time = sort_time + setfiber_time + allocate_time;
    printf("[Total preprocess time]: %lf\n", preprocess_time);

    /* Set indices */
    sptStartTimer(timer);
    sptSemiSparseTensorSetIndices(Y, &fiberidx, X);
    sptStopTimer(timer);
    copy_time = sptPrintElapsedTime(timer, "Copy indices");

    /* Computation */
    sptStartTimer(timer);
    #pragma omp parallel for
    for(sptNnzIndex i = 0; i < Y->nnz; ++i) {
        sptNnzIndex inz_begin = fiberidx.data[i];
        sptNnzIndex inz_end = fiberidx.data[i+1];
        for(sptNnzIndex j = inz_begin; j < inz_end; ++j) {
            sptIndex r = X->inds[mode].data[j];
            #pragma omp simd
            for(sptIndex k = 0; k < U->ncols; ++k) {
                Y->values.values[i * stride + k] += X->values.data[j] * U->values[r * stride + k];
            }
        }
    }
    sptStopTimer(timer);
    comp_time = sptPrintElapsedTime(timer, "Omp SpTns * Mtx");
    
    sptFreeTimer(timer);
    sptFreeNnzIndexVector(&fiberidx);

    total_time = copy_time + comp_time;
    printf("[Total time]: %lf\n", total_time);
    printf("\n");

    return 0;
}
#endif