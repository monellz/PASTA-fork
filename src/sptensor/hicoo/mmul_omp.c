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
#include <stdlib.h>

/**
 * Sparse tensor times a dense matrix (SpTTM)
 * @param[out] Y    the result of X*U, should be uninitialized
 * @param[in]  X    the sparse tensor input X
 * @param[in]  U    the dense matrix input U
 * @param      mode the mode on which the multiplication is done on
 *
 * This function will sort Y with `sptSparseTensorSortIndexAtMode`
 * automatically, this operation can be undone with `sptSparseTensorSortIndex`
 * if you need to access raw data.
 * Anyway, you do not have to take this side-effect into consideration if you
 * do not need to access raw data.
 */
int sptOmpSparseTensorMulMatrixHiCOO(sptSemiSparseTensorHiCOO *Y, sptSparseTensorHiCOOGeneral *X, const sptMatrix *U, sptIndex const mode) 
{
    if(mode >= X->nmodes) {
        spt_CheckError(SPTERR_SHAPE_MISMATCH, "Omp HiSpTns * Mtx", "shape mismatch");
    }
    if(X->ndims[mode] != U->nrows) {
        spt_CheckError(SPTERR_SHAPE_MISMATCH, "Omp HiSpTns * Mtx", "shape mismatch");
    }
    if(X->nmodes != X->ncmodes + 1) {
        spt_CheckError(SPTERR_SHAPE_MISMATCH, "Omp HiSpTns * Mtx", "shape mismatch");
    }
    
    sptIndex stride = U->stride;
    int result;
    sptIndex m;
    sptNnzIndexVector fiberidx;
    sptNnzIndexVector bptr; // Do NOT free it
    sptIndex *ind_buf;
    sptTimer timer;
    sptNewTimer(&timer, 0);
    double setfiber_time, allocate_time, preprocess_time, copy_time, comp_time, total_time;

    /* Set fibers */
    sptStartTimer(timer);
    sptSemiSparseTensorSetFibersHiCOO(&bptr, &fiberidx, X);
    sptStopTimer(timer);
    setfiber_time = sptPrintElapsedTime(timer, "sptSemiSparseTensorSetFibersHiCOO");

    /* Allocate output Y */
    sptStartTimer(timer);
    ind_buf = malloc(X->nmodes * sizeof *ind_buf);
    spt_CheckOSError(!ind_buf, "Omp HiSpTns * Mtx");
    for(m = 0; m < X->nmodes; ++m) {
        ind_buf[m] = X->ndims[m];
    }
    ind_buf[mode] = U->ncols;
    result = sptNewSemiSparseTensorHiCOOWithBptr(Y, X->nmodes, ind_buf, fiberidx.len - 1, mode, X->sb_bits, &bptr);
    spt_CheckError(result, "Omp HiSpTns * Mtx", NULL);
    free(ind_buf);
    if(Y->values.stride != stride) {
        spt_CheckError(SPTERR_SHAPE_MISMATCH, "Omp HiSpTns * Mtx", "shape mismatch");
    }
    sptStopTimer(timer);
    allocate_time = sptPrintElapsedTime(timer, "sptNewSemiSparseTensorHiCOOWithBptr");

    preprocess_time = setfiber_time + allocate_time;
    printf("[Total preprocess time]: %lf\n", preprocess_time);

    /* Set indices */
    sptStartTimer(timer);
    sptSemiSparseTensorSetIndicesHiCOO(Y, &fiberidx, X);
    sptStopTimer(timer);
    copy_time = sptPrintElapsedTime(timer, "Copy indices");
    // printf("fiberidx-final:\n");
    // sptDumpNnzIndexVector(&fiberidx, stdout);

    /* Computation */
    sptStartTimer(timer);
    #pragma omp parallel for schedule(dynamic)
    for(sptNnzIndex i = 0; i < Y->nnz; ++i) {
        sptNnzIndex inz_begin = fiberidx.data[i];
        sptNnzIndex inz_end = fiberidx.data[i+1];
        for(sptNnzIndex j = inz_begin; j < inz_end; ++j) {
            sptIndex k = X->inds[0].data[j];    // non-compressed modes
            #pragma omp simd
            for(sptIndex r = 0; r < U->ncols; ++r) {
                Y->values.values[i * stride + r] += X->values.data[j] * U->values[k * stride + r];
            }
        }
    }

    sptStopTimer(timer);
    comp_time = sptPrintElapsedTime(timer, "Omp HiSpTns * Mtx");
    
    sptFreeTimer(timer);
    sptFreeNnzIndexVector(&fiberidx);

    total_time = copy_time + comp_time;
    printf("[Total time]: %lf\n", total_time);
    printf("\n");
    
    return 0;
}

