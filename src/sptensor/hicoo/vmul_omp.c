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
 * Sparse tensor times a vector (SpTTV) in HiCOO format
 */
int sptOmpSparseTensorMulVectorHiCOO(sptSparseTensorHiCOO *hiY, sptSparseTensorHiCOOGeneral *hiX, const sptValueVector *V, sptIndex mode) 
{
    if(mode >= hiX->nmodes) {
        spt_CheckError(SPTERR_SHAPE_MISMATCH, "Omp HiSpTns * Vec", "shape mismatch");
    }
    if(hiX->ndims[mode] != V->len) {
        spt_CheckError(SPTERR_SHAPE_MISMATCH, "Omp HiSpTns * Vec", "shape mismatch");
    }

    int result;
    sptIndex *ind_buf;
    sptNnzIndexVector fiberidx;
    sptNnzIndexVector bptr; // Do NOT free it
    sptTimer timer;
    sptNewTimer(&timer, 0);
    double setfiber_time, allocate_time, preprocess_time, copy_time, comp_time, total_time;

    /* Set fibers */
    sptStartTimer(timer);
    sptSparseTensorSetFibersHiCOO(&bptr, &fiberidx, hiX);
    sptStopTimer(timer);
    setfiber_time = sptPrintElapsedTime(timer, "sptSparseTensorSetFibersHiCOO");

    /* Allocate output Y */
    sptStartTimer(timer);
    ind_buf = malloc(hiX->nmodes * sizeof *ind_buf);
    spt_CheckOSError(!ind_buf, "Omp HiSpTns * Vec");
    for(sptIndex m = 0; m < hiX->nmodes; ++m) {
        if(m < mode)
            ind_buf[m] = hiX->ndims[m];
        else if(m > mode)
            ind_buf[m - 1] = hiX->ndims[m];
    }

    result = sptNewSparseTensorHiCOOWithBptr(hiY, hiX->nmodes - 1, ind_buf, fiberidx.len - 1, hiX->sb_bits, &bptr);
    spt_CheckError(result, "Omp HiSpTns * Vec", NULL);
    free(ind_buf);
    sptStopTimer(timer);
    allocate_time = sptPrintElapsedTime(timer, "sptNewSparseTensorHiCOOWithBptr");

    preprocess_time = setfiber_time + allocate_time;
    printf("[Total preprocess time]: %lf\n", preprocess_time);

    /* Set indices */
    sptStartTimer(timer);
    sptSparseTensorSetIndicesHiCOO(hiY, &fiberidx, hiX);
    sptStopTimer(timer);
    copy_time = sptPrintElapsedTime(timer, "Copy indices");

    /* Computation */
    sptStartTimer(timer);
    #pragma omp parallel for
    for(sptNnzIndex i = 0; i < hiY->nnz; ++i) {
        sptNnzIndex inz_begin = fiberidx.data[i];
        sptNnzIndex inz_end = fiberidx.data[i+1];
        sptNnzIndex j;
        for(j = inz_begin; j < inz_end; ++j) {
            sptIndex r = hiX->inds[0].data[j];
            hiY->values.data[i] += hiX->values.data[j] * V->data[r];
        }
    }
    sptStopTimer(timer);
    comp_time = sptPrintElapsedTime(timer, "Omp HiSpTns * Vec");

    sptFreeTimer(timer);
    sptFreeNnzIndexVector(&fiberidx);

    total_time = copy_time + comp_time;
    printf("[Total time]: %lf\n", total_time);
    printf("\n");

    return 0;
}
