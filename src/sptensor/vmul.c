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
#include "sptensor.h"

/**
 * Sparse tensor times a vector (SpTTV)
 */
int sptSparseTensorMulVector(sptSparseTensor *Y, sptSparseTensor *X, const sptValueVector *V, sptIndex mode) 
{
    if(mode >= X->nmodes) {
        spt_CheckError(SPTERR_SHAPE_MISMATCH, "Cpu SpTns * Vec", "shape mismatch");
    }
    if(X->ndims[mode] != V->len) {
        spt_CheckError(SPTERR_SHAPE_MISMATCH, "Cpu SpTns * Vec", "shape mismatch");
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
    ind_buf = malloc(X->nmodes * sizeof *ind_buf);
    spt_CheckOSError(!ind_buf, "Cpu SpTns * Vec");
    for(sptIndex m = 0; m < X->nmodes; ++m) {
        if(m < mode)
            ind_buf[m] = X->ndims[m];
        else if(m > mode)
            ind_buf[m - 1] = X->ndims[m];
    }

    result = sptNewSparseTensor(Y, X->nmodes - 1, ind_buf);
    free(ind_buf);
    spt_CheckError(result, "Cpu SpTns * Vec", NULL);
    sptSparseTensorSetIndices(Y, &fiberidx, mode, X);
    sptStopTimer(timer);
    sptPrintElapsedTime(timer, "Allocate output tensor");


    sptStartTimer(timer);
    for(sptNnzIndex i = 0; i < Y->nnz; ++i) {
        sptNnzIndex inz_begin = fiberidx.data[i];
        sptNnzIndex inz_end = fiberidx.data[i+1];
        sptNnzIndex j;
        for(j = inz_begin; j < inz_end; ++j) {
            sptIndex r = X->inds[mode].data[j];
            Y->values.data[i] += X->values.data[j] * V->data[r];
        }
    }
    sptStopTimer(timer);
    sptPrintElapsedTime(timer, "Cpu SpTns * Vec");
    sptFreeTimer(timer);
    sptFreeNnzIndexVector(&fiberidx);

    return 0;
}
