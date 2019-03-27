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
#include <stdlib.h>

/**
 * Sparse tensor times a vector (SpTTV) in HiCOO format
 */
int sptSparseTensorMulVectorHiCOO(sptSparseTensorHiCOO *hiY, sptSparseTensorHiCOOGeneral *hiX, const sptValueVector *V, sptIndex mode) 
{
    if(mode >= hiX->nmodes) {
        spt_CheckError(SPTERR_SHAPE_MISMATCH, "CPU  HiSpTns * Vec", "shape mismatch");
    }
    if(hiX->ndims[mode] != V->len) {
        spt_CheckError(SPTERR_SHAPE_MISMATCH, "CPU  HiSpTns * Vec", "shape mismatch");
    }

    int result;
    sptIndex *ind_buf;
    sptNnzIndexVector fiberidx;
    sptTimer timer;
    sptNewTimer(&timer, 0);

    sptStartTimer(timer);
    ind_buf = malloc(hiX->nmodes * sizeof *ind_buf);
    spt_CheckOSError(!ind_buf, "CPU  HiSpTns * Vec");
    for(sptIndex m = 0; m < hiX->nmodes; ++m) {
        if(m < mode)
            ind_buf[m] = hiX->ndims[m];
        else if(m > mode)
            ind_buf[m - 1] = hiX->ndims[m];
    }

    result = sptNewSparseTensorHiCOO(hiY, hiX->nmodes - 1, ind_buf, 0, hiX->sb_bits);
    free(ind_buf);
    spt_CheckError(result, "CPU  HiSpTns * Vec", NULL);
    sptSparseTensorSetIndicesHiCOO(hiY, &fiberidx, hiX);
    sptStopTimer(timer);
    sptPrintElapsedTime(timer, "Allocate output tensor");


    sptStartTimer(timer);
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
    sptPrintElapsedTime(timer, "Cpu  HiSpTns * Vec");
    sptFreeTimer(timer);
    sptFreeNnzIndexVector(&fiberidx);

    return 0;
}
