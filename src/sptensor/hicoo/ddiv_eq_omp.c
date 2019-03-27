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

/**
 * Element wise divide two sparse tensors, with exactly the same nonzero
 * distribution.
 * @param[out] Z the result of X.+Y, should be uninitialized
 * @param[in]  X the input X
 * @param[in]  Y the input Y
 */
int sptOmpSparseTensorDotDivEqHiCOO(sptSparseTensorHiCOO *hiZ, const sptSparseTensorHiCOO *hiX, const sptSparseTensorHiCOO *hiY, int collectZero)
{
    sptAssert(collectZero == 0);
    /* Ensure X and Y are in same shape */
    if(hiY->nmodes != hiX->nmodes) {
        spt_CheckError(SPTERR_SHAPE_MISMATCH, "SpTns DotDivHiCOO", "shape mismatch");
    }
    for(sptIndex i = 0; i < hiX->nmodes; ++i) {
        if(hiY->ndims[i] != hiX->ndims[i]) {
            spt_CheckError(SPTERR_SHAPE_MISMATCH, "SpTns DotDivHiCOO", "shape mismatch");
        }
    }
    /* Ensure X and Y have exactly the same nonzero distribution */
    if(hiY->nnz != hiX->nnz) {
        spt_CheckError(SPTERR_SHAPE_MISMATCH, "SpTns DotDivHiCOO", "nonzero distribution mismatch");
    }
    sptNnzIndex nnz = hiX->nnz;

    sptTimer timer;
    sptNewTimer(&timer, 0);

    sptStartTimer(timer);
    sptCopySparseTensorHiCOO(hiZ, hiX);
    sptStopTimer(timer);
    sptPrintElapsedTime(timer, "sptCopySparseTensorHiCOO");

    sptStartTimer(timer);
    #pragma omp parallel for schedule(static)
    for(sptNnzIndex i=0; i< nnz; ++i)
        hiZ->values.data[i] = hiX->values.data[i] / hiY->values.data[i];
    sptStopTimer(timer);
    sptPrintElapsedTime(timer, "Cpu SpTns DotDivHiCOO");

    /* TODO: Check whether elements become zero after adding.
       If so, fill the gap with the [nnz-1]'th element.
    */
    
    return 0;
}
