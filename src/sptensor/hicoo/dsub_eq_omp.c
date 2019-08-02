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
 * Element wise subtract two sparse tensors, with exactly the same nonzero
 * distribution.
 * @param[out] Z the result of X.+Y, should be uninitialized
 * @param[in]  X the input X
 * @param[in]  Y the input Y
 */
int sptOmpSparseTensorDotSubEqHiCOO(sptSparseTensorHiCOO *hiZ, const sptSparseTensorHiCOO *hiX, const sptSparseTensorHiCOO *hiY, int collectZero)
{
    sptAssert(collectZero == 0);
    /* Ensure X and Y are in same shape */
    if(hiY->nmodes != hiX->nmodes) {
        spt_CheckError(SPTERR_SHAPE_MISMATCH, "Omp HiSpTns DotSub", "shape mismatch");
    }
    for(sptIndex i = 0; i < hiX->nmodes; ++i) {
        if(hiY->ndims[i] != hiX->ndims[i]) {
            spt_CheckError(SPTERR_SHAPE_MISMATCH, "Omp HiSpTns DotSub", "shape mismatch");
        }
    }
    /* Ensure X and Y have exactly the same nonzero distribution */
    if(hiY->nnz != hiX->nnz) {
        spt_CheckError(SPTERR_SHAPE_MISMATCH, "Omp HiSpTns DotSub", "nonzero distribution mismatch");
    }
    sptNnzIndex nnz = hiX->nnz;

    sptTimer timer;
    sptNewTimer(&timer, 0);
    double copy_time, comp_time, total_time;

    /* Allocate space */
    sptCopySparseTensorHiCOOAllocateOnly(hiZ, hiX);

    /* Set values */
    sptStartTimer(timer);
    sptCopySparseTensorHiCOOCopyOnly(hiZ, hiX);
    sptStopTimer(timer);
    copy_time = sptPrintElapsedTime(timer, "sptCopySparseTensorHiCOO");

    /* Computation */
    sptStartTimer(timer);
    #pragma omp parallel for schedule(static)
    for(sptNnzIndex i=0; i< nnz; ++i)
        hiZ->values.data[i] = hiX->values.data[i] - hiY->values.data[i];
    sptStopTimer(timer);
    comp_time = sptPrintElapsedTime(timer, "Omp HiSpTns DotSub");

    /* TODO: Check whether elements become zero after adding.
       If so, fill the gap with the [nnz-1]'th element.
    */
    
    total_time = copy_time + comp_time;
    printf("[Total time]: %lf\n", total_time);
    printf("\n");
    
    return 0;
}
