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
#include "sptensor.h"

/**
 * Element wise multiply two sparse tensors
 * @param[out] Z the result of X.*Y, should be uninitialized
 * @param[in]  X the input X
 * @param[in]  Y the input Y
 */
int sptSparseTensorDotMul(sptSparseTensor *Z, const sptSparseTensor * X, const sptSparseTensor *Y, int collectZero) 
{
    /* Ensure X and Y are in same number of dimensions */
    if(Y->nmodes != X->nmodes) {
        spt_CheckError(SPTERR_SHAPE_MISMATCH, "Cpu SpTns DotMul", "shape mismatch");
    }
    sptIndex * max_ndims = (sptIndex*)malloc(X->nmodes * sizeof(sptIndex));
    for(sptIndex i = 0; i < X->nmodes; ++i) {
        if(Y->ndims[i] > X->ndims[i]) {
            max_ndims[i] = Y->ndims[i];
        } else {
            max_ndims[i] = X->ndims[i];
        }
    }

    sptTimer timer;
    sptNewTimer(&timer, 0);

    sptStartTimer(timer);
    sptNewSparseTensor(Z, X->nmodes, max_ndims);
    free(max_ndims);
    sptStopTimer(timer);
    sptPrintElapsedTime(timer, "sptNewSparseTensor");

    sptStartTimer(timer);
    /* Multiple elements one by one, assume indices are ordered */
    sptNnzIndex i, j;
    int result;
    i = 0;
    j = 0;
    while(i < X->nnz && j < Y->nnz) {
        int compare = spt_SparseTensorCompareIndices(X, i, Y, j);
        if(compare > 0) {  // X[i] > Y[j]
            ++j;
        } else if(compare < 0) {  // X[i] < Y[j]
            ++i;
        } else {  // X[i] == Y[j]
            for(sptIndex mode = 0; mode < X->nmodes; ++mode) {
                result = sptAppendIndexVector(&Z->inds[mode], X->inds[mode].data[i]);
                spt_CheckError(result, "Cpu SpTns DotMul", NULL);
            }
            result = sptAppendValueVector(&Z->values, X->values.data[i] * Y->values.data[j]);
            spt_CheckError(result, "Cpu SpTns DotMul", NULL);

            ++Z->nnz;
            ++i;
            ++j;
        }
    }
    sptStopTimer(timer);
    sptPrintElapsedTime(timer, "Cpu SpTns DotMul");

    /* Check whether elements become zero after adding.
       If so, fill the gap with the [nnz-1]'th element.
    */
    sptStartTimer(timer);
    if(collectZero == 1) {
        sptSparseTensorCollectZeros(Z);
    }
    sptStopTimer(timer);
    sptPrintElapsedTime(timer, "sptSparseTensorCollectZeros");

    return 0;
}
