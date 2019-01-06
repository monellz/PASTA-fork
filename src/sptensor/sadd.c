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

/**
 * Multiply a sparse tensors with a scalar.
 * @param[out] Z the result of a*X, should be uninitialized
 * @param[in]  a the input scalar
 * @param[in]  X the input X
 */
int sptSparseTensorAddScalar(sptSparseTensor *Z, sptSparseTensor *X, sptValue a)
{
    sptAssert(a != 0.0);

    sptTimer timer;
    sptNewTimer(&timer, 0);

    sptStartTimer(timer);
    sptCopySparseTensor(Z, X, 1);
    sptStopTimer(timer);
    sptPrintElapsedTime(timer, "sptCopySparseTensor");

    sptStartTimer(timer);
#ifdef PARTI_USE_OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for(sptNnzIndex i = 0; i < Z->nnz; ++i) {
        Z->values.data[i] += a;
    }
    sptStopTimer(timer);
#ifdef PARTI_USE_OPENMP
    sptPrintElapsedTime(timer, "Omp SpTns MulScalar");
#else
    sptPrintElapsedTime(timer, "Cpu SpTns MulScalar");
#endif
    sptFreeTimer(timer);
    printf("\n");

    return 0;
}
