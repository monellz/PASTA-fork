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
 * Multiply a HiCOO sparse tensor with a scalar.
 * @param[out] Z the result of a+X in HiCOO, should be uninitialized
 * @param[in]  a the input scalar
 * @param[in]  X the input X in HiCOO
 */
int sptOmpSparseTensorAddScalarHiCOO(sptSparseTensorHiCOO *hiZ, sptSparseTensorHiCOO *hiX, sptValue a)
{
    sptAssert(a != 0.0);

    sptTimer timer;
    sptNewTimer(&timer, 0);

    sptStartTimer(timer);
    sptCopySparseTensorHiCOO(hiZ, hiX);
    // sptSparseTensorStatusHiCOO(hiZ, stdout);
    sptStopTimer(timer);
    sptPrintElapsedTime(timer, "sptCopySparseTensorHiCOO");

    sptStartTimer(timer);
    #pragma omp parallel for schedule(static)
    for(sptNnzIndex i = 0; i < hiZ->nnz; ++i) {
        hiZ->values.data[i] += a;
    }
    sptStopTimer(timer);
    sptPrintElapsedTime(timer, "Omp HiSpTns AddScalar");
    sptFreeTimer(timer);
    printf("\n");

    return 0;
}
