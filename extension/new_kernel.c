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

#if 0
int newKernel(sptSparseTensor *Z, sptSparseTensor *X, sptValue a)
{
    sptAssert(a != 0.0);

    sptTimer timer;
    sptNewTimer(&timer, 0);
    double copy_time, comp_time, total_time;

    /* Allocate space */
    sptCopySparseTensorAllocateOnly(Z, X);

    /* Set values */
    sptStartTimer(timer);
    sptCopySparseTensorCopyOnly(Z, X);
    sptStopTimer(timer);
    copy_time = sptPrintElapsedTime(timer, "sptCopySparseTensor");

    sptStartTimer(timer);
    for(sptNnzIndex i = 0; i < Z->nnz; ++i) {
        Z->values.data[i] *= a;
    }
    sptStopTimer(timer);
    comp_time = sptPrintElapsedTime(timer, "Cpu SpTns MulScalar");
    sptFreeTimer(timer);

    total_time = copy_time + comp_time;
    printf("[Total time]: %lf\n", total_time);
    printf("\n");

    return 0;
}
#endif
