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
#include "matrix.h"
#include "mex.h"
#include "sptmx.h"

spt_DefineCastArray(spt_mxArrayToSize, size_t)

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    spt_mxCheckArgs("sptOmpMTTKRP", 0, "No", 5, "Five");

    sptSparseTensor *X = spt_mxGetPointer(prhs[0], 0);
    size_t nmodes = X->nmodes;
    size_t m;
    sptMatrix **mats = malloc((nmodes+1) * sizeof *mats);
    for(m = 0; m < nmodes+1; ++m) {
        mats[m] = spt_mxGetPointer(prhs[1], m);
    }
    size_t *mats_order = spt_mxArrayToSize(prhs[2]);
    size_t mode = mxGetScalar(prhs[3])-1;
    sptVector *scratch = spt_mxGetPointer(prhs[4], 0);

    for(m = 0; m < mats_order->len; ++m) {
        --mats_order[m];
    }
    sptOmpMTTKRP(X, mats, mats_order, mode, scratch);

    free(mats_order);
    free(mats);
}
