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
#include "../sptmx.h"

spt_DefineSetScalar(spt_mxSetSize, size_t)

void mexFunction1(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    spt_mxCheckArgs("sptSizeVector:data", 1, "One", 1, "One");

    sptSizeVector *vec = spt_mxGetPointer(prhs[0], 0);

    mxDestroyArray(plhs[0]);
    plhs[0] = mxCreateNumericMatrix(1, vec->len, mxUINT64_CLASS, mxREAL);
    size_t i;
    for(i = 0; i < vec->len; ++i) {
        spt_mxSetSize(plhs[0], i, vec->data[i]);
    }
}

void mexFunction2(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    spt_mxCheckArgs("sptSizeVector:data", 1, "One", 2, "Two");

    sptSizeVector *vec = spt_mxGetPointer(prhs[0], 0);
    size_t i = mxGetScalar(prhs[1])-1;

    mxDestroyArray(plhs[0]);
    plhs[0] = mxCreateNumericMatrix(1, 1, mxUINT64_CLASS, mxREAL);
    spt_mxSetSize(plhs[0], 0, vec->data[i]);
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    if(nrhs == 2) {
        mexFunction2(nlhs, plhs, nrhs, prhs);
    } else {
        mexFunction1(nlhs, plhs, nrhs, prhs);
    }
}
