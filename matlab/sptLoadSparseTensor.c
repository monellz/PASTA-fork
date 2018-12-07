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
#include <stdio.h>
#include <stdlib.h>
#include "matrix.h"
#include "mex.h"
#include "sptmx.h"

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    spt_mxCheckArgs("sptLoadSparseTensor", 1, "One", 2, "Two");

    size_t start_idx = mxGetScalar(prhs[0]);
    char *fn = mxArrayToString(prhs[1]);
    FILE *fp = fopen(fn, "r");
    mxFree(fn);
    if(!fp) {
        mexErrMsgIdAndTxt("ParTI:sptLoadSparseTensor", "Cannot open file.");
    }

    sptSparseTensor *tsr = malloc(sizeof *tsr);
    int result = sptLoadSparseTensor(tsr, start_idx, fp);
    fclose(fp);
    if(result) {
        free(tsr);
        tsr = NULL;
    }

    mexCallMATLAB(nlhs, plhs, 0, NULL, "sptSparseTensor");
    spt_mxSetPointer(plhs[0], 0, tsr);
}
