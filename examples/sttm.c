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

#include <stdio.h>
#include <stdlib.h>
#include <ParTI.h>

int main(int argc, char const *argv[]) {
    FILE *fX, *fU, *fY;
    sptSparseTensor spX, spU, spY;
    sptSemiSparseTensor X, Y;
    sptMatrix U;
    sptIndex mode = 0;
    int cuda_dev_id = -1;

    if(argc < 5) {
        printf("Usage: %s X U Y mode [cuda_dev_id]\n\n", argv[0]);
        return 1;
    }

    fX = fopen(argv[1], "r");
    sptAssert(fX != NULL);
    sptAssert(sptLoadSparseTensor(&spX, 1, fX) == 0);
    fclose(fX);

    fU = fopen(argv[2], "r");
    sptAssert(fU != NULL);
    sptAssert(sptLoadSparseTensor(&spU, 1, fU) == 0);
    fclose(fU);

    sscanf(argv[4], "%"PARTI_SCN_INDEX, &mode);
    if(argc >= 6) {
        sscanf(argv[5], "%d", &cuda_dev_id);
    }

    sptAssert(sptSparseTensorToSemiSparseTensor(&X, &spX, mode) == 0);
    sptFreeSparseTensor(&spX);
    sptAssert(sptSparseTensorToMatrix(&U, &spU) == 0);
    sptFreeSparseTensor(&spU);

    if(cuda_dev_id == -1) {
        sptAssert(sptSemiSparseTensorMulMatrix(&Y, &X, &U, mode) == 0);
    } else {
        fprintf(stderr, "This build does not support GPU. Refer to sttm_gpu for GPU-enabled build.\n");
        abort();
        //sptCudaSetDevice(cuda_dev_id);
        //sptAssert(sptCudaSemiSparseTensorMulMatrix(&Y, &X, &U, mode) == 0);
    }

    sptAssert(sptSemiSparseTensorToSparseTensor(&spY, &Y, 1e-9) == 0);

    sptFreeSemiSparseTensor(&Y);
    sptFreeMatrix(&U);
    sptFreeSemiSparseTensor(&X);

    fY = fopen(argv[3], "w");
    sptAssert(fY != NULL);
    sptAssert(sptDumpSparseTensor(&spY, 1, fY) == 0);
    fclose(fY);

    sptFreeSparseTensor(&spY);

    return 0;
}
