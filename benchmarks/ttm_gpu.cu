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
    FILE *fX, *fY;
    sptSparseTensor X, spY;
    sptSemiSparseTensor Y;
    sptMatrix U;
    sptIndex mode = 0;
    sptIndex R = 16;
    int cuda_dev_id = -2;
    int niters = 5;

    if(argc < 5) {
        printf("Usage: %s X mode impl_num smem_size [cuda_dev_id, R, Y]\n\n", argv[0]);
        return 1;
    }

    fX = fopen(argv[1], "r");
    sptAssert(fX != NULL);
    sptAssert(sptLoadSparseTensor(&X, 1, fX) == 0);
    fclose(fX);

    sscanf(argv[2], "%"PARTI_SCN_INDEX, &mode);
    sptIndex impl_num = 0;
    sscanf(argv[3], "%"PARTI_SCN_INDEX, &impl_num);
    sptNnzIndex smem_size = 0;
    sscanf(argv[4], "%"PARTI_SCN_NNZ_INDEX, &smem_size);

    if(argc > 5) {
        sscanf(argv[5], "%d", &cuda_dev_id);
    }
    if(argc > 6) {
        sscanf(argv[6], "%"PARTI_SCN_INDEX, &R);
    }

    fprintf(stderr, "sptRandomizeMatrix(&U, %"PARTI_PRI_INDEX ", %"PARTI_PRI_INDEX ")\n", X.ndims[mode], R);
    // sptAssert(sptRandomizeMatrix(&U, X.ndims[mode], R) == 0);
    sptAssert(sptNewMatrix(&U, X.ndims[mode], R) == 0);
    sptAssert(sptConstantMatrix(&U, 1) == 0);

    /* For warm-up caches, timing not included */
    if(cuda_dev_id == -2) {
        sptAssert(sptSparseTensorMulMatrix(&Y, &X, &U, mode) == 0);
    } else if(cuda_dev_id == -1) {
        sptAssert(sptOmpSparseTensorMulMatrix(&Y, &X, &U, mode) == 0);
    } else {
        sptCudaSetDevice(cuda_dev_id);
        sptAssert(sptCudaSparseTensorMulMatrix(&Y, &X, &U, mode, impl_num, smem_size) == 0);
    }

    for(int it=0; it<niters; ++it) {
        sptFreeSemiSparseTensor(&Y);
        if(cuda_dev_id == -2) {
            sptAssert(sptSparseTensorMulMatrix(&Y, &X, &U, mode) == 0);
        } else if(cuda_dev_id == -1) {
            sptAssert(sptOmpSparseTensorMulMatrix(&Y, &X, &U, mode) == 0);
        } else {
            sptCudaSetDevice(cuda_dev_id);
            sptAssert(sptCudaSparseTensorMulMatrix(&Y, &X, &U, mode, impl_num, smem_size) == 0);
        }
    }


    if(argc > 7) {
        sptAssert(sptSemiSparseTensorToSparseTensor(&spY, &Y, 1e-9) == 0);

        fY = fopen(argv[7], "w");
        sptAssert(fY != NULL);
        sptAssert(sptDumpSparseTensor(&spY, 0, fY) == 0);
        fclose(fY);

        sptFreeSparseTensor(&spY);
    }

    sptFreeSemiSparseTensor(&Y);
    sptFreeMatrix(&U);
    sptFreeSparseTensor(&X);

    return 0;
}
