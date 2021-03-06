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
#include <pasta.h>

/**
 * Benchmark obtaining a COO tensor nonzero entry.
 */
int main(int argc, char *argv[]) {
    FILE *fo;
    sptSparseTensor tsr;

    if(argc != 3) {
        printf("Usage: %s input location\n\n", argv[0]);
        return 1;
    }

    sptAssert(sptLoadSparseTensor(&tsr, 1, argv[1]) == 0);
    sptSparseTensorStatus(&tsr, stdout);


    sptNnzIndex loc = atol(argv[2]);
    sptIndex const nmodes = tsr.nmodes;

    printf("Entry %"PASTA_PRI_NNZ_INDEX": ( ", loc);
    for(sptIndex m = 0; m < nmodes - 1; ++m) {
        printf("%" PASTA_PRI_INDEX ", ", tsr.inds[m].data[loc]);
    }
    printf("%" PASTA_PRI_INDEX " ) ", tsr.inds[nmodes-1].data[loc]);
    printf("%" PASTA_PRI_VALUE "\n", tsr.values.data[loc]);

    sptFreeSparseTensor(&tsr);

    return 0;
}
