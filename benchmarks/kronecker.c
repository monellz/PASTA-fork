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
#include <pasta.h>

int main(int argc, char *argv[]) {
    FILE *fo;
    sptSparseTensor a, b, out;

    if(argc != 4) {
        printf("Usage: %s a b out\n\n", argv[0]);
        return 1;
    }

    sptAssert(sptLoadSparseTensor(&a, 1, argv[1]) == 0);
    sptAssert(sptLoadSparseTensor(&b, 1, argv[2]) == 0);

    sptAssert(sptSparseTensorKroneckerMul(&out, &a, &b) == 0);

    fo = fopen(argv[3], "w");
    sptAssert(fo != NULL);
    sptAssert(sptDumpSparseTensor(&out, 1, fo) == 0);
    fclose(fo);

    return 0;
}
