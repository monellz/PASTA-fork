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
#include <getopt.h>
#include <ParTI.h>
#include "../src/sptensor/sptensor.h"

static void print_usage(char ** argv) {
    printf("Usage: %s [options] \n\n", argv[0]);
    printf("Options: -i INPUT, --input=INPUT (.tns file)\n");
    printf("         -o OUTPUT, --output=OUTPUT (output file name)\n");
    printf("         -m MODE, --mode=MODE (Specify a mode, e.g., 0 (default) or 1 or 2 for third-order tensors.)\n");
    printf("         -s sortcase, --sortcase=SORTCASE (0:default,1,2,3,4. Different tensor sorting.)\n");
    printf("         -b BLOCKSIZE, --blocksize=BLOCKSIZE (in bits) (only for sortcase=3)\n");
    printf("         -k KERNELSIZE, --kernelsize=KERNELSIZE (in bits) (only for sortcase=3)\n");
    printf("         --help\n");
    printf("\n");
}


/**
 * Benchmark COO tensor sortings in different mode orders, randomized sorting, and Z-Morton sorting.
 */
int main(int argc, char ** argv) {
    FILE *fi = NULL, *fo = NULL;
    sptSparseTensor X;

    sptIndex mode = 0;
    int dev_id = -2;
    int nthreads = 1;
    /* sortcase:
     * = 0 : the same with the old COO code.
     * = 1 : best case. Sort order: [mode, (ordered by increasing dimension sizes)]
     * = 2 : worse case. Sort order: [(ordered by decreasing dimension sizes)]
     * = 3 : Z-Morton ordering (same with HiCOO format order)
     * = 4 : random shuffling.
     */
    int sortcase = 0;
    sptElementIndex sb_bits = 7;
    sptElementIndex sk_bits = 20;

    if(argc <= 3) { // #Required arguments
        print_usage(argv);
        exit(1);
    }

    static struct option long_options[] = {
        {"input", required_argument, 0, 'i'},
        {"output", optional_argument, 0, 'o'},
        {"mode", required_argument, 0, 'm'},
        {"bs", required_argument, 0, 'b'},
        {"ks", required_argument, 0, 'k'},
        {"sortcase", optional_argument, 0, 's'},
        {"dev-id", optional_argument, 0, 'd'},
        {"help", no_argument, 0, 0},
        {0, 0, 0, 0}
    };
    int c;
    for(;;) {
        int option_index = 0;
        c = getopt_long(argc, argv, "i:o:m:b:k:s:d:", long_options, &option_index);
        if(c == -1) {
            break;
        }
        switch(c) {
        case 'i':
            fi = fopen(optarg, "r");
            sptAssert(fi != NULL);
            printf("input file: %s\n", optarg); fflush(stdout);
            break;
        case 'o':
            fo = fopen(optarg, "w");
            sptAssert(fo != NULL);
            printf("output file: %s\n", optarg); fflush(stdout);
            break;
        case 'm':
            sscanf(optarg, "%"PARTI_SCN_INDEX, &mode);
            break;
        case 'b':
            sscanf(optarg, "%"PARTI_SCN_ELEMENT_INDEX, &sb_bits);
            break;
        case 'k':
            sscanf(optarg, "%"PARTI_SCN_ELEMENT_INDEX, &sk_bits);
            break;
        case 's':
            sscanf(optarg, "%d", &sortcase);
            break;
        case 'd':
            sscanf(optarg, "%d", &dev_id);
            break;
        case '?':   /* invalid option */
        case 'h':
        default:
            print_usage(argv);
            exit(1);
        }
    }
    printf("mode: %"PARTI_PRI_INDEX "\n", mode);
    printf("dev_id: %d\n", dev_id);
    printf("sortcase: %d\n", sortcase);
    if(sortcase == 3) {
        printf("sb_bits: %"PARTI_PRI_ELEMENT_INDEX", sk_bits: %"PARTI_PRI_ELEMENT_INDEX"\n", sb_bits, sk_bits);
    }

    /* Load a sparse tensor from file as it is */
    sptAssert(sptLoadSparseTensor(&X, 1, fi) == 0);
    fclose(fi);
    sptSparseTensorStatus(&X, stdout);

    sptIndex nmodes = X.nmodes;
    sptIndex * mode_order = (sptIndex*) malloc(X.nmodes * sizeof(*mode_order));
    memset(mode_order, 0, X.nmodes * sizeof(*mode_order));

    /* Sort sparse tensor */
    sptTimer timer;
    sptNewTimer(&timer, 0);
    sptStartTimer(timer);
    switch (sortcase) {
        case 0:
            sptSparseTensorSortIndex(&X, 1);
            break;
        case 1:
            sptGetBestModeOrder(mode_order, mode, X.ndims, X.nmodes);
            sptSparseTensorSortIndexCustomOrder(&X, mode_order, 1);
            break;
        case 2:
            sptGetWorstModeOrder(mode_order, mode, X.ndims, X.nmodes);
            sptSparseTensorSortIndexCustomOrder(&X, mode_order, 1);
            break;
        case 3:
            /* Pre-process tensor, the same with the one used in HiCOO.
             * Only difference is not setting kptr and kschr in this function.
             */
            sptSparseTensorMixedOrder(&X, sb_bits, sk_bits);
            break;
        case 4:
            // sptGetBestModeOrder(mode_order, 0, X.ndims, X.nmodes);
            sptGetRandomShuffleElements(&X);
            break;
        default:
            printf("Error: Wrong sortcase number, reset by -s. \n");
    }
    if(sortcase == 1 || sortcase == 2) {
        printf("mode_order:\n");
        sptDumpIndexArray(mode_order, X.nmodes, stdout);
    }
    sptStopTimer(timer);
    sptPrintElapsedTime(timer, "Sort");
    sptFreeTimer(timer);


    if(fo != NULL) {
        sptAssert(sptDumpSparseTensor(&X, 1, fo) == 0);
        fclose(fo);
    }

    sptFreeSparseTensor(&X);
    free(mode_order);
 
    return 0;
}