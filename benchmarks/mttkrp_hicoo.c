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
#include <pasta.h>
#include "../src/sptensor/sptensor.h"

void print_usage(char ** argv) {
    printf("Usage: %s [options] \n", argv[0]);
    printf("Options: -i INPUT, --input=INPUT\n");
    printf("         -o OUTPUT, --output=OUTPUT\n");
    printf("         -b BLOCKSIZE (bits), --blocksize=BLOCKSIZE (bits)\n");
    printf("         -m MODE, --mode=MODE\n");
    printf("         -d DEV_ID, --dev-id=DEV_ID\n");
    printf("         -r RANK\n");
    printf("         OpenMP options: \n");
    printf("         -t NTHREADS, --nthreads=NT (1:default)\n");
    printf("\n");
}

/**
 * Benchmark Matriced Tensor Times Khatri-Rao Product (MTTKRP), tensor in HiCOO format, matrices are dense.
 */
int main(int argc, char ** argv) 
{
    FILE *fo = NULL;
    char fname[1000];
    sptSparseTensor tsr;
    sptMatrix ** U;
    sptSparseTensorHiCOO hitsr;
    sptElementIndex sb_bits = 7;

    sptIndex mode = 0;
    sptIndex R = 16;
    int dev_id = -2;
    int niters = 5;
    int nthreads = 1;
    int sort_impl = 1;  // 1: Morton order; 2: Rowblock sorting
    printf("niters: %d\n", niters);
    int retval;

    if(argc <= 3) { // #Required arguments
        print_usage(argv);
        exit(1);
    }

    for(;;) {
        static struct option long_options[] = {
            {"input", required_argument, 0, 'i'},
            {"output", required_argument, 0, 'o'},
            {"bs", optional_argument, 0, 'b'},
            {"mode", optional_argument, 0, 'm'},
            {"dev-id", optional_argument, 0, 'd'},
            {"rank", optional_argument, 0, 'r'},
            {"nthreads", optional_argument, 0, 't'},
            {0, 0, 0, 0}
        };
        int option_index = 0;
        int c = 0;
        // c = getopt_long(argc, argv, "i:o:b:k:c:m:", long_options, &option_index);
        c = getopt_long(argc, argv, "i:o:b:m:d:r:t:", long_options, &option_index);
        if(c == -1) {
            break;
        }
        switch(c) {
        case 'i':
            strcpy(fname, optarg);
            printf("input file: %s\n", fname); fflush(stdout);
            break;
        case 'o':
            fo = fopen(optarg, "w");
            sptAssert(fo != NULL);
            printf("output file: %s\n", optarg); fflush(stdout);
            break;
        case 'b':
            sscanf(optarg, "%"PASTA_SCN_ELEMENT_INDEX, &sb_bits);
            break;
        case 'm':
            sscanf(optarg, "%"PASTA_SCN_INDEX, &mode);
            break;
        case 'd':
            sscanf(optarg, "%d", &dev_id);
            if(dev_id < -2 || dev_id >= 0) {
                fprintf(stderr, "Error: set dev_id to -2/-1.\n");
                exit(1);
            }
            break;
        case 'r':
            sscanf(optarg, "%"PASTA_SCN_INDEX, &R);
            break;
        case 't':
            sscanf(optarg, "%d", &nthreads);
            break;
        case '?':   /* invalid option */
        case 'h':
        default:
            print_usage(argv);
            exit(1);
        }
    }
    printf("mode: %"PASTA_PRI_INDEX "\n", mode);
    printf("dev_id: %d\n", dev_id);
    printf("Sorting implementation: %d\n", sort_impl);

    sptAssert(sptLoadSparseTensor(&tsr, 1, fname) == 0);
    // sptSparseTensorSortIndex(&tsr, 1);
    sptSparseTensorStatus(&tsr, stdout);
    // sptAssert(sptDumpSparseTensor(&tsr, 0, stdout) == 0);

    sptTimer convert_timer;
    sptNewTimer(&convert_timer, 0);
    sptStartTimer(convert_timer);

    /* Convert to HiCOO tensor */
    sptNnzIndex max_nnzb = 0;
    sptAssert(sptSparseTensorToHiCOO(&hitsr, &max_nnzb, &tsr, sb_bits, sort_impl, nthreads) == 0);
    sptFreeSparseTensor(&tsr);
    sptSparseTensorStatusHiCOO(&hitsr, stdout);
    // sptAssert(sptDumpSparseTensorHiCOO(&hitsr, stdout) == 0);

    sptStopTimer(convert_timer);
    sptPrintElapsedTime(convert_timer, "Convert HiCOO");
    sptFreeTimer(convert_timer);

    sptIndex nmodes = hitsr.nmodes;
    U = (sptMatrix **)malloc((nmodes+1) * sizeof(sptMatrix*));
    for(sptIndex m=0; m<nmodes+1; ++m) {
      U[m] = (sptMatrix *)malloc(sizeof(sptMatrix));
    }
    sptIndex max_ndims = 0;
    for(sptIndex m=0; m<nmodes; ++m) {
      sptAssert(sptNewMatrix(U[m], hitsr.ndims[m], R) == 0);
      // sptAssert(sptConstantMatrix(U[m], 1) == 0);
      sptAssert(sptRandomizeMatrix(U[m]) == 0);
      if(hitsr.ndims[m] > max_ndims)
        max_ndims = hitsr.ndims[m];
    }
    sptAssert(sptNewMatrix(U[nmodes], max_ndims, R) == 0);
    sptAssert(sptConstantMatrix(U[nmodes], 0) == 0);

    sptIndex * mats_order = (sptIndex*)malloc(nmodes * sizeof(*mats_order));
    mats_order[0] = mode;
    for(sptIndex i=1; i<nmodes; ++i)
        mats_order[i] = (mode+i) % nmodes;

    /* For warm-up caches, timing not included */
    if(dev_id == -2) {
        nthreads = 1;
        sptAssert(sptMTTKRPHiCOO(&hitsr, U, mats_order, mode) == 0);
    } else if(dev_id == -1) {
#ifdef PASTA_USE_OPENMP
        #pragma omp parallel
        {
            nthreads = omp_get_num_threads();
        }
        printf("nthreads: %d\n", nthreads);
        sptAssert(sptOmpMTTKRPHiCOO(&hitsr, U, mats_order, mode, nthreads) == 0);
#endif
    }

    sptTimer timer;
    sptNewTimer(&timer, 0);
    sptStartTimer(timer);

    for(int it=0; it<niters; ++it) {
        if(dev_id == -2) {
            sptAssert(sptMTTKRPHiCOO(&hitsr, U, mats_order, mode) == 0);
        } else if(dev_id == -1) {
#ifdef PASTA_USE_OPENMP
            sptAssert(sptOmpMTTKRPHiCOO(&hitsr, U, mats_order, mode, nthreads) == 0);
#endif
        }
    }

    sptStopTimer(timer);
    sptPrintAverageElapsedTime(timer, niters, "Average HiCooMTTKRP");
    sptFreeTimer(timer);

    if(fo != NULL) {
        sptAssert(sptDumpMatrix(U[nmodes], fo) == 0);
        fclose(fo);
    }


    for(sptIndex m=0; m<nmodes; ++m) {
        sptFreeMatrix(U[m]);
    }
    free(mats_order);
    sptFreeMatrix(U[nmodes]);
    free(U);
    sptFreeSparseTensorHiCOO(&hitsr);

    return 0;
}
