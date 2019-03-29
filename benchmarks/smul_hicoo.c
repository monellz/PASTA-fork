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

static void print_usage(char ** argv) {
    printf("Usage: %s [options] \n\n", argv[0]);
    printf("Options: -X INPUT (.tns file)\n");
    printf("         -a INPUT (a scalar)\n");
    printf("         -Z OUTPUT (output file name)\n");
    printf("         -b BLOCKSIZE (bits), --blocksize=BLOCKSIZE (bits)\n");
    printf("         -d DEV_ID, --dev-id=DEV_ID (-2:sequential,default; -1:OpenMP parallel)\n");
    printf("         --help\n");
    printf("\n");
}

/**
 * Benchmark HiCOO tensor multiplication with a scalar. 
 */
int main(int argc, char *argv[]) 
{
    FILE *fX = NULL, *fZ = NULL;
    sptValue a;
    sptSparseTensor X, Z;
    sptSparseTensorHiCOO hiX, hiZ;
    sptElementIndex sb_bits = 7;
    int dev_id = -2;
    int niters = 5;
    int nthreads;
    int sort_impl = 1;  // 1: Morton order; 2: Rowblock sorting
    sptTimer timer;
    sptNewTimer(&timer, 0);

    if(argc < 3) {
        print_usage(argv);
        exit(1);
    }

    static struct option long_options[] = {
        {"Xinput", required_argument, 0, 'X'},
        {"ainput", required_argument, 0, 'a'},
        {"Zoutput", optional_argument, 0, 'Z'},
        {"bs", optional_argument, 0, 'b'},
        {"dev-id", optional_argument, 0, 'd'},
        {"help", no_argument, 0, 0},
        {0, 0, 0, 0}
    };
    int c;
    for(;;) {
        int option_index = 0;
        c = getopt_long(argc, argv, "X:a:Z:b:d:", long_options, &option_index);
        if(c == -1) {
            break;
        }
        switch(c) {
        case 'X':
            fX = fopen(optarg, "r");
            sptAssert(fX != NULL);
            printf("X input file: %s\n", optarg); fflush(stdout);
            break;
        case 'a':
            sscanf(optarg, "%"PASTA_SCN_VALUE, &a);
            break;
        case 'Z':
            fZ = fopen(optarg, "w");
            sptAssert(fZ != NULL);
            printf("Z output file: %s\n", optarg); fflush(stdout);
            break;
        case 'b':
            sscanf(optarg, "%"PASTA_SCN_ELEMENT_INDEX, &sb_bits);
            break;
        case 'd':
            sscanf(optarg, "%d", &dev_id);
            if(dev_id < -2 || dev_id >= 0) {
                fprintf(stderr, "Error: set dev_id to -2/-1.\n");
                exit(1);
            }
            break;
        case '?':   /* invalid option */
        case 'h':
        default:
            print_usage(argv);
            exit(1);
        }
    }
    printf("Scaling a: %"PASTA_PRI_VALUE"\n", a); 
    printf("Block size (bit-length): %"PASTA_PRI_ELEMENT_INDEX"\n", sb_bits);
    printf("dev_id: %d\n", dev_id);
    printf("Sorting implementation: %d\n", sort_impl); fflush(stdout);

    sptAssert(sptLoadSparseTensor(&X, 1, fX) == 0);
    fclose(fX);
    sptSparseTensorStatus(&X, stdout);
    // sptAssert(sptDumpSparseTensor(&X, 0, stdout) == 0);

    /* Convert to HiCOO tensor */
    sptStartTimer(timer);
    sptNnzIndex max_nnzb = 0;
    sptAssert(sptSparseTensorToHiCOO(&hiX, &max_nnzb, &X, sb_bits, sort_impl, 1) == 0);
    sptFreeSparseTensor(&X);
    // sptSparseTensorStatusHiCOO(&hiX, stdout);
    // sptAssert(sptDumpSparseTensorHiCOO(&hiX, stdout) == 0);
    sptStopTimer(timer);
    sptPrintElapsedTime(timer, "Convert COO -> HiCOO");

    /* For warm-up caches, timing not included */
    if(dev_id == -2) {
        sptAssert(sptSparseTensorMulScalarHiCOO(&hiZ, &hiX, a) == 0);
    } else if(dev_id == -1) {
#ifdef PASTA_USE_OPENMP
        #pragma omp parallel
        {
            nthreads = omp_get_num_threads();
        }
        printf("\nnthreads: %d\n", nthreads);
        sptAssert(sptOmpSparseTensorMulScalarHiCOO(&hiZ, &hiX, a) == 0);
#endif
    }

    sptStartTimer(timer);
    for(int it=0; it<niters; ++it) {
        sptFreeSparseTensorHiCOO(&hiZ);
        if(dev_id == -2) {
            sptAssert(sptSparseTensorMulScalarHiCOO(&hiZ, &hiX, a) == 0);
        } else if(dev_id == -1) {
#ifdef PASTA_USE_OPENMP
            #pragma omp parallel
            {
                nthreads = omp_get_num_threads();
            }
            printf("nthreads: %d\n", nthreads);
            sptAssert(sptOmpSparseTensorMulScalarHiCOO(&hiZ, &hiX, a) == 0);
#endif
        }
    }
    sptStopTimer(timer);
    sptPrintAverageElapsedTime(timer, niters, "Average HiCooMulScalar");


    if(fZ != NULL) {
        // sptDumpSparseTensorHiCOO(&hiZ, stdout);

        /* Convert HiCOO to COO tensor */
        sptStartTimer(timer);
        sptAssert(sptHiCOOToSparseTensor(&Z, &hiZ) == 0);
        sptFreeSparseTensorHiCOO(&hiZ);
        // sptSparseTensorStatus(&Z, stdout);
        // sptAssert(sptDumpSparseTensor(&Z, stdout) == 0);
        sptStopTimer(timer);
        sptPrintElapsedTime(timer, "Convert HiCOO -> COO");
        sptFreeTimer(timer);

        sptSparseTensorSortIndex(&Z, 0, Z.nnz, 1);
        sptAssert(sptDumpSparseTensor(&Z, 1, fZ) == 0);
        fclose(fZ);
        sptFreeSparseTensor(&Z);
    }

    sptFreeSparseTensorHiCOO(&hiX);

    return 0;
}
