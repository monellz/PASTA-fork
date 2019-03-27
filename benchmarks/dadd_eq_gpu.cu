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

static void print_usage(char ** argv) {
    printf("Usage: %s [options] \n\n", argv[0]);
    printf("Options: -X INPUT (.tns file)\n");
    printf("         -Y INPUT (.tns file)\n");
    printf("         -Z OUTPUT (output file name)\n");
    printf("         -d DEV_ID, --dev-id=DEV_ID (-2:sequential,default; -1:OpenMP parallel; >=0:GPU parallel)\n");
    printf("         -c collectZero (0:default; 1)\n");
    printf("         --help\n");
    printf("\n");
}

/**
 * Benchmark element-wise COO tensor addition. 
 * Require two tensors has the same number of dimensions, the same shape and the same nonzero distribution.
 */
int main(int argc, char *argv[]) {
    FILE *fX, *fY, *fZ;
    sptSparseTensor X, Y, Z;
    int dev_id = -2;
    int niters = 5;
    int collectZero = 0;
    int nthreads;

    if(argc < 3) {
        print_usage(argv);
        exit(1);
    }

    static struct option long_options[] = {
        {"Xinput", required_argument, 0, 'X'},
        {"Yinput", required_argument, 0, 'Y'},
        {"Zoutput", optional_argument, 0, 'Z'},
        {"dev-id", optional_argument, 0, 'd'},
        {"collectZero", optional_argument, 0, 'c'},
        {"help", no_argument, 0, 0},
        {0, 0, 0, 0}
    };
    int c;
    for(;;) {
        int option_index = 0;
        c = getopt_long(argc, argv, "X:Y:Z:d:c:", long_options, &option_index);
        if(c == -1) {
            break;
        }
        switch(c) {
        case 'X':
            fX = fopen(optarg, "r");
            sptAssert(fX != NULL);
            printf("X input file: %s\n", optarg); fflush(stdout);
            break;
        case 'Y':
            fY = fopen(optarg, "r");
            sptAssert(fY != NULL);
            printf("Y input file: %s\n", optarg); fflush(stdout);
            break;
        case 'Z':
            fZ = fopen(optarg, "w");
            sptAssert(fZ != NULL);
            printf("Z output file: %s\n", optarg); fflush(stdout);
            break;
        case 'c':
            sscanf(optarg, "%d", &collectZero);
            if(collectZero != 0 && collectZero != 1) {
                fprintf(stderr, "Error: set collectZero to 0/1.\n");
                exit(1);
            }
            break;
        case 'd':
            sscanf(optarg, "%d", &dev_id);
            if(dev_id < -2) {
                fprintf(stderr, "Error: set dev_id to -2/-1/>=0.\n");
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
    printf("dev_id: %d\n", dev_id);
    printf("collectZero: %d\n", collectZero); fflush(stdout);


    sptAssert(sptLoadSparseTensor(&X, 1, fX) == 0);
    fclose(fX);
    sptAssert(sptLoadSparseTensor(&Y, 1, fY) == 0);
    fclose(fY);

    sptTimer timer;
    sptNewTimer(&timer, 0);

    /* For warm-up caches, timing not included */
    if(dev_id == -2) {
        sptAssert(sptSparseTensorDotAddEq(&Z, &X, &Y, collectZero) == 0);
    } else if(dev_id == -1) {
#ifdef PARTI_USE_OPENMP
        #pragma omp parallel
        {
            nthreads = omp_get_num_threads();
        }
        printf("\nnthreads: %d\n", nthreads);
        sptAssert(sptOmpSparseTensorDotAddEq(&Z, &X, &Y, collectZero) == 0);
#endif
    } else {
        sptCudaSetDevice(dev_id);
        sptAssert(sptCudaSparseTensorDotAddEq(&Z, &X, &Y, collectZero) == 0);
    }

    sptStartTimer(timer);
    for(int it=0; it<niters; ++it) {
        sptFreeSparseTensor(&Z);
        if(dev_id == -2) {
            sptAssert(sptSparseTensorDotAddEq(&Z, &X, &Y, collectZero) == 0);
        } else if(dev_id == -1) {
#ifdef PARTI_USE_OPENMP
            sptAssert(sptOmpSparseTensorDotAddEq(&Z, &X, &Y, collectZero) == 0);
#endif
        } else {
            sptCudaSetDevice(dev_id);
            sptAssert(sptCudaSparseTensorDotAddEq(&Z, &X, &Y, collectZero) == 0);
        }
    }
    sptStopTimer(timer);
    sptPrintAverageElapsedTime(timer, niters, "Average CooDotAddEq");
    sptFreeTimer(timer);

    if(fZ != NULL) {
        sptAssert(sptDumpSparseTensor(&Z, 1, fZ) == 0);
        fclose(fZ);
    }

    sptFreeSparseTensor(&X);
    sptFreeSparseTensor(&Y);
    sptFreeSparseTensor(&Z);

    return 0;
}