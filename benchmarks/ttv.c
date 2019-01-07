/*
    Internal code for ParTI!
    (c) Sam Bliss, 2018, all rights reserved.
*/

#include <stdio.h>
#include <stdlib.h>
#include <getopt.h>
#include <ParTI.h>
#include "../src/error/error.h"


static void print_usage(char ** argv) {
    printf("Usage: %s [options] \n\n", argv[0]);
    printf("Options: -i INPUT, --input=INPUT (.tns file)\n");
    printf("         -o OUTPUT, --output=OUTPUT (output file name)\n");
    printf("         -m MODE, --mode=MODE (specify a mode, e.g., 0 (default) or 1 or 2 for third-order tensors.)\n");
    printf("         -d DEV_ID, --dev-id=DEV_ID (-2:sequential,default; -1:OpenMP parallel)\n");
    printf("         OpenMP options: \n");
    printf("         -t NTHREADS, --nt=NT (1:default)\n");
    printf("         --help\n");
    printf("\n");
}


/**
 * Benchmark COO tensor times a dense matrix.
 */
int main(int argc, char ** argv) 
{
    FILE *fi = NULL, *fo = NULL;
    sptSparseTensor X;
    sptSparseTensor Y;
    sptValueVector V;
    sptIndex mode = 0;
    int dev_id = -2;
    int niters = 5;
    int nthreads = 1;
    printf("niters: %d\n", niters);

    if(argc < 3) {
        print_usage(argv);
        return 0;
    }

    static struct option long_options[] = {
        {"input", required_argument, 0, 'i'},
        {"mode", optional_argument, 0, 'm'},
        {"output", optional_argument, 0, 'o'},
        {"dev-id", optional_argument, 0, 'd'},
        {"nt", optional_argument, 0, 't'},
        {"help", no_argument, 0, 0},
        {0, 0, 0, 0}
    };
    int c;
    for(;;) {
        int option_index = 0;
        c = getopt_long(argc, argv, "i:m:o:d:t:", long_options, &option_index);
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
        case 'd':
            sscanf(optarg, "%d", &dev_id);
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

    printf("mode: %"PARTI_PRI_INDEX "\n", mode);
    printf("dev_id: %d\n", dev_id);

    sptAssert(sptLoadSparseTensor(&X, 1, fi) == 0);
    fclose(fi);

    sptAssert(sptNewValueVector(&V, X.ndims[mode], X.ndims[mode]) == 0);
    sptAssert(sptRandomizeValueVector(&V) == 0);

    /* For warm-up caches, timing not included */
    int result;
    if(dev_id == -2) {
        sptAssert(sptSparseTensorMulVector(&Y, &X, &V, mode) == 0);
    } else if(dev_id == -1) {
#ifdef PARTI_USE_OPENMP
        #pragma omp parallel
        {
            nthreads = omp_get_num_threads();
        }
        printf("\nnthreads: %d\n", nthreads);
        sptAssert(sptOmpSparseTensorMulVector(&Y, &X, &V, mode) == 0);
#endif
    }  

    sptTimer timer;
    sptNewTimer(&timer, 0);
    sptStartTimer(timer);

    for(int i = 0; i < niters; i++) {
        if(dev_id == -2) {
            sptAssert(sptSparseTensorMulVector(&Y, &X, &V, mode) == 0);
        } else if(dev_id == -1) {
    #ifdef PARTI_USE_OPENMP
            sptAssert(sptOmpSparseTensorMulVector(&Y, &X, &V, mode) == 0);
    #endif
        }
    }

    sptStopTimer(timer);
    sptPrintAverageElapsedTime(timer, niters, "Average CooTtv");
    sptFreeTimer(timer);

    if(fo != NULL) {
        sptAssert(sptDumpSparseTensor(&Y, 1, fo) == 0);      
    }

    sptFreeSparseTensor(&Y);
    sptFreeSparseTensor(&X);
    sptFreeValueVector(&V);

    return 0;
}
