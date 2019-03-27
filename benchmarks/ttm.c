/*
    Internal code for ParTI!
    (c) Sam Bliss, 2018, all rights reserved.
*/

#include <stdio.h>
#include <stdlib.h>
#include <getopt.h>
#include <ParTI.h>


static void print_usage(char ** argv) {
    printf("Usage: %s [options] \n\n", argv[0]);
    printf("Options: -i INPUT, --input=INPUT (.tns file)\n");
    printf("         -o OUTPUT, --output=OUTPUT (output file name)\n");
    printf("         -m MODE, --mode=MODE (specify a mode, e.g., 0 (default) or 1 or 2 for third-order tensors.)\n");
    printf("         -d DEV_ID, --dev-id=DEV_ID (-2:sequential,default; -1:OpenMP parallel)\n");
    printf("         -r RANK (the number of matrix columns, 16:default)\n");
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
    sptSemiSparseTensor Y;
    sptMatrix U;
    sptIndex mode = 0;
    sptIndex R = 16;
    int dev_id = -2;
    int niters = 5;
    int nthreads = 1;
    printf("niters: %d\n", niters);

    if(argc < 3) {
        print_usage(argv);
        exit(1);
    }

    static struct option long_options[] = {
        {"input", required_argument, 0, 'i'},
        {"mode", optional_argument, 0, 'm'},
        {"output", optional_argument, 0, 'o'},
        {"dev-id", optional_argument, 0, 'd'},
        {"rank", optional_argument, 0, 'r'},
        {"help", no_argument, 0, 0},
        {0, 0, 0, 0}
    };
    int c;
    for(;;) {
        int option_index = 0;
        c = getopt_long(argc, argv, "i:m:o:d:r:", long_options, &option_index);
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
        case 'r':
            sscanf(optarg, "%u"PARTI_SCN_INDEX, &R);
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

    sptAssert(sptNewMatrix(&U, X.ndims[mode], R) == 0);
    sptAssert(sptConstantMatrix(&U, 1.0) == 0);
    // sptAssert(sptRandomizeMatrix(&U) == 0);

    /* For warm-up caches, timing not included */
    int result;
    if(dev_id == -2) {
        sptAssert(sptSparseTensorMulMatrix(&Y, &X, &U, mode) == 0);
    } else if(dev_id == -1) {
#ifdef PARTI_USE_OPENMP
        #pragma omp parallel
        {
            nthreads = omp_get_num_threads();
        }
        printf("\nnthreads: %d\n", nthreads);
        sptAssert(sptOmpSparseTensorMulMatrix(&Y, &X, &U, mode) == 0);
#endif
    }  

    sptTimer timer;
    sptNewTimer(&timer, 0);
    sptStartTimer(timer);

    for(int i = 0; i < niters; i++) {
        sptFreeSemiSparseTensor(&Y);
        if(dev_id == -2) {
            sptAssert(sptSparseTensorMulMatrix(&Y, &X, &U, mode) == 0);
        } else if(dev_id == -1) {
    #ifdef PARTI_USE_OPENMP
            sptAssert(sptOmpSparseTensorMulMatrix(&Y, &X, &U, mode) == 0);
    #endif
        }
    }

    sptStopTimer(timer);
    sptPrintAverageElapsedTime(timer, niters, "Average CooTtm");

    if(fo != NULL) {
        sptAssert(sptDumpSemiSparseTensor(&Y, fo) == 0);

        /* Convert Semi-COO to COO tensor */
        // sptSparseTensor Y_coo;
        // sptAssert(sptSemiSparseTensorToSparseTensor(&Y_coo, &Y, 1e-6) == 0);
        // sptAssert(sptDumpSparseTensor(&Y_coo, 1, fo) == 0);
        // sptFreeSparseTensor(&Y_coo);  
        fclose(fo);      
    }

    sptFreeTimer(timer);
    sptFreeSemiSparseTensor(&Y);
    sptFreeSparseTensor(&X);
    sptFreeMatrix(&U);

    return 0;
}
