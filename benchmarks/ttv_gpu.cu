/*
    Internal code for ParTI!
    (c) Sam Bliss, 2018, all rights reserved.
*/

#include <stdio.h>
#include <stdlib.h>
#include <getopt.h>
#include <pasta.h>


static void print_usage(char ** argv) {
    printf("Usage: %s [options] \n\n", argv[0]);
    printf("Options: -i INPUT, --input=INPUT (.tns file)\n");
    printf("         -o OUTPUT, --output=OUTPUT (output file name)\n");
    printf("         -m MODE, --mode=MODE (specify a mode, e.g., 0 (default) or 1 or 2 for third-order tensors.)\n");
    printf("         -d DEV_ID, --dev-id=DEV_ID (-2:sequential,default; -1:OpenMP parallel; >=0:GPU parallel)\n");
    printf("         CUDA options: \n");
    printf("         -p IMPL_NUM, --impl-num=IMPL_NUM\n");
    printf("         -s SMEM_SIZE, --smem-size=SMEM_SIZE\n");
    printf("         --help\n");
    printf("\n");
}


/**
 * Benchmark COO tensor times a dense matrix.
 */
int main(int argc, char ** argv) 
{
    FILE *fo = NULL;
    char fname[1000];
    sptSparseTensor X;
    sptSparseTensor Y;
    sptValueVector V;
    sptIndex mode = 0;
    int dev_id = -2;
    int impl_num = 11;
    sptNnzIndex smem_size = 0;
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
        {"impl-num", optional_argument, 0, 'p'},
        {"smem-size", optional_argument, 0, 's'},
        {"help", no_argument, 0, 0},
        {0, 0, 0, 0}
    };
    int c;
    for(;;) {
        int option_index = 0;
        c = getopt_long(argc, argv, "i:m:o:d:p:s:", long_options, &option_index);
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
        case 'm':
            sscanf(optarg, "%" PASTA_SCN_INDEX, &mode);
            break;
        case 'd':
            sscanf(optarg, "%d", &dev_id);
            break;
        case 't':
            sscanf(optarg, "%d", &nthreads);
            break;
        case 'p':
            sscanf(optarg, "%d", &impl_num);
            break;
        case 's':
            sscanf(optarg, "%" PASTA_SCN_NNZ_INDEX, &smem_size);
            break;
        case '?':   /* invalid option */
        case 'h':
        default:
            print_usage(argv);
            exit(1);
        }
    }

    printf("mode: %" PASTA_PRI_INDEX "\n", mode);
    printf("dev_id: %d\n", dev_id);
    if(dev_id >= 0)
        printf("impl_num: %d\n", impl_num);

    sptAssert(sptLoadSparseTensor(&X, 1, fname) == 0);

    sptAssert(sptNewValueVector(&V, X.ndims[mode], X.ndims[mode]) == 0);
    // sptAssert(sptConstantValueVector(&V, 1.0) == 0);
    sptAssert(sptRandomizeValueVector(&V) == 0);

    /* For warm-up caches, timing not included */
    if(dev_id == -2) {
        sptAssert(sptSparseTensorMulVector(&Y, &X, &V, mode) == 0);
    } else if(dev_id == -1) {
#ifdef PASTA_USE_OPENMP
        #pragma omp parallel
        {
            nthreads = omp_get_num_threads();
        }
        printf("\nnthreads: %d\n", nthreads);
        sptAssert(sptOmpSparseTensorMulVector(&Y, &X, &V, mode) == 0);
#endif
    } else {
        sptCudaSetDevice(dev_id);
        sptAssert(sptCudaSparseTensorMulVector(&Y, &X, &V, mode, impl_num, smem_size) == 0);
    }

    sptTimer timer;
    sptNewTimer(&timer, 0);
    sptStartTimer(timer);

    for(int i = 0; i < niters; i++) {
        if(dev_id == -2) {
            sptAssert(sptSparseTensorMulVector(&Y, &X, &V, mode) == 0);
        } else if(dev_id == -1) {
#ifdef PASTA_USE_OPENMP
            sptAssert(sptOmpSparseTensorMulVector(&Y, &X, &V, mode) == 0);
#endif
        } else {
        sptCudaSetDevice(dev_id);
        sptAssert(sptCudaSparseTensorMulVector(&Y, &X, &V, mode, impl_num, smem_size) == 0);
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
