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
    printf("         -b BLOCKSIZE (bits), --blocksize=BLOCKSIZE (bits)\n");
    printf("         -d DEV_ID, --dev-id=DEV_ID (-2:sequential,default; -1:OpenMP parallel; >=0:GPU parallel)\n");
    printf("         -r RANK (the number of matrix columns, 16:default)\n");
    printf("         CUDA options: \n");
    printf("         -p IMPL_NUM, --impl-num=IMPL_NUM\n");
    printf("         -s SMEM_SIZE, --smem-size=SMEM_SIZE\n");
    printf("         --help\n");
    printf("\n");
}


/**
 * Benchmark HiCOO tensor times a dense matrix.
 */
int main(int argc, char ** argv) 
{
    FILE *fo = NULL;
    char fname[1000];
    sptSparseTensor X;
    sptSemiSparseTensor Y;
    sptSparseTensorHiCOOGeneral hiX;
    sptSemiSparseTensorHiCOO hiY;
    sptMatrix U;
    sptIndex mode = 0;
    sptElementIndex sb_bits = 7;
    sptIndex R = 16;
    int dev_id = -2;
    int impl_num = 14;
    sptNnzIndex smem_size = 40000;
    int niters = 5;
    int nthreads = 1;
    printf("niters: %d\n", niters);
    sptTimer timer;
    sptNewTimer(&timer, 0);

    if(argc < 3) {
        print_usage(argv);
        exit(1);
    }

    static struct option long_options[] = {
        {"input", required_argument, 0, 'i'},
        {"mode", optional_argument, 0, 'm'},
        {"output", optional_argument, 0, 'o'},
        {"bs", optional_argument, 0, 'b'},
        {"dev-id", optional_argument, 0, 'd'},
        {"rank", optional_argument, 0, 'r'},
        {"impl-num", optional_argument, 0, 'p'},
        {"smem-size", optional_argument, 0, 's'},
        {"help", no_argument, 0, 0},
        {0, 0, 0, 0}
    };
    int c;
    for(;;) {
        int option_index = 0;
        c = getopt_long(argc, argv, "i:m:o:b:d:r:p:s:", long_options, &option_index);
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
            sscanf(optarg, "%" PASTA_SCN_ELEMENT_INDEX, &sb_bits);
            break;
        case 'm':
            sscanf(optarg, "%" PASTA_SCN_INDEX, &mode);
            break;
        case 'd':
            sscanf(optarg, "%d", &dev_id);
            break;
        case 'r':
            sscanf(optarg, "%u" PASTA_SCN_INDEX, &R);
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
    printf("Block size (bit-length): %" PASTA_PRI_ELEMENT_INDEX"\n", sb_bits);
    printf("dev_id: %d\n", dev_id);
    if(dev_id >= 0)
        printf("impl_num: %d\n", impl_num);

    sptAssert(sptLoadSparseTensor(&X, 1, fname) == 0);
    sptSparseTensorStatus(&X, stdout);
    // sptAssert(sptDumpSparseTensor(&X, 0, stdout) == 0);

    sptAssert(sptNewMatrix(&U, X.ndims[mode], R) == 0);
    sptAssert(sptConstantMatrix(&U, 1.0) == 0);
    // sptAssert(sptRandomizeMatrix(&U) == 0);

    sptIndex ncmodes = X.nmodes - 1;
    sptIndex * flags = (sptIndex *)malloc(X.nmodes * sizeof(*flags));
    for(sptIndex m = 0; m < X.nmodes; ++m) {
        flags[m] = 1;
    }
    flags[mode] = 0;

    /* Convert to HiCOO tensor */
    sptStartTimer(timer);
    sptNnzIndex max_nnzb = 0;
    sptAssert(sptSparseTensorToHiCOOGeneral(&hiX, &max_nnzb, &X, sb_bits, ncmodes, flags, 1) == 0);
    // printf("Sort inside blocks:\n");
    // sptAssert(sptDumpSparseTensor(&X, 0, stdout) == 0);
    sptFreeSparseTensor(&X);
    sptSparseTensorStatusHiCOOGeneral(&hiX, stdout);
    // sptAssert(sptDumpSparseTensorHiCOOGeneral(&hiX, stdout) == 0);
    sptStopTimer(timer);
    sptPrintElapsedTime(timer, "Convert COO -> HiCOO");

    /* For warm-up caches, timing not included */
    int result;
    if(dev_id == -2) {
        sptAssert(sptSparseTensorMulMatrixHiCOO(&hiY, &hiX, &U, mode) == 0);
    } else if(dev_id == -1) {
#ifdef PASTA_USE_OPENMP
        #pragma omp parallel
        {
            nthreads = omp_get_num_threads();
        }
        printf("\nnthreads: %d\n", nthreads);
        sptAssert(sptOmpSparseTensorMulMatrixHiCOO(&hiY, &hiX, &U, mode) == 0);
#endif
    } else {
        sptCudaSetDevice(dev_id);
        sptAssert(sptCudaSparseTensorMulMatrixHiCOO(&hiY, &hiX, &U, mode, impl_num, smem_size) == 0);
    } 


    sptStartTimer(timer);

    for(int i = 0; i < niters; i++) {
        sptFreeSemiSparseTensorHiCOO(&hiY);
        if(dev_id == -2) {
            sptAssert(sptSparseTensorMulMatrixHiCOO(&hiY, &hiX, &U, mode) == 0);
        } else if(dev_id == -1) {
    #ifdef PASTA_USE_OPENMP
            sptAssert(sptOmpSparseTensorMulMatrixHiCOO(&hiY, &hiX, &U, mode) == 0);
    #endif
        } else {
            sptCudaSetDevice(dev_id);
            sptAssert(sptCudaSparseTensorMulMatrixHiCOO(&hiY, &hiX, &U, mode, impl_num, smem_size) == 0);
        }
    }

    sptStopTimer(timer);
    sptPrintAverageElapsedTime(timer, niters, "Average HiCooTtm");

    if(fo != NULL) {
        sptDumpSemiSparseTensorHiCOO(&hiY, fo);

        /* Convert Semi-HiCOO to Semi-COO tensor to compare with COO output */
        // sptStartTimer(timer);
        // sptAssert(sptSemiHiCOOToSemiSparseTensor(&Y, &hiY) == 0);
        // sptSemiSparseTensorSortIndex(&Y);
        // sptStopTimer(timer);
        // sptPrintElapsedTime(timer, "Convert Semi-HiCOO -> Semi-COO");
        // sptAssert(sptDumpSemiSparseTensor(&Y, fo) == 0);

        /* Convert Semi-COO to COO tensor */
        // sptStartTimer(timer);
        // sptSparseTensor Y_coo;
        // sptAssert(sptSemiSparseTensorToSparseTensor(&Y_coo, &Y, 1e-6) == 0);
        // sptStopTimer(timer);
        // sptPrintElapsedTime(timer, "Convert Semi-COO -> COO");

        // sptAssert(sptDumpSparseTensor(&Y_coo, 1, fo) == 0);
        // sptFreeSparseTensor(&Y_coo);

        // sptFreeSemiSparseTensor(&Y);
        fclose(fo);      
    }

    sptFreeSemiSparseTensorHiCOO(&hiY);
    sptFreeSparseTensorHiCOOGeneral(&hiX);
    sptFreeTimer(timer);
    sptFreeMatrix(&U);

    return 0;
}
