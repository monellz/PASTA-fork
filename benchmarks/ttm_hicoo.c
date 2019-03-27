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
    printf("         -b BLOCKSIZE (bits), --blocksize=BLOCKSIZE (bits)\n");
    printf("         -d DEV_ID, --dev-id=DEV_ID (-2:sequential,default; -1:OpenMP parallel)\n");
    printf("         -r RANK (the number of matrix columns, 16:default)\n");
    printf("         --help\n");
    printf("\n");
}


/**
 * Benchmark HiCOO tensor times a dense matrix.
 */
int main(int argc, char ** argv) 
{
    FILE *fi = NULL, *fo = NULL;
    sptSparseTensor X;
    sptSemiSparseTensor Y;
    sptSparseTensorHiCOOGeneral hiX;
    sptSemiSparseTensorHiCOO hiY;
    sptMatrix U;
    sptIndex mode = 0;
    sptElementIndex sb_bits = 7;
    sptIndex R = 16;
    int dev_id = -2;
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
        {"help", no_argument, 0, 0},
        {0, 0, 0, 0}
    };
    int c;
    for(;;) {
        int option_index = 0;
        c = getopt_long(argc, argv, "i:m:o:b:d:r:", long_options, &option_index);
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
        case 'b':
            sscanf(optarg, "%"PARTI_SCN_ELEMENT_INDEX, &sb_bits);
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
    printf("Block size (bit-length): %"PARTI_PRI_ELEMENT_INDEX"\n", sb_bits);
    printf("dev_id: %d\n", dev_id);

    sptAssert(sptLoadSparseTensor(&X, 1, fi) == 0);
    fclose(fi);
    sptSparseTensorStatus(&X, stdout);
    // sptAssert(sptDumpSparseTensor(&X, 0, stdout) == 0);

    sptAssert(sptNewMatrix(&U, X.ndims[mode], R) == 0);
    // sptAssert(sptConstantMatrix(&U, 1.0) == 0);
    sptAssert(sptRandomizeMatrix(&U) == 0);

    sptIndex ncmodes = 2;
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
#ifdef PARTI_USE_OPENMP
        #pragma omp parallel
        {
            nthreads = omp_get_num_threads();
        }
        printf("\nnthreads: %d\n", nthreads);
        // sptAssert(sptOmpSparseTensorMulMatrixHiCOO(&hiY, &hiX, &U, mode) == 0);
#endif
    }  


    sptStartTimer(timer);

    for(int i = 0; i < niters; i++) {
        sptFreeSemiSparseTensorHiCOO(&hiY);
        if(dev_id == -2) {
            sptAssert(sptSparseTensorMulMatrixHiCOO(&hiY, &hiX, &U, mode) == 0);
        } else if(dev_id == -1) {
    #ifdef PARTI_USE_OPENMP
            // sptAssert(sptOmpSparseTensorMulMatrixHiCOO(&hiY, &hiX, &U, mode) == 0);
    #endif
        }
    }

    sptStopTimer(timer);
    sptPrintAverageElapsedTime(timer, niters, "Average CooTtmHiCOO");

    if(fo != NULL) {
        // sptDumpSemiSparseTensorHiCOO(&hiY, stdout);

        /* Convert Semi-HiCOO to Semi-COO tensor */
        sptStartTimer(timer);
        sptAssert(sptSemiHiCOOToSemiSparseTensor(&Y, &hiY) == 0);
        sptSemiSparseTensorSortIndex(&Y);
        sptStopTimer(timer);
        sptPrintElapsedTime(timer, "Convert Semi-HiCOO -> Semi-COO");
        sptAssert(sptDumpSemiSparseTensor(&Y, fo) == 0);

        /* Convert Semi-COO to COO tensor */
        // sptStartTimer(timer);
        // sptSparseTensor Y_coo;
        // sptAssert(sptSemiSparseTensorToSparseTensor(&Y_coo, &Y, 1e-6) == 0);
        // sptStopTimer(timer);
        // sptPrintElapsedTime(timer, "Convert Semi-COO -> COO");
        sptFreeSemiSparseTensor(&Y);

        // sptAssert(sptDumpSparseTensor(&Y_coo, 1, fo) == 0);
        // sptFreeSparseTensor(&Y_coo);  
        fclose(fo);      
    }

    sptFreeSemiSparseTensorHiCOO(&hiY);
    sptFreeSparseTensorHiCOOGeneral(&hiX);
    sptFreeTimer(timer);
    sptFreeMatrix(&U);

    return 0;
}
