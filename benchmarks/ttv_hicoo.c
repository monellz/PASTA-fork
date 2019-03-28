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
    printf("         -d DEV_ID, --dev-id=DEV_ID (-2:sequential,default; -1:OpenMP parallel)\n");
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
    sptSparseTensor Y;
    sptSparseTensorHiCOOGeneral hiX;
    sptSparseTensorHiCOO hiY;
    sptValueVector V;
    sptIndex mode = 0;
    sptElementIndex sb_bits = 7;
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
        {"help", no_argument, 0, 0},
        {0, 0, 0, 0}
    };
    int c;
    for(;;) {
        int option_index = 0;
        c = getopt_long(argc, argv, "i:m:o:b:d:", long_options, &option_index);
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
            sscanf(optarg, "%"PASTA_SCN_ELEMENT_INDEX, &sb_bits);
            break;
        case 'm':
            sscanf(optarg, "%"PASTA_SCN_INDEX, &mode);
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

    printf("mode: %"PASTA_PRI_INDEX "\n", mode);
    printf("Block size (bit-length): %"PASTA_PRI_ELEMENT_INDEX"\n", sb_bits);
    printf("dev_id: %d\n", dev_id);

    sptAssert(sptLoadSparseTensor(&X, 1, fi) == 0);
    fclose(fi);
    sptSparseTensorStatus(&X, stdout);
    // sptAssert(sptDumpSparseTensor(&X, 0, stdout) == 0);

    sptAssert(sptNewValueVector(&V, X.ndims[mode], X.ndims[mode]) == 0);
    // sptAssert(sptConstantValueVector(&V, 1.0) == 0);
    sptAssert(sptRandomizeValueVector(&V) == 0);


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
        sptAssert(sptSparseTensorMulVectorHiCOO(&hiY, &hiX, &V, mode) == 0);
    } else if(dev_id == -1) {
#ifdef PASTA_USE_OPENMP
        #pragma omp parallel
        {
            nthreads = omp_get_num_threads();
        }
        printf("\nnthreads: %d\n", nthreads);
        sptAssert(sptOmpSparseTensorMulVectorHiCOO(&hiY, &hiX, &V, mode) == 0);
#endif
    }  


    sptStartTimer(timer);

    for(int i = 0; i < niters; i++) {
        sptFreeSparseTensorHiCOO(&hiY);
        if(dev_id == -2) {
            sptAssert(sptSparseTensorMulVectorHiCOO(&hiY, &hiX, &V, mode) == 0);
        } else if(dev_id == -1) {
    #ifdef PASTA_USE_OPENMP
            sptAssert(sptOmpSparseTensorMulVectorHiCOO(&hiY, &hiX, &V, mode) == 0);
    #endif
        }
    }

    sptStopTimer(timer);
    sptPrintAverageElapsedTime(timer, niters, "Average CooTtvHiCOO");

    if(fo != NULL) {
        // sptDumpSparseTensorHiCOO(&hiY, stdout);

        /* Convert Semi-HiCOO to Semi-COO tensor */
        sptStartTimer(timer);
        sptAssert(sptHiCOOToSparseTensor(&Y, &hiY) == 0);
        sptStopTimer(timer);
        sptPrintElapsedTime(timer, "Convert HiCOO -> COO");

        sptSparseTensorSortIndex(&Y, 0, Y.nnz, 1);
        sptAssert(sptDumpSparseTensor(&Y, 1, fo) == 0);
        sptFreeSparseTensor(&Y); 
        fclose(fo);
    }

    sptFreeTimer(timer);
    sptFreeSparseTensorHiCOO(&hiY);
    sptFreeSparseTensorHiCOOGeneral(&hiX);
    sptFreeValueVector(&V);

    return 0;
}
