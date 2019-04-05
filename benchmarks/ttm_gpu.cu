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
    printf("Options: -i INPUT, --input=INPUT (.tns file)\n");
    printf("         -o OUTPUT, --output=OUTPUT (output file name)\n");
    printf("         -m MODE, --mode=MODE (specify a mode, e.g., 0 (default) or 1 or 2 for third-order tensors.)\n");
    printf("         -d DEV_ID, --dev-id=DEV_ID (-2:sequential,default; -1:OpenMP parallel)\n");
    printf("         -r RANK (the number of matrix columns, 16:default)\n");
    printf("         CUDA options: \n");
    printf("         -p IMPL_NUM, --impl-num=IMPL_NUM\n");
    printf("         -s SMEM_SIZE, --smem-size=SMEM_SIZE\n");
    printf("         --help\n");
    printf("\n");
}

int main(int argc, char ** argv) 
{
    FILE *fo = NULL;
    char fname[1000];
    sptSparseTensor X;
    sptSemiSparseTensor Y;
    sptMatrix U;
    sptIndex mode = 0;
    sptIndex R = 16;
    int dev_id = -2;
    int impl_num = 14;
    sptNnzIndex smem_size = 40000;
    int niters = 5;
    int nthreads = 1;
    printf("niters: %d\n", niters);

    if(argc < 5) { // #Required arguments
        print_usage(argv);
        exit(1);
    }

    static struct option long_options[] = {
        {"input", required_argument, 0, 'i'},
        {"mode", required_argument, 0, 'm'},
        {"output", optional_argument, 0, 'o'},
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
        c = getopt_long(argc, argv, "i:m:o:d:r:p:s:", long_options, &option_index);
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
    printf("dev_id: %d\n", dev_id);
    if(dev_id >= 0)
        printf("impl_num: %d\n", impl_num);

    sptAssert(sptLoadSparseTensor(&X, 1, fname) == 0);

    sptAssert(sptNewMatrix(&U, X.ndims[mode], R) == 0);
    // sptAssert(sptConstantMatrix(&U, 1.0) == 0);
    sptAssert(sptRandomizeMatrix(&U) == 0);

    /* For warm-up caches, timing not included */
    if(dev_id == -2) {
        sptAssert(sptSparseTensorMulMatrix(&Y, &X, &U, mode) == 0);
    } else if(dev_id == -1) {
#ifdef PASTA_USE_OPENMP
        #pragma omp parallel
        {
            nthreads = omp_get_num_threads();
        }
        printf("\nnthreads: %d\n", nthreads);
        sptAssert(sptOmpSparseTensorMulMatrix(&Y, &X, &U, mode) == 0);
#endif
    } else {
        sptCudaSetDevice(dev_id);
        sptAssert(sptCudaSparseTensorMulMatrix(&Y, &X, &U, mode, impl_num, smem_size) == 0);
    }

    sptTimer timer;
    sptNewTimer(&timer, 0);
    sptStartTimer(timer);

    for(int it=0; it<niters; ++it) {
        sptFreeSemiSparseTensor(&Y);
        if(dev_id == -2) {
            sptAssert(sptSparseTensorMulMatrix(&Y, &X, &U, mode) == 0);
        } else if(dev_id == -1) {
#ifdef PASTA_USE_OPENMP
            sptAssert(sptOmpSparseTensorMulMatrix(&Y, &X, &U, mode) == 0);
#endif
        } else {
            sptCudaSetDevice(dev_id);
            sptAssert(sptCudaSparseTensorMulMatrix(&Y, &X, &U, mode, impl_num, smem_size) == 0);
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
    sptFreeMatrix(&U);
    sptFreeSparseTensor(&X);

    return 0;
}
