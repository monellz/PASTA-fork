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
#include "../src/sptensor/sptensor.h"

void print_usage(int argc, char ** argv) {
    printf("Usage: %s [options] \n", argv[0]);
    printf("Options: -i INPUT, --input=INPUT\n");
    printf("         -o OUTPUT, --output=OUTPUT\n");
    printf("         -b BLOCKSIZE (bits), --blocksize=BLOCKSIZE (bits)\n");
    printf("         -m MODE, --mode=MODE\n");
    printf("         -d DEV_ID, --dev-id=DEV_ID (-2:sequential,default; -1:OpenMP parallel; >=0:GPU parallel)\n");
    printf("         -r RANK (the number of matrix columns, 16:default)\n");
    printf("         CUDA options: \n");
    printf("         -p IMPL_NUM, --impl-num=IMPL_NUM\n");
    printf("         -s SMEM_SIZE, --smem-size=SMEM_SIZE\n");
    printf("\n");
}

/**
 * Benchmark Matriced Tensor Times Khatri-Rao Product (MTTKRP), tensor in HiCOO format, matrices are dense.
 */
int main(int argc, char ** argv) 
{
    FILE *fi = NULL, *fo = NULL;
    sptSparseTensor tsr;
    sptMatrix ** U;
    sptSparseTensorHiCOO hitsr;
    sptElementIndex sb_bits = 7;

    sptIndex mode = 0;
    sptIndex R = 16;
    int dev_id = -2;
    int impl_num = 14;
    sptNnzIndex smem_size = 40000;
    int niters = 5;
    int nthreads = 1;
    int sort_impl = 1;  // 1: Morton order; 2: Rowblock sorting
    printf("niters: %d\n", niters);
    int retval;

    if(argc <= 3) { // #Required arguments
        print_usage(argc, argv);
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
            {"impl-num", optional_argument, 0, 'p'},
            {"smem-size", optional_argument, 0, 's'},
            {0, 0, 0, 0}
        };
        int option_index = 0;
        int c = 0;
        // c = getopt_long(argc, argv, "i:o:b:k:c:m:", long_options, &option_index);
        c = getopt_long(argc, argv, "i:o:b:m:d:r:p:s:", long_options, &option_index);
        if(c == -1) {
            break;
        }
        switch(c) {
        case 'i':
            fi = fopen(optarg, "r");
            sptAssert(fi != NULL);
            break;
        case 'o':
            fo = fopen(optarg, "w");
            sptAssert(fo != NULL);
            break;
        case 'b':
            sscanf(optarg, "%"PARTI_SCN_ELEMENT_INDEX, &sb_bits);
            break;
        case 'm':
            sscanf(optarg, "%"PARTI_SCN_INDEX, &mode);
            break;
        case 'd':
            sscanf(optarg, "%d", &dev_id);
            if(dev_id < -2) {
                fprintf(stderr, "Error: set dev_id to -2/-1/>=0.\n");
                exit(1);
            }
            break;
        case 'r':
            sscanf(optarg, "%"PARTI_SCN_INDEX, &R);
            break;
        case 'p':
            sscanf(optarg, "%d", &impl_num);
            break;
        case 's':
            sscanf(optarg, "%"PARTI_SCN_NNZ_INDEX, &smem_size);
            break;
        case '?':   /* invalid option */
        case 'h':
        default:
            print_usage(argc, argv);
            exit(1);
        }
    }
    printf("mode: %"PARTI_PRI_INDEX "\n", mode);
    printf("Block size (bit-length): %"PARTI_PRI_ELEMENT_INDEX"\n", sb_bits);
    printf("dev_id: %d\n", dev_id);
    if(dev_id >= 0)
        printf("impl_num: %d\n", impl_num);
    // printf("Sorting implementation: %d\n", sort_impl);

    sptAssert(sptLoadSparseTensor(&tsr, 1, fi) == 0);
    // sptSparseTensorSortIndex(&tsr, 1);
    fclose(fi);
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
      // sptAssert(sptRandomizeMatrix(U[m], tsr.ndims[m], R) == 0);
      sptAssert(sptNewMatrix(U[m], hitsr.ndims[m], R) == 0);
      sptAssert(sptConstantMatrix(U[m], 1) == 0);
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
        sptAssert(sptMTTKRPHiCOO(&hitsr, U, mats_order, mode) == 0);
    } else if(dev_id == -1) {
#ifdef PARTI_USE_OPENMP
        #pragma omp parallel
        {
            nthreads = omp_get_num_threads();
        }
        printf("\nnthreads: %d\n", nthreads);
        sptAssert(sptOmpMTTKRPHiCOO(&hitsr, U, mats_order, mode, nthreads) == 0);
#endif
    } else {
        sptCudaSetDevice(dev_id);
        sptAssert(sptCudaMTTKRPHiCOO(&hitsr, U, mats_order, mode, max_nnzb, impl_num) == 0);
    }

    sptTimer timer;
    sptNewTimer(&timer, 0);
    sptStartTimer(timer);

    for(int it=0; it<niters; ++it) {
        if(dev_id == -2) {
            sptAssert(sptMTTKRPHiCOO(&hitsr, U, mats_order, mode) == 0);
        } else if(dev_id == -1) {
#ifdef PARTI_USE_OPENMP
            sptAssert(sptOmpMTTKRPHiCOO(&hitsr, U, mats_order, mode, nthreads) == 0);
#endif
        } else {
            sptCudaSetDevice(dev_id);
            sptAssert(sptCudaMTTKRPHiCOO(&hitsr, U, mats_order, mode, max_nnzb, impl_num) == 0);
        }
    }

    sptStopTimer(timer);
    sptPrintAverageElapsedTime(timer, niters, "CPU  SpTns MTTKRP HiCOO");
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
