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
#include "../src/sptensor/sptensor.h"

void print_usage(char ** argv) {
    printf("Usage: %s [options] \n", argv[0]);
    printf("Options: -i INPUT, --input=INPUT\n");
    printf("         -b BLOCKSIZE (bits), --blocksize=BLOCKSIZE (bits)\n");
    printf("         -d DEV_ID, --dev-id=DEV_ID\n");
    printf("         OpenMP options: \n");
    printf("         -t NTHREADS, --nthreads=NT (1:default)\n");
    printf("\n");
}

/**
 * Benchmark Matriced Tensor Times Khatri-Rao Product (MTTKRP), tensor in HiCOO format, matrices are dense.
 */
int main(int argc, char ** argv) 
{
    FILE *fo = NULL;
    char fname[1000];
    sptSparseTensor tsr;
    sptMatrix ** U;
    sptSparseTensorHiCOO hitsr;
    sptElementIndex sb_bits = 7;

    int dev_id = -2;
    int nthreads = 1;
    int sort_impl = 1;  // 1: Morton order; 2: Rowblock sorting

    if(argc <= 3) { // #Required arguments
        print_usage(argv);
        exit(1);
    }

    for(;;) {
        static struct option long_options[] = {
            {"input", required_argument, 0, 'i'},
            {"bs", optional_argument, 0, 'b'},
            {"dev-id", optional_argument, 0, 'd'},
            {"nthreads", optional_argument, 0, 't'},
            {0, 0, 0, 0}
        };
        int option_index = 0;
        int c = 0;
        // c = getopt_long(argc, argv, "i:o:b:k:c:m:", long_options, &option_index);
        c = getopt_long(argc, argv, "i:b:d:t:", long_options, &option_index);
        if(c == -1) {
            break;
        }
        switch(c) {
        case 'i':
            strcpy(fname, optarg);
            printf("input file: %s\n", fname); fflush(stdout);
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
    printf("dev_id: %d\n", dev_id);
    printf("Sorting implementation: %d\n", sort_impl);

    sptAssert(sptLoadSparseTensor(&tsr, 1, fname) == 0);
    // sptSparseTensorSortIndex(&tsr, 1);
    sptSparseTensorStatus(&tsr, stdout);
    // sptAssert(sptDumpSparseTensor(&tsr, 0, stdout) == 0);

    /* Convert to HiCOO tensor */
    sptNnzIndex max_nnzb = 0;
    sptAssert(sptSparseTensorToHiCOO(&hitsr, &max_nnzb, &tsr, sb_bits, sort_impl, nthreads) == 0);
    sptSparseTensorStatusHiCOO(&hitsr, stdout);
    sptFreeSparseTensorHiCOO(&hitsr);
    // sptAssert(sptDumpSparseTensorHiCOO(&hitsr, stdout) == 0);

    /* Set fibers */
    for(sptIndex m = 0; m < tsr.nmodes; ++m) {
        /* Sort tensor except mode */
        sptSparseTensorSortIndexAtMode(&tsr, m, 1);

        sptNnzIndexVector fiberidx;
        sptSemiSparseTensorSetFibers(&fiberidx, &tsr, m);

        double avg_flen = (double)tsr.nnz / (fiberidx.len - 1);
        printf("[ mode %u ] nfibs: %lu, Average flen = %.2f .\n", m, fiberidx.len - 1, avg_flen);
        sptFreeNnzIndexVector(&fiberidx);
    }

    sptFreeSparseTensor(&tsr);

    return 0;
}
