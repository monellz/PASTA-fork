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
#ifdef PASTA_USE_OPENMP

#include <pasta.h>
#include "sptensor.h"


/**
 * OpenMP parallelized element-wise addition of two sparse tensors
 * @param[out] Z the result of X.+Y, should be uninitialized
 * @param[in]  X the input X
 * @param[in]  Y the input Y
 */
int sptOmpSparseTensorDotAdd(sptSparseTensor *Z, sptSparseTensor *X, sptSparseTensor *Y, int collectZero, int nthreads) 
{
    /* Ensure X and Y are in same shape */
    if(Y->nmodes != X->nmodes) {
        spt_CheckError(SPTERR_SHAPE_MISMATCH, "OMP SpTns Add", "shape mismatch");
    }
    sptIndex * max_ndims = (sptIndex*)malloc(X->nmodes * sizeof(sptIndex));
    for(sptIndex i = 0; i < X->nmodes; ++i) {
        if(Y->ndims[i] > X->ndims[i]) {
            max_ndims[i] = Y->ndims[i];
        } else {
            max_ndims[i] = X->ndims[i];
        }
    }

    sptTimer timer;
    sptNewTimer(&timer, 0);

    /* Allocate output tensor Z */
    sptStartTimer(timer);
    sptAssert(sptNewSparseTensor(Z, X->nmodes, max_ndims) == 0);
    free(max_ndims);
    sptStopTimer(timer);
    sptPrintElapsedTime(timer, "sptNewSparseTensor");

    /* Determine partitioning strategy. */
    sptStartTimer(timer);
    sptNnzIndex * dist_nnzs_X = (sptNnzIndex*)malloc((nthreads+1)*sizeof(sptNnzIndex));
    sptNnzIndex * dist_nnzs_Y = (sptNnzIndex*)malloc((nthreads+1)*sizeof(sptNnzIndex));
    sptIndex * dist_nrows_X = (sptIndex*)malloc(nthreads*sizeof(sptIndex));

    spt_DistSparseTensor(X, nthreads, dist_nnzs_X, dist_nrows_X);
    spt_DistSparseTensorFixed(Y, nthreads, dist_nrows_X, dist_nnzs_Y);
    for(sptIndex i = 0; i < nthreads; ++ i)
        if(dist_nrows_X[i] == 0) {
            printf("Error: Reduce nthreads to remove 0 rows allocation or put the larger tensor first.\n");
            exit(1);
        }

    printf("dist_nnzs_X:\n");
    for(int i=0; i<nthreads + 1; ++i) {
        printf("%zu ", dist_nnzs_X[i]);
    }
    printf("\n");
    printf("dist_nrows_X:\n");
    for(int i=0; i<nthreads; ++i) {
        printf("%u ", dist_nrows_X[i]);
    }
    printf("\n");
    printf("dist_nnzs_Y:\n");
    for(int i=0; i<nthreads + 1; ++i) {
        printf("%zu ", dist_nnzs_Y[i]);
    }
    printf("\n");
    fflush(stdout);

    free(dist_nrows_X);
    sptStopTimer(timer);
    sptPrintElapsedTime(timer, "Distribute two input tensors");


    sptStartTimer(timer);
    /* Build a private arrays to append values. */
    sptNnzIndex nnz_gap = llabs((long long) X->nnz - (long long) Y->nnz);
    sptNnzIndex increase_size = 0;
    if(nnz_gap == 0) increase_size = 10;
    else increase_size = nnz_gap;

    sptIndexVector **local_inds = (sptIndexVector**)malloc(nthreads* sizeof *local_inds);
    for(int k=0; k<nthreads; ++k) {
        local_inds[k] = (sptIndexVector*)malloc(Y->nmodes* sizeof *(local_inds[k]));
        for(sptIndex m=0; m<Y->nmodes; ++m) {
            sptNewIndexVector(&(local_inds[k][m]), 0, increase_size);
        }
    }

    sptValueVector *local_vals = (sptValueVector*)malloc(nthreads* sizeof *local_vals);
    for(int k=0; k<nthreads; ++k) {
        sptNewValueVector(&(local_vals[k]), 0, increase_size);
    }

    /* Add elements one by one, assume indices are ordered */
    sptNnzIndex Znnz = 0;
    omp_set_dynamic(0);
    omp_set_num_threads(nthreads);
    #pragma omp parallel reduction(+:Znnz)
    {
        int tid = omp_get_thread_num();
        int result;
        sptNnzIndex i=dist_nnzs_X[tid], j=dist_nnzs_Y[tid];

        while(i < dist_nnzs_X[tid+1] && j < dist_nnzs_Y[tid+1]) {
            int compare = spt_SparseTensorCompareIndices(X, i, Y, j);
            if(compare > 0) {    // X(i) > Y(j)
                for(sptIndex mode = 0; mode < X->nmodes; ++mode) {
                    sptAssert(sptAppendIndexVector(&(local_inds[tid][mode]), Y->inds[mode].data[j]) == 0);
                }
                sptAssert(sptAppendValueVector(&(local_vals[tid]), Y->values.data[j]) == 0);

                ++ Znnz;
                ++j;
            } else if(compare < 0) {    // X(i) < Y(j)
                for(sptIndex mode = 0; mode < X->nmodes; ++mode) {
                    sptAssert(sptAppendIndexVector(&(local_inds[tid][mode]), X->inds[mode].data[i]) == 0);
                }
                sptAssert(sptAppendValueVector(&(local_vals[tid]), X->values.data[i]) == 0);

                ++Znnz;
                ++i;
            } else {    // X(i) = Y(j)
                for(sptIndex mode = 0; mode < X->nmodes; ++mode) {
                    sptAssert(sptAppendIndexVector(&(local_inds[tid][mode]), X->inds[mode].data[i]) == 0);
                }
                sptAssert(sptAppendValueVector(&(local_vals[tid]), X->values.data[i] + Y->values.data[j]) == 0);

                ++ Znnz;
                ++i;
                ++j;
            }
        }

        /* Append remaining elements of X to Z */
        while(i < dist_nnzs_X[tid+1]) {
            for(sptIndex mode = 0; mode < X->nmodes; ++mode) {
                sptAssert(sptAppendIndexVector(&(local_inds[tid][mode]), X->inds[mode].data[i]) == 0);
            }
            sptAssert(sptAppendValueVector(&(local_vals[tid]), X->values.data[i]) == 0);

            ++Znnz;
            ++i;
        }

        /* Append remaining elements of Y to Z */
        while(j < dist_nnzs_Y[tid+1]) {
            for(sptIndex mode = 0; mode < Y->nmodes; ++mode) {
                sptAssert(sptAppendIndexVector(&(local_inds[tid][mode]), Y->inds[mode].data[j]) == 0);
            }
            sptAssert(sptAppendValueVector(&(local_vals[tid]), Y->values.data[j]) == 0);

            ++Znnz;
            ++j;
        }

    }
    Z->nnz = Znnz;  

    /* Append all the local arrays to Y. */
    for(int k=0; k<nthreads; ++k) {
        for(sptIndex m=0; m<Z->nmodes; ++m) {
            sptAssert(sptAppendIndexVectorWithVector(&(Z->inds[m]), &(local_inds[k][m])) == 0);
        }
        sptAssert(sptAppendValueVectorWithVector(&(Z->values), &(local_vals[k])) == 0);
    }
    sptStopTimer(timer);
    sptPrintElapsedTime(timer, "Omp  SpTns DotAdd");


    for(int k=0; k<nthreads; ++k) {
        for(sptIndex m=0; m<Y->nmodes; ++m) {
            sptFreeIndexVector(&(local_inds[k][m]));
        }
        free(local_inds[k]);
        sptFreeValueVector(&(local_vals[k]));
    }
    free(local_inds);
    free(local_vals);
    free(dist_nnzs_X);
    free(dist_nnzs_Y);


    /* Check whether elements become zero after adding.
       If so, fill the gap with the [nnz-1]'th element.
    */
    sptStartTimer(timer);
    if(collectZero == 1) {
        sptSparseTensorCollectZeros(Z);
    }
    sptStopTimer(timer);
    sptPrintElapsedTime(timer, "sptSparseTensorCollectZeros");

    return 0;
}

#endif
