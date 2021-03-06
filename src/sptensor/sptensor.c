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

#include <pasta.h>
#include "sptensor.h"
#include <stdlib.h>
#include <string.h>

/**
 * Create a new sparse tensor
 * @param tsr    a pointer to an uninitialized sparse tensor
 * @param nmodes number of modes the tensor will have
 * @param ndims  the dimension of each mode the tensor will have
 */
int sptNewSparseTensor(sptSparseTensor *tsr, sptIndex nmodes, const sptIndex ndims[]) {
    sptIndex i;
    int result;
    tsr->nmodes = nmodes;
    tsr->sortorder = malloc(nmodes * sizeof tsr->sortorder[0]);
    for(i = 0; i < nmodes; ++i) {
        tsr->sortorder[i] = i;
    }
    tsr->ndims = malloc(nmodes * sizeof *tsr->ndims);
    spt_CheckOSError(!tsr->ndims, "SpTns New");
    memcpy(tsr->ndims, ndims, nmodes * sizeof *tsr->ndims);
    tsr->nnz = 0;
    tsr->inds = malloc(nmodes * sizeof *tsr->inds);
    spt_CheckOSError(!tsr->inds, "SpTns New");
    for(i = 0; i < nmodes; ++i) {
        result = sptNewIndexVector(&tsr->inds[i], 0, 0);
        spt_CheckError(result, "SpTns New", NULL);
    }
    result = sptNewValueVector(&tsr->values, 0, 0);
    spt_CheckError(result, "SpTns New", NULL);
    return 0;
}


/**
 * Create a new sparse tensor with given number of nonzeros.
 * @param tsr    a pointer to an uninitialized sparse tensor
 * @param nmodes number of modes the tensor will have
 * @param ndims  the dimension of each mode the tensor will have
 */
int sptNewSparseTensorWithNnz(sptSparseTensor *tsr, sptIndex nmodes, const sptIndex ndims[], sptNnzIndex nnz) {
    sptIndex i;
    int result;
    tsr->nmodes = nmodes;
    tsr->sortorder = malloc(nmodes * sizeof tsr->sortorder[0]);
    for(i = 0; i < nmodes; ++i) {
        tsr->sortorder[i] = i;
    }
    tsr->ndims = malloc(nmodes * sizeof *tsr->ndims);
    spt_CheckOSError(!tsr->ndims, "SpTns New");
    memcpy(tsr->ndims, ndims, nmodes * sizeof *tsr->ndims);
    tsr->nnz = nnz;
    tsr->inds = malloc(nmodes * sizeof *tsr->inds);
    spt_CheckOSError(!tsr->inds, "SpTns New");
    for(i = 0; i < nmodes; ++i) {
        result = sptNewIndexVector(&tsr->inds[i], nnz, nnz);
        spt_CheckError(result, "SpTns New", NULL);
    }
    result = sptNewValueVector(&tsr->values, nnz, nnz);
    spt_CheckError(result, "SpTns New", NULL);
    return 0;
}


/**
 * Copy a sparse tensor
 * @param[out] dest a pointer to an uninitialized sparse tensor
 * @param[in]  src  a pointer to a valid sparse tensor
 */
int sptCopySparseTensor(sptSparseTensor *dest, const sptSparseTensor *src) {
    sptIndex i;
    int result;
    dest->nmodes = src->nmodes;
    dest->sortorder = malloc(src->nmodes * sizeof src->sortorder[0]);
    memcpy(dest->sortorder, src->sortorder, src->nmodes * sizeof src->sortorder[0]);
    dest->ndims = malloc(dest->nmodes * sizeof *dest->ndims);
    spt_CheckOSError(!dest->ndims, "SpTns Copy");
    memcpy(dest->ndims, src->ndims, src->nmodes * sizeof *src->ndims);
    dest->nnz = src->nnz;
    dest->inds = malloc(dest->nmodes * sizeof *dest->inds);
    spt_CheckOSError(!dest->inds, "SpTns Copy");
    for(i = 0; i < dest->nmodes; ++i) {
        result = sptCopyIndexVector(&dest->inds[i], &src->inds[i]);
        spt_CheckError(result, "SpTns Copy", NULL);
    }
    result = sptCopyValueVector(&dest->values, &src->values);
    spt_CheckError(result, "SpTns Copy", NULL);
    return 0;
}

/**
 * Allocate a sparse tensor from another tensor.
 * @param[out] dest a pointer to an uninitialized sparse tensor
 * @param[in]  src  a pointer to a valid sparse tensor
 */
int sptCopySparseTensorAllocateOnly(sptSparseTensor *dest, const sptSparseTensor *src) {
    sptIndex i;
    int result;
    dest->nmodes = src->nmodes;
    dest->sortorder = malloc(src->nmodes * sizeof src->sortorder[0]);
    memcpy(dest->sortorder, src->sortorder, src->nmodes * sizeof src->sortorder[0]);
    dest->ndims = malloc(dest->nmodes * sizeof *dest->ndims);
    spt_CheckOSError(!dest->ndims, "SpTns Copy");
    memcpy(dest->ndims, src->ndims, src->nmodes * sizeof *src->ndims);
    dest->nnz = src->nnz;
    dest->inds = malloc(dest->nmodes * sizeof *dest->inds);
    spt_CheckOSError(!dest->inds, "SpTns Copy");
    for(i = 0; i < dest->nmodes; ++i) {
        result = sptNewIndexVector(&dest->inds[i], src->inds[i].len, src->inds[i].len);
        spt_CheckError(result, "SpTns Copy", NULL);
    }
    result = sptNewValueVector(&dest->values, src->values.len, src->values.len);
    spt_CheckError(result, "SpTns Copy", NULL);
    return 0;
}


/**
 * Copy a sparse tensor without allocation.
 * @param[out] dest a pointer to an uninitialized sparse tensor
 * @param[in]  src  a pointer to a valid sparse tensor
 */
int sptCopySparseTensorCopyOnly(sptSparseTensor *dest, const sptSparseTensor *src) {
    sptIndex i;
    int result;
    for(i = 0; i < dest->nmodes; ++i) {
        result = sptCopyIndexVectorCopyOnly(&dest->inds[i], &src->inds[i]);
        spt_CheckError(result, "SpTns Copy", NULL);
    }
    result = sptCopyValueVectorCopyOnly(&dest->values, &src->values);
    spt_CheckError(result, "SpTns Copy", NULL);
    return 0;
}

/**
 * Release any memory the sparse tensor is holding
 * @param tsr the tensor to release
 */
void sptFreeSparseTensor(sptSparseTensor *tsr) {
    sptIndex i;
    for(i = 0; i < tsr->nmodes; ++i) {
        sptFreeIndexVector(&tsr->inds[i]);
    }
    free(tsr->sortorder);
    free(tsr->ndims);
    free(tsr->inds);
    sptFreeValueVector(&tsr->values);
    tsr->nmodes = 0;
}


double SparseTensorFrobeniusNormSquared(sptSparseTensor const * const spten) 
{
  double norm = 0;
  sptValue const * const restrict vals = spten->values.data;
  
#ifdef PASTA_USE_OPENMP
  #pragma omp parallel for reduction(+:norm)
#endif
  for(sptNnzIndex n=0; n < spten->nnz; ++n) {
    norm += vals[n] * vals[n];
  }
  return norm;
}


int spt_DistSparseTensor(sptSparseTensor * tsr,
    int nthreads,
    sptNnzIndex * dist_nnzs,
    sptIndex * dist_nrows) {

    sptNnzIndex global_nnz = tsr->nnz;
    sptNnzIndex aver_nnz = global_nnz / nthreads;
    memset(dist_nnzs, 0, (nthreads + 1) * sizeof(sptNnzIndex));
    memset(dist_nrows, 0, nthreads*sizeof(sptIndex));

    sptSparseTensorSortIndex(tsr, 0, tsr->nnz, 0);
    sptIndex * ind0 = tsr->inds[0].data;

    int tid = 0;
    sptNnzIndex tmp_nnzs = 1;
    dist_nnzs[0] = 0;
    dist_nrows[0] = 1;
    for(sptNnzIndex x=1; x<global_nnz; ++x) {
        if(ind0[x] == ind0[x-1]) {
            ++ tmp_nnzs;
        } else if (ind0[x] > ind0[x-1]) {
            if(tmp_nnzs < aver_nnz || tid == nthreads-1) {
                ++ tmp_nnzs;
                ++ dist_nrows[tid];
            } else {
                ++ tid;
                dist_nnzs[tid] = dist_nnzs[tid - 1] + tmp_nnzs;
                ++ dist_nrows[tid];
                tmp_nnzs = 1;
            }
        } else {
            spt_CheckError(SPTERR_VALUE_ERROR, "SpTns Dist", "tensor unsorted on mode-0");
        }
    }
    dist_nnzs[nthreads] = global_nnz;

    return 0;

}


int spt_DistSparseTensorFixed(sptSparseTensor * tsr,
    int nthreads,
    sptIndex * dist_nrows,
    sptNnzIndex * dist_nnzs) 
{

    memset(dist_nnzs, 0, (nthreads + 1) * sizeof(sptNnzIndex));

    sptSparseTensorSortIndex(tsr, 0, tsr->nnz, 0);
    sptIndex * ind0 = tsr->inds[0].data;

    int tid = 0;
    sptIndex tmp_nrows = 1;
    sptNnzIndex tmp_nnzs = 1;
    dist_nnzs[0] = 0;
    for(sptNnzIndex x=1; x<tsr->nnz; ++x) {
        if(ind0[x] == ind0[x-1]) {
            ++ tmp_nnzs;
        } else if (ind0[x] > ind0[x-1]) {
            if(tmp_nrows + 1 <= dist_nrows[tid]) {
                ++ tmp_nnzs;
                ++ tmp_nrows;
            } else {
                ++ tid;
                dist_nnzs[tid] = dist_nnzs[tid - 1] + tmp_nnzs;
                tmp_nrows = 1;
                tmp_nnzs = 1;
            }
        } else {
            spt_CheckError(SPTERR_VALUE_ERROR, "SpTns Dist", "tensor unsorted on mode-0");
        }
    }
    if(tid < nthreads - 1) {
        ++ tid;
        dist_nnzs[tid] = dist_nnzs[tid - 1] + tmp_nnzs;
        sptAssert(dist_nnzs[tid] == tsr->nnz);
        while(tid < nthreads) {
            ++ tid;
            dist_nnzs[tid] = dist_nnzs[tid - 1];
        }
    }
    dist_nnzs[nthreads] = tsr->nnz;

    return 0;
}



/**
 * Shuffle all indices.
 *
 * @param[in] tsr tensor to be shuffled
 * @param[out] map_inds is the renumbering mapping
 *
 */
void sptSparseTensorShuffleIndices(sptSparseTensor *tsr, sptIndex ** map_inds) {
    /* Renumber nonzero elements */
    sptIndex tmp_ind;
    for(sptNnzIndex z = 0; z < tsr->nnz; ++z) {
        for(sptIndex m = 0; m < tsr->nmodes; ++m) {
            tmp_ind = tsr->inds[m].data[z];
            tsr->inds[m].data[z] = map_inds[m][tmp_ind];
        }
    }
    
}
