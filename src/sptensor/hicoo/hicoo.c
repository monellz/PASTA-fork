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

#include <ParTI.h>

/**
 * Create a new sparse tensor in HiCOO format
 * @param hitsr    a pointer to an uninitialized sparse tensor
 * @param nmodes number of modes the tensor will have
 * @param ndims  the dimension of each mode the tensor will have
 * @param nnz number of nonzeros the tensor will have
 */
int sptNewSparseTensorHiCOO(
    sptSparseTensorHiCOO *hitsr, 
    const sptIndex nmodes, 
    const sptIndex ndims[],
    const sptNnzIndex nnz,
    const sptElementIndex sb_bits,
    const sptElementIndex sk_bits,
    const sptElementIndex sc_bits)
{
    sptIndex i;
    int result;

    hitsr->nmodes = nmodes;
    hitsr->sortorder = malloc(nmodes * sizeof hitsr->sortorder[0]);
    for(i = 0; i < nmodes; ++i) {
        hitsr->sortorder[i] = i;
    }
    hitsr->ndims = malloc(nmodes * sizeof *hitsr->ndims);
    spt_CheckOSError(!hitsr->ndims, "HiSpTns New");
    memcpy(hitsr->ndims, ndims, nmodes * sizeof *hitsr->ndims);
    hitsr->nnz = nnz;

    /* Parameters */
    hitsr->sb_bits = sb_bits; // block size by nnz
    hitsr->sk_bits = sk_bits; // kernel size by nnz
    hitsr->sc_bits = sc_bits; // chunk size by blocks
    sptIndex sk = (sptIndex)pow(2, sk_bits);

    hitsr->kschr = (sptIndexVector**)malloc(nmodes * sizeof *hitsr->kschr);
    spt_CheckOSError(!hitsr->kschr, "HiSpTns New");
    for(sptIndex m = 0; m < nmodes; ++m) {
        sptIndex kernel_ndim = (ndims[m] + sk - 1)/sk;
        hitsr->kschr[m] = (sptIndexVector*)malloc(kernel_ndim * sizeof(*(hitsr->kschr[m])));
        spt_CheckOSError(!hitsr->kschr[m], "HiSpTns New");
        for(sptIndex i = 0; i < kernel_ndim; ++i) {
            result = sptNewIndexVector(&(hitsr->kschr[m][i]), 0, 0);
            spt_CheckError(result, "HiSpTns New", NULL);
        }
    }
    hitsr->nkiters = (sptIndex*)malloc(nmodes * sizeof *hitsr->nkiters);

    result = sptNewNnzIndexVector(&hitsr->kptr, 0, 0);
    spt_CheckError(result, "HiSpTns New", NULL);
    result = sptNewNnzIndexVector(&hitsr->cptr, 0, 0);
    spt_CheckError(result, "HiSpTns New", NULL);

    /* Balanced structures */
    hitsr->kschr_balanced = (sptIndexVector**)malloc(nmodes * sizeof *hitsr->kschr_balanced);
    spt_CheckOSError(!hitsr->kschr_balanced, "HiSpTns New");
    for(sptIndex m = 0; m < nmodes; ++m) {
        sptIndex kernel_ndim = (ndims[m] + sk - 1)/sk;
        hitsr->kschr_balanced[m] = (sptIndexVector*)malloc(kernel_ndim * sizeof(*(hitsr->kschr_balanced[m])));
        spt_CheckOSError(!hitsr->kschr_balanced[m], "HiSpTns New");
        for(sptIndex i = 0; i < kernel_ndim; ++i) {
            result = sptNewIndexVector(&(hitsr->kschr_balanced[m][i]), 0, 0);
            spt_CheckError(result, "HiSpTns New", NULL);
        }
    }
    hitsr->kschr_balanced_pos = (sptIndexVector**)malloc(nmodes * sizeof *hitsr->kschr_balanced_pos);
    spt_CheckOSError(!hitsr->kschr_balanced_pos, "HiSpTns New");
    for(sptIndex m = 0; m < nmodes; ++m) {
        sptIndex kernel_ndim = (ndims[m] + sk - 1)/sk;
        hitsr->kschr_balanced_pos[m] = (sptIndexVector*)malloc(kernel_ndim * sizeof(*(hitsr->kschr_balanced_pos[m])));
        spt_CheckOSError(!hitsr->kschr_balanced_pos[m], "HiSpTns New");
        for(sptIndex i = 0; i < kernel_ndim; ++i) {
            result = sptNewIndexVector(&(hitsr->kschr_balanced_pos[m][i]), 0, 0);
            spt_CheckError(result, "HiSpTns New", NULL);
        }
    }
    hitsr->nkpars = (sptIndex*)malloc(nmodes * sizeof(sptIndex));
    spt_CheckOSError(!hitsr->nkpars, "HiSpTns New");
    hitsr->kschr_rest = (sptIndexVector*)malloc(nmodes * sizeof *hitsr->kschr_rest);
    spt_CheckOSError(!hitsr->kschr_rest, "HiSpTns New");
    for(sptIndex m = 0; m < nmodes; ++m) {
        result = sptNewIndexVector(&(hitsr->kschr_rest[m]), 0, 0);
        spt_CheckError(result, "HiSpTns New", NULL);
    }
    result = sptNewNnzIndexVector(&hitsr->knnzs, 0, 0);
    spt_CheckError(result, "HiSpTns New", NULL);

    result = sptNewNnzIndexVector(&hitsr->bptr, 0, 0);
    spt_CheckError(result, "HiSpTns New", NULL);
    hitsr->binds = malloc(nmodes * sizeof *hitsr->binds);
    spt_CheckOSError(!hitsr->binds, "HiSpTns New");
    for(i = 0; i < nmodes; ++i) {
        result = sptNewBlockIndexVector(&hitsr->binds[i], 0, 0);
        spt_CheckError(result, "HiSpTns New", NULL);
    }

    hitsr->einds = malloc(nmodes * sizeof *hitsr->einds);
    spt_CheckOSError(!hitsr->einds, "HiSpTns New");
    for(i = 0; i < nmodes; ++i) {
        result = sptNewElementIndexVector(&hitsr->einds[i], 0, 0);
        spt_CheckError(result, "HiSpTns New", NULL);
    }
    result = sptNewValueVector(&hitsr->values, 0, 0);
    spt_CheckError(result, "HiSpTns New", NULL);


    return 0;
}


/**
 * Copy a sparse tensor
 * @param[out] dest a pointer to an uninitialized sparse tensor
 * @param[in]  src  a pointer to a valid sparse tensor
 */
int sptCopySparseTensorHiCOO(sptSparseTensorHiCOO *dest, const sptSparseTensorHiCOO *src) 
{
    sptIndex i;
    int result;
    dest->nmodes = src->nmodes;
    dest->sortorder = malloc(src->nmodes * sizeof src->sortorder[0]);
    memcpy(dest->sortorder, src->sortorder, src->nmodes * sizeof src->sortorder[0]);
    dest->ndims = malloc(dest->nmodes * sizeof *dest->ndims);
    spt_CheckOSError(!dest->ndims, "SpTns Copy");
    memcpy(dest->ndims, src->ndims, src->nmodes * sizeof *src->ndims);
    dest->nnz = src->nnz;

    dest->sb_bits = src->sb_bits;
    dest->sk_bits = src->sk_bits;
    dest->sc_bits = src->sc_bits;

    // if (src->kptr.len > 0) {
    //     result = sptCopyNnzIndexVector(&dest->kptr, &src->kptr);
    //     spt_CheckError(result, "HiSpTns Copy", NULL);
    //     if(src->kschr != NULL) {
    //         // TODO
    //     }
    //     if(src->nkiters != NULL) {
    //         // TODO
    //     }
    // }
    // if (src->cptr.len > 0) {
    //     result = sptCopyNnzIndexVector(&dest->cptr, &src->cptr);
    //     spt_CheckError(result, "HiSpTns Copy", NULL);
    // }

    /* Ignore balanced scheduler */

    result = sptCopyNnzIndexVector(&dest->bptr, &src->bptr);
    spt_CheckError(result, "HiSpTns Copy", NULL);
    dest->binds = malloc(dest->nmodes * sizeof *dest->binds);
    for(i = 0; i < dest->nmodes; ++i) {
        result = sptCopyBlockIndexVector(&(dest->binds[i]), &(src->binds[i]));
        spt_CheckError(result, "HiSpTns Copy", NULL);
        
    }
    dest->einds = malloc(dest->nmodes * sizeof *dest->einds);
    for(i = 0; i < dest->nmodes; ++i) {
        result = sptCopyElementIndexVector(&(dest->einds[i]), &(src->einds[i]));
        spt_CheckError(result, "HiSpTns Copy", NULL);
    }

    result = sptCopyValueVector(&dest->values, &src->values, 1);
    spt_CheckError(result, "HiSpTns Copy", NULL);

    return 0;
}

/**
 * Release any memory the HiCOO sparse tensor is holding
 * @param hitsr the tensor to release
 */
void sptFreeSparseTensorHiCOO(sptSparseTensorHiCOO *hitsr)
{
    sptIndex i;
    sptIndex nmodes = hitsr->nmodes;
    sptIndex sk = (sptIndex)pow(2, hitsr->sk_bits);

    sptFreeNnzIndexVector(&hitsr->bptr);
    for(i = 0; i < nmodes; ++i) {
        sptFreeBlockIndexVector(&hitsr->binds[i]);
        sptFreeElementIndexVector(&hitsr->einds[i]);
    }
    free(hitsr->binds);
    free(hitsr->einds);
    sptFreeValueVector(&hitsr->values);

    hitsr->nmodes = 0;
    hitsr->nnz = 0;
    hitsr->sb_bits = 0;

    free(hitsr->sortorder);
    free(hitsr->ndims);
}


double SparseTensorFrobeniusNormSquaredHiCOO(sptSparseTensorHiCOO const * const hitsr) 
{
  double norm = 0;
  sptValue const * const restrict vals = hitsr->values.data;

#ifdef PARTI_USE_OPENMP
  #pragma omp parallel for reduction(+:norm)
#endif
  for(size_t n=0; n < hitsr->nnz; ++n) {
    norm += vals[n] * vals[n];
  }
  return norm;
}