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
    const sptElementIndex sb_bits)
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


/**
 * Create a new sparse tensor in HiCOO-General format
 * @param hitsr    a pointer to an uninitialized sparse tensor
 * @param nmodes number of modes the tensor will have
 * @param ndims  the dimension of each mode the tensor will have
 * @param nnz number of nonzeros the tensor will have
 */
int sptNewSparseTensorHiCOOGeneral(
    sptSparseTensorHiCOOGeneral *hitsr, 
    const sptIndex nmodes, 
    const sptIndex ndims[],
    const sptNnzIndex nnz,
    const sptElementIndex sb_bits,
    const sptIndex ncmodes,
    const sptIndex *flags)
{
    sptIndex i;
    int result;

    hitsr->nmodes = nmodes;
    hitsr->ncmodes = ncmodes;
    hitsr->sortorder = malloc(nmodes * sizeof hitsr->sortorder[0]);
    sptIndex fm = 0;
    for(i = 0; i < nmodes; ++i) {
        if(flags[i] == 1) {
            hitsr->sortorder[fm] = i;
            ++ fm;
        }
    }
    for(i = 0; i < nmodes; ++i) {
        if(flags[i] == 0) {
            hitsr->sortorder[fm] = i;
            ++ fm;
        }
    }
    sptAssert(fm == nmodes);
    
    hitsr->ndims = malloc(nmodes * sizeof *hitsr->ndims);
    spt_CheckOSError(!hitsr->ndims, "HiSpTnsGen New");
    memcpy(hitsr->ndims, ndims, nmodes * sizeof *hitsr->ndims);
    hitsr->flags = malloc(nmodes * sizeof *hitsr->flags);
    spt_CheckOSError(!hitsr->flags, "HiSpTnsGen New");
    memcpy(hitsr->flags, flags, nmodes * sizeof *hitsr->flags);
    hitsr->nnz = nnz;

    /* Parameters */
    hitsr->sb_bits = sb_bits; // block size by nnz

    result = sptNewNnzIndexVector(&hitsr->bptr, 0, 0);
    spt_CheckError(result, "HiSpTnsGen New", NULL);
    hitsr->binds = malloc( ncmodes * sizeof *hitsr->binds);
    spt_CheckOSError(!hitsr->binds, "HiSpTnsGen New");
    for(i = 0; i < ncmodes; ++i) {
        result = sptNewBlockIndexVector(&hitsr->binds[i], 0, 0);
        spt_CheckError(result, "HiSpTnsGen New", NULL);
    }

    hitsr->einds = malloc( ncmodes * sizeof *hitsr->einds);
    spt_CheckOSError(!hitsr->einds, "HiSpTnsGen New");
    for(i = 0; i < ncmodes; ++i) {
        result = sptNewElementIndexVector(&hitsr->einds[i], 0, 0);
        spt_CheckError(result, "HiSpTnsGen New", NULL);
    }

    hitsr->inds = malloc( (nmodes - ncmodes) * sizeof *hitsr->inds);
    spt_CheckOSError(!hitsr->inds, "HiSpTnsGen New");
    for(i = 0; i < nmodes - ncmodes; ++i) {
        result = sptNewIndexVector(&hitsr->inds[i], 0, 0);
        spt_CheckError(result, "HiSpTnsGen New", NULL);
    }

    result = sptNewValueVector(&hitsr->values, 0, 0);
    spt_CheckError(result, "HiSpTnsGen New", NULL);


    return 0;
}


/**
 * Release any memory the HiCOO-General sparse tensor is holding
 * @param hitsr the tensor to release
 */
void sptFreeSparseTensorHiCOOGeneral(sptSparseTensorHiCOOGeneral *hitsr)
{
    sptIndex i;
    sptIndex nmodes = hitsr->nmodes;

    sptFreeNnzIndexVector(&hitsr->bptr);
    for(i = 0; i < hitsr->ncmodes; ++i) {
        sptFreeBlockIndexVector(&hitsr->binds[i]);
        sptFreeElementIndexVector(&hitsr->einds[i]);
    }
    for(i = 0; i < nmodes - hitsr->ncmodes; ++i) {
        sptFreeIndexVector(&hitsr->inds[i]);
    }
    free(hitsr->binds);
    free(hitsr->einds);
    free(hitsr->inds);
    sptFreeValueVector(&hitsr->values);

    hitsr->nmodes = 0;
    hitsr->ncmodes = 0;
    hitsr->nnz = 0;
    hitsr->sb_bits = 0;

    free(hitsr->sortorder);
    free(hitsr->ndims);
    free(hitsr->flags);
}
