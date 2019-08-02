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


/**
 * Create a new semi-sparse tensor in HiCOO format
 * @param histsr    a pointer to an uninitialized sparse tensor
 * @param nmodes number of modes the tensor will have
 * @param ndims  the dimension of each mode the tensor will have
 * @param nnz number of nonzeros the tensor will have
 */
int sptNewSemiSparseTensorHiCOO(
    sptSemiSparseTensorHiCOO *histsr, 
    const sptIndex nmodes, 
    const sptIndex ndims[],
    const sptIndex mode,
    const sptElementIndex sb_bits)
{
    sptIndex ncmodes = nmodes - 1;
    sptIndex i;
    int result;
    if(nmodes < 2) {
        spt_CheckError(SPTERR_SHAPE_MISMATCH, "SspTnsHiCOO New", "nmodes < 2");
    }

    histsr->nmodes = nmodes;
    histsr->ndims = malloc(nmodes * sizeof *histsr->ndims);
    spt_CheckOSError(!histsr->ndims, "HiSpTns New");
    memcpy(histsr->ndims, ndims, nmodes * sizeof *histsr->ndims);
    histsr->mode = mode;
    histsr->nnz = 0;

    /* Parameters */
    histsr->sb_bits = sb_bits; // block size by nnz

    result = sptNewNnzIndexVector(&histsr->bptr, 0, 0);
    spt_CheckError(result, "HiSpTns New", NULL);
    histsr->binds = malloc(ncmodes * sizeof *histsr->binds);
    spt_CheckOSError(!histsr->binds, "HiSpTns New");
    for(i = 0; i < ncmodes; ++i) {
        result = sptNewBlockIndexVector(&histsr->binds[i], 0, 0);
        spt_CheckError(result, "HiSpTns New", NULL);
    }

    histsr->einds = malloc(ncmodes * sizeof *histsr->einds);
    spt_CheckOSError(!histsr->einds, "HiSpTns New");
    for(i = 0; i < ncmodes; ++i) {
        result = sptNewElementIndexVector(&histsr->einds[i], 0, 0);
        spt_CheckError(result, "HiSpTns New", NULL);
    }
    result = sptNewMatrix(&histsr->values, 0, histsr->ndims[mode]);
    spt_CheckError(result, "HiSpTns New", NULL);

    return 0;
}


/**
 * Create a new semi-sparse tensor in HiCOO format
 * @param histsr    a pointer to an uninitialized sparse tensor
 * @param nmodes number of modes the tensor will have
 * @param ndims  the dimension of each mode the tensor will have
 * @param nnz number of nonzeros the tensor will have
 */
int sptNewSemiSparseTensorHiCOOWithBptr(
    sptSemiSparseTensorHiCOO *histsr, 
    const sptIndex nmodes, 
    const sptIndex ndims[],
    const sptNnzIndex nfibers,
    const sptIndex mode,
    const sptElementIndex sb_bits,
    sptNnzIndexVector * bptr)
{
    sptIndex ncmodes = nmodes - 1;
    sptIndex i;
    int result;
    sptNnzIndex nb = bptr->len - 1;
    if(nmodes < 2) {
        spt_CheckError(SPTERR_SHAPE_MISMATCH, "SspTnsHiCOO New", "nmodes < 2");
    }

    histsr->nmodes = nmodes;
    histsr->ndims = malloc(nmodes * sizeof *histsr->ndims);
    spt_CheckOSError(!histsr->ndims, "HiSpTns New");
    memcpy(histsr->ndims, ndims, nmodes * sizeof *histsr->ndims);
    histsr->mode = mode;
    histsr->nnz = nfibers;

    /* Parameters */
    histsr->sb_bits = sb_bits; // block size by nnz

    /* Soft copy bptr */
    histsr->bptr.len = bptr->len;
    histsr->bptr.cap = bptr->cap;
    histsr->bptr.data = bptr->data;

    histsr->binds = malloc(ncmodes * sizeof *histsr->binds);
    spt_CheckOSError(!histsr->binds, "HiSpTns New");
    for(i = 0; i < ncmodes; ++i) {
        result = sptNewBlockIndexVector(&histsr->binds[i], nb, nb);
        spt_CheckError(result, "HiSpTns New", NULL);
    }

    histsr->einds = malloc(ncmodes * sizeof *histsr->einds);
    spt_CheckOSError(!histsr->einds, "HiSpTns New");
    for(i = 0; i < ncmodes; ++i) {
        result = sptNewElementIndexVector(&histsr->einds[i], nfibers, nfibers);
        spt_CheckError(result, "HiSpTns New", NULL);
    }
    result = sptNewMatrix(&histsr->values, nfibers, histsr->ndims[mode]);
    spt_CheckError(result, "HiSpTns New", NULL);

    return 0;
}


/**
 * Release any memory the HiCOO semi-sparse tensor is holding
 * @param histsr the tensor to release
 */
void sptFreeSemiSparseTensorHiCOO(sptSemiSparseTensorHiCOO *histsr)
{
    sptIndex i;
    sptIndex nmodes = histsr->nmodes;

    sptFreeNnzIndexVector(&histsr->bptr);
    for(i = 0; i < nmodes - 1; ++i) {
        sptFreeBlockIndexVector(&histsr->binds[i]);
        sptFreeElementIndexVector(&histsr->einds[i]);
    }
    free(histsr->binds);
    free(histsr->einds);
    sptFreeMatrix(&histsr->values);

    histsr->nmodes = 0;
    histsr->nnz = 0;
    histsr->sb_bits = 0;

    free(histsr->ndims);
}