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

#include <assert.h>
#include <pasta.h>

static int spt_SparseTensorCompare(const sptSparseTensorHiCOOGeneral *tsr1, sptNnzIndex bloc1, sptNnzIndex eloc1, const sptSparseTensorHiCOOGeneral *tsr2, sptNnzIndex bloc2, sptNnzIndex eloc2) 
{
    sptIndex i;
    sptIndex index1, index2;
    sptBlockIndex bidx1, bidx2;
    sptElementIndex eidx1, eidx2;
    assert(tsr1->nmodes == tsr2->nmodes);
    assert(tsr1->ncmodes == tsr2->ncmodes);
    for(i = 0; i < tsr1->ncmodes; ++i) {
        bidx1 = tsr1->binds[i].data[bloc1];
        bidx2 = tsr2->binds[i].data[bloc2];
        eidx1 = tsr1->einds[i].data[eloc1];
        eidx2 = tsr2->einds[i].data[eloc2];

        if(bidx1 < bidx2) {
            return -1;
        } else if(bidx1 > bidx2) {
            return 1;
        } else {
            if(eidx1 < eidx2) {
                return -1;
            } else if(eidx1 > eidx2) {
                return 1;
            }
        }
    }
    return 0;
}


/**
 * Set the fibers of a gHiCOO sparse tensor.
 * without setting any actual data
 * @param[out] dest     a pointer to an initialized semi sparse tensor
 * @param[out] fiberidx a vector to store the starting position of each fiber, should be uninitialized
 * @param[in]  ref      a pointer to a valid sparse tensor
 */
int sptSparseTensorSetFibersHiCOO (
    sptNnzIndexVector *bptr,
    sptNnzIndexVector *fiberidx,
    sptSparseTensorHiCOOGeneral *ref) 
{
    sptNnzIndex lastidx = ref->nnz;
    sptNnzIndex lastbidx = ref->bptr.len - 1;
    sptNnzIndex tmp_nnz = 0;
    int result;

    result = sptNewNnzIndexVector(fiberidx, 0, 0);
    spt_CheckError(result, "SpTns SetIndices", NULL);
    result = sptNewNnzIndexVector(bptr, 0, 0);
    spt_CheckError(result, "SpTns SetIndices", NULL);
    
    result = sptAppendNnzIndexVector(bptr, 0);
    for(sptNnzIndex b = 0; b < ref->bptr.len - 1; ++ b) {
        sptNnzIndex b_begin = ref->bptr.data[b];
        sptNnzIndex b_end = ref->bptr.data[b + 1];
        for(sptNnzIndex z = b_begin; z < b_end; ++z) {

            if(lastidx == ref->nnz || spt_SparseTensorCompare(ref, lastbidx, lastidx, ref, b, z) != 0) 
            {
                lastidx = z;
                lastbidx = b;
                ++ tmp_nnz;
                result = sptAppendNnzIndexVector(fiberidx, z);
                spt_CheckError(result, "SpTns SetIndices", NULL);
            }
        }   // End elements in a block
        result = sptAppendNnzIndexVector(bptr, tmp_nnz);
        spt_CheckError(result, "SpTns SetIndices", NULL);
    }   // End blocks

    result = sptAppendNnzIndexVector(fiberidx, ref->nnz);
    spt_CheckError(result, "SpTns SetIndices", NULL);
    
    return 0;
}



/**
 * Set the indices of a sparse tensor.
 * without setting any actual data
 * @param[out] dest     a pointer to an initialized semi sparse tensor
 * @param[out] fiberidx a vector to store the starting position of each fiber, should be uninitialized
 * @param[in]  ref      a pointer to a valid sparse tensor
 */
int sptSparseTensorSetIndicesHiCOO(
    sptSparseTensorHiCOO *dest,
    sptNnzIndexVector *fiberidx,
    sptSparseTensorHiCOOGeneral *ref) 
{
    int result;
    assert(dest->nmodes == ref->nmodes - 1);

    /* binds is the same with ref gHiCOO tensor's */
    for(sptIndex m = 0; m < ref->ncmodes; ++m) {
        result = sptCopyBlockIndexVectorCopyOnly(&dest->binds[m], &ref->binds[m]);
        spt_CheckError(result, "SpTns SetIndices", NULL);
    }
    
#ifdef PASTA_USE_OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for(sptNnzIndex f = 0; f < fiberidx->len - 1; ++f) {
        sptNnzIndex z = fiberidx->data[f];
        for(sptIndex m = 0; m < ref->ncmodes; ++m) {
            dest->einds[m].data[f] = ref->einds[m].data[z];
        }
    }
    
    return 0;
}
