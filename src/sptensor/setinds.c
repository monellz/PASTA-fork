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
#include "../sptensor/sptensor.h"

static int spt_SparseTensorCompareExceptMode(const sptSparseTensor *tsr1, sptNnzIndex ind1, const sptSparseTensor *tsr2, sptNnzIndex ind2, sptIndex mode) {
    sptIndex i;
    sptIndex eleind1, eleind2;
    assert(tsr1->nmodes == tsr2->nmodes);
    for(i = 0; i < tsr1->nmodes; ++i) {
        if(i != mode) {
            eleind1 = tsr1->inds[i].data[ind1];
            eleind2 = tsr2->inds[i].data[ind2];
            if(eleind1 < eleind2) {
                return -1;
            } else if(eleind1 > eleind2) {
                return 1;
            }
        }
    }
    return 0;
}


/**
 * Set the fibers of a sparse tensor.
 * @param[out] dest     a pointer to an initialized sparse tensor
 * @param[out] fiberidx a vector to store the starting position of each fiber, should be uninitialized
 * @param[in]  ref      a pointer to a valid sparse tensor
 */
int sptSparseTensorSetFibers(
    sptNnzIndexVector *fiberidx,
    sptIndex mode,
    sptSparseTensor *ref
) {
    sptNnzIndex lastidx = ref->nnz;
    sptNnzIndex i;
    sptIndex m;
    int result;

    // sptSparseTensorSortIndexAtMode(ref, mode, 0);
    result = sptNewNnzIndexVector(fiberidx, 0, 0);
    spt_CheckError(result, "SpTns SetFibers", NULL);
    for(i = 0; i < ref->nnz; ++i) {
        if(lastidx == ref->nnz || spt_SparseTensorCompareExceptMode(ref, lastidx, ref, i, mode) != 0) 
        {
            lastidx = i;
            result = sptAppendNnzIndexVector(fiberidx, i);
            spt_CheckError(result, "SpTns SetFibers", NULL);
        }
    }
    result = sptAppendNnzIndexVector(fiberidx, ref->nnz);
    spt_CheckError(result, "SpTns SetFibers", NULL);

    return 0;
}


/**
 * Set the indices of a sparse tensor.
 * @param[out] dest     a pointer to an initialized sparse tensor
 * @param[out] fiberidx a vector to store the starting position of each fiber, should be uninitialized
 * @param[in]  ref      a pointer to a valid sparse tensor
 */
int sptSparseTensorSetIndices(
    sptSparseTensor *dest,
    sptNnzIndexVector *fiberidx,
    sptIndex mode,
    sptSparseTensor *ref
) {
    int result;
    assert(dest->nmodes == ref->nmodes - 1);

#ifdef PASTA_USE_OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for(sptNnzIndex f = 0; f < fiberidx->len - 1; ++f) {
        sptNnzIndex i = fiberidx->data[f];
        for(sptIndex m = 0; m < ref->nmodes; ++m) {
            if(m < mode) {
                dest->inds[m].data[f] = ref->inds[m].data[i];
            } else if(m > mode) {
                dest->inds[m - 1].data[f] = ref->inds[m].data[i];
            }
        }
    }

    return 0;
}