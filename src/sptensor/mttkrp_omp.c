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

int sptOmpMTTKRP_3D(sptSparseTensor const * const X,
    sptMatrix * mats[],     // mats[nmodes] as temporary space.
    sptIndex const mats_order[],    // Correspond to the mode order of X.
    sptIndex const mode,
    const int tk);

/**
 * OpenMP parallelized Matriced sparse tensor times a sequence of dense matrix Khatri-Rao products (MTTKRP) on a specified mode
 * @param[out] mats[nmodes]    the result of MTTKRP, a dense matrix, with size
 * ndims[mode] * R
 * @param[in]  X    the sparse tensor input X
 * @param[in]  mats    (N+1) dense matrices, with mats[nmodes] as temporary
 * @param[in]  mats_order    the order of the Khatri-Rao products
 * @param[in]  mode   the mode on which the MTTKRP is performed
 * @param[in]  scratch an temporary array to store intermediate results, space assigned before this function
 *
 * This function uses support arbitrary-order sparse tensors with Khatri-Rao
 * products of dense factor matrices, the output is the updated dense matrix for the "mode".
 * In this version, a large scratch is used to maximize parallelism. (To be optimized)
 */
int sptOmpMTTKRP(sptSparseTensor const * const X,
    sptMatrix * mats[],     // mats[nmodes] as temporary space.
    sptIndex const mats_order[],    // Correspond to the mode order of X.
    sptIndex const mode,
    const int tk)
{
    sptIndex const nmodes = X->nmodes;

    if(nmodes == 3) {
        sptAssert(sptOmpMTTKRP_3D(X, mats, mats_order, mode, tk) == 0);
        return 0;
    }

    sptNnzIndex const nnz = X->nnz;
    sptIndex const * const ndims = X->ndims;
    sptValue const * const vals = X->values.data;
    sptIndex const stride = mats[0]->stride;

    /* Check the mats. */
    for(sptIndex i=0; i<nmodes; ++i) {
        if(mats[i]->ncols != mats[nmodes]->ncols) {
            spt_CheckError(SPTERR_SHAPE_MISMATCH, "CPU  SpTns MTTKRP", "mats[i]->cols != mats[nmodes]->ncols");
        }
        if(mats[i]->nrows != ndims[i]) {
            spt_CheckError(SPTERR_SHAPE_MISMATCH, "CPU  SpTns MTTKRP", "mats[i]->nrows != ndims[i]");
        }
    }

    sptIndex const tmpI = mats[mode]->nrows;
    sptIndex const R = mats[mode]->ncols;
    sptIndex const * const mode_ind = X->inds[mode].data;
    sptValue * const restrict mvals = mats[nmodes]->values;
    memset(mvals, 0, tmpI*stride*sizeof(sptValue));

    #pragma omp parallel for schedule(static) num_threads(tk)
    for(sptNnzIndex x=0; x<nnz; ++x) {
        sptValueVector scratch;  // Temporary array
        sptNewValueVector(&scratch, R, R);
        sptConstantValueVector(&scratch, 0);

        sptIndex times_mat_index = mats_order[1];
        sptMatrix * times_mat = mats[times_mat_index];
        sptIndex * times_inds = X->inds[times_mat_index].data;
        sptIndex tmp_i = times_inds[x];
        sptValue const entry = vals[x];
        #pragma omp simd
        for(sptIndex r=0; r<R; ++r) {
            scratch.data[r] = entry * times_mat->values[tmp_i * stride + r];
        }

        for(sptIndex i=2; i<nmodes; ++i) {
            times_mat_index = mats_order[i];
            times_mat = mats[times_mat_index];
            times_inds = X->inds[times_mat_index].data;
            tmp_i = times_inds[x];

            #pragma omp simd
            for(sptIndex r=0; r<R; ++r) {
                scratch.data[r] *= times_mat->values[tmp_i * stride + r];
            }
        }

        sptIndex const mode_i = mode_ind[x];
        sptValue * const restrict mvals_row = mvals + mode_i * stride;
        for(sptIndex r=0; r<R; ++r) {
            #pragma omp atomic update
            mvals_row[r] += scratch.data[r];
        }

        sptFreeValueVector(&scratch);
    }   // End loop nnzs

    return 0;
}


int sptOmpMTTKRP_3D(sptSparseTensor const * const X,
    sptMatrix * mats[],     // mats[nmodes] as temporary space.
    sptIndex const mats_order[],    // Correspond to the mode order of X.
    sptIndex const mode,
    const int tk) 
{
    sptIndex const nmodes = X->nmodes;
    sptNnzIndex const nnz = X->nnz;
    sptIndex const * const ndims = X->ndims;
    sptValue const * const restrict vals = X->values.data;
    sptIndex const stride = mats[0]->stride;

    /* Check the mats. */
    sptAssert(nmodes ==3);
    for(sptIndex i=0; i<nmodes; ++i) {
        if(mats[i]->ncols != mats[nmodes]->ncols) {
            spt_CheckError(SPTERR_SHAPE_MISMATCH, "CPU  SpTns MTTKRP", "mats[i]->cols != mats[nmodes]->ncols");
        }
        if(mats[i]->nrows != ndims[i]) {
            spt_CheckError(SPTERR_SHAPE_MISMATCH, "CPU  SpTns MTTKRP", "mats[i]->nrows != ndims[i]");
        }
    }

    sptIndex const tmpI = mats[mode]->nrows;
    sptIndex const R = mats[mode]->ncols;
    sptIndex const * const restrict mode_ind = X->inds[mode].data;
    sptValue * const restrict mvals = mats[nmodes]->values;
    memset(mvals, 0, tmpI*stride*sizeof(sptValue));

    sptIndex times_mat_index_1 = mats_order[1];
    sptMatrix * restrict times_mat_1 = mats[times_mat_index_1];
    sptIndex * restrict times_inds_1 = X->inds[times_mat_index_1].data;
    sptIndex times_mat_index_2 = mats_order[2];
    sptMatrix * restrict times_mat_2 = mats[times_mat_index_2];
    sptIndex * restrict times_inds_2 = X->inds[times_mat_index_2].data;

    #pragma omp parallel for schedule(static) num_threads(tk)
    for(sptNnzIndex x=0; x<nnz; ++x) {
        sptIndex mode_i = mode_ind[x];
        sptValue * const restrict mvals_row = mvals + mode_i * stride;
        sptIndex tmp_i_1 = times_inds_1[x];
        sptIndex tmp_i_2 = times_inds_2[x];
        sptValue entry = vals[x];

        for(sptIndex r=0; r<R; ++r) {
            #pragma omp atomic update
            mvals_row[r] += entry * times_mat_1->values[tmp_i_1 * stride + r] * times_mat_2->values[tmp_i_2 * stride + r];
        }
    }

    return 0;
}
