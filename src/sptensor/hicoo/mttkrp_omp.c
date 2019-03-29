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
#include <omp.h>

#define CHUNKSIZE 1

int sptOmpMTTKRPHiCOO_3D(
    sptSparseTensorHiCOO const * const hitsr,
    sptMatrix * mats[],     // mats[nmodes] as temporary space.
    sptIndex const mats_order[],    // Correspond to the mode order of X.
    sptIndex const mode,
    const int nthreads);



/**
 * Matriced sparse tensor in HiCOO format times a sequence of dense matrix Khatri-Rao products (MTTKRP) on a specified mode
 * @param[out] mats[nmodes]    the result of MTTKRP, a dense matrix, with size
 * ndims[mode] * R
 * @param[in]  hitsr    the HiCOO sparse tensor input
 * @param[in]  mats    (N+1) dense matrices, with mats[nmodes] as temporary
 * @param[in]  mats_order    the order of the Khatri-Rao products
 * @param[in]  mode   the mode on which the MTTKRP is performed
 * @param[in]  scratch an temporary array to store intermediate results, space assigned before this function
 *
 * This function uses support arbitrary-order sparse tensors with Khatri-Rao
 * products of dense factor matrices, the output is the updated dense matrix for the "mode".
 */
int sptOmpMTTKRPHiCOO(
    sptSparseTensorHiCOO const * const hitsr,
    sptMatrix * mats[],     // mats[nmodes] as temporary space.
    sptIndex const mats_order[],    // Correspond to the mode order of X.
    sptIndex const mode,
    const int nthreads) 
{
    sptIndex const nmodes = hitsr->nmodes;

    if(nmodes == 3) {
        sptAssert(sptOmpMTTKRPHiCOO_3D(hitsr, mats, mats_order, mode, nthreads) == 0);
        return 0;
    }

    sptIndex const * const ndims = hitsr->ndims;
    sptValue const * const vals = hitsr->values.data;
    sptIndex const stride = mats[0]->stride;

    /* Check the mats. */
    for(sptIndex i=0; i<nmodes; ++i) {
        if(mats[i]->ncols != mats[nmodes]->ncols) {
            spt_CheckError(SPTERR_SHAPE_MISMATCH, "Omp HiSpTns MTTKRP", "mats[i]->cols != mats[nmodes]->ncols");
        }
        if(mats[i]->nrows != ndims[i]) {
            spt_CheckError(SPTERR_SHAPE_MISMATCH, "Omp HiSpTns MTTKRP", "mats[i]->nrows != ndims[i]");
        }
    }

    sptIndex const tmpI = mats[mode]->nrows;
    sptIndex const R = mats[mode]->ncols;
    sptMatrix * const M = mats[nmodes];
    sptValue * const mvals = M->values;
    memset(mvals, 0, tmpI*stride*sizeof(*mvals));

    // omp_lock_t lock;
    // omp_init_lock(&lock);

    sptTimer timer;
    sptNewTimer(&timer, 0);
    sptStartTimer(timer);

    /* Loop kernels */
    #pragma omp parallel for num_threads(nthreads)
    for(sptIndex b=0; b<hitsr->bptr.len - 1; ++b) {
        /* Allocate thread-private data */
        sptIndex * block_coord = (sptIndex*)malloc(nmodes * sizeof(*block_coord));
        sptIndex * ele_coord = (sptIndex*)malloc(nmodes * sizeof(*ele_coord));
        sptValueVector scratch; // Temporary array
        sptNewValueVector(&scratch, R, R);       

        /* Block indices */
        for(sptIndex m=0; m<nmodes; ++m)
            block_coord[m] = hitsr->binds[m].data[b];

        sptNnzIndex bptr_begin = hitsr->bptr.data[b];
        sptNnzIndex bptr_end = hitsr->bptr.data[b+1];
        /* Loop entries in a block */
        for(sptNnzIndex z=bptr_begin; z<bptr_end; ++z) {
            /* Element indices */
            for(sptIndex m=0; m<nmodes; ++m)
                ele_coord[m] = (block_coord[m] << hitsr->sb_bits) + hitsr->einds[m].data[z];

            /* Multiply the 1st matrix */
            sptIndex times_mat_index = mats_order[1];
            sptMatrix * times_mat = mats[times_mat_index];
            sptIndex tmp_i = ele_coord[times_mat_index];
            sptValue const entry = vals[z];
            for(sptIndex r=0; r<R; ++r) {
                scratch.data[r] = entry * times_mat->values[tmp_i * stride + r];
            }
            /* Multiply the rest matrices */
            for(sptIndex m=2; m<nmodes; ++m) {
                times_mat_index = mats_order[m];
                times_mat = mats[times_mat_index];
                tmp_i = ele_coord[times_mat_index];
                for(sptIndex r=0; r<R; ++r) {
                    scratch.data[r] *= times_mat->values[tmp_i * stride + r];
                }
            }

            sptIndex const mode_i = ele_coord[mode];
            // omp_set_lock(&lock);
            for(sptIndex r=0; r<R; ++r) {
                #pragma omp atomic update
                mvals[mode_i * stride + r] += scratch.data[r];
            }
            // omp_unset_lock(&lock);
        }   // End loop entries

        /* Free thread-private space */
        free(block_coord);
        free(ele_coord);
        sptFreeValueVector(&scratch);
    }   // End loop blocks

    sptStopTimer(timer);
    sptPrintElapsedTime(timer, "Omp HiSpTns MTTKRP");
    sptFreeTimer(timer);

    // omp_destroy_lock(&lock);

    return 0;
}


int sptOmpMTTKRPHiCOO_3D(
    sptSparseTensorHiCOO const * const hitsr,
    sptMatrix * mats[],     // mats[nmodes] as temporary space.
    sptIndex const mats_order[],    // Correspond to the mode order of X.
    sptIndex const mode,
    const int nthreads) 
{
    sptIndex const nmodes = hitsr->nmodes;
    sptIndex const * const ndims = hitsr->ndims;
    sptValue const * const restrict vals = hitsr->values.data;
    sptIndex const stride = mats[0]->stride;

    /* Check the mats. */
    sptAssert(nmodes ==3);
    for(sptIndex i=0; i<nmodes; ++i) {
        if(mats[i]->ncols != mats[nmodes]->ncols) {
            spt_CheckError(SPTERR_SHAPE_MISMATCH, "Omp HiSpTns MTTKRP", "mats[i]->cols != mats[nmodes]->ncols");
        }
        if(mats[i]->nrows != ndims[i]) {
            spt_CheckError(SPTERR_SHAPE_MISMATCH, "Omp HiSpTns MTTKRP", "mats[i]->nrows != ndims[i]");
        }
    }

    sptIndex const tmpI = mats[mode]->nrows;
    sptIndex const R = mats[mode]->ncols;
    sptMatrix * const restrict M = mats[nmodes];
    sptValue * const restrict mvals = M->values;
    memset(mvals, 0, tmpI*stride*sizeof(*mvals));

    sptIndex times_mat_index_1 = mats_order[1];
    sptMatrix * restrict times_mat_1 = mats[times_mat_index_1];
    sptIndex times_mat_index_2 = mats_order[2];
    sptMatrix * restrict times_mat_2 = mats[times_mat_index_2];


    sptTimer timer;
    sptNewTimer(&timer, 0);
    sptStartTimer(timer);
    
    /* Loop kernels */
    #pragma omp parallel for num_threads(nthreads)
    for(sptIndex b=0; b<hitsr->bptr.len - 1; ++b) {

        sptBlockIndex block_coord_mode = hitsr->binds[mode].data[b];
        sptBlockIndex block_coord_1 = hitsr->binds[times_mat_index_1].data[b];
        sptBlockIndex block_coord_2 = hitsr->binds[times_mat_index_2].data[b];

        sptNnzIndex bptr_begin = hitsr->bptr.data[b];
        sptNnzIndex bptr_end = hitsr->bptr.data[b+1];
        /* Loop entries in a block */
        for(sptIndex z=bptr_begin; z<bptr_end; ++z) {
            
            sptIndex mode_i = (block_coord_mode << hitsr->sb_bits) + hitsr->einds[mode].data[z];
            sptIndex tmp_i_1 = (block_coord_1 << hitsr->sb_bits) + hitsr->einds[times_mat_index_1].data[z];
            sptIndex tmp_i_2 = (block_coord_2 << hitsr->sb_bits) + hitsr->einds[times_mat_index_2].data[z];
            sptValue entry = vals[z];
            for(sptIndex r=0; r<R; ++r) {
                #pragma omp atomic update
                mvals[mode_i * stride + r] += entry * times_mat_1->values[tmp_i_1 * stride + r] * times_mat_2->values[tmp_i_2 * stride + r];
            }
            
        }   // End loop entries
    }   // End loop blocks

    sptStopTimer(timer);
    sptPrintElapsedTime(timer, "Omp HiSpTns MTTKRP");
    sptFreeTimer(timer);

    return 0;
}

#endif
