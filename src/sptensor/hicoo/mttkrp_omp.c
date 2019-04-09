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
    sptElementIndex const sb_bits = hitsr->sb_bits;

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
    sptValue * const mvals = mats[nmodes]->values;
    memset(mvals, 0, tmpI*stride*sizeof(*mvals));

    // omp_lock_t lock;
    // omp_init_lock(&lock);

    sptTimer timer;
    sptNewTimer(&timer, 0);
    sptStartTimer(timer);

    /* Loop kernels */
    #pragma omp parallel for schedule(dynamic) num_threads(nthreads)
    for(sptIndex b=0; b<hitsr->bptr.len - 1; ++b) {
        /* Allocate thread-private data */
        sptValue ** block_values = (sptValue**)malloc(nmodes * sizeof(*block_values));
        sptValueVector scratch; // Temporary array
        sptNewValueVector(&scratch, R, R);    

        /* Block indices */
        block_values[mode] = mvals + (hitsr->binds[mode].data[b] << sb_bits) * stride;
        for(sptIndex m=0; m<nmodes; ++m) {
            if(m != mode) {
                block_values[m] = mats[m]->values + (hitsr->binds[m].data[b] << sb_bits) * stride;
            }
        }

        sptNnzIndex bptr_begin = hitsr->bptr.data[b];
        sptNnzIndex bptr_end = hitsr->bptr.data[b+1];
        /* Loop entries in a block */
        for(sptNnzIndex z=bptr_begin; z<bptr_end; ++z) {
            /* Multiply the 1st matrix */
            sptIndex times_mat_index = mats_order[1];
            sptValue * times_matval = block_values[times_mat_index];
            sptElementIndex tmp_i = hitsr->einds[times_mat_index].data[z];
            sptValue const entry = vals[z];
            #pragma omp simd
            for(sptIndex r=0; r<R; ++r) {
                scratch.data[r] = entry * times_matval[tmp_i * stride + r];
            }
            /* Multiply the rest matrices */
            for(sptIndex m=2; m<nmodes; ++m) {
                times_mat_index = mats_order[m];
                times_matval = block_values[times_mat_index];
                tmp_i = hitsr->einds[times_mat_index].data[z];
                
                #pragma omp simd
                for(sptIndex r=0; r<R; ++r) {
                    scratch.data[r] *= times_matval[tmp_i * stride + r];
                }
            }

            sptIndex const mode_i = hitsr->einds[mode].data[z];
            times_matval = block_values[mode];
            // omp_set_lock(&lock);
            for(sptIndex r=0; r<R; ++r) {
                #pragma omp atomic update
                times_matval[mode_i * stride + r] += scratch.data[r];
            }
            // omp_unset_lock(&lock);
        }   // End loop entries

        /* Free thread-private space */
        free(block_values);
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
    sptElementIndex const sb_bits = hitsr->sb_bits;

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
    #pragma omp parallel for schedule(dynamic) num_threads(nthreads)
    for(sptIndex b=0; b<hitsr->bptr.len - 1; ++b) {

        sptBlockIndex block_coord_mode = hitsr->binds[mode].data[b];
        sptBlockIndex block_coord_1 = hitsr->binds[times_mat_index_1].data[b];
        sptBlockIndex block_coord_2 = hitsr->binds[times_mat_index_2].data[b];

        sptValue * restrict block_mvals = mvals + (block_coord_mode << sb_bits) * stride;
        sptValue * restrict block_times_matval_1 = times_mat_1->values + (block_coord_1 << sb_bits) * stride;
        sptValue * restrict block_times_matval_2 = times_mat_2->values + (block_coord_2 << sb_bits) * stride;

        sptNnzIndex bptr_begin = hitsr->bptr.data[b];
        sptNnzIndex bptr_end = hitsr->bptr.data[b+1];
        /* Loop entries in a block */
        for(sptIndex z=bptr_begin; z<bptr_end; ++z) {
            
            sptElementIndex mode_i = hitsr->einds[mode].data[z];
            sptValue * const restrict block_mvals_row = block_mvals + mode_i * stride;
            sptElementIndex const tmp_i_1 = hitsr->einds[times_mat_index_1].data[z];
            sptElementIndex const tmp_i_2 = hitsr->einds[times_mat_index_2].data[z];
            sptValue const entry = vals[z];
            for(sptIndex r=0; r<R; ++r) {
                #pragma omp atomic update
                block_mvals_row[r] += entry * block_times_matval_1[tmp_i_1 * stride + r] * block_times_matval_2[tmp_i_2 * stride + r];
            }
            
        }   // End loop entries
    }   // End loop blocks

    sptStopTimer(timer);
    sptPrintElapsedTime(timer, "Omp HiSpTns MTTKRP");
    sptFreeTimer(timer);

    return 0;
}

#endif
