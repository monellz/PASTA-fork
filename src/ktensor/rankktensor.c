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
#include <stdlib.h>
#include <string.h>
#include "../error/error.h"

/**
 * Assign a new Kruskal tensor with sptElementIndex as the columns of their factor matrices.
 *
 * @param[out] ktsr Kruskal tensor
 * @param[in] nmodes the number of dimensions/modes/tensor order
 * @param[in] ndims the mode sizes
 * @param[in] rank tensor rank or the number of columns of factor matrices
 *
 */
int sptNewRankKruskalTensor(sptRankKruskalTensor *ktsr, sptIndex nmodes, const sptIndex ndims[], sptElementIndex rank)
{
    ktsr->nmodes = nmodes;
    ktsr->rank = rank;
    ktsr->ndims = (sptIndex*)malloc(nmodes*sizeof(sptIndex));
    for(sptIndex i=0; i<nmodes; ++i)
        ktsr->ndims[i] = ndims[i];
    ktsr->lambda = (sptValue*)malloc(rank*sizeof(sptValue));
    ktsr->fit = 0.0;
    
	return 0;
}

/**
 * Shuffle factor matrices row indices.
 *
 * @param[out] map_inds is the renumbering mapping 
 * @param[in] ktsr Kruskal tensor to be shuffled
 *
 */
void sptRankKruskalTensorInverseShuffleIndices(sptRankKruskalTensor * ktsr, sptIndex ** map_inds) {
    /* Renumber factor matrices rows */
    sptIndex new_i;
    for(sptIndex m=0; m < ktsr->nmodes; ++m) {
        sptRankMatrix * mtx = ktsr->factors[m];
        sptIndex * mode_map_inds = map_inds[m];
        sptValue * tmp_values = malloc(mtx->cap * mtx->stride * sizeof (sptValue));

        for(sptIndex i=0; i<mtx->nrows; ++i) {
            new_i = mode_map_inds[i];
            for(sptElementIndex j=0; j<mtx->ncols; ++j) {
                tmp_values[i * mtx->stride + j] = mtx->values[new_i * mtx->stride + j];
            }
        }
        free(mtx->values);
        mtx->values = tmp_values;
    }    
}

/**
 * Free a new Kruskal tensor with sptElementIndex as the columns of their factor matrices.
 *
 * @param[in] ktsr Kruskal tensor
 *
 */
void sptFreeRankKruskalTensor(sptRankKruskalTensor *ktsr)
{
	ktsr->rank = 0;
	ktsr->fit = 0.0;
	free(ktsr->ndims);
	free(ktsr->lambda);
	for(sptIndex i=0; i<ktsr->nmodes; ++i)
		sptFreeRankMatrix(ktsr->factors[i]);
    free(ktsr->factors);
	ktsr->nmodes = 0;
}


/**
 * Compute the fit of a Kruskal tensor (with sptElementIndex as the columns of their factor matrices) to a sparse tensor.
 *
 * @param[in] hitsr   a HiCOO sparse tensor.
 * @param[in] lambda  the weight array
 * @param[in] mats    factor matrices
 * @param[in] ata    the results of ATA, A is a factor matrix
 * @return fit  a double-precision float-point value
 *
 */
double sptKruskalTensorFitHiCOO(
  sptSparseTensorHiCOO const * const hitsr,
  sptValue const * const __restrict lambda,
  sptRankMatrix ** mats,
  sptRankMatrix ** ata) 
{
  sptIndex const nmodes = hitsr->nmodes;

  double spten_normsq = sptSparseTensorFrobeniusNormSquaredHiCOO(hitsr);
  double const norm_mats = sptKruskalTensorFrobeniusNormSquaredRank(nmodes, lambda, ata);
  double const inner = sptSparseKruskalTensorInnerProductRank(nmodes, lambda, mats);
  double residual = spten_normsq + norm_mats - 2 * inner;
  if (residual > 0.0) {
    residual = sqrt(residual);
  }
  double fit = 1 - (residual / sqrt(spten_normsq));

  return fit;
}


// Column-major. 
/* Compute a Kruskal tensor's norm is compute on "ata"s. Check Tammy's sparse  */
double sptKruskalTensorFrobeniusNormSquaredRank(
  sptIndex const nmodes,
  sptValue const * const __restrict lambda,
  sptRankMatrix ** ata) // ata: column-major
{
  sptElementIndex const rank = ata[0]->ncols;
  sptElementIndex const stride = ata[0]->stride;
  sptValue * const __restrict tmp_atavals = ata[nmodes]->values;    // Column-major
  double norm_mats = 0;

#ifdef PARTI_USE_OPENMP
  #pragma omp parallel for schedule(static)
#endif
  for(sptIndex x=0; x < rank*stride; ++x) {
    tmp_atavals[x] = 1.;
  }

  /* Compute Hadamard product for all "ata"s */
  for(sptIndex m=0; m < nmodes; ++m) {
    sptValue const * const __restrict atavals = ata[m]->values;
#ifdef PARTI_USE_OPENMP
  #pragma omp parallel for schedule(static)
#endif
    for(sptElementIndex i=0; i < rank; ++i) {
        for(sptElementIndex j=i; j < rank; ++j) {
            tmp_atavals[j * stride + i] *= atavals[j * stride + i];
        }
    }
  }

  /* compute lambda^T * aTa[MAX_NMODES] * lambda, only compute a half of them because of its symmetric */
// #ifdef PARTI_USE_OPENMP
//   #pragma omp parallel for schedule(static) reduction(+:norm_mats)
// #endif
  for(sptElementIndex i=0; i < rank; ++i) {
    norm_mats += tmp_atavals[i+(i*stride)] * lambda[i] * lambda[i];
    for(sptElementIndex j=i+1; j < rank; ++j) {
      norm_mats += tmp_atavals[i+(j*stride)] * lambda[i] * lambda[j] * 2;
    }
  }

  return fabs(norm_mats);
}



// Row-major, compute via MTTKRP result (mats[nmodes]) and mats[nmodes-1].
double sptSparseKruskalTensorInnerProductRank(
  sptIndex const nmodes,
  sptValue const * const __restrict lambda,
  sptRankMatrix ** mats) 
{
  sptElementIndex const rank = mats[0]->ncols;
  sptElementIndex const stride = mats[0]->stride;
  sptIndex const last_mode = nmodes - 1;
  sptIndex const I = mats[last_mode]->nrows;
  
  sptValue const * const last_vals = mats[last_mode]->values;
  sptValue const * const tmp_vals = mats[nmodes]->values;
  sptValue * buffer_accum;

  double inner = 0;

  double * const __restrict accum = (double *) malloc(rank*sizeof(*accum));

#ifdef PARTI_USE_OPENMP
  #pragma omp parallel for schedule(static)
#endif
  for(sptElementIndex r=0; r < rank; ++r) {
    accum[r] = 0.0; 
  }

#ifdef PARTI_USE_OPENMP
  #pragma omp parallel
  {
    int const nthreads = omp_get_num_threads();
    #pragma omp master
    {
      buffer_accum = (sptValue *)malloc(nthreads * rank * sizeof(sptValue));
      for(sptIndex j=0; j < (sptIndex)nthreads * rank; ++j)
          buffer_accum[j] = 0.0;
    }
  }
#endif

#ifdef PARTI_USE_OPENMP
  #pragma omp parallel
  {
    int const tid = omp_get_thread_num();
    int const nthreads = omp_get_num_threads();
    sptValue * loc_accum = buffer_accum + tid * rank;

    #pragma omp for
    for(sptIndex i=0; i < I; ++i) {
      for(sptElementIndex r=0; r < rank; ++r) {
        loc_accum[r] += last_vals[r+(i*stride)] * tmp_vals[r+(i*stride)];
      }
    }

    #pragma omp for schedule(static)
    for(sptElementIndex j=0; j < rank; ++j) {
      for(sptIndex i=0; i < (sptIndex)nthreads; ++i) {
        accum[j] += buffer_accum[i*rank + j];
      }
    }

  }

#else

  for(sptIndex i=0; i < I; ++i) {
    for(sptElementIndex r=0; r < rank; ++r) {
      accum[r] += last_vals[r+(i*stride)] * tmp_vals[r+(i*stride)];
    }
  }

#endif

#ifdef PARTI_USE_OPENMP
  #pragma omp parallel for schedule(static) reduction(+:inner)
#endif
  for(sptElementIndex r=0; r < rank; ++r) {
    inner += accum[r] * lambda[r];
  }

#ifdef PARTI_USE_OPENMP
  free(buffer_accum);
#endif

  return inner;
}