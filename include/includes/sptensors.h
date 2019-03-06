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

#ifndef PARTI_SPTENSORS_H
#define PARTI_SPTENSORS_H

/* Sparse tensor */
int sptNewSparseTensor(sptSparseTensor *tsr, sptIndex nmodes, const sptIndex ndims[]);
int sptCopySparseTensor(sptSparseTensor *dest, const sptSparseTensor *src, int const nt);
void sptFreeSparseTensor(sptSparseTensor *tsr);
double SparseTensorFrobeniusNormSquared(sptSparseTensor const * const spten);
int sptLoadSparseTensor(sptSparseTensor *tsr, sptIndex start_index, FILE *fp);
int sptDumpSparseTensor(const sptSparseTensor *tsr, sptIndex start_index, FILE *fp);
int sptMatricize(sptSparseTensor const * const X,
    sptIndex const m,
    sptSparseMatrix * const A,
    int const transpose);
void sptGetBestModeOrder(
    sptIndex * mode_order,
    sptIndex const mode,
    sptIndex const * ndims,
    sptIndex const nmodes);
void sptGetWorstModeOrder(
    sptIndex * mode_order,
    sptIndex const mode,
    sptIndex const * ndims,
    sptIndex const nmodes);
void sptGetRandomShuffleElements(sptSparseTensor *tsr);
void sptGetRandomShuffledIndices(sptSparseTensor *tsr, sptIndex ** map_inds);
void sptSparseTensorShuffleIndices(sptSparseTensor *tsr, sptIndex ** map_inds);
void sptSparseTensorSortIndex(sptSparseTensor *tsr, int force);
void sptSparseTensorSortIndexAtMode(sptSparseTensor *tsr, sptIndex const mode, int force);
void sptSparseTensorSortIndexCustomOrder(sptSparseTensor *tsr, sptIndex const *  mode_order, int force);
void sptSparseTensorSortIndexMorton(
    sptSparseTensor *tsr, 
    int force,
    const sptNnzIndex begin,
    const sptNnzIndex end,
    const sptElementIndex sb_bits);
void sptSparseTensorSortIndexRowBlock(
    sptSparseTensor *tsr, 
    int force,
    const sptNnzIndex begin,
    const sptNnzIndex end,
    const sptElementIndex sk_bits);
void sptSparseTensorSortIndexSingleMode(sptSparseTensor *tsr, int force, sptIndex mode);
void sptSparseTensorSortIndexExceptSingleMode(sptSparseTensor *tsr, int force, sptIndex * mode_order);
int sptSparseTensorMixedOrder(
    sptSparseTensor *tsr, 
    const sptElementIndex sb_bits,
    const sptElementIndex sk_bits);
int sptSparseTensorSortPartialIndex(
    sptSparseTensor *tsr, 
    sptIndex const *  mode_order,
    const sptElementIndex sb_bits);
void sptSparseTensorCalcIndexBounds(sptIndex inds_low[], sptIndex inds_high[], const sptSparseTensor *tsr);
int spt_ComputeSliceSizes(
    sptNnzIndex * slice_nnzs, 
    sptSparseTensor * const tsr,
    sptIndex const mode);
void sptSparseTensorStatus(sptSparseTensor *tsr, FILE *fp);
double sptSparseTensorDensity(sptSparseTensor const * const tsr);
int sptSparseTensorSetIndices(
    sptSparseTensor *dest,
    sptNnzIndexVector *fiberidx,
    sptIndex mode,
    sptSparseTensor *ref
);
int sptSetKernelPointers(
    sptNnzIndexVector *kptr,
    sptSparseTensor *tsr, 
    const sptElementIndex sk_bits);

/* Sparse tensor unary operations */
int sptSparseTensorAddScalar(sptSparseTensor *Z, sptSparseTensor *X, sptValue a);
int sptOmpSparseTensorAddScalar(sptSparseTensor *Z, sptSparseTensor *X, sptValue a);
int sptSparseTensorMulScalar(sptSparseTensor *Z, sptSparseTensor *X, sptValue a);
int sptOmpSparseTensorMulScalar(sptSparseTensor *Z, sptSparseTensor *X, sptValue a);

/* Sparse tensor binary operations */
int sptSparseTensorDotAdd(sptSparseTensor *Z, const sptSparseTensor *X, const sptSparseTensor *Y, int collectZero);
int sptOmpSparseTensorDotAdd(sptSparseTensor *Z, sptSparseTensor *X, sptSparseTensor *Y, int collectZero, int nthreads);
int sptSparseTensorDotAddEq(sptSparseTensor *Z, const sptSparseTensor *X, const sptSparseTensor *Y, int collectZero);
int sptOmpSparseTensorDotAddEq(sptSparseTensor *Z, const sptSparseTensor *X, const sptSparseTensor *Y, int collectZero);

int sptSparseTensorDotSub(sptSparseTensor *Z, const sptSparseTensor *X, const sptSparseTensor *Y, int collectZero);
int sptOmpSparseTensorDotSub(sptSparseTensor *Z, sptSparseTensor *X, sptSparseTensor *Y, int collectZero, int nthreads);
int sptSparseTensorDotSubEq(sptSparseTensor *Z, const sptSparseTensor *X, const sptSparseTensor *Y, int collectZero);
int sptOmpSparseTensorDotSubEq(sptSparseTensor *Z, const sptSparseTensor *X, const sptSparseTensor *Y, int collectZero);

int sptSparseTensorDotMul(sptSparseTensor *Z, const sptSparseTensor * X, const sptSparseTensor *Y, int collectZero);
int sptOmpSparseTensorDotMul(sptSparseTensor *Z, sptSparseTensor *X, sptSparseTensor *Y, int collectZero, int nthreads);
int sptSparseTensorDotMulEq(sptSparseTensor *Z, const sptSparseTensor *X, const sptSparseTensor *Y, int collectZero);
int sptOmpSparseTensorDotMulEq(sptSparseTensor *Z, const sptSparseTensor *X, const sptSparseTensor *Y, int collectZero);

int sptSparseTensorDotDiv(sptSparseTensor *Z, const sptSparseTensor *X, const sptSparseTensor *Y);
int sptSparseTensorDotDivEq(sptSparseTensor *Z, const sptSparseTensor *X, const sptSparseTensor *Y, int collectZero);
int sptOmpSparseTensorDotDivEq(sptSparseTensor *Z, const sptSparseTensor *X, const sptSparseTensor *Y, int collectZero);

int sptSparseTensorMulMatrix(sptSemiSparseTensor *Y, sptSparseTensor *X, const sptMatrix *U, sptIndex const mode);
int sptOmpSparseTensorMulMatrix(sptSemiSparseTensor *Y, sptSparseTensor *X, const sptMatrix *U, sptIndex const mode);
int sptCudaSparseTensorMulMatrix(
    sptSemiSparseTensor *Y,
    sptSparseTensor *X,
    const sptMatrix *U,
    sptIndex const mode,
    sptIndex const impl_num,
    sptNnzIndex const smen_size);

int sptSparseTensorMulVector(sptSparseTensor *Y, sptSparseTensor *X, const sptValueVector *V, sptIndex mode);
int sptOmpSparseTensorMulVector(sptSparseTensor *Y, sptSparseTensor *X, const sptValueVector *V, sptIndex mode);


/**
 * Kronecker product
 */
int sptSparseTensorKroneckerMul(sptSparseTensor *Y, const sptSparseTensor *A, const sptSparseTensor *B);

/**
 * Khatri-Rao product
 */
int sptSparseTensorKhatriRaoMul(sptSparseTensor *Y, const sptSparseTensor *A, const sptSparseTensor *B);


/**
 * Matricized tensor times Khatri-Rao product.
 */
int sptMTTKRP(
    sptSparseTensor const * const X,
    sptMatrix * mats[],     // mats[nmodes] as temporary space.
    sptIndex const mats_order[],    // Correspond to the mode order of X.
    sptIndex const mode);
int sptOmpMTTKRP(
    sptSparseTensor const * const X,
    sptMatrix * mats[],     // mats[nmodes] as temporary space.
    sptIndex const mats_order[],    // Correspond to the mode order of X.
    sptIndex const mode,
    const int tk);
int sptOmpMTTKRP_Reduce(sptSparseTensor const * const X,
    sptMatrix * mats[],     // mats[nmodes] as temporary space.
    sptMatrix * copy_mats[],    // temporary matrices for reduction
    sptIndex const mats_order[],    // Correspond to the mode order of X.
    sptIndex const mode,
    const int tk);
int sptOmpMTTKRP_Lock(sptSparseTensor const * const X,
    sptMatrix * mats[],     // mats[nmodes] as temporary space.
    sptIndex const mats_order[],    // Correspond to the mode order of X.
    sptIndex const mode,
    const int tk,
    sptMutexPool * lock_pool);
int sptCudaMTTKRP(
    sptSparseTensor const * const X,
    sptMatrix ** const mats,     // mats[nmodes] as temporary space.
    sptIndex * const mats_order,    // Correspond to the mode order of X.
    sptIndex const mode,
    sptIndex const impl_num);


#endif