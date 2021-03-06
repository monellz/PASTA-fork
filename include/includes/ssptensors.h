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

#ifndef PASTA_SSPTENSORS_H
#define PASTA_SSPTENSORS_H


/**
 * epsilon is a small positive value, every -epsilon < x < x would be considered as zero
 */
int sptSemiSparseTensorToSparseTensor(sptSparseTensor *dest, const sptSemiSparseTensor *src, sptValue epsilon);

int sptNewSemiSparseTensor(sptSemiSparseTensor *tsr, sptIndex nmodes, sptIndex mode, const sptIndex ndims[]);
int sptNewSemiSparseTensorWithNnz(sptSemiSparseTensor *tsr, sptIndex nmodes, sptIndex mode, const sptIndex ndims[], sptNnzIndex nfibers);
int sptCopySemiSparseTensor(sptSemiSparseTensor *dest, const sptSemiSparseTensor *src);
void sptFreeSemiSparseTensor(sptSemiSparseTensor *tsr);
int sptDumpSemiSparseTensor(sptSemiSparseTensor * const tsr, FILE *fp);
int sptSparseTensorToSemiSparseTensor(sptSemiSparseTensor *dest, const sptSparseTensor *src, sptIndex mode);
int sptSemiSparseTensorSortIndex(sptSemiSparseTensor *tsr);

/* Semi-sparse tensor HiCOO */
int sptNewSemiSparseTensorHiCOO(
    sptSemiSparseTensorHiCOO *histsr, 
    const sptIndex nmodes, 
    const sptIndex ndims[],
    const sptIndex mode,
    const sptElementIndex sb_bits);
int sptNewSemiSparseTensorHiCOOWithBptr(
    sptSemiSparseTensorHiCOO *histsr, 
    const sptIndex nmodes, 
    const sptIndex ndims[],
    const sptNnzIndex nfibers,
    const sptIndex mode,
    const sptElementIndex sb_bits,
    sptNnzIndexVector * bptr);
void sptFreeSemiSparseTensorHiCOO(sptSemiSparseTensorHiCOO *histsr);
int sptSemiHiCOOToSemiSparseTensor(
    sptSemiSparseTensor *stsr, 
    sptSemiSparseTensorHiCOO *histsr);
int sptDumpSemiSparseTensorHiCOO(sptSemiSparseTensorHiCOO * const hitsr, FILE *fp);

/**
 * Set indices of a semi-sparse according to a reference sparse
 * Call sptSparseTensorSortIndexAtMode on ref first
 */
int sptSemiSparseTensorSetFibers(
    sptNnzIndexVector *fiberidx,
    sptSparseTensor *ref,
    sptIndex mode
);
int sptSemiSparseTensorSetIndices(sptSemiSparseTensor *dest, sptNnzIndexVector *fiberidx, sptSparseTensor *ref);
int sptSemiSparseTensorSetFibersHiCOO (
    sptNnzIndexVector *bptr,
    sptNnzIndexVector *fiberidx,
    sptSparseTensorHiCOOGeneral *ref);
int sptSemiSparseTensorSetIndicesHiCOO(
    sptSemiSparseTensorHiCOO *dest,
    sptNnzIndexVector *fiberidx,
    sptSparseTensorHiCOOGeneral *ref);

/**
 * Semi-sparse tensor times a dense matrix (TTM)
 * Input: semi-sparse tensor X[I][J][K], dense matrix U[I][R}, mode n={0, 1, 2}
 * Output: sparse tensor Y[I][J][R] (e.g. n=2)
 */
int sptSemiSparseTensorMulMatrix(sptSemiSparseTensor *Y, const sptSemiSparseTensor *X, const sptMatrix *U, sptIndex mode);
#endif