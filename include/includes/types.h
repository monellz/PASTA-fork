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

#ifndef PASTA_TYPES_H
#define PASTA_TYPES_H

#include <stdint.h>

/**
 * Define types, TODO: check the bit size of them, add branch for different settings
 */
#define PASTA_INDEX_TYPEWIDTH 32
#define PASTA_VALUE_TYPEWIDTH 32
#define PASTA_ELEMENT_INDEX_TYPEWIDTH 8

#if PASTA_INDEX_TYPEWIDTH == 32
  typedef uint32_t sptIndex;
  typedef uint32_t sptBlockIndex;
  #define PASTA_INDEX_MAX UINT32_MAX
  #define PASTA_PRI_INDEX PRIu32
  #define PASTA_SCN_INDEX SCNu32
  #define PASTA_PRI_BLOCK_INDEX PRIu32
  #define PASTA_SCN_BLOCK_INDEX SCNu32
#elif PASTA_INDEX_TYPEWIDTH == 64
  typedef uint64_t sptIndex;
  typedef uint64_t sptBlockIndex;
  #define PASTA_INDEX_MAX UINT64_MAX
  #define PASTA_PRI_INDEX PRIu64
  #define PASTA_SCN_INDEX SCNu64
  #define PASTA_PRI_BLOCK_INDEX PRIu64
  #define PASTA_SCN_BLOCK_INDEX SCNu64
#else
  #error "Unrecognized PASTA_INDEX_TYPEWIDTH."
#endif

#if PASTA_VALUE_TYPEWIDTH == 32
  typedef float sptValue;
  #define PASTA_PRI_VALUE "f"
  #define PASTA_SCN_VALUE "f"
#elif PASTA_VALUE_TYPEWIDTH == 64
  typedef double sptValue;
  #define PASTA_PRI_VALUE "lf"
  #define PASTA_SCN_VALUE "lf"
#else
  #error "Unrecognized PASTA_VALUE_TYPEWIDTH."
#endif

#if PASTA_ELEMENT_INDEX_TYPEWIDTH == 8
  typedef uint8_t sptElementIndex;
  typedef uint16_t sptBlockMatrixIndex;  // R < 256
  #define PASTA_PRI_ELEMENT_INDEX PRIu8
  #define PASTA_SCN_ELEMENT_INDEX SCNu8
  #define PASTA_PRI_BLOCKMATRIX_INDEX PRIu16
  #define PASTA_SCN_BLOCKMATRIX_INDEX SCNu16
#elif PASTA_ELEMENT_INDEX_TYPEWIDTH == 16
  typedef uint16_t sptElementIndex;
  typedef uint32_t sptBlockMatrixIndex;
  #define PASTA_PRI_ELEMENT_INDEX PRIu16
  #define PASTA_SCN_ELEMENT_INDEX SCNu16
  #define PASTA_PRI_BLOCKMATRIX_INDEX PRIu32
  #define PASTA_SCN_BLOCKMATRIX_INDEX SCNu32
#elif PASTA_ELEMENT_INDEX_TYPEWIDTH == 32
  typedef uint32_t sptElementIndex;
  typedef uint32_t sptBlockMatrixIndex;
  #define PASTA_PRI_ELEMENT_INDEX PRIu32
  #define PASTA_SCN_ELEMENT_INDEX SCNu32
  #define PASTA_PRI_BLOCKMATRIX_INDEX PRIu32
  #define PASTA_SCN_BLOCKMATRIX_INDEX SCNu32
#else
  #error "Unrecognized PASTA_ELEMENT_INDEX_TYPEWIDTH."
#endif

typedef sptBlockIndex sptBlockNnzIndex;
#define PASTA_PRI_BLOCKNNZ_INDEX PASTA_PRI_BLOCK_INDEX
#define PASTA_SCN_BLOCKNNZ_INDEX PASTA_SCN_BLOCK_INDEX

typedef uint64_t sptNnzIndex;
#define PASTA_NNZ_INDEX_MAX UINT64_MAX
#define PASTA_PRI_NNZ_INDEX PRIu64
#define PASTA_SCN_NNZ_INDEX PRIu64

typedef unsigned __int128 sptMortonIndex;
// typedef __uint128_t sptMortonIndex;


#endif