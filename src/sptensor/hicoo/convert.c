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
#include "../sptensor.h"

/**
 * Compare two coordinates, in the order of mode-0,...,N-1. One from the sparse tensor, the other is specified.
 * @param tsr    a pointer to a sparse tensor
 * @return      1, z > item; otherwise, 0.
 */
static int sptLargerThanCoordinates(
    sptSparseTensor *tsr,
    const sptNnzIndex z,
    const sptIndex * item)
{
    sptIndex nmodes = tsr->nmodes;
    sptIndex i1, i2;

    for(sptIndex m=0; m<nmodes; ++m) {
        i1 = tsr->inds[m].data[z];
        i2 = item[m];
        if(i1 > i2) {
            return 1;
            break;
        }
    }
    return 0;
}


/**
 * Compare two coordinates, in the order of mode-0,...,N-1. One from the sparse tensor, the other is specified.
 * @param tsr    a pointer to a sparse tensor
 * @return      1, z < item; otherwise, 0.
 */
static int sptSmallerThanCoordinates(
    sptSparseTensor *tsr,
    const sptNnzIndex z,
    const sptIndex * item)
{
    sptIndex nmodes = tsr->nmodes;
    sptIndex i1, i2;

    for(sptIndex m=0; m<nmodes; ++m) {
        i1 = tsr->inds[m].data[z];
        i2 = item[m];
        if(i1 < i2) {
            return 1;
            break;
        }
    }
    return 0;
}


/**
 * Compare two coordinates, in the order of mode-0,...,N-1. One from the sparse tensor, the other is specified.
 * @param tsr    a pointer to a sparse tensor
 * @return      1, z = item; otherwise, 0.
 */
static int sptEqualWithCoordinates(
    sptSparseTensor *tsr,
    const sptNnzIndex z,
    const sptIndex * item)
{
    sptIndex nmodes = tsr->nmodes;
    sptIndex i1, i2;

    for(sptIndex m=0; m<nmodes; ++m) {
        i1 = tsr->inds[m].data[z];
        i2 = item[m];
        if(i1 != i2) {
            return 0;
            break;
        }
    }
    return 1;
}


/**
 * Compare two specified coordinates.
 * @param tsr    a pointer to a sparse tensor
 * @return      1, z == item; otherwise, 0.
 */
static int sptEqualWithTwoCoordinates(
    const sptIndex * item1,
    const sptIndex * item2,
    const sptIndex nmodes)
{
    sptIndex i1, i2;
    for(sptIndex m=0; m<nmodes; ++m) {
        i1 = item1[m];
        i2 = item2[m];
        if(i1 != i2) {
            return 0;
            break;
        }
    }
    return 1;
}

/**
 * Check if a nonzero item is in the range of two given coordinates, in the order of mode-0,...,N-1. 
 * @param tsr    a pointer to a sparse tensor
 * @return      1, yes; 0, no.
 */
static int sptCoordinatesInRange(
    sptSparseTensor *tsr,
    const sptNnzIndex z,
    const sptIndex * range_begin,
    const sptIndex * range_end)
{
    if ( (sptLargerThanCoordinates(tsr, z, range_begin) == 1 ||
        sptEqualWithCoordinates(tsr, z, range_begin) == 1) &&
        sptSmallerThanCoordinates(tsr, z, range_end) == 1) {
        return 1;
    }
    return 0;
}

/**
 * Compute the beginning of the next block
 * @param tsr    a pointer to a sparse tensor
 * @return out_item     the beginning indices of the next block
 */
static int sptNextBlockBegin(
    sptIndex * out_item,
    sptSparseTensor *tsr,
    const sptIndex * in_item,
    const sptElementIndex sb)
{
    sptIndex nmodes = tsr->nmodes;

    for(int32_t m=nmodes-1; m>=0; --m) {
        if(in_item[m] < tsr->ndims[m]-1) {
            out_item[m] = in_item[m]+sb-1 < tsr->ndims[m] ? in_item[m]+sb-1 : tsr->ndims[m] - 1;
            break;
        }
    }

    return 0;
}


/**
 * Compute the end of this block
 * @param tsr    a pointer to a sparse tensor
 * @return out_item     the end indices of this block
 */
static int sptBlockEnd(
    sptIndex * out_item,
    sptIndex nmodes,
    sptIndex * ndims,
    sptIndex * flags,
    const sptIndex * in_item,
    const sptElementIndex sb)
{
    if(flags != NULL) {
        for(sptIndex m=0; m<nmodes; ++m) {
            sptAssert(in_item[m] < ndims[m]);
            out_item[m] = in_item[m]+sb < ndims[m] ? in_item[m]+sb : ndims[m];    // exclusive
        }
    } else {
        sptIndex fm = 0;
        for(sptIndex m=0; m<nmodes; ++m) {
            if(flags[m] == 1) {
                sptAssert(in_item[fm] < ndims[m]);
                out_item[fm] = in_item[fm]+sb < ndims[m] ? in_item[fm]+sb : ndims[m];    // exclusive
                ++ fm;
            }
        }
    }

    return 0;
}


/**
 * Locate the beginning of the block/kernel containing the coordinates
 * @param tsr    a pointer to a sparse tensor
 * @return out_item     the beginning indices of this block
 */
static int sptLocateBeginCoord(
    sptIndex * out_item,
    sptIndex nmodes,
    const sptIndex * in_item,
    const sptElementIndex bits)
{   
    for(sptIndex m=0; m<nmodes; ++m) {
        out_item[m] = in_item[m] >> bits;
    }

    return 0;
}


/**
 * Compute the strides for kernels, mode order N-1, ..., 0 (row-like major)
 * @param tsr    a pointer to a sparse tensor
 * @return out_item     the beginning indices of this block
 */
static int sptKernelStrides(
    sptIndex * strides,
    sptSparseTensor *tsr,
    const sptIndex sk)
{
    sptIndex nmodes = tsr->nmodes;
    sptIndex kernel_size = 0;
    
    // TODO: efficiently use bitwise operation
    strides[nmodes-1] = 1;
    for(sptIndex m=nmodes-2; m>=1; --m) {
        kernel_size = (sptIndex)(tsr->ndims[m+1] + sk - 1) / sk;
        strides[m] = strides[m+1] * kernel_size;
    }
    kernel_size = (sptIndex)(tsr->ndims[1] + sk - 1) / sk;
    strides[0] = strides[1] * kernel_size;

    return 0;
}





/**
 * Compute the end of this kernel
 * @param tsr    a pointer to a sparse tensor
 * @return out_item     the end indices of this block
 */
static int sptKernelEnd(
    sptIndex * out_item,
    sptSparseTensor *tsr,
    const sptIndex * in_item,
    const sptIndex sk)
{
    sptIndex nmodes = tsr->nmodes;

    for(sptIndex m=0; m<nmodes; ++m) {
        sptAssert(in_item[m] < tsr->ndims[m]);
        out_item[m] = in_item[m]+sk < tsr->ndims[m] ? in_item[m]+sk : tsr->ndims[m];    // exclusive
    }

    return 0;
}


/**
 * Record mode pointers for kernel rows, from a sorted tensor.
 * @param kptr  a vector of kernel pointers
 * @param tsr    a pointer to a sparse tensor
 * @return      mode pointers
 */
int sptSetKernelPointers(
    sptNnzIndexVector *kptr,
    sptNnzIndexVector *knnzs,
    sptSparseTensor *tsr, 
    const sptElementIndex sk_bits)
{
    sptIndex nmodes = tsr->nmodes;
    sptNnzIndex nnz = tsr->nnz;
    sptNnzIndex k = 0;  // count kernels
    sptNnzIndex knnz = 0;   // #Nonzeros per kernel
    int result = 0;
    result = sptAppendNnzIndexVector(kptr, 0);
    spt_CheckError(result, "HiSpTns Convert", NULL);

    sptIndex * coord = (sptIndex *)malloc(nmodes * sizeof(*coord));
    sptIndex * kernel_coord = (sptIndex *)malloc(nmodes * sizeof(*kernel_coord));
    sptIndex * kernel_coord_prior = (sptIndex *)malloc(nmodes * sizeof(*kernel_coord_prior));

    /* Process first nnz to get the first kernel_coord_prior */
    for(sptIndex m=0; m<nmodes; ++m) 
        coord[m] = tsr->inds[m].data[0];    // first nonzero indices
    result = sptLocateBeginCoord(kernel_coord_prior, nmodes, coord, sk_bits);
    spt_CheckError(result, "HiSpTns Convert", NULL);

    for(sptNnzIndex z=0; z<nnz; ++z) {
        for(sptIndex m=0; m<nmodes; ++m) 
            coord[m] = tsr->inds[m].data[z];
        result = sptLocateBeginCoord(kernel_coord, nmodes, coord, sk_bits);
        spt_CheckError(result, "HiSpTns Convert", NULL);

        if(sptEqualWithTwoCoordinates(kernel_coord, kernel_coord_prior, nmodes) == 1) {
            ++ knnz;
        } else {
            ++ k;
            result = sptAppendNnzIndexVector(kptr, knnz + kptr->data[k-1]);
            spt_CheckError(result, "HiSpTns Convert", NULL);
            result = sptAppendNnzIndexVector(knnzs, knnz);
            spt_CheckError(result, "HiSpTns Convert", NULL);
            for(sptIndex m=0; m<nmodes; ++m) 
                kernel_coord_prior[m] = kernel_coord[m];
            knnz = 1;
        }
    }
    sptAssert(k < kptr->len);
    sptAssert(kptr->data[kptr->len-1] + knnz == nnz);

    /* Set the last element for kptr */
    sptAppendNnzIndexVector(kptr, nnz); 
    sptAppendNnzIndexVector(knnzs, knnz);

    free(coord);
    free(kernel_coord);
    free(kernel_coord_prior);

    return 0;
}


/**
 * Pre-process COO sparse tensor by permuting, sorting, and record pointers to blocked rows. Kernels in Row-major order, blocks and elements are in Z-Morton order.
 * @param tsr    a pointer to a sparse tensor
 * @return      mode pointers
 */
static int sptPreprocessSparseTensor(
    sptSparseTensor *tsr, 
    const sptElementIndex sb_bits,
    int sort_impl)
{
    sptTimer timer;
    sptNewTimer(&timer, 0);
    sptStartTimer(timer);

    /* Sort blocks in each kernel in Morton-order */
    if(sort_impl == 1)
        sptSparseTensorSortIndexMorton(tsr, 1, 0, tsr->nnz, sb_bits);
    else if(sort_impl == 2)
        sptSparseTensorSortIndexRowBlock(tsr, 1, 0, tsr->nnz, sb_bits);
    else {
        printf("Specify a valid sorting implementation. \n");
        exit(1);
    }

    sptStopTimer(timer);
    if(sort_impl == 1)
        sptPrintElapsedTime(timer, "\t\tMorton sorting");
    else if(sort_impl == 2)
        sptPrintElapsedTime(timer, "\t\tRowblock sorting");
    sptFreeTimer(timer);

    return 0;
}


int sptSparseTensorToHiCOO(
    sptSparseTensorHiCOO *hitsr,
    sptNnzIndex *max_nnzb,
    sptSparseTensor *tsr, 
    const sptElementIndex sb_bits,
    int sort_impl,
    int const tk)
{
    sptIndex i;
    int result;
    sptIndex nmodes = tsr->nmodes;
    sptNnzIndex nnz = tsr->nnz;
    sptElementIndex sb = pow(2, sb_bits);

    /* Set HiCOO parameters. ndims for type conversion, size_t -> sptIndex */
    sptIndex * ndims = malloc(nmodes * sizeof *ndims);
    spt_CheckOSError(!ndims, "HiSpTns Convert");
    for(i = 0; i < nmodes; ++i) {
        ndims[i] = (sptIndex)tsr->ndims[i];
    }

    result = sptNewSparseTensorHiCOO(hitsr, (sptIndex)tsr->nmodes, ndims, (sptNnzIndex)tsr->nnz, sb_bits);
    spt_CheckError(result, "HiSpTns Convert", NULL);

    /* Pre-process tensor to get hitsr->kptr, values are nonzero locations. */
    sptTimer sort_timer;
    sptNewTimer(&sort_timer, 0);
    sptStartTimer(sort_timer);

    sptPreprocessSparseTensor(tsr, sb_bits, sort_impl);

    sptStopTimer(sort_timer);
    sptPrintElapsedTime(sort_timer, "\tHiCOO sorting (Morton)");
    sptFreeTimer(sort_timer);
#if PARTI_DEBUG >= 2
    printf("Blocks: Morton-order sorted:\n");
    sptAssert(sptDumpSparseTensor(tsr, 0, stdout) == 0);
#endif

    sptTimer gen_timer;
    sptNewTimer(&gen_timer, 0);
    sptStartTimer(gen_timer);

    /* Temporary storage */
    sptIndex * block_begin = (sptIndex *)malloc(nmodes * sizeof(*block_begin));
    sptIndex * block_end = (sptIndex *)malloc(nmodes * sizeof(*block_end));
    sptIndex * block_begin_prior = (sptIndex *)malloc(nmodes * sizeof(*block_begin_prior));
    sptIndex * block_coord = (sptIndex *)malloc(nmodes * sizeof(*block_coord));

    sptNnzIndex nb = 1; // #Blocks  // counting from the first nnz
    sptNnzIndex ne = 0; // #Nonzeros per block
    sptIndex eindex = 0;

    /* different appending methods:
     * elements: append every nonzero entry
     * blocks: append when seeing a new block.
     * chunks: appending when seeting a new chunk. Notice the boundary of kernels and the last chunk of the whole tensor may be larger than the sc.
     * kernels: append when seeing a new kernel. Not appending a vector, just write data into an allocated array.
     */
    /* Process first nnz */
    for(sptIndex m=0; m<nmodes; ++m) 
        block_coord[m] = tsr->inds[m].data[0];    // first nonzero indices
    result = sptLocateBeginCoord(block_begin_prior, nmodes, block_coord, sb_bits);
    spt_CheckError(result, "HiSpTns Convert", NULL);
    for(sptIndex m=0; m<nmodes; ++m)
        sptAppendBlockIndexVector(&hitsr->binds[m], (sptBlockIndex)block_begin_prior[m]);
    sptAppendNnzIndexVector(&hitsr->bptr, 0);


    /* Loop nonzeros in each kernel */
    for(sptNnzIndex z = 0; z < hitsr->nnz; ++z) {
        #if PARTI_DEBUG == 5
            printf("z: %"PARTI_PRI_NNZ_INDEX "\n", z);
        #endif

        for(sptIndex m=0; m<nmodes; ++m) 
            block_coord[m] = tsr->inds[m].data[z];    // first nonzero indices
        #if PARTI_DEBUG == 5
            printf("block_coord:\n");
            sptAssert(sptDumpIndexArray(block_coord, nmodes, stdout) == 0);
        #endif

        result = sptLocateBeginCoord(block_begin, nmodes, block_coord, sb_bits);
        // spt_CheckError(result, "HiSpTns Convert", NULL);
        #if PARTI_DEBUG == 5
            printf("block_begin_prior:\n");
            sptAssert(sptDumpIndexArray(block_begin_prior, nmodes, stdout) == 0);
            printf("block_begin:\n");
            sptAssert(sptDumpIndexArray(block_begin, nmodes, stdout) == 0);
        #endif

        result = sptBlockEnd(block_end, nmodes, tsr->ndims, NULL, block_begin, sb);  // exclusive
        // spt_CheckError(result, "HiSpTns Convert", NULL);

        /* Append einds and values */
        for(sptIndex m=0; m<nmodes; ++m) {
            eindex = tsr->inds[m].data[z] < (block_begin[m] << sb_bits) ? tsr->inds[m].data[z] : tsr->inds[m].data[z] - (block_begin[m] << sb_bits);
            sptAssert(eindex < sb);
            sptAppendElementIndexVector(&hitsr->einds[m], (sptElementIndex)eindex);
        }
        sptAppendValueVector(&hitsr->values, tsr->values.data[z]);


        /* z in the same block with last z */
        if (sptEqualWithTwoCoordinates(block_begin, block_begin_prior, nmodes) == 1)
        {
            /* ne: #Elements in current block */
            ++ ne;
        } else { /* New block */
            /* ne: #Elements in the last block */
            /* Append block bptr and bidx */
            sptAppendNnzIndexVector(&hitsr->bptr, (sptBlockIndex)z);
            for(sptIndex m=0; m<nmodes; ++m)
                sptAppendBlockIndexVector(&hitsr->binds[m], (sptBlockIndex)block_begin[m]);
            for(sptIndex m=0; m<nmodes; ++m)
                block_begin_prior[m] = block_begin[m];

            ++ nb;
            ne = 1;              
        } // End new block
        #if PARTI_DEBUG == 5
            printf("nb: %u, ne: %u\n\n", nb, ne);
        #endif

    }   // End z loop
    
    sptAssert(nb <= nnz);
    sptAssert(nb == hitsr->binds[0].len); 

    /* Last element for bptr */
    sptAppendNnzIndexVector(&hitsr->bptr, nnz);


    *max_nnzb = hitsr->bptr.data[1] - hitsr->bptr.data[0];
    sptNnzIndex sum_nnzb = 0;
    for(sptIndex i=0; i < hitsr->bptr.len - 1; ++i) {
        sptNnzIndex nnzb = hitsr->bptr.data[i+1] - hitsr->bptr.data[i];
        sum_nnzb += nnzb;
        if(*max_nnzb < nnzb) {
          *max_nnzb = nnzb;
        }
    }
    sptAssert(sum_nnzb == hitsr->nnz);

    sptStopTimer(gen_timer);
    sptPrintElapsedTime(gen_timer, "\tGenerate HiCOO");
    sptFreeTimer(gen_timer);


    free(block_begin);
    free(block_end);
    free(block_begin_prior);
    free(block_coord);

	return 0;
}


int sptHiCOOToSparseTensor(
    sptSparseTensor *tsr, 
    sptSparseTensorHiCOO *hitsr)
{
    sptIndex const nmodes = hitsr->nmodes;
    sptNnzIndex const nnz = hitsr->nnz;
    int result;

    result = sptNewSparseTensor(tsr, nmodes, hitsr->ndims);
    spt_CheckOSError(result, "Convert HiCOO -> COO");
    tsr->nnz = hitsr->nnz;
    for(sptIndex m=0; m<nmodes; ++m) {
        result = sptResizeIndexVector(&(tsr->inds[m]), nnz);
        spt_CheckOSError(result, "Convert HiCOO -> COO");
    }
    result = sptResizeValueVector(&tsr->values, nnz);
    spt_CheckOSError(result, "Convert HiCOO -> COO");

    sptIndex * block_coord = (sptIndex*)malloc(nmodes * sizeof(*block_coord));
    sptIndex ele_coord;


    /* Loop blocks in a kernel */
    for(sptIndex b=0; b<hitsr->bptr.len - 1; ++b) {
        /* Block indices */
        for(sptIndex m=0; m<nmodes; ++m)
            block_coord[m] = hitsr->binds[m].data[b] << hitsr->sb_bits;

        sptNnzIndex bptr_begin = hitsr->bptr.data[b];
        sptNnzIndex bptr_end = hitsr->bptr.data[b+1];
        /* Loop entries in a block */
        for(sptNnzIndex z=bptr_begin; z<bptr_end; ++z) {
            /* Element indices */
            for(sptIndex m=0; m<nmodes; ++m) {
                ele_coord = block_coord[m] + hitsr->einds[m].data[z];
                tsr->inds[m].data[z] = ele_coord;
            }
            tsr->values.data[z] = hitsr->values.data[z];
        }
    }

    free(block_coord);

    return 0;
}



int sptSparseTensorToHiCOOGeneral(
    sptSparseTensorHiCOOGeneral *hitsr,
    sptNnzIndex *max_nnzb,
    sptSparseTensor *tsr, 
    const sptElementIndex sb_bits,
    sptIndex ncmodes,
    sptIndex *flags,
    int const tk)
{
    sptAssert(ncmodes > 0);
    sptIndex i;
    int result;
    sptIndex nmodes = tsr->nmodes;
    sptNnzIndex nnz = tsr->nnz;
    sptElementIndex sb = pow(2, sb_bits);

    /* Set HiCOO parameters. ndims for type conversion, size_t -> sptIndex */
    sptIndex * ndims = malloc(nmodes * sizeof *ndims);
    spt_CheckOSError(!ndims, "HiSpTns Convert");
    for(i = 0; i < nmodes; ++i) {
        ndims[i] = (sptIndex)tsr->ndims[i];
    }

    result = sptNewSparseTensorHiCOOGeneral(hitsr, (sptIndex)tsr->nmodes, ndims, (sptNnzIndex)tsr->nnz, sb_bits, ncmodes, flags);
    spt_CheckError(result, "HiSpTns Convert", NULL);

    /* Pre-process tensor to get hitsr->kptr, values are nonzero locations. */
    sptTimer sort_timer;
    sptNewTimer(&sort_timer, 0);
    sptStartTimer(sort_timer);

    /* Sort naturally */
    sptSparseTensorSortIndex(tsr, 1);
    /* Sort only block indices. Keep the sorting order in the same fiber. */
    sptPreprocessSparseTensor(tsr, sb_bits, 2);

    sptStopTimer(sort_timer);
    sptPrintElapsedTime(sort_timer, "\tHiCOO sorting (Morton)");
    sptFreeTimer(sort_timer);
#if PARTI_DEBUG >= 2
    printf("Blocks: Morton-order sorted:\n");
    sptAssert(sptDumpSparseTensor(tsr, 0, stdout) == 0);
#endif

    sptTimer gen_timer;
    sptNewTimer(&gen_timer, 0);
    sptStartTimer(gen_timer);

    /* Temporary storage */
    sptIndex * block_begin = (sptIndex *)malloc(ncmodes * sizeof(*block_begin));
    sptIndex * block_end = (sptIndex *)malloc(ncmodes * sizeof(*block_end));
    sptIndex * block_begin_prior = (sptIndex *)malloc(ncmodes * sizeof(*block_begin_prior));
    sptIndex * block_coord = (sptIndex *)malloc(ncmodes * sizeof(*block_coord));

    sptNnzIndex nb = 1; // #Blocks  // counting from the first nnz
    sptNnzIndex ne = 0; // #Nonzeros per block
    sptIndex eindex = 0;
    sptIndex fm = 0;

    /* different appending methods:
     * elements: append every nonzero entry
     * blocks: append when seeing a new block.
     * chunks: appending when seeting a new chunk. Notice the boundary of kernels and the last chunk of the whole tensor may be larger than the sc.
     * kernels: append when seeing a new kernel. Not appending a vector, just write data into an allocated array.
     */
    /* Process first nnz */
    fm = 0;
    for(sptIndex m=0; m<nmodes; ++m) {
        if(flags[m] == 1) {
            block_coord[fm] = tsr->inds[m].data[0];    // first nonzero indices
            ++ fm;
        }
    }
    sptAssert(fm == ncmodes);
    result = sptLocateBeginCoord(block_begin_prior, ncmodes, block_coord, sb_bits);
    spt_CheckError(result, "HiSpTns Convert", NULL);
    for(sptIndex m=0; m<ncmodes; ++m) {
        sptAppendBlockIndexVector(&hitsr->binds[m], (sptBlockIndex)block_begin_prior[m]);
    }
    sptAppendNnzIndexVector(&hitsr->bptr, 0);


    /* Loop nonzeros in each kernel */
    for(sptNnzIndex z = 0; z < hitsr->nnz; ++z) {
        sptIndex fm = 0;
        #if PARTI_DEBUG == 5
            printf("z: %"PARTI_PRI_NNZ_INDEX "\n", z);
        #endif

        fm = 0;
        for(sptIndex m=0; m<nmodes; ++m) {
            if(flags[m] == 1) {
                block_coord[fm] = tsr->inds[m].data[z];    // first nonzero indices
                ++ fm;
            }
        }
        sptAssert(fm == ncmodes);
        #if PARTI_DEBUG == 5
            printf("block_coord:\n");
            sptAssert(sptDumpIndexArray(block_coord, ncmodes, stdout) == 0);
        #endif

        result = sptLocateBeginCoord(block_begin, ncmodes, block_coord, sb_bits);
        // spt_CheckError(result, "HiSpTns Convert", NULL);
        #if PARTI_DEBUG == 5
            printf("block_begin_prior:\n");
            sptAssert(sptDumpIndexArray(block_begin_prior, ncmodes, stdout) == 0);
            printf("block_begin:\n");
            sptAssert(sptDumpIndexArray(block_begin, ncmodes, stdout) == 0);
        #endif

        result = sptBlockEnd(block_end, nmodes, tsr->ndims, flags, block_begin, sb);  // exclusive
        // spt_CheckError(result, "HiSpTns Convert", NULL);

        /* Append einds */
        fm = 0;
        for(sptIndex m=0; m<nmodes; ++m) {
            if(flags[m] == 1) {
                eindex = tsr->inds[m].data[z] < (block_begin[fm] << sb_bits) ? tsr->inds[m].data[z] : tsr->inds[m].data[z] - (block_begin[fm] << sb_bits);
                sptAssert(eindex < sb);
                sptAppendElementIndexVector(&hitsr->einds[fm], (sptElementIndex)eindex);
                ++ fm;
            }
        }
        sptAssert(fm == ncmodes);

        /* Append uncompressed inds and values */
        fm = 0;
        for(sptIndex m=0; m<nmodes; ++m) {
            if(flags[m] == 0) {
                sptAppendIndexVector(&hitsr->inds[fm], tsr->inds[m].data[z]);
                ++ fm;
            }
        }
        sptAssert(fm == nmodes - ncmodes);
        sptAppendValueVector(&hitsr->values, tsr->values.data[z]);


        /* z in the same block with last z */
        if (sptEqualWithTwoCoordinates(block_begin, block_begin_prior, ncmodes) == 1)
        {
            /* ne: #Elements in current block */
            ++ ne;
        } else { /* New block */
            /* ne: #Elements in the last block */
            /* Append block bptr and bidx */
            sptAppendNnzIndexVector(&hitsr->bptr, (sptBlockIndex)z);
            for(sptIndex m=0; m<ncmodes; ++m)
                sptAppendBlockIndexVector(&hitsr->binds[m], (sptBlockIndex)block_begin[m]);
            for(sptIndex m=0; m<ncmodes; ++m)
                block_begin_prior[m] = block_begin[m];

            ++ nb;
            ne = 1;              
        } // End new block
        #if PARTI_DEBUG == 5
            printf("nb: %u, ne: %u\n\n", nb, ne);
        #endif

    }   // End z loop
    
    sptAssert(nb <= nnz);
    sptAssert(nb == hitsr->binds[0].len); 

    /* Last element for bptr */
    sptAppendNnzIndexVector(&hitsr->bptr, nnz);


    *max_nnzb = hitsr->bptr.data[1] - hitsr->bptr.data[0];
    sptNnzIndex sum_nnzb = 0;
    for(sptIndex i=0; i < hitsr->bptr.len - 1; ++i) {
        sptNnzIndex nnzb = hitsr->bptr.data[i+1] - hitsr->bptr.data[i];
        sum_nnzb += nnzb;
        if(*max_nnzb < nnzb) {
          *max_nnzb = nnzb;
        }
    }
    sptAssert(sum_nnzb == hitsr->nnz);

    sptStopTimer(gen_timer);
    sptPrintElapsedTime(gen_timer, "\tGenerate HiCOO-General");
    sptFreeTimer(gen_timer);


    free(block_begin);
    free(block_end);
    free(block_begin_prior);
    free(block_coord);

    return 0;
}