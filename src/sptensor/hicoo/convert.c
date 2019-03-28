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
#include "../sptensor.h"


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
 * Compute the end of this block
 * @param tsr    a pointer to a sparse tensor
 * @return out_item     the end indices of this block
 */
static int sptBlockEnd(
    sptIndex * out_item,
    sptIndex nmodes,
    sptIndex * ndims,
    sptIndex * sortorder,
    const sptIndex * in_item,
    const sptElementIndex sb)
{
    if(sortorder == NULL) {
        for(sptIndex m=0; m<nmodes; ++m) {
            sptAssert(in_item[m] < ndims[m]);
            out_item[m] = in_item[m]+sb < ndims[m] ? in_item[m]+sb : ndims[m];    // exclusive
        }
    } else {
        sptIndex fm = 0;
        for(sptIndex m=0; m<nmodes; ++m) {
            sptIndex fm = sortorder[m];
            sptAssert(in_item[m] < ndims[fm]);
            out_item[m] = in_item[m]+sb < ndims[fm] ? in_item[m]+sb : ndims[fm];    // exclusive
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
        sptSparseTensorSortIndexRowBlock(tsr, 1, 0, tsr->nnz, sb_bits, NULL);
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
#if PASTA_DEBUG >= 2
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
        #if PASTA_DEBUG == 5
            printf("z: %"PASTA_PRI_NNZ_INDEX "\n", z);
        #endif

        for(sptIndex m=0; m<nmodes; ++m) 
            block_coord[m] = tsr->inds[m].data[z];    // first nonzero indices
        #if PASTA_DEBUG == 5
            printf("block_coord:\n");
            sptAssert(sptDumpIndexArray(block_coord, nmodes, stdout) == 0);
        #endif

        result = sptLocateBeginCoord(block_begin, nmodes, block_coord, sb_bits);
        // spt_CheckError(result, "HiSpTns Convert", NULL);
        #if PASTA_DEBUG == 5
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
        #if PASTA_DEBUG == 5
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
    sptIndex * sortorder = hitsr->sortorder;

    /* Pre-process tensor to get hitsr->kptr, values are nonzero locations. */
    sptTimer sort_timer;
    sptNewTimer(&sort_timer, 0);
    sptStartTimer(sort_timer);

    /* Sort only block indices. Keep the sorting order in the same fiber. */
    sptSparseTensorSortIndexRowBlock(tsr, 1, 0, tsr->nnz, sb_bits, flags);
    // printf("Sort by blocks:\n");
    // sptAssert(sptDumpSparseTensor(tsr, 0, stdout) == 0);

    sptStopTimer(sort_timer);
    sptPrintElapsedTime(sort_timer, "\tHiCOO RowBlock sorting");
    sptFreeTimer(sort_timer);
    // printf("RowBlock sorted:\n");
    // sptAssert(sptDumpSparseTensor(tsr, 0, stdout) == 0);

    sptTimer timer;
    sptNewTimer(&timer, 0);
    sptStartTimer(timer);

    /* Temporary storage */
    sptIndex * block_begin = (sptIndex *)malloc(ncmodes * sizeof(*block_begin));
    sptIndex * block_end = (sptIndex *)malloc(ncmodes * sizeof(*block_end));
    sptIndex * block_begin_prior = (sptIndex *)malloc(ncmodes * sizeof(*block_begin_prior));
    sptIndex * block_coord = (sptIndex *)malloc(ncmodes * sizeof(*block_coord));

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
    for(sptIndex m=0; m<ncmodes; ++m) {
        sptIndex fm = sortorder[m];
        block_coord[m] = tsr->inds[fm].data[0];    // first nonzero indices
    }

    result = sptLocateBeginCoord(block_begin_prior, ncmodes, block_coord, sb_bits);
    spt_CheckError(result, "HiSpTns Convert", NULL);

    for(sptIndex m=0; m<ncmodes; ++m) {
        sptAppendBlockIndexVector(&hitsr->binds[m], (sptBlockIndex)block_begin_prior[m]);
    }
    sptAppendNnzIndexVector(&hitsr->bptr, 0);


    /* Loop nonzeros in each kernel */
    for(sptNnzIndex z = 0; z < hitsr->nnz; ++z) {
        // printf("z: %"PASTA_PRI_NNZ_INDEX "\n", z);
        for(sptIndex m=0; m<ncmodes; ++m) {
            sptIndex fm = sortorder[m];
            block_coord[m] = tsr->inds[fm].data[z];    // first nonzero indices
        }
        // printf("block_coord:\n");
        // sptAssert(sptDumpIndexArray(block_coord, ncmodes, stdout) == 0);

        result = sptLocateBeginCoord(block_begin, ncmodes, block_coord, sb_bits);
        spt_CheckError(result, "HiSpTns Convert", NULL);
        // printf("block_begin_prior:\n");
        // sptAssert(sptDumpIndexArray(block_begin_prior, ncmodes, stdout) == 0);
        // printf("block_begin:\n");
        // sptAssert(sptDumpIndexArray(block_begin, ncmodes, stdout) == 0);

        result = sptBlockEnd(block_end, ncmodes, tsr->ndims, sortorder, block_begin, sb);  // exclusive
        spt_CheckError(result, "HiSpTns Convert", NULL);
        // printf("block_end:\n");
        // sptAssert(sptDumpIndexArray(block_end, ncmodes, stdout) == 0);

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
        // printf("nb: %lu, ne: %lu\n\n", nb, ne);

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

    sptStopTimer(timer);
    sptPrintElapsedTime(timer, "\tGenerate HiCOO-General");

    free(block_begin);
    free(block_end);
    free(block_begin_prior);

    sptStartTimer(timer);
    for(sptNnzIndex b = 0; b < hitsr->bptr.len - 1; ++ b) {

        sptNnzIndex b_begin = hitsr->bptr.data[b];
        sptNnzIndex b_end = hitsr->bptr.data[b + 1];
        sptSparseTensorSortIndexCustomOrder(tsr, sortorder, b_begin, b_end, 1);
        
        for(sptIndex m=0; m<ncmodes; ++m) {
            sptIndex fm = sortorder[m];
            block_coord[m] = (sptIndex)hitsr->binds[m].data[b];
        }

        for(sptNnzIndex z = b_begin; z < b_end; ++z) {
            /* Append einds */
            for(sptIndex m=0; m<ncmodes; ++m) {
                sptIndex fm = sortorder[m];
                    eindex = tsr->inds[fm].data[z] < (block_coord[m] << sb_bits) ? tsr->inds[fm].data[z] : tsr->inds[fm].data[z] - (block_coord[m] << sb_bits);
                sptAssert(eindex < sb);
                sptAppendElementIndexVector(&hitsr->einds[m], (sptElementIndex)eindex);
            }

            /* Append uncompressed inds and values */
            for(sptIndex m=ncmodes; m<nmodes; ++m) {
                sptIndex fm = sortorder[m];
                sptAppendIndexVector(&hitsr->inds[m - ncmodes], tsr->inds[fm].data[z]);
            }
            sptAppendValueVector(&hitsr->values, tsr->values.data[z]);
        }
    }
    free(block_coord);

    sptStopTimer(timer);
    sptPrintElapsedTime(timer, "\tSort inside each block and set einds and values");
    sptFreeTimer(timer);

    return 0;
}