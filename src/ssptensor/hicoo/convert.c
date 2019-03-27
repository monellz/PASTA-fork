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

int sptSemiHiCOOToSemiSparseTensor(
    sptSemiSparseTensor *stsr, 
    sptSemiSparseTensorHiCOO *histsr)
{
    sptIndex const nmodes = histsr->nmodes;
    sptIndex const mode = histsr->mode;
    sptNnzIndex const nnz = histsr->nnz;
    sptIndex const stride = histsr->values.stride;
    int result;

    result = sptNewSemiSparseTensor(stsr, nmodes, mode, histsr->ndims);
    spt_CheckOSError(result, "Convert Semi-HiCOO -> Semi-COO");
    stsr->nnz = histsr->nnz;
    for(sptIndex m=0; m<nmodes - 1; ++m) {
        result = sptResizeIndexVector(&(stsr->inds[m]), nnz);
        spt_CheckOSError(result, "Convert Semi-HiCOO -> Semi-COO");
    }
    result = sptResizeMatrix(&(stsr->values), nnz);
    spt_CheckOSError(result, "Convert Semi-HiCOO -> Semi-COO");

    sptIndex * block_coord = (sptIndex*)malloc((nmodes - 1) * sizeof(*block_coord));
    sptIndex ele_coord;


    /* Loop blocks in a kernel */
    for(sptIndex b=0; b<histsr->bptr.len - 1; ++b) {
        /* Block indices */
        for(sptIndex m=0; m<nmodes - 1; ++m)
            block_coord[m] = histsr->binds[m].data[b] << histsr->sb_bits;

        sptNnzIndex bptr_begin = histsr->bptr.data[b];
        sptNnzIndex bptr_end = histsr->bptr.data[b+1];
        /* Loop entries in a block */
        for(sptNnzIndex z=bptr_begin; z<bptr_end; ++z) {
            /* Element indices */
            for(sptIndex m=0; m<nmodes - 1; ++m) {
                ele_coord = block_coord[m] + histsr->einds[m].data[z];
                stsr->inds[m].data[z] = ele_coord;
            }
            for(sptIndex s=0; s<histsr->values.ncols; ++s) {
                stsr->values.values[z * stride + s] = histsr->values.values[z * stride + s];
            }   
        }
    }

    free(block_coord);

    return 0;
}