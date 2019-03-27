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
#include <stdio.h>


/**
 * Save the contents of a HiCOO-General sparse tensor into a text file
 * @param hitsr         th sparse tensor used to write
 * @param start_index the index of the first element in array. Set to 1 for MATLAB compability, else set to 0
 * @param fp          the file to write into
 */
int sptDumpSemiSparseTensor(sptSemiSparseTensor * const tsr, FILE *fp) 
{
    int iores;
    sptIndex mode;

    iores = fprintf(fp, "NNZ: %"PASTA_PRI_NNZ_INDEX"\n", tsr->nnz);
    spt_CheckOSError(iores < 0, "SpTns Dump");
    iores = fprintf(fp, "dense mode: %"PASTA_PRI_INDEX"\n", tsr->mode);
    spt_CheckOSError(iores < 0, "SpTns Dump");
    fprintf(fp, "ndims:\n");
    for(mode = 0; mode < tsr->nmodes; ++mode) {
        if(mode != 0) {
            iores = fputs("x", fp);
            spt_CheckOSError(iores < 0, "SpTns Dump");
        }
        iores = fprintf(fp, "%u", tsr->ndims[mode]);
        spt_CheckOSError(iores < 0, "SpTns Dump");
    }
    fputs("\n", fp);

    fprintf(fp, "inds:\n");
    for(mode = 0; mode < tsr->nmodes - 1; ++mode) {
        sptDumpIndexVector(&tsr->inds[mode], fp);
    }
    fprintf(fp, "values:\n");
    sptDumpMatrix(&tsr->values, fp);

    return 0;
}
