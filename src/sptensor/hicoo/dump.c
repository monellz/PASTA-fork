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
 * Save the contents of a HiCOO sparse tensor into a text file
 * @param hitsr         th sparse tensor used to write
 * @param start_index the index of the first element in array. Set to 1 for MATLAB compability, else set to 0
 * @param fp          the file to write into
 */
int sptDumpSparseTensorHiCOO(sptSparseTensorHiCOO * const hitsr, FILE *fp) 
{
    int iores;
    sptIndex mode;

    iores = fprintf(fp, "%"PASTA_PRI_INDEX ", NNZ: %"PASTA_PRI_NNZ_INDEX"\n", hitsr->nmodes, hitsr->nnz);
    spt_CheckOSError(iores < 0, "SpTns Dump");
    iores = fprintf(fp, "sb_bits: %"PASTA_PRI_ELEMENT_INDEX"\n", hitsr->sb_bits);
    spt_CheckOSError(iores < 0, "SpTns Dump");
    fprintf(fp, "ndims:\n");
    for(mode = 0; mode < hitsr->nmodes; ++mode) {
        if(mode != 0) {
            iores = fputs(" ", fp);
            spt_CheckOSError(iores < 0, "SpTns Dump");
        }
        iores = fprintf(fp, "%u", hitsr->ndims[mode]);
        spt_CheckOSError(iores < 0, "SpTns Dump");
    }
    fputs("\n", fp);
    fprintf(fp, "bptr:\n");
    sptDumpNnzIndexVector(&hitsr->bptr, fp);
    fprintf(fp, "binds:\n");
    for(mode = 0; mode < hitsr->nmodes; ++mode) {
        sptDumpBlockIndexVector(&hitsr->binds[mode], fp);
    }
    fprintf(fp, "einds:\n");
    for(mode = 0; mode < hitsr->nmodes; ++mode) {
        sptDumpElementIndexVector(&hitsr->einds[mode], fp);
    }
    fprintf(fp, "values:\n");
    sptDumpValueVector(&hitsr->values, fp);

    return 0;
}


/**
 * Save the contents of a HiCOO-General sparse tensor into a text file
 * @param hitsr         th sparse tensor used to write
 * @param start_index the index of the first element in array. Set to 1 for MATLAB compability, else set to 0
 * @param fp          the file to write into
 */
int sptDumpSparseTensorHiCOOGeneral(sptSparseTensorHiCOOGeneral * const hitsr, FILE *fp) 
{
    int iores;
    sptIndex mode;

    iores = fprintf(fp, "%u (Compressed %u) \n", hitsr->nmodes, hitsr->ncmodes);
    spt_CheckOSError(iores < 0, "SpTns Dump");
    iores = fprintf(fp, "NNZ: %"PASTA_PRI_NNZ_INDEX"\n", hitsr->nnz);
    spt_CheckOSError(iores < 0, "SpTns Dump");
    iores = fprintf(fp, "sb_bits: %"PASTA_PRI_ELEMENT_INDEX"\n", hitsr->sb_bits);
    spt_CheckOSError(iores < 0, "SpTns Dump");
    fprintf(fp, "ndims:\n");
    for(mode = 0; mode < hitsr->nmodes; ++mode) {
        if(mode != 0) {
            iores = fputs(" ", fp);
            spt_CheckOSError(iores < 0, "SpTns Dump");
        }
        iores = fprintf(fp, "%u", hitsr->ndims[mode]);
        spt_CheckOSError(iores < 0, "SpTns Dump");
    }
    fputs("\n", fp);

    fprintf(fp, "flags:\n");
    for(mode = 0; mode < hitsr->nmodes; ++mode) {
        if(mode != 0) {
            iores = fputs(" ", fp);
            spt_CheckOSError(iores < 0, "SpTns Dump");
        }
        iores = fprintf(fp, "%u", hitsr->flags[mode]);
        spt_CheckOSError(iores < 0, "SpTns Dump");
    }
    fputs("\n", fp);
    
    fprintf(fp, "sortorder:\n");
    for(mode = 0; mode < hitsr->nmodes; ++mode) {
        if(mode != 0) {
            iores = fputs(" ", fp);
            spt_CheckOSError(iores < 0, "SpTns Dump");
        }
        iores = fprintf(fp, "%u", hitsr->sortorder[mode]);
        spt_CheckOSError(iores < 0, "SpTns Dump");
    }
    fputs("\n", fp);

    fprintf(fp, "bptr:\n");
    sptDumpNnzIndexVector(&hitsr->bptr, fp);
    fprintf(fp, "binds:\n");
    for(mode = 0; mode < hitsr->ncmodes; ++mode) {
        sptDumpBlockIndexVector(&hitsr->binds[mode], fp);
    }
    fprintf(fp, "einds:\n");
    for(mode = 0; mode < hitsr->ncmodes; ++mode) {
        sptDumpElementIndexVector(&hitsr->einds[mode], fp);
    }
    fprintf(fp, "inds:\n");
    for(mode = 0; mode < hitsr->nmodes - hitsr->ncmodes; ++mode) {
        sptDumpIndexVector(&hitsr->inds[mode], fp);
    }
    fprintf(fp, "values:\n");
    sptDumpValueVector(&hitsr->values, fp);

    return 0;
}
