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
#include <assert.h>

void sptSparseTensorStatusHiCOO(sptSparseTensorHiCOO *hitsr, FILE *fp)
{
  sptIndex nmodes = hitsr->nmodes;
  fprintf(fp, "HiCOO Sparse Tensor information ---------\n");
  fprintf(fp, "DIMS = %"PASTA_PRI_INDEX, hitsr->ndims[0]);
  for(sptIndex m=1; m < nmodes; ++m) {
    fprintf(fp, "x%"PASTA_PRI_INDEX, hitsr->ndims[m]);
  }
  fprintf(fp, "\n");
  fprintf(fp, "NNZ = %"PASTA_PRI_NNZ_INDEX"\n", hitsr->nnz);
  fprintf(fp, "sb = %"PASTA_PRI_INDEX"\n", (sptIndex)pow(2, hitsr->sb_bits));
  fprintf(fp, "nb = %"PASTA_PRI_NNZ_INDEX"\n", hitsr->bptr.len - 1);

  sptNnzIndex bytes = hitsr->nnz * ( sizeof(sptValue) + nmodes * sizeof(sptElementIndex) );
  bytes += hitsr->binds[0].len * nmodes * sizeof(sptBlockIndex);
  bytes += hitsr->bptr.len * sizeof(sptNnzIndex);

  char * bytestr = sptBytesString(bytes);
  fprintf(fp, "HiCOO-STORAGE=%s\n", bytestr);
  free(bytestr);

  sptIndex sb = (sptIndex)pow(2, hitsr->sb_bits);
  sptNnzIndex max_nnzb = hitsr->bptr.data[1] - hitsr->bptr.data[0];
  sptNnzIndex min_nnzb = hitsr->bptr.data[1] - hitsr->bptr.data[0];
  sptNnzIndex sum_nnzb = 0;
  double geo_mean_nnzb = 1;
  sptNnzIndex nb = hitsr->bptr.len - 1;
  sptNnzIndex * nnzb_array = (sptNnzIndex *)malloc(nb * sizeof(* nnzb_array));
  for(sptNnzIndex i=0; i < hitsr->bptr.len - 1; ++i) {
    sptNnzIndex nnzb = hitsr->bptr.data[i+1] - hitsr->bptr.data[i];
    if(max_nnzb < nnzb) {
      max_nnzb = nnzb;
    }
    if(min_nnzb > nnzb) {
      min_nnzb = nnzb;
    }
    sum_nnzb += nnzb;
    geo_mean_nnzb *= pow( (double)nnzb / sb, 1.0/nb );
    nnzb_array[i] = nnzb;
  }
  assert(sum_nnzb == hitsr->nnz);
  sptNnzIndex avg_nnzb = (sptNnzIndex)sum_nnzb / (hitsr->bptr.len - 1);

  /* Compute median */
  sptQuickSortNnzIndexArray(nnzb_array, 0, nb);
  sptNnzIndex median_loc = (nb + 1) / 2 - 1;
  assert (median_loc >= 0);
  sptNnzIndex median_nnzb = nnzb_array[median_loc];
  free(nnzb_array);
  
  fprintf(fp, "block nnzs:\n");
  fprintf(fp, "Nnzb: Max = %" PASTA_PRI_NNZ_INDEX ", Min = %" PASTA_PRI_NNZ_INDEX ", Avg = %" PASTA_PRI_NNZ_INDEX "\n", max_nnzb, min_nnzb, avg_nnzb);
  fprintf(fp, "cb: Max = %.3lf, Min = %.3lf, Avg = %.3lf\n", (double)max_nnzb / sb, (double)min_nnzb / sb, (double)avg_nnzb / sb);
  fprintf(fp, "median cb: %.3lf, geometric mean cb: %.3lf\n", (double)median_nnzb / sb, geo_mean_nnzb);
  fprintf(fp, "alpha_b: %lf\n", (double)(hitsr->bptr.len - 1) / hitsr->nnz);

  fprintf(fp, "\nParameter configuration --------\n");
  fprintf(fp, "Suggest B (sb) <= %.2lf / R. For cache efficiency\n", (double)L1_SIZE / hitsr->nmodes / sizeof(sptValue));
  fprintf(fp, "Suggest alpha_b in (0,1], small is better. For tensor storage\n");
  fprintf(fp, "Suggest cb > 1, large is better. For MTTKRP performance\n");
  fprintf(fp, "Suggest num_tasks should in [%d, %d] PAR_DEGREE: [%d, %d]. For parallel efficiency\n", PAR_MIN_DEGREE * NUM_CORES, PAR_MAX_DEGREE * NUM_CORES, PAR_MIN_DEGREE, PAR_MAX_DEGREE);
  fprintf(fp, "\n\n");

}



void sptSparseTensorStatusHiCOOGeneral(sptSparseTensorHiCOOGeneral *hitsr, FILE *fp)
{
  sptIndex nmodes = hitsr->nmodes;
  sptIndex ncmodes = hitsr->ncmodes;
  fprintf(fp, "HiCOO-General Sparse Tensor information ---------\n");
  fprintf(fp, "%u (Compressed %u) \n", hitsr->nmodes, hitsr->ncmodes);
  fprintf(fp, "DIMS=%"PASTA_PRI_INDEX, hitsr->ndims[0]);
  for(sptIndex m=1; m < nmodes; ++m) {
    fprintf(fp, "x%"PASTA_PRI_INDEX, hitsr->ndims[m]);
  }
  fprintf(fp, "\n");
  fprintf(fp, "NNZ=%"PASTA_PRI_NNZ_INDEX"\n", hitsr->nnz);
  fprintf(fp, "sb=%"PASTA_PRI_INDEX"\n", (sptIndex)pow(2, hitsr->sb_bits));
  fprintf(fp, "nb=%"PASTA_PRI_NNZ_INDEX"\n", hitsr->bptr.len - 1);

  sptNnzIndex bytes = hitsr->nnz * ( sizeof(sptValue) + ncmodes * sizeof(sptElementIndex) + (nmodes - ncmodes) * sizeof(sptIndex) );
  bytes += hitsr->binds[0].len * ncmodes * sizeof(sptBlockIndex);
  bytes += hitsr->bptr.len * sizeof(sptNnzIndex);

  char * bytestr = sptBytesString(bytes);
  fprintf(fp, "HiCOO-STORAGE=%s\n", bytestr);
  free(bytestr);

}
