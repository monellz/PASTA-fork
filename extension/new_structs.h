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

#ifndef PASTA_NEW_STRUCTS_H
#define PASTA_NEW_STRUCTS_H

/* Add new tensor format below */
/* For example:
typedef struct {
    sptIndex nmodes;      /// # modes
    sptIndex * sortorder;  /// the order in which the indices are sorted
    sptIndex * ndims;      /// size of each mode, length nmodes
    sptNnzIndex nnz;         /// # non-zeros
    sptIndexVector * inds;       /// indices of each element, length [nmodes][nnz]
    sptValueVector values;      /// non-zero values, length nnz
} sptSparseTensor;
*/

#endif