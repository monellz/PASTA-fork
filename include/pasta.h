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

#ifndef PASTA_H
#define PASTA_H

#include <stddef.h>
#include <stdio.h>
#include <inttypes.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <time.h>
#include <assert.h>
#ifdef PASTA_USE_OPENMP
    #include <omp.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif


/*************************************************
 * TYPES
 *************************************************/
#include "includes/types.h"


/*************************************************
 * MACROS
 *************************************************/
#include "includes/macros.h"


/*************************************************
 * STRUCTS
 *************************************************/
#include "includes/structs.h"


/*************************************************
 * HELPER FUNCTIONS
 *************************************************/
#include "includes/helper_funcs.h"
#include "includes/error.h"

/*************************************************
 * FUNCTIONS
 *************************************************/
/* Vector functions */
#include "includes/vectors.h"
/* Dense matrix functions */
#include "includes/matrices.h"
/* Sparse tensor functions */
#include "includes/sptensors.h"
/* Semi-sparse tensor functions */
#include "includes/ssptensors.h"


#ifdef __cplusplus
}
#endif

#endif
