Getting started with ParTI!
===========================


To use ParTI! in a C/C++ project, simply add
```c
#include <ParTI.h>
```
to your source code.

Link your code with
```sh
-fopenmp -lParTI -lm
```

This intro document can help you get used to the basics of ParTI!.


Data types
----------

`sptIndex`: the default integer value data type for tensor indices. It is defined as `uint32_t` or `uint64_t` type. For the large tensors we tested, `uint32_t` is enough for tensor indices. (More data types are used for HiCOO, `sptElementIndex` (`uint8_t` by default) and `sptBlockIndex` (also `uint32_t` by default).)

`sptNnzIndex`: the default integer value data type for tensor nonzeros. It is defined as `uint64_t` type to index the nonzero location.

`sptValue`: the default real value data type. It is defined as `float` or `double` type. For some devices without 64-bit float point support or data is out-of-memory in `double` type, you might need to define `sptValue` as `float`.

`sptIndexVector`: dense dynamic array of `sptIndex` type scalars. This is implemented twice to avoid templates for CUDA code.

`sptValueVector`: dense dynamic array of `sptValue` type scalars. It is implemented as a one-dimensional array. It uses preallocation to reduce the overhead of the append operation.

`sptMatrix`: dense matrix type. It is implemented as a two-dimensional array. Column count is aligned as multiples of 8.

`sptSparseMatrix`: sparse matrix type in coordinate (COO) format. It stores the coordinates and the value of every non-zero entry.

`sptSparseTensorHiCOO`: sparse tensor type in coordinate (HiCOO) format (details explained in [our SC19 paper](http://fruitfly1026.github.io/static/files/sc18-li.pdf).

`sptSemiSparseTensor`: semi-sparse tensor type in sCOO format (details explained in [our SC16-IA3 paper](http://fruitfly1026.github.io/static/files/sc16-ia3.pdf). It can be considered as "a sparse tensor with dense fibers".

`sptSemiSparseTensor`: semi-sparse tensor type in sCOO format (details explained in [our SC16-IA3 paper](http://fruitfly1026.github.io/static/files/sc16-ia3.pdf). It can be considered as "a sparse tensor with dense fibers".


Creating objects
----------------

Most data types can fit themselves into stack memory, as local variables. They will handle extra memory allocations on demand.

For example, to construct an `sptVector` and use it.

```c
// Construct it
sptValueVector my_vector;
sptNewValueVector(&my_vector, 0, 0);

// Add values to it
sptAppendValueVector(&my_vector, 42);
sptAppendValueVector(&my_vector, 31);

// Copy it to another uninitialized vector
sptValueVector another_vector;
sptCopyValueVector(&another_vector, &my_vector);

// Access data
printf("%lf %lf\n", another_vector.data[0], another_vector.data[1]);

// Free memory
sptFreeValueVector(&my_vector);
sptFreeValueVector(&another_vector);
```

Most functions require initialized data structures. While functions named `New` or `Copy` require uninitialized data structions. They are states in the Doxygen document on a function basis. Failing to supply data with correct initialization state may result in memory leak or program crash.


Validation
----------

For the sake of simplicity, properties are not designed. You can directly modify any field of any struct.

Every function assumes the input is valid, and guarantees the output is valid. This reduces the the need to check the input for most of the time, and improves the performance as a math library.

But if you modify the data structure directly, you must keep it valid. Some functions expect ordered input, you should sort them with functions like `sptSparseTensorSortIndex` after your modification (other sortings are also supported, e.g., Z-Morton order sorting, sorting in all but one mode), or the functions may not work correctly. These functions usually also produces ordered output.


Error reporting
---------------

Most functions return 0 when it succeeded, non-zero when failed.

By invoking `sptGetLastError`, you can extract the last error information.

Operating system `errno` and CUDA error code are also captured and converted.

If you need to make sure a procedure produces no error, call `sptClearLastError` first, since succes procedures does not clear last error status automatically.

Limitation: Memory might not be released properly when an error happened. The application will be in an inconsistent state. This might be fixed in future releases.
