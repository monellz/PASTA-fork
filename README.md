PASTA
------

PASTA, A Parallel Sparse Tensor Algorithm Benchmark Suite, provides essential sparse tensor operations from various real-world applications for multicore CPU and GPU architectures. These tensor operations are critical to the overall performance of tensor analysis algorithms (such as tensor decompositions). 


## Supported sparse tensor operations:

* Scala-tensor mul/div [CPU]
* Kronecker product [CPU]
* Khatri-Rao product [CPU]
* Sparse tensor matricization [CPU]
* Element-wise tensor add/sub/mul/div [CPU, Multicore, GPU]
* Sparse tensor-times-dense matrix (SpTTM) [CPU, Multicore, GPU]
* Sparse matricized tensor times Khatri-Rao product (SpMTTKRP) [CPU, Multicore, GPU]
* Sparse tensor matricization [CPU]


## Supported sparse tensor formats:

* Coordinate (COO) format
* Hierarchical coordinate (HiCOO) format [[Paper]](http://fruitfly1026.github.io/static/files/sc18-li.pdf)


## Build requirements:

- C Compiler (GCC or ICC or Clang)

- [CMake](https://cmake.org) (>v3.2)

- [CUDA SDK](https://developer.nvidia.com/cuda-downloads) [Required for GPU algorithms]


## Build:

1. Create a file by `cp build.config build-sample.config' and change the settings appropriately

2. Type `./build.sh`

3. Check `build` for resulting library

4. Check `build/tests` for testing programs, for basic functionality

5. Check `build/examples` for example programs including MTTKRP, TTM
    

## Build docs:

1. Install Doxygen

2. Go to `docs`

3. Type `make`


## Run examples:

Please refer to `GettingStarted.md` for more general cases. Only some major functions are shown below.

**_MTTKRP_**: 
1. COO-MTTKRP (CPU, Multicore)

    * Usage: ./build/examples/mttkrp [options], Options:
      * -i INPUT, --input=INPUT (.tns file)
      * -o OUTPUT, --output=OUTPUT (output file name)
      * -m MODE, --mode=MODE (default -1: loop all modes, or specify a mode, e.g., 0 or 1 or 2 for third-order tensors.)
      * -s sortcase, --sortcase=SORTCASE (0:default,1,2,3,4. Different tensor sorting.)
      * -b BLOCKSIZE, --blocksize=BLOCKSIZE (in bits) (Only for sortcase=3)
      * -k KERNELSIZE, --kernelsize=KERNELSIZE (in bits) (Only for sortcase=3)
      * -d DEV_ID, --dev-id=DEV_ID (-2:sequential,default; -1:OpenMP parallel)
      * -r RANK (the number of matrix columns, 16:default)
      * OpenMP options: 
      * -t NTHREADS, --nt=NT (1:default)
      * -u use_reduce, --ur=use_reduce (use privatization or not)
      * --help

<!---
    tsr mode impl_num [cuda_dev_id, R, output]
    * tsr: input sparse tensor
    * mode: specify tensor mode, e.g. (0, or 1, or 2) for third-order tensors
    * impl_num: 11, 12, 15, where 15 should be the best case
    * cuda_dev_id: -2, -1, or 0, 1, -2: sequential code; -1: omp code; 0, or other possible integer: GPU devide id. [Optinal, -2 by default]
    * R: rank number (matrix column size), an integer. [Optinal, 16 by default]
    * output: the file name for output. [Optinal]
    * An example: ./build/examples/mttkrp example.tns 0 15 0 16 result.txt
--->

2. COO-MTTKRP (GPU)

    * Usage: ./build/examples/mttkrp_gpu [options], Options:
      * -i INPUT, --input=INPUT (.tns file)
      * -o OUTPUT, --output=OUTPUT (output file name)
      * -m MODE, --mode=MODE (specify a mode, e.g., 0 or 1 or 2 for third-order tensors. Default:0.)
      * -s sortcase, --sortcase=SORTCASE (0:default,1,2,3,4. Different tensor sorting.)
      * -b BLOCKSIZE, --blocksize=BLOCKSIZE (in bits) (Only for sortcase=3)
      * -k KERNELSIZE, --kernelsize=KERNELSIZE (in bits) (Only for sortcase=3)
      * -d CUDA_DEV_ID,, --cuda-dev-id=CUDA_DEV_ID (>0:GPU device id)
      * -r RANK (the number of matrix columns, 16:default)
      * GPU options: 
      * -p IMPL_NUM, --impl-num=IMPL_NUM (11, 12, 15, where 15 should be the best case)
      * --help


3. HiCOO-MTTKRP (CPU, Multicore)

    * Usage: ./build/examples/mttkrp_hicoo [options], Options:
      * -i INPUT, --input=INPUT (.tns file)
      * -o OUTPUT, --output=OUTPUT (output file name)
      * -m MODE, --mode=MODE (default -1: loop all modes, or specify a mode, e.g., 0 or 1 or 2 for third-order tensors.)
      * -b BLOCKSIZE, --blocksize=BLOCKSIZE (in bits) (required)
      * -k KERNELSIZE, --kernelsize=KERNELSIZE (in bits) (required)
      * -d DEV_ID, --dev-id=DEV_ID (-2:sequential,default; -1:OpenMP parallel)
      * -r RANK (the number of matrix columns, 16:default)
      * OpenMP options: 
      * -t NTHREADS, --nt=NT (1:default)
      * --help

    
**_TTM_**: 
1. COO-TTM (CPU, Multicore, GPU)

    * Usage: ./build/examples/ttm tsr mode impl_num smem_size [cuda_dev_id, R, output]
    * tsr: input sparse tensor
    * mode: specify tensor mode, e.g. (0, or 1, or 2) for third-order tensors
    * impl_num: 11, 12, 13, 14, 15, where either 14 or 15 should be the best case
    * smem_size: shared memory size in bytes (0, or 16000, or 32000, or 48000) 
    * cuda_dev_id: -2, -1, or 0, 1, ... -2: sequential code; -1: omp code; 0, or other possible integer: GPU devide id. [Optinal, -2 by default]
    * R: rank number (matrix column size), an integer. [Optinal, 16 by default]
    * output: the file name for output. [Optinal]
    * An example: ./build/examples/ttm example.tns 0 15 16000 0 16 result.txt

2. SCOO-TTM (CPU, GPU)

    * Usage: ./build/examples/sttm tsr U Y mode [cuda_dev_id]
    * tsr: input semi-sparse tensor
    * U: input dense matrix
    * Y: output semi-sparse tensor
    * mode: specify tensor mode, e.g. (0, or 1, or 2) for third-order tensors
    * cuda_dev_id: -1, or 0, 1, ... -1: sequential code; 0, or other possible integer: GPU devide id. [Optinal, -1 by default]


<br/>The algorithms and details are described in the following publications.

## Publication
* **Scalable Tensor Decompositions in High Performance Computing Environments**. Jiajia Li. PhD Dissertation. Georgia Institute of Technology, Atlanta, GA, USA. July 2018. [[pdf]](http://fruitfly1026.github.io/static/files/LI-DISSERTATION-2018.pdf) [[bib]](http://fruitfly1026.github.io/static/files/LI-DISSERTATION-2018-bib.txt) 

* **HiCOO: Hierarchical Storage of Sparse Tensors**. Jiajia Li, Jimeng Sun, Richard Vuduc. ACM/IEEE International Conference for High-Performance Computing, Networking, Storage, and Analysis (SC). 2018 (accepted). [[pdf]](http://fruitfly1026.github.io/static/files/sc18-li.pdf) [[bib]](http://fruitfly1026.github.io/static/files/sc18-li-bib.txt)

* **Optimizing Sparse Tensor Times Matrix on GPUs**. Yuchen Ma, Jiajia Li, Xiaolong Wu, Chenggang Yan, Jimeng Sun, Richard Vuduc. Journal of Parallel and Distributed Computing (Special Issue on Systems for Learning, Inferencing, and Discovering). [[pdf]](http://fruitfly1026.github.io/static/files/jpdc-ma.pdf) [[bib]](http://fruitfly1026.github.io/static/files/jpdc-ma-bib.txt) 

* **Optimizing Sparse Tensor Times Matrix on multi-core and many-core architectures**. Jiajia Li, Yuchen Ma, Chenggang Yan, Richard Vuduc. The sixth Workshop on Irregular Applications: Architectures and Algorithms (IA^3), co-located with SC’16. 2016. [[pdf]](http://fruitfly1026.github.io/static/files/sc16-ia3.pdf)

* **ParTI!: a Parallel Tensor Infrastructure for Data Analysis**. Jiajia Li, Yuchen Ma, Chenggang Yan, Jimeng Sun, Richard Vuduc. Tensor-Learn Workshop @ NIPS'16. [[pdf]](http://fruitfly1026.github.io/static/files/nips16-tensorlearn.pdf)

## Citation
```
@misc{pasta,
author="Jiajia Li and Kevin Barker",
title="{PASTA}: A Parallel Sparse Tensor Algorithm Benchmark Suite",
month="Dec",
year="2019",
url="https://gitlab.com/tensorworld/pasta"
}
```

## Contributiors

* Jiajia Li (Contact: jiajia.li@pnnl.gov)

## License

PASTA is free software: you can redistribute it and/or modify it under the terms of the GNU Lesser General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
