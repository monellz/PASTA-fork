# Requires
cmake_minimum_required(VERSION 3.2)  # CMake 3.2 supports CUDA 7.0
project(ParTI)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake-modules")

# Enable unit test
include(CTest)
if(BUILD_TESTING)
  find_program(MEMORYCHECK_COMMAND valgrind)
  set(MEMORYCHECK_COMMAND_OPTIONS "--trace-children=yes --leak-check=full --show-reachable=yes --suppressions=${CMAKE_CURRENT_LIST_DIR}/valgrind.supp")
  option(CODE_COVERAGE "Enable code coverage test" OFF)
  if(CODE_COVERAGE)
    find_program(CTEST_COVERAGE_COMMAND gcov)
    include(CodeCoverage)
    APPEND_COVERAGE_COMPILER_FLAGS()
  endif()
endif()

# Define build ParTI library
option(BUILD_STATIC "Build static library" OFF)
option(BUILD_SHARED "Build shared library" ON)

# Define data types
add_definitions(-DPARTI_INDEX_TYPEWIDTH=32)
add_definitions(-DPARTI_VALUE_TYPEWIDTH=32)
add_definitions(-DPARTI_ELEMENT_INDEX_TYPEWIDTH=8)

# Check for implementations
option(USE_ICC "Use ICC" OFF)
option(USE_OPENMP "Use OPENMP" ON)
option(USE_CUDA "Use NVIDIA CUDA library" OFF)


# Check for debug mode
if (DEFINED DEBUG)
  message("Building in DEBUG mode. Expect for plenty of prints.")
  add_definitions(-DPARTI_DEBUG=${DEBUG})
  set(CMAKE_BUILD_TYPE Debug)
else()
  message("Building in RELEASE mode.")
  set(CMAKE_BUILD_TYPE Release)
endif()


if(USE_CUDA)
    find_package(CUDA REQUIRED)
    add_definitions(-DPARTI_USE_CUDA)
    link_libraries(${CUDA_cusparse_LIBRARY})
    link_libraries(${CUDA_cusolver_LIBRARY})
    if(DEFINED DEBUG)
        set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -O0 -g -gencode arch=compute_${CUDA_ARCH_BIN},code=sm_${CUDA_ARCH_BIN}")
    else()
        set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -O3 -gencode arch=compute_${CUDA_ARCH_BIN},code=sm_${CUDA_ARCH_BIN}")
    endif()
endif()

if(USE_OPENMP)
    add_definitions(-DPARTI_USE_OPENMP)
    if(USE_ICC)
        if(DEFINED DEBUG)
            set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=gnu99 -Wall -Wextra -Wno-maybe-uninitialized -Wno-unused-variable -lrt -qopenmp -O0 -g -DADD_")
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wno-maybe-uninitialized -lrt -qopenmp -O0 -g -DADD_")
        else()
            set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=gnu99 -Wall -Wextra -Wno-maybe-uninitialized -Wno-unused-variable -lrt -qopenmp -O3 -DADD_")
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wno-maybe-uninitialized -lrt -qopenmp -O3 -DADD_")
        endif()
    else()
        if(DEFINED DEBUG)
            set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -D_GNU_SOURCE -Wall -Wextra -Wno-maybe-uninitialized -Wno-unused-variable -lrt -fopenmp -O0 -g -DADD_")
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wno-maybe-uninitialized -fopenmp -O0 -g -DADD_")
        else()
            set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -D_GNU_SOURCE -Wall -Wextra -Wno-maybe-uninitialized -Wno-unused-variable -lrt -fopenmp -O3 -DADD_")
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wno-maybe-uninitialized -lrt -fopenmp -O3 -DADD_")
        endif()
    endif()
else()
    if(DEFINED DEBUG)
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -D_GNU_SOURCE -Wall -Wextra -Wno-maybe-uninitialized -Wno-unused-variable -lrt -O0 -g -DADD_")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wno-maybe-uninitialized -lrt -O0 -g -DADD_")
    else()
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -D_GNU_SOURCE -Wall -Wextra -Wno-maybe-uninitialized -Wno-unused-variable -lrt -O3 -DADD_")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wno-maybe-uninitialized -lrt -O3 -DADD_")
    endif()
endif()



# A bug in FindCUDA module forces us to put "include_directories" before "cuda_add_library".
# So we cannot use "target_include_directories" for target-wise include tracking.
include_directories("include")
link_libraries("m")

if(USE_CUDA)
    file(GLOB_RECURSE PARTI_SRC RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "src/*.c" "src/*.cu" "src/*.h" "include/*.h")
    # MESSAGE( STATUS "PARTI_SRC:         " ${PARTI_SRC} )

    if(BUILD_SHARED)
        cuda_add_library(ParTI SHARED ${PARTI_SRC})
        CUDA_ADD_CUBLAS_TO_TARGET(ParTI)
    endif()
    if(BUILD_STATIC)
        cuda_add_library(ParTI_s STATIC ${PARTI_SRC})
        CUDA_ADD_CUBLAS_TO_TARGET(ParTI_s)
    endif()
else()
    file(GLOB_RECURSE PARTI_SRC RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "src/*.c" "src/*.h" "include/*.h")

    if(BUILD_SHARED)
        add_library(ParTI SHARED ${PARTI_SRC})
    endif()
    if(BUILD_STATIC)
        add_library(ParTI_s STATIC ${PARTI_SRC})
    endif()
endif()

if(BUILD_SHARED)
    set_target_properties(ParTI PROPERTIES C_STANDARD 99)
endif()
if(BUILD_STATIC)
    set_target_properties(ParTI_s PROPERTIES C_STANDARD 99)
endif()

add_subdirectory("benchmarks")
add_subdirectory("tests")
