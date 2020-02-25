cmake_minimum_required(VERSION 3.1)
project(chainerx CXX C)

# CMake setup
if(POLICY CMP0054)
    cmake_policy(SET CMP0054 NEW)
endif()

set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake;${CMAKE_MODULE_PATH}")
if(NOT APPLE)
    set(CMAKE_INSTALL_RPATH $ORIGIN)
else()
    set(CMAKE_INSTALL_RPATH "@loader_path")
endif()


# includes
include(cmake/clang-tidy.cmake)
include(cmake/third-party.cmake)

#---------
# Options
#---------

# CHAINERX_CUDNN_USE_CUPY
if(DEFINED ENV{CHAINERX_CUDNN_USE_CUPY})
    set(DEFAULT_CHAINERX_CUDNN_USE_CUPY $ENV{CHAINERX_CUDNN_USE_CUPY})
else()
    set(DEFAULT_CHAINERX_CUDNN_USE_CUPY OFF)
endif()

# CHAINERX_ENABLE_COVERAGE
if(DEFINED ENV{CHAINERX_ENABLE_COVERAGE})
    set(DEFAULT_CHAINERX_ENABLE_COVERAGE $ENV{CHAINERX_ENABLE_COVERAGE})
else()
    set(DEFAULT_CHAINERX_ENABLE_COVERAGE OFF)
endif()

# CHAINERX_BUILD_CUDA
if(DEFINED ENV{CHAINERX_BUILD_CUDA})
    set(DEFAULT_CHAINERX_BUILD_CUDA $ENV{CHAINERX_BUILD_CUDA})
else()
    set(DEFAULT_CHAINERX_BUILD_CUDA OFF)
endif()

# CHAINERX_ENABLE_BLAS
if(DEFINED ENV{CHAINERX_ENABLE_BLAS})
    set(DEFAULT_CHAINERX_ENABLE_BLAS $ENV{CHAINERX_ENABLE_BLAS})
else()
    set(DEFAULT_CHAINERX_ENABLE_BLAS ON)
endif()

# CHAINERX_ENABLE_LAPACK
if(DEFINED ENV{CHAINERX_ENABLE_LAPACK})
    set(DEFAULT_CHAINERX_ENABLE_LAPACK $ENV{CHAINERX_ENABLE_LAPACK})
else()
    set(DEFAULT_CHAINERX_ENABLE_LAPACK ON)
endif()

option(CHAINERX_BUILD_PYTHON "Build Python binding" OFF)
option(CHAINERX_BUILD_TEST "Build test" OFF)
option(CHAINERX_BUILD_EXAMPLES "Build examples" OFF)
option(CHAINERX_WARNINGS_AS_ERRORS "Make all warnings of compilers into errors" ON)
option(CHAINERX_ENABLE_THREAD_SANITIZER "Enable thread sanitizer." OFF)
option(CHAINERX_CUDNN_USE_CUPY "Use existing CuPy installation for cuDNN headers and libraries" ${DEFAULT_CHAINERX_CUDNN_USE_CUPY})
option(CHAINERX_ENABLE_COVERAGE "Enable test coverage with gcov" ${DEFAULT_CHAINERX_ENABLE_COVERAGE})
option(CHAINERX_BUILD_CUDA "Build CUDA backend (if CUDA is available)" ${DEFAULT_CHAINERX_BUILD_CUDA})
option(CHAINERX_ENABLE_BLAS "Use BLAS if available" ${DEFAULT_CHAINERX_ENABLE_BLAS})
option(CHAINERX_ENABLE_LAPACK "Use LAPACK if available" ${DEFAULT_CHAINERX_ENABLE_LAPACK})

if(MSVC)
    option(CUDA_USE_STATIC_CUDA_RUNTIME "Use the static version of the CUDA runtime library if available" OFF)
    set(CUDA_USE_STATIC_CUDA_RUNTIME OFF CACHE INTERNAL "")
    add_definitions(-DWIN32_LEAN_AND_MEAN)
endif()

#-------------
# Build setup
#-------------

message(STATUS "CMake generator: " ${CMAKE_GENERATOR})

# Set CMAKE_BUILD_TYPE (defaults to Release).
# CMake's specification is case-insensitive, but we only accept capitalized ones.
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release")
endif()

if(NOT ((CMAKE_BUILD_TYPE STREQUAL "Release") OR (CMAKE_BUILD_TYPE STREQUAL "Debug")))
    message(FATAL_ERROR "Invalid CMAKE_BUILD_TYPE: " ${CMAKE_BUILD_TYPE})
endif()

#-----------
# C++ setup
#-----------
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_OUTPUT_EXTENSION_REPLACE 1)  # ref. https://texus.me/2015/09/06/cmake-and-gcov/

if("${CMAKE_CXX_COMPILER_ID}" MATCHES "GNU|Clang|Intel")
    if(MSVC)
        add_definitions(-D_CRT_SECURE_NO_WARNINGS)
    else()
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -pthread")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -Wall -Wextra -fPIC -pthread")
    endif()
    set(CMAKE_CXX_FLAGS_RELEASE "-O2 -DNDEBUG")  # cmake -DCMAKE_BUILD_TYPE=Release
    set(CMAKE_CXX_FLAGS_DEBUG "-O0")  # cmake -DCMAKE_BUILD_TYPE=Debug

    # cmake -DCHAINERX_ENABLE_COVERAGE=ON|OFF
    if(CHAINERX_ENABLE_COVERAGE)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --coverage")
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} --coverage")
    endif()

    # cmake -DCHAINERX_WARNINGS_AS_ERRORS=ON|OFF
    if (CHAINERX_WARNINGS_AS_ERRORS)
        if (MSVC)
            set(CHAINERX_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W4")
        else()
            set(CHAINERX_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror")
        endif()
    endif()

    # cmake -DCHAINERX_ENABLE_THREAD_SANITIZER=ON|OFF
    if(CHAINERX_ENABLE_THREAD_SANITIZER)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=thread")
    endif()
endif()

# Compiler-specific workarounds

if(CMAKE_CXX_COMPILER_ID MATCHES "^(Clang|AppleClang)$" AND NOT MSVC)
    EXECUTE_PROCESS(COMMAND ${CMAKE_CXX_COMPILER} --version OUTPUT_VARIABLE clang_full_version_string)
    string(REGEX REPLACE ".*clang version ([0-9]+\\.[0-9]+).*" "\\1" CLANG_VERSION_STRING ${clang_full_version_string})

    if(CLANG_VERSION_STRING VERSION_LESS 6.0)
        # clang<6.0 suggests superfluous braces for std::array initialization.
        # https://bugs.llvm.org/show_bug.cgi?id=21629
        set(CHAINERX_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-missing-braces")
    endif()
endif()


#--------------
# Dependencies
#--------------

# CUDA
if(${CHAINERX_BUILD_CUDA})
    find_package(CUDA REQUIRED QUIET)

    add_definitions(-DCHAINERX_ENABLE_CUDA=1)

    list(APPEND CUDA_NVCC_FLAGS "-std=c++${CMAKE_CXX_STANDARD}")

    # Ignore unused function warnings in NVCC that comes from CUDA system headers.
    # It's not possible to ignore warnings selectively in system headers because CMake's CUDA module doesn't support SYSTEM specification for include directories
    # (https://gitlab.kitware.com/cmake/cmake/issues/16464),
    # so relevant warnings from all the sources are ignored.
    list(APPEND CUDA_NVCC_FLAGS -Xcompiler "-Wno-unused-function")

    # CUDA_cublas_device_LIBRARY is required for CUDA > 9.2 with cmake < to 3.12.2.
    # This is because cublas_device was deprecated in CUDA 9.2, but FindCUDA supported it in 3.12.2.
    set(CUDA_cublas_device_LIBRARY ${CUDA_LIBRARIES})

    # If CHAINERX_CUDNN_USE_CUPY is set, retrieve cuDNN files from CuPy installation.
    if(CHAINERX_CUDNN_USE_CUPY)
        execute_process(
            # This counts on sys.stdout.write to raise an error if None is given, resulting ret != 0.
            COMMAND python -c "import sys, cupyx; info = cupyx.runtime.get_install_info(); sys.stdout.write(info.get_data_path('include'))"
            OUTPUT_VARIABLE CUDNN_INCLUDE_DIR
            RESULT_VARIABLE ret)
        if(NOT ret EQUAL 0)
            message(FATAL_ERROR "Failed to retrieve cuDNN include directory from CuPy installation.")
        endif()
        execute_process(
            # The same as above about sys.stdout.write.
            COMMAND python -c "import sys, cupyx; info = cupyx.runtime.get_install_info(); sys.stdout.write(info.get_data_path('lib'))"
            OUTPUT_VARIABLE CUDNN_LIB_DIR
            RESULT_VARIABLE ret)
        if(NOT ret EQUAL 0)
            message(FATAL_ERROR "Failed to retrieve cuDNN library directory from CuPy installation.")
        endif()
        # find_library() cannot find the library file 'libcudnn.so.<ver>' without a symlink 'libcudnn.so', which CuPy does not provide due to pip restriction.
        # Here using glob to actively find the target file.
        file(GLOB CUDNN_LIBRARY ${CUDNN_LIB_DIR}/libcudnn.so*)
        if(NOT CUDNN_LIBRARY)
            message(FATAL_ERROR "Could not find libcudnn shared library in ${CUDNN_LIB_DIR}.")
        endif()
    endif()

    # Allow to specify --generate-code option of the nvcc command.
    # Supposed usage is to avoid slowness of PTX JIT compilation on development.
    # Multiple code specs can be given delimited by semicolons.
    set(CHAINERX_NVCC_GENERATE_CODE "$ENV{CHAINERX_NVCC_GENERATE_CODE}" CACHE STRING "nvcc --generate-code option")

    find_package(CuDNN 7 REQUIRED)
    include_directories(${CUDNN_INCLUDE_DIRS})
    if(NOT CHAINERX_NVCC_GENERATE_CODE STREQUAL "")
        foreach(generate_code ${CHAINERX_NVCC_GENERATE_CODE})
            list(APPEND CUDA_NVCC_FLAGS --generate-code=${generate_code})
        endforeach(generate_code)
    elseif(CMAKE_BUILD_TYPE STREQUAL "Release")
        list(APPEND CUDA_NVCC_FLAGS --generate-code=arch=compute_30,code=sm_30)
        list(APPEND CUDA_NVCC_FLAGS --generate-code=arch=compute_50,code=sm_50)
        list(APPEND CUDA_NVCC_FLAGS --generate-code=arch=compute_60,code=sm_60)
        list(APPEND CUDA_NVCC_FLAGS --generate-code=arch=compute_70,code=sm_70)
    endif()

    # cuSOLVER requires OpenMP
    if(MSVC)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /openmp")
    else()
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
    endif()

    if(MSVC)
        if(CMAKE_CXX_COMPILER_ID MATCHES "GCC|Clang")
            list(APPEND CUDA_NVCC_FLAGS -Xcompiler /MD)
        else()
            list(APPEND CUDA_NVCC_FLAGS_RELEASE -Xcompiler /MD)
            list(APPEND CUDA_NVCC_FLAGS_DEBUG -Xcompiler /MDd)
        endif()
    endif()
endif()

# BLAS
if(${CHAINERX_ENABLE_BLAS})
    find_package(BLAS QUIET)
    if (${BLAS_FOUND})
        message(STATUS "Found BLAS (library: ${BLAS_LIBRARIES})")
    else()
        message(STATUS "BLAS not found")
    endif()
endif()

# LAPACK
if(${CHAINERX_ENABLE_LAPACK})
    find_package(LAPACK QUIET)
    if (${LAPACK_FOUND})
        message(STATUS "Found LAPACK (library: ${LAPACK_LIBRARIES})")
        message(STATUS "Defining CHAINERX_ENABLE_LAPACK variable.")
    else()
        message(STATUS "LAPACK not found")
        set(CHAINERX_ENABLE_LAPACK OFF)
    endif()
endif()

if(${CHAINERX_ENABLE_LAPACK})
    add_definitions(-DCHAINERX_ENABLE_LAPACK=1)
else()
    add_definitions(-DCHAINERX_ENABLE_LAPACK=0)
endif()

# dl libs
if(DEFINED CMAKE_DL_LIBS)
else()
    message(FATAL_ERROR "libdl not found")
endif()

# pybind11
if(${CHAINERX_BUILD_PYTHON})
    get_third_party(pybind11)
    include_directories(${CMAKE_CURRENT_BINARY_DIR}/pybind11/include)
    add_subdirectory(${CMAKE_CURRENT_BINARY_DIR}/pybind11 ${CMAKE_CURRENT_BINARY_DIR}/pybind11-build)
endif()

# gsl-lite
# gsl-lite is a header-only library, we do not need to build and run tests
get_third_party(gsl-lite)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/gsl-lite/include)
# ref. https://github.com/martinmoene/gsl-lite#api-macro
# As default, functions (methods) are decorated with __host__ __device__ for the CUDA platform.
# We want to stop it because we get many warnings with nvcc.
add_definitions(-Dgsl_api=)
if(${CUDA_FOUND})
    list(APPEND CUDA_NVCC_FLAGS "-Dgsl_api=")
endif()

# abseil
get_third_party(abseil)
include_directories("${CMAKE_CURRENT_BINARY_DIR}/abseil")
add_subdirectory("${CMAKE_CURRENT_BINARY_DIR}/abseil" "${CMAKE_CURRENT_BINARY_DIR}/abseil-build")

# gtest
if(MSVC)
    if(CMAKE_CXX_COMPILER_ID MATCHES "GCC|Clang")
        # TODO(durswd): Remove it. It is hack for MSVC+LLVM
        target_compile_options(gtest INTERFACE -Wno-global-constructors PRIVATE -w)
    else()
        # ChainerX is linked with MultiThreaded DLL, including gtest
        set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    endif()
endif()

# TODO(niboshi): Remove gtest dependency from testing
get_third_party(gtest)
add_subdirectory(${CMAKE_CURRENT_BINARY_DIR}/googletest-src
    ${CMAKE_CURRENT_BINARY_DIR}/googletest-build
    EXCLUDE_FROM_ALL)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/googletest-src/googletest/include)


#------
# Test
#------
if(${CHAINERX_BUILD_TEST})
    enable_testing()
endif()

#----------
# Examples
#----------
if(${CHAINERX_BUILD_EXAMPLES})
    add_subdirectory(examples)
endif()

#----------
# ChainerX
#----------
add_subdirectory(chainerx)
