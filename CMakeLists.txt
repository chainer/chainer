cmake_minimum_required(VERSION 3.1)
project(xchainer CXX C)

# CMake setup
if(POLICY ${CMP0054})
    cmake_policy(SET ${CMP0054} NEW)
endif()

SET(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake ${CMAKE_MODULE_PATH})

# includes
include(cmake/clang-tidy.cmake)
include(cmake/third-party.cmake)

# Configure options
option(BUILD_PYTHON "Build Python binding" OFF)
option(ENABLE_TEST "Enable building test" ON)
option(WARNINGS_AS_ERRORS "Make all warnings of compilers into errors" ON)

if(DEFINED ENV{ENABLE_COVERAGE})
    set(DEFAULT_ENABLE_COVERAGE $ENV{ENABLE_COVERAGE})
else()
    set(DEFAULT_ENABLE_COVERAGE OFF)
endif()
option(ENABLE_COVERAGE "Enable test coverage with gcov" ${DEFAULT_ENABLE_COVERAGE})

if(DEFINED ENV{BUILD_CUDA})
    set(DEFAULT_BUILD_CUDA $ENV{BUILD_CUDA})
else()
    set(DEFAULT_BUILD_CUDA ON)
endif()
option(BUILD_CUDA "Build CUDA backend (if CUDA is available)" ${DEFAULT_BUILD_CUDA})

# Allow to specify *one* --generate-code option of the nvcc command.
# Supposed usage is to avoid slowness of PTX JIT compilation on development.
set(NVCC_GENERATE_CODE "$ENV{NVCC_GENERATE_CODE}" CACHE STRING "nvcc --generate-code option")

# defaults to cmake -DCMAKE_BUILD_TYPE=Debug
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug)
endif()

# CUDA
if(${BUILD_CUDA})
    find_package(CUDA QUIET)
    if(${CUDA_FOUND})
        add_definitions(-DXCHAINER_ENABLE_CUDA=1)
        find_package(CuDNN 7 REQUIRED)
        if(NOT NVCC_GENERATE_CODE STREQUAL "")
            list(APPEND CUDA_NVCC_FLAGS --generate-code=${NVCC_GENERATE_CODE})
        elseif(CMAKE_BUILD_TYPE MATCHES Release)
            list(APPEND CUDA_NVCC_FLAGS --generate-code=arch=compute_30,code=sm_30)
            list(APPEND CUDA_NVCC_FLAGS --generate-code=arch=compute_50,code=sm_50)
            list(APPEND CUDA_NVCC_FLAGS --generate-code=arch=compute_60,code=sm_60)
            list(APPEND CUDA_NVCC_FLAGS --generate-code=arch=compute_70,code=sm_70)
        endif()
    endif()
endif()

# C++ setup
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_OUTPUT_EXTENSION_REPLACE 1)  # ref. https://texus.me/2015/09/06/cmake-and-gcov/

if("${CMAKE_CXX_COMPILER_ID}" MATCHES "GNU|Clang|Intel")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -pthread")
    set(CMAKE_CXX_FLAGS "-g -Wall -Wextra -fPIC -pthread")
    set(CMAKE_CXX_FLAGS_RELEASE "-O2 -DNDEBUG")  # cmake -DCMAKE_BUILD_TYPE=Release
    set(CMAKE_CXX_FLAGS_DEBUG "-O0")  # cmake -DCMAKE_BUILD_TYPE=Debug
    if (${CUDA_FOUND})
        list(APPEND CUDA_NVCC_FLAGS -std=c++14)
    endif()

    # cmake -DENABLE_COVERAGE=ON|OFF
    if(ENABLE_COVERAGE)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --coverage")
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} --coverage")
    endif()

    # cmake -DWARNINGS_AS_ERRORS=ON|OFF
    if (WARNINGS_AS_ERRORS)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror")
    endif()
endif()
include_directories("${PROJECT_SOURCE_DIR}")

# dl libs
if(DEFINED CMAKE_DL_LIBS)
else()
    message(FATAL_ERROR "libdl not found")
endif()

# pybind11
if(${BUILD_PYTHON})
    get_third_party(pybind11)
    include_directories(${CMAKE_BINARY_DIR}/pybind11/include)
    add_subdirectory(${CMAKE_BINARY_DIR}/pybind11)
endif()

# gsl-lite
get_third_party(gsl-lite)
include_directories(${CMAKE_BINARY_DIR}/gsl-lite/include)
# ref. https://github.com/martinmoene/gsl-lite#api-macro
# As default, functions (methods) are decorated with __host__ __device__ for the CUDA platform. 
# We want to stop it because we get many warnings with nvcc.
add_definitions(-Dgsl_api=)
if(${CUDA_FOUND})
    list(APPEND CUDA_NVCC_FLAGS "-Dgsl_api=")
endif()
# gsl-lite is a header-only library, we do not need to build and run tests
# add_subdirectory(${CMAKE_BINARY_DIR}/gsl-lite)

# optional-lite
get_third_party(optional-lite)
include_directories(${CMAKE_BINARY_DIR}/optional-lite/include)

# Test
if(ENABLE_TEST)
    get_third_party(gtest)
    enable_testing()
    add_subdirectory(${CMAKE_BINARY_DIR}/googletest-src
        ${CMAKE_BINARY_DIR}/googletest-build
        EXCLUDE_FROM_ALL)
    include_directories(${CMAKE_BINARY_DIR}/googletest-src/googletest/include)
endif()

add_subdirectory(xchainer)
