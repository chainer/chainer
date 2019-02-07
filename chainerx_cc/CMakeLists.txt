cmake_minimum_required(VERSION 3.1)
project(chainerx CXX C)

# CMake setup
if(POLICY CMP0054)
    cmake_policy(SET CMP0054 NEW)
endif()

SET(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake ${CMAKE_MODULE_PATH})

# includes
include(cmake/clang-tidy.cmake)
include(cmake/third-party.cmake)

# Configure options
option(CHAINERX_BUILD_PYTHON "Build Python binding" OFF)
option(CHAINERX_BUILD_TEST "Build test" OFF)
option(CHAINERX_BUILD_EXAMPLES "Build examples" OFF)
option(CHAINERX_WARNINGS_AS_ERRORS "Make all warnings of compilers into errors" ON)
option(CHAINERX_ENABLE_THREAD_SANITIZER "Enable thread sanitizer." OFF)

if(DEFINED ENV{CHAINERX_ENABLE_COVERAGE})
    set(DEFAULT_CHAINERX_ENABLE_COVERAGE $ENV{CHAINERX_ENABLE_COVERAGE})
else()
    set(DEFAULT_CHAINERX_ENABLE_COVERAGE OFF)
endif()
option(CHAINERX_ENABLE_COVERAGE "Enable test coverage with gcov" ${DEFAULT_CHAINERX_ENABLE_COVERAGE})

if(DEFINED ENV{CHAINERX_BUILD_CUDA})
    set(DEFAULT_CHAINERX_BUILD_CUDA $ENV{CHAINERX_BUILD_CUDA})
else()
    set(DEFAULT_CHAINERX_BUILD_CUDA OFF)
endif()
option(CHAINERX_BUILD_CUDA "Build CUDA backend (if CUDA is available)" ${DEFAULT_CHAINERX_BUILD_CUDA})

# Allow to specify *one* --generate-code option of the nvcc command.
# Supposed usage is to avoid slowness of PTX JIT compilation on development.
set(CHAINERX_NVCC_GENERATE_CODE "$ENV{CHAINERX_NVCC_GENERATE_CODE}" CACHE STRING "nvcc --generate-code option")

if(DEFINED ENV{CHAINERX_ENABLE_BLAS})
    set(DEFAULT_CHAINERX_ENABLE_BLAS $ENV{CHAINERX_ENABLE_BLAS})
else()
    set(DEFAULT_CHAINERX_ENABLE_BLAS ON)
endif()
option(CHAINERX_ENABLE_BLAS "Use BLAS if available" ${DEFAULT_CHAINERX_ENABLE_BLAS})

# defaults to cmake -DCMAKE_BUILD_TYPE=Release
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# CUDA
if(${CHAINERX_BUILD_CUDA})
    if(MSVC)
        option(CUDA_USE_STATIC_CUDA_RUNTIME "Use the static version of the CUDA runtime library if available" OFF)
        set(CUDA_USE_STATIC_CUDA_RUNTIME OFF CACHE INTERNAL "")
    endif()

    find_package(CUDA QUIET)
    if(${CUDA_FOUND})
        # CUDA_cublas_device_LIBRARY is required for CUDA > 9.2 with cmake < to 3.12.2.
        # This is because cublas_device was deprecated in CUDA 9.2, but FindCUDA supported it in 3.12.2.
        set(CUDA_cublas_device_LIBRARY ${CUDA_LIBRARIES})
        add_definitions(-DCHAINERX_ENABLE_CUDA=1)
        find_package(CuDNN 7 REQUIRED)
        include_directories(${CUDNN_INCLUDE_DIRS})
        if(NOT CHAINERX_NVCC_GENERATE_CODE STREQUAL "")
            list(APPEND CUDA_NVCC_FLAGS --generate-code=${CHAINERX_NVCC_GENERATE_CODE})
        elseif(CMAKE_BUILD_TYPE MATCHES Release)
            list(APPEND CUDA_NVCC_FLAGS --generate-code=arch=compute_30,code=sm_30)
            list(APPEND CUDA_NVCC_FLAGS --generate-code=arch=compute_50,code=sm_50)
            list(APPEND CUDA_NVCC_FLAGS --generate-code=arch=compute_60,code=sm_60)
            list(APPEND CUDA_NVCC_FLAGS --generate-code=arch=compute_70,code=sm_70)
        endif()

        if(MSVC)
            list(APPEND CUDA_NVCC_FLAGS -Xcompiler /MD)
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

# C++ setup
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
    if (${CUDA_FOUND})
        list(APPEND CUDA_NVCC_FLAGS -std=c++14)
    endif()

    # cmake -DCHAINERX_ENABLE_COVERAGE=ON|OFF
    if(CHAINERX_ENABLE_COVERAGE)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --coverage")
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} --coverage")
    endif()

    # cmake -DCHAINERX_WARNINGS_AS_ERRORS=ON|OFF
    if (CHAINERX_WARNINGS_AS_ERRORS)
        if (MSVC)
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W4")
        else()
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror")
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
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-missing-braces")
    endif()
endif()


include_directories("${PROJECT_SOURCE_DIR}")

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

# optional-lite
get_third_party(optional-lite)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/optional-lite/include)

# Test
# TODO(niboshi): Remove gtest dependency from testing
get_third_party(gtest)
add_subdirectory(${CMAKE_CURRENT_BINARY_DIR}/googletest-src
    ${CMAKE_CURRENT_BINARY_DIR}/googletest-build
    EXCLUDE_FROM_ALL)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/googletest-src/googletest/include)

# TODO(durswd): Remove it. It is hack for MSVC+LLVM
if(CMAKE_CXX_COMPILER_ID MATCHES "GCC|Clang" AND MSVC)
    target_compile_options(gtest INTERFACE -Wno-global-constructors PRIVATE -w)
endif()

if(${CHAINERX_BUILD_TEST})
    enable_testing()
endif()

# Examples
if(${CHAINERX_BUILD_EXAMPLES})
    add_subdirectory(examples)
endif()

add_subdirectory(chainerx)
