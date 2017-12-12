cmake_minimum_required(VERSION 3.1)
project(xchainer CXX C)

# CMake setup
if(POLICY ${CMP0054})
    cmake_policy(SET ${CMP0054} NEW)
endif()

# includes
include(cmake/clang-tidy.cmake)
include(cmake/third-party.cmake)

# Configure options
option(BUILD_PYTHON "Build Python binding" OFF)
option(ENABLE_TEST "Enable building test" ON)
option(ENABLE_COVERAGE "Enable test coverage with gcov" OFF)

# C++ setup
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_OUTPUT_EXTENSION_REPLACE 1)  # ref. https://texus.me/2015/09/06/cmake-and-gcov/

# defaults to cmake -DCMAKE_BUILD_TYPE=Debug
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug)
endif()

if("${CMAKE_CXX_COMPILER_ID}" MATCHES "GNU|Clang|Intel")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -pthread")
    set(CMAKE_CXX_FLAGS "-g -Wall -Wextra -fPIC -pthread")
    set(CMAKE_CXX_FLAGS_RELEASE "-O2")  # cmake -DCMAKE_BULD_TYPE=Release
    set(CMAKE_CXX_FLAGS_DEBUG "-O0")  # cmake -DCMAKE_BUILD_TYPE=Debug

    # cmake -DENABLE_COVERAGE=ON|OFF
    if(ENABLE_COVERAGE)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --coverage")
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} --coverage")
    endif()
endif()
include_directories("${PROJECT_SOURCE_DIR}")

# pybind11
if(${BUILD_PYTHON})
    get_third_party(pybind11)
    include_directories(${CMAKE_BINARY_DIR}/pybind11/include)
    add_subdirectory(${CMAKE_BINARY_DIR}/pybind11)
endif()

# gsl-lite
get_third_party(gsl-lite)
include_directories(${CMAKE_BINARY_DIR}/gsl-lite/include)
# add_subdirectory(${CMAKE_BINARY_DIR}/gsl-lite)

# Test
if(ENABLE_TEST)
    get_third_party(gtest)
    enable_testing()
    add_subdirectory(${CMAKE_BINARY_DIR}/googletest-src
        ${CMAKE_BINARY_DIR}/googletest-build
        EXCLUDE_FROM_ALL)
endif()

add_subdirectory(xchainer)
