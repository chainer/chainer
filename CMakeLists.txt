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

# C++ setup
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_OUTPUT_EXTENSION_REPLACE 1)  # ref. https://texus.me/2015/09/06/cmake-and-gcov/

if("${CMAKE_CXX_COMPILER_ID}" MATCHES "GNU|Clang|Intel")
    set(CMAKE_CXX_FLAGS "-g -Wall -Wextra -fPIC -pthread")
    set(CMAKE_CXX_FLAGS_DEBUG "--coverage")  # cmake -DCMAKE_BUILD_TYPE=Debug
    # set(CMAKE_CXX_FLAGS_RELEASE "")  # cmake -DCMAKE_BULD_TYPE=Release
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -pthread")
endif()
include_directories("${PROJECT_SOURCE_DIR}")

# pybind11
if(${BUILD_PYTHON})
    get_third_party(pybind11)
    include_directories(${CMAKE_BINARY_DIR}/pybind11/include)
    add_subdirectory(${CMAKE_BINARY_DIR}/pybind11)
endif()

# Test
if(ENABLE_TEST)
    get_third_party(gtest)
    enable_testing()
    add_subdirectory(${CMAKE_BINARY_DIR}/googletest-src
        ${CMAKE_BINARY_DIR}/googletest-build
        EXCLUDE_FROM_ALL)
endif()

add_subdirectory(xchainer)
