cmake_minimum_required(VERSION 3.1)
project(xchainer CXX C)

# CMake setup
if(POLICY ${CMP0054})
    cmake_policy(SET ${CMP0054} NEW)
endif()
include(util.cmake)

# Configure options
option(BUILD_PYTHON "Build Python binding" OFF)

# C++ setup
set(CMAKE_CXX_STANDARD 14)
if("${CMAKE_CXX_COMPILER_ID}" MATCHES "GNU|Clang|Intel")
    add_definitions(-g -Wall -Wextra -fPIC -pthread)
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -pthread")
endif()
include_directories("${PROJECT_SOURCE_DIR}")

# pybind11
if(${BUILD_PYTHON})
    get_third_party(pybind11)
    include_directories(${CMAKE_BINARY_DIR}/pybind11/include)
    add_subdirectory(${CMAKE_BINARY_DIR}/pybind11)
endif()

# includes
include(cmake/clang-tidy.cmake)

add_subdirectory(xchainer)
