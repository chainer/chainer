cmake_minimum_required(VERSION 3.1)
project(xchainer CXX C)

# CMake setup
if(POLICY ${CMP0054})
    cmake_policy(SET ${CMP0054} NEW)
endif()

# includes
include(cmake/clang-tidy.cmake)
include(cmake/third-party.cmake)

# C++ setup
set(CMAKE_CXX_STANDARD 14)
if("${CMAKE_CXX_COMPILER_ID}" MATCHES "GNU|Clang|Intel")
    add_definitions(-g -Wall -Wextra -fPIC -pthread)
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -pthread")
endif()
include_directories("${PROJECT_SOURCE_DIR}")

# Test
if($ENV{DISABLE_TEST})
    set(ENABLE_TEST OFF)
else()
    set(ENABLE_TEST ON)
endif()
if(${ENABLE_TEST})
    get_third_party(gtest)
    enable_testing()
    add_subdirectory(${CMAKE_BINARY_DIR}/googletest-src
        ${CMAKE_BINARY_DIR}/googletest-build
        EXCLUDE_FROM_ALL)
endif()

add_subdirectory(xchainer)
