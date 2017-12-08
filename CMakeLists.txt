cmake_minimum_required(VERSION 3.1)
project(xchainer CXX C)

# CMake setup
if(POLICY ${CMP0054})
    cmake_policy(SET ${CMP0054} NEW)
endif()

# C++ setup
set(CMAKE_CXX_STANDARD 14)
if("${CMAKE_CXX_COMPILER_ID}" MATCHES "GNU|Clang|Intel")
    add_definitions(-g -Wall -Wextra -fPIC -pthread)
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -pthread")
endif()
include_directories("${PROJECT_SOURCE_DIR}")

add_subdirectory(xchainer)
