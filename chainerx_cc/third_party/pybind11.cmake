cmake_minimum_required(VERSION 3.1)
project(pybind11-download NONE)

include(ExternalProject)
ExternalProject_Add(pybind11
    GIT_REPOSITORY    https://github.com/pybind/pybind11.git
    GIT_TAG           v2.3.0
    SOURCE_DIR        "${CMAKE_CURRENT_BINARY_DIR}/pybind11"
    BINARY_DIR        ""
    CONFIGURE_COMMAND ""
    BUILD_COMMAND     ""
    INSTALL_COMMAND   ""
    TEST_COMMAND      ""
    CMAKE_ARGS
        -DCMAKE_INSTALL_MESSAGE=LAZY
        -DCMAKE_POSITION_INDEPENDENT_CODE=ON
    )
