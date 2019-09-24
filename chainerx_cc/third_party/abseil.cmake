cmake_minimum_required(VERSION 3.1)
project(abseil-download NONE)

include(ExternalProject)
ExternalProject_Add(pybind11
    GIT_REPOSITORY    https://github.com/abseil/abseil-cpp.git
    GIT_TAG           20190808
    SOURCE_DIR        "${CMAKE_CURRENT_BINARY_DIR}/abseil"
    BINARY_DIR        ""
    CONFIGURE_COMMAND ""
    BUILD_COMMAND     ""
    INSTALL_COMMAND   ""
    TEST_COMMAND      ""
    CMAKE_ARGS
        -DCMAKE_INSTALL_MESSAGE=LAZY
        -DCMAKE_POSITION_INDEPENDENT_CODE=ON
    )
