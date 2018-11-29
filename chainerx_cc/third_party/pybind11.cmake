cmake_minimum_required(VERSION 3.1)
project(pybind11-download NONE)

include(ExternalProject)
ExternalProject_Add(pybind11
    GIT_REPOSITORY    https://github.com/pybind/pybind11.git
    GIT_TAG           5ef1af138dc3ef94c05274ae554e70d64cd589bf # 2.2.3 + #1371 merged
    SOURCE_DIR        "${CMAKE_BINARY_DIR}/pybind11"
    BINARY_DIR        ""
    CONFIGURE_COMMAND ""
    BUILD_COMMAND     ""
    INSTALL_COMMAND   ""
    TEST_COMMAND      ""
    )
