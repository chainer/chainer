cmake_minimum_required(VERSION 3.1)
project(abseil-download NONE)

include(ExternalProject)
ExternalProject_Add(abseil
    GIT_REPOSITORY    https://github.com/abseil/abseil-cpp.git
    # TODO(niboshi): Should use a tagged version. 20190808 has a problem causing duplicate symbol errors.
    GIT_TAG           0514227d2547793b23e209809276375e41c76617
    SOURCE_DIR        "${CMAKE_CURRENT_BINARY_DIR}/abseil"
    BINARY_DIR        ""
    CONFIGURE_COMMAND ""
    BUILD_COMMAND     ""
    INSTALL_COMMAND   ""
    TEST_COMMAND      ""
    )
