cmake_minimum_required(VERSION 3.1)
project(abseil-download NONE)

include(ExternalProject)
ExternalProject_Add(pybind11
    GIT_REPOSITORY    https://github.com/abseil/abseil-cpp.git
    # TODO(take-cheeze): Use tag instead as soon as 201906 is out
    # https://github.com/abseil/abseil-cpp/releases
    GIT_TAG           72e09a54d993b192db32be14c65adf7e9bd08c31
    SOURCE_DIR        "${CMAKE_CURRENT_BINARY_DIR}/abseil"
    BINARY_DIR        ""
    CONFIGURE_COMMAND ""
    BUILD_COMMAND     ""
    INSTALL_COMMAND   ""
    TEST_COMMAND      ""
    )
