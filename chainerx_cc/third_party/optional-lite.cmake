cmake_minimum_required(VERSION 2.8.2)
project(optional-lite-download NONE)

include(ExternalProject)
ExternalProject_Add(optional-lite
    GIT_REPOSITORY    https://github.com/martinmoene/optional-lite
    GIT_TAG           v2.3.0
    SOURCE_DIR        "${CMAKE_BINARY_DIR}/optional-lite"
    BINARY_DIR        ""
    CONFIGURE_COMMAND ""
    BUILD_COMMAND     ""
    INSTALL_COMMAND   ""
    TEST_COMMAND      ""
    )
