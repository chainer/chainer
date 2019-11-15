set(BACKEND_TESTDATA_LINK_OPTIONS "")
if(APPLE)
    set(BACKEND_TESTDATA_LINK_OPTIONS "-undefined dynamic_lookup")
endif()

add_library(backend0.so MODULE
    backend0.cc)
set_target_properties(backend0.so
    PROPERTIES
    PREFIX ""
    SUFFIX ""
    LINK_FLAGS "${BACKEND_TESTDATA_LINK_OPTIONS}"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/backends")

add_library(backend1.so MODULE
    backend1.cc)
set_target_properties(backend1.so
    PROPERTIES
    PREFIX ""
    SUFFIX ""
    LINK_FLAGS "${BACKEND_TESTDATA_LINK_OPTIONS}"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/backends")
