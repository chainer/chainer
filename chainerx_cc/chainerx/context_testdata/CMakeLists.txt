add_library(backend0.so MODULE
    backend0.cc)
target_link_libraries(backend0.so chainerx)
set_target_properties(backend0.so
    PROPERTIES
    PREFIX ""
    SUFFIX ""
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/backends")

add_library(backend1.so MODULE
    backend1.cc)
target_link_libraries(backend1.so chainerx)
set_target_properties(backend1.so
    PROPERTIES
    PREFIX ""
    SUFFIX ""
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/backends")
