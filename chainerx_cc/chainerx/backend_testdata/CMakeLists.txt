add_library(backend0.so MODULE
    backend0.cc)
set_target_properties(backend0.so
    PROPERTIES
    PREFIX ""
    SUFFIX ""
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/backends")
if(APPLE)
    target_link_libraries(backend0.so
        PRIVATE
        "-undefined dynamic_lookup")
endif()

add_library(backend1.so MODULE
    backend1.cc)
set_target_properties(backend1.so
    PROPERTIES
    PREFIX ""
    SUFFIX ""
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/backends")
if(APPLE)
    target_link_libraries(backend1.so
        PRIVATE
        "-undefined dynamic_lookup")
endif()
