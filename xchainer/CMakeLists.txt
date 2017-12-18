if(${BUILD_PYTHON})
    add_subdirectory(python)
endif()

install(FILES
    array.h
    array_repr.h
    constant.h
    device.h
    dtype.h
    error.h
    scalar.h
    shape.h
    xchainer.h
    DESTINATION include/xchainer
    )

add_library(xchainer SHARED
    array_repr.cc
    device.cc
    dtype.cc
    scalar.cc
    shape.cc
    )

install(TARGETS xchainer DESTINATION lib)

if(ENABLE_TEST)
    add_executable(xchainer_test
        array_test.cc
        array_repr_test.cc
        device_test.cc
        dtype_test.cc
        scalar_test.cc
        shape_test.cc
        )
    target_link_libraries(xchainer_test
        xchainer
        gtest_main)
    add_test(NAME xchainer_test COMMAND xchainer_test)
endif()
