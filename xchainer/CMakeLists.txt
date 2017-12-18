if(${BUILD_PYTHON})
    add_subdirectory(python)
endif()

install(FILES
    array.h
    constant.h
    dtype.h
    error.h
    scalar.h
    shape.h
    xchainer.h
    DESTINATION include/xchainer
    )

add_library(xchainer SHARED
    array_repr.cc
    dtype.cc
    scalar.cc
    shape.cc
    )

install(TARGETS xchainer DESTINATION lib)

if(ENABLE_TEST)
    add_executable(xchainer_test
        array_repr_test.cc
        dtype_test.cc
        scalar_test.cc
        shape_test.cc
        )
    target_link_libraries(xchainer_test
        xchainer
        gtest_main)
    add_test(NAME xchainer_test COMMAND xchainer_test)
endif()
