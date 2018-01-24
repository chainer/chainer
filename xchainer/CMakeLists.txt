if(${CUDA_FOUND})
    add_subdirectory(cuda)
endif()
if(${BUILD_PYTHON})
    add_subdirectory(python)
endif()

install(FILES
    array.h
    array_math.h
    array_repr.h
    backend.h
    backprop.h
    constant.h
    device.h
    dtype.h
    error.h
    native_backend.h
    memory.h
    scalar.h
    shape.h
    xchainer.h
    DESTINATION include/xchainer
    )

add_library(xchainer SHARED
    array.cc
    array_fill.cc
    array_math.cc
    array_repr.cc
    backprop.cc
    device.cc
    dtype.cc
    gradient_check.cc
    native_backend.cc
    memory.cc
    numeric.cc
    scalar.cc
    shape.cc
    )

target_link_libraries(xchainer
    PRIVATE
    -Wl,--whole-archive)
if(${CUDA_FOUND})
    target_link_libraries(xchainer PRIVATE xchainer_cuda)
endif()
target_link_libraries(xchainer
    PRIVATE
    -Wl,--no-whole-archive)
install(TARGETS xchainer DESTINATION lib)

if(ENABLE_TEST)
    set(srcs
        array_repr_test.cc
        array_test.cc
        check_backward.cc
        check_backward_test.cc
        backprop_test.cc
        device_test.cc
        dtype_test.cc
        gradient_check_test.cc
        memory_test.cc
        numeric_test.cc
        scalar_test.cc
        shape_test.cc
        )
    if(${CUDA_FOUND})
        CUDA_ADD_EXECUTABLE(xchainer_test ${srcs})
    else()
        add_executable(xchainer_test ${srcs})
    endif()

    target_link_libraries(xchainer_test
        xchainer
        gtest_main)
    add_test(NAME xchainer_test COMMAND xchainer_test)
endif()
