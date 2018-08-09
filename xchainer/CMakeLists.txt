add_subdirectory(routines)
add_subdirectory(native)
add_subdirectory(testing)
if(${CUDA_FOUND})
    add_subdirectory(cuda)
endif()
if(${BUILD_PYTHON})
    add_subdirectory(python)
endif()

install(FILES
    array.h
    array_body.h
    array_body_leak_detection.h
    array_fwd.h
    array_index.h
    array_node.h
    array_repr.h
    axes.h
    backend.h
    backend_util.h
    backprop_mode.h
    backprop_scope.h
    backward.h
    backward_builder.h
    backward_context.h
    backward_fwd.h
    constant.h
    context.h
    device.h
    device_id.h
    dtype.h
    enum.h
    error.h
    graph.h
    hash_combine.h
    index_iterator.h
    indexable_array.h
    indexer.h
    macro.h
    numeric_limits.h
    op_node.h
    optional_container_arg.h
    reduction_kernel_arg.h
    scalar.h
    shape.h
    slice.h
    squash_dims.h
    stack_vector.h
    strides.h
    xchainer.h
    DESTINATION include/xchainer
    )

add_library(xchainer SHARED
    array.cc
    array_body.cc
    array_body_leak_detection.cc
    array_repr.cc
    axes.cc
    backend.cc
    backprop_mode.cc
    backward.cc
    backward_builder.cc
    backward_context.cc
    check_backward.cc
    context.cc
    device.cc
    device_id.cc
    dtype.cc
    graph.cc
    numeric.cc
    numerical_gradient.cc
    op_node.cc
    reduction_kernel_arg.cc
    scalar.cc
    shape.cc
    strides.cc
    )

target_link_libraries(xchainer
    PUBLIC
    ${CMAKE_DL_LIBS}
    PRIVATE
    -Wl,--whole-archive)
target_link_libraries(xchainer PRIVATE xchainer_routines)
target_link_libraries(xchainer PRIVATE xchainer_native)
if(${CUDA_FOUND})
    target_link_libraries(xchainer PRIVATE xchainer_cuda)
endif()
target_link_libraries(xchainer
    PRIVATE
    -Wl,--no-whole-archive)
install(TARGETS xchainer DESTINATION lib)

if(ENABLE_TEST)
    add_subdirectory(context_testdata)
    set(srcs
        array_body_leak_detection_test.cc
        array_device_test.cc
        array_repr_test.cc
        array_test.cc
        array_to_device_test.cc
        axes_test.cc
        backprop_mode_test.cc
        backward_test.cc
        check_backward_test.cc
        context_test.cc
        device_test.cc
        dtype_test.cc
        index_iterator_test.cc
        indexable_array_test.cc
        indexer_test.cc
        numeric_limits_test.cc
        numerical_gradient_test.cc
        numeric_test.cc
        optional_container_arg_test.cc
        scalar_test.cc
        shape_test.cc
        squash_dims_test.cc
        stack_vector_test.cc
        strides_test.cc
        )
    if(${CUDA_FOUND})
        CUDA_ADD_EXECUTABLE(xchainer_test ${srcs})
    else()
        add_executable(xchainer_test ${srcs})
    endif()

    target_compile_definitions(xchainer_test PRIVATE
        XCHAINER_TEST_DIR="${CMAKE_CURRENT_BINARY_DIR}")

    target_link_libraries(xchainer_test
        xchainer
        xchainer_testing
        gtest)
    add_test(NAME xchainer_test COMMAND xchainer_test)
endif()
