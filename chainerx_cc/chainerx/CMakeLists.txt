include_directories("${PROJECT_SOURCE_DIR}")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${CHAINERX_CXX_FLAGS}")

add_subdirectory(kernels)
add_subdirectory(routines)
add_subdirectory(native)
add_subdirectory(testing)
if(${CUDA_FOUND})
    add_subdirectory(cuda)
endif()
if(${CHAINERX_BUILD_PYTHON})
    add_subdirectory(python)
endif()

install(FILES
    arithmetic_ops.h
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
    chainerx.h
    check_backward.h
    constant.h
    context.h
    device.h
    device_id.h
    dims.h
    dtype.h
    dynamic_lib.h
    enum.h
    error.h
    float16.h
    graph.h
    hash_combine.h
    index_iterator.h
    indexable_array.h
    indexer.h
    kernel.h
    kernel_registry.h
    macro.h
    numerical_gradient.h
    numeric.h
    numeric_limits.h
    op_node.h
    optional_container_arg.h
    platform.h
    reduction_kernel_arg.h
    scalar.h
    shape.h
    slice.h
    squash_dims.h
    stack_vector.h
    strides.h
    thread_local_state.h
    util.h
    DESTINATION include/chainerx
    )

set(chainerx_srcs
    array.cc
    array_body.cc
    array_body_leak_detection.cc
    array_index.cc
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
    dims.cc
    dtype.cc
    dynamic_lib.cc
    float16.cc
    graph.cc
    numeric.cc
    numerical_gradient.cc
    op_node.cc
    platform.cc
    reduction_kernel_arg.cc
    scalar.cc
    shape.cc
    strides.cc
    thread_local_state.cc
    util.cc
    )

add_library(chainerx_base STATIC ${chainerx_srcs})

if(MSVC)
    install(FILES
        platform/windows.h
        DESTINATION include/chainerx/platform
        )

    set(chainerx_srcs
        "${chainerx_srcs}"
        platform/windows.cc
        )
    add_library(chainerx STATIC empty.cc)
else()
    add_library(chainerx SHARED empty.cc)
endif()

# abseil
target_link_libraries(
    chainerx
    PUBLIC
    absl::failure_signal_handler
    absl::bad_optional_access
    PRIVATE
    absl::flat_hash_map
    )

# dlopen / dlclose
target_link_libraries(chainerx PUBLIC ${CMAKE_DL_LIBS})

# ChainerX
set(chainerx_sub_libs
    chainerx_base
    chainerx_routines
    chainerx_native
    chainerx_testing
    )

if(${CUDA_FOUND})
    set(chainerx_sub_libs ${chainerx_sub_libs} chainerx_cuda)
endif()

if(MSVC)
    target_link_libraries(chainerx PRIVATE ${chainerx_sub_libs})
    foreach(lib ${chainerx_sub_libs})
        target_link_options(chainerx PUBLIC /wholearchive:$<TARGET_FILE:${lib}>)
    endforeach(lib)
elseif(${APPLE})
    foreach(lib ${chainerx_sub_libs})
        target_link_libraries(chainerx PRIVATE -Wl,-force_load ${lib})
    endforeach(lib)
    target_link_libraries(chainerx PRIVATE -Wl,-noall_load)
else()
    target_link_libraries(chainerx
        PRIVATE
        -Wl,--whole-archive
        ${chainerx_sub_libs}
        -Wl,--no-whole-archive
        )
endif()

install(TARGETS chainerx DESTINATION lib)

if(${CHAINERX_BUILD_TEST})
    add_subdirectory(backend_testdata)
    set(srcs
        array_body_leak_detection_test.cc
        array_device_test.cc
        array_repr_test.cc
        array_test.cc
        array_to_device_test.cc
        axes_test.cc
        backprop_mode_test.cc
        backward_builder_test.cc
        backward_test.cc
        check_backward_test.cc
        context_test.cc
        device_test.cc
        dims_test.cc
        dtype_test.cc
        float16_test.cc
        index_iterator_test.cc
        indexable_array_test.cc
        indexer_test.cc
        kernel_registry_test.cc
        numeric_limits_test.cc
        numerical_gradient_test.cc
        numeric_test.cc
        optional_container_arg_test.cc
        scalar_test.cc
        shape_test.cc
        squash_dims_test.cc
        stack_vector_test.cc
        strides_test.cc
        thread_local_state_test.cc
        )
    if(${CUDA_FOUND})
        CUDA_ADD_EXECUTABLE(chainerx_test ${srcs})
    else()
        add_executable(chainerx_test ${srcs})
    endif()

    target_compile_definitions(chainerx_test PRIVATE
        CHAINERX_TEST_DIR="${CMAKE_CURRENT_BINARY_DIR}")

    target_link_libraries(chainerx_test
        chainerx
        chainerx_test_main
        # gtest is linked via chainerx_testing.
        )
    add_test(NAME chainerx_test COMMAND chainerx_test)
endif()
