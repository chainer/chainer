install(FILES
    native_device.h
    native_backend.h
    elementwise.h
    reduce.h
    col2im.h
    im2col.h
    tensor_dot.h
    DESTINATION include/chainerx/native
    )

add_library(chainerx_native STATIC
    native_device.cc
    native_device/activation.cc
    native_device/arithmetic.cc
    native_device/conv.cc
    native_device/copy.cc
    native_device/comparison.cc
    native_device/dot.cc
    native_device/exp_log.cc
    native_device/fill.cc
    native_device/indexing.cc
    native_device/memory.cc
    native_device/misc.cc
    native_device/pool.cc
    native_device/reduction.cc
    native_backend.cc
    col2im.cc
    im2col.cc
    tensor_dot.cc)

if(${BLAS_FOUND})
    # Workaround to guess the BLAS include directory from its library path (.so).
    # This workaround is required since CMake does not set the variable as expected.
    get_filename_component(BLAS_INCLUDE_DIRS ${BLAS_LIBRARIES} PATH)  # Directory of .so.
    get_filename_component(BLAS_INCLUDE_DIRS ${BLAS_INCLUDE_DIRS} PATH)  # Directory of lib.
    set(BLAS_INCLUDE_DIRS "${BLAS_INCLUDE_DIRS}/include")

    include_directories(${BLAS_INCLUDE_DIRS})
    target_link_libraries(chainerx_native ${BLAS_LIBRARIES})
endif()

if(${CHAINERX_BUILD_TEST})
  add_executable(chainerx_native_test
      native_backend_test.cc
      native_device_test.cc
  )
  target_link_libraries(chainerx_native_test
      chainerx
      chainerx_native
      chainerx_testing
      gtest)
  add_test(NAME chainerx_native_test COMMAND chainerx_native_test)
endif()
