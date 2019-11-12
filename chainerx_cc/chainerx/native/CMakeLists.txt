install(FILES
    native_device.h
    native_backend.h
    data_type.h
    elementwise.h
    kernel_regist.h
    reduce.h
    col2im.h
    im2col.h
    tensor_dot.h
    DESTINATION include/chainerx/native
    )

add_library(chainerx_native STATIC
    native_device.cc
    native_device/arithmetic.cc
    native_device/batch_norm.cc
    native_device/binary.cc
    native_device/conv.cc
    native_device/copy.cc
    native_device/logic.cc
    native_device/dot.cc
    native_device/exp_log.cc
    native_device/fill.cc
    native_device/hyperbolic.cc
    native_device/indexing.cc
    native_device/linalg.cc
    native_device/memory.cc
    native_device/misc.cc
    native_device/pool.cc
    native_device/reduction.cc
    native_device/rnn.cc
    native_device/rounding.cc
    native_device/statistics.cc
    native_device/trigonometric.cc
    native_backend.cc
    col2im.cc
    im2col.cc
    tensor_dot.cc)

if(${BLAS_FOUND})
    add_definitions(-DCHAINERX_ENABLE_BLAS=1)
    target_link_libraries(chainerx_native ${BLAS_LIBRARIES})
endif()

if(${LAPACK_FOUND})
    MESSAGE(STATUS "Found LAPACK (library: ${LAPACK_LIBRARIES})")
    include_directories(${LAPACK_INCLUDE_DIRS})
    target_link_libraries(chainerx_native ${LAPACK_LIBRARIES})
endif()

if(${CHAINERX_BUILD_TEST})
  add_executable(chainerx_native_test
      native_backend_test.cc
      native_device_test.cc
  )
  target_link_libraries(chainerx_native_test
      chainerx
      chainerx_test_main
      # gtest is linked via chainerx_testing.
      )
  add_test(NAME chainerx_native_test COMMAND chainerx_native_test)
endif()
