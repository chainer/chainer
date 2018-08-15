install(FILES
    native_device.h
    native_backend.h
    elementwise.h
    reduce.h
    col2im.h
    im2col.h
    tensor_dot.h
    DESTINATION include/xchainer/native
    )

add_library(xchainer_native STATIC
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

if(${XCHAINER_BUILD_TEST})
  add_executable(xchainer_native_test
      native_backend_test.cc
      native_device_test.cc
  )
  target_link_libraries(xchainer_native_test
      xchainer
      xchainer_native
      xchainer_testing
      gtest)
  add_test(NAME xchainer_native_test COMMAND xchainer_native_test)
endif()
