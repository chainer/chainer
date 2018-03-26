add_library(xchainer_native STATIC
    native_device.cc
    native_backend.cc)

if(${ENABLE_TEST})
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
