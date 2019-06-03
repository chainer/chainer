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
    native_device/comparison.cc
    native_device/dot.cc
    native_device/exp_log.cc
    native_device/fill.cc
    native_device/hyperbolic.cc
    native_device/indexing.cc
    native_device/memory.cc
    native_device/misc.cc
    native_device/pool.cc
    native_device/reduction.cc
    native_device/statistics.cc
    native_device/trigonometric.cc
    native_backend.cc
    col2im.cc
    im2col.cc
    tensor_dot.cc)

if(${BLAS_FOUND})
    if(DEFINED ENV{CHAINERX_BLAS_INCLUDE_DIRS})
        set(BLAS_INCLUDE_DIRS $ENV{CHAINERX_BLAS_INCLUDE_DIRS})
    else()
        # Workaround for UNIX environments to guess the BLAS include directory from its library path (.so).
        # This workaround is required since CMake does not set the BLAS_INCLUDE_DIRS variable.
        list(GET BLAS_LIBRARIES 0 BLAS_LIBRARY)
        get_filename_component(BLAS_ROOT_DIR ${BLAS_LIBRARY} DIRECTORY)  # Directory of .so.
        get_filename_component(BLAS_ROOT_DIR ${BLAS_ROOT_DIR} DIRECTORY)  # Directory of lib.
        find_path(BLAS_INCLUDE_DIRS cblas.h
            HINTS ${BLAS_ROOT_DIR}
            PATH_SUFFIXES include)
    endif()
    if(BLAS_INCLUDE_DIRS)
        message(STATUS "Found BLAS (include directory: ${BLAS_INCLUDE_DIRS})")
        add_definitions(-DCHAINERX_ENABLE_BLAS=1)
        include_directories(${BLAS_INCLUDE_DIRS})
        target_link_libraries(chainerx_native ${BLAS_LIBRARIES})
    else()
        message(WARNING "Found the BLAS library but not the include directory. Skipping BLAS.")
    endif()
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
