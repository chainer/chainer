if(${CUDA_FOUND})
    add_subdirectory(cuda)
endif()

add_library(_core.so MODULE
    core_module.cc
    array.cc
    device.cc
    dtype.cc
    error.cc
    scalar.cc
    shape.cc
    )
target_link_libraries(_core.so
    PRIVATE
    "-Wl,-flto" pybind11::module
    "-Wl,-R,'$ORIGIN/.'" xchainer)
if(${CUDA_FOUND})
    target_link_libraries(_core.so PRIVATE core_cuda)
endif()
set_target_properties(_core.so
    PROPERTIES
    PREFIX "${PYTHON_MODULE_PREFIX}"
    SUFFIX "${PYTHON_MODULE_SUFFIX}")
