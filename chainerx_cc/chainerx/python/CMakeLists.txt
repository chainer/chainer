set(srcs
    core_module.cc
    array.cc
    array_index.cc
    axes.cc
    backend.cc
    backward.cc
    backprop_mode.cc
    chainer_interop.cc
    check_backward.cc
    context.cc
    device.cc
    dtype.cc
    error.cc
    graph.cc
    routines.cc
    scalar.cc
    shape.cc
    slice.cc
    strides.cc
    testing/device_buffer.cc
    testing/testing_module.cc
    )

pybind11_add_module(_core.so MODULE ${srcs})

# CUDA specific bindings.
if(${CUDA_FOUND})
    CUDA_ADD_LIBRARY(_core_cuda STATIC cuda/cuda_module.cc)
    target_link_libraries(_core_cuda pybind11::module)
    target_link_libraries(_core.so PRIVATE _core_cuda)
endif()

if(${APPLE})
    target_link_libraries(_core.so PRIVATE "-Wl,-rpath,@loader_path" chainerx)
else()
    target_link_libraries(_core.so PRIVATE "-Wl,-R,'$ORIGIN/.'" chainerx)
endif()

# Visibility (CXX_VISIBILITY_PRESET) must be set to "default" to register custom exceptions, overriding the visibility configured by pybind11_add_module.
# Note however that this only seems to be an issue when building with libc++.
set_target_properties(_core.so
    PROPERTIES
    PREFIX "${PYTHON_MODULE_PREFIX}"
    SUFFIX "${PYTHON_MODULE_SUFFIX}"
    CXX_VISIBILITY_PRESET "default")
