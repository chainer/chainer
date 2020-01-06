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

set(module_name _core.so)

if(MSVC)
    set(PYBIND11_LTO_CXX_FLAGS "" CACHE INTERNAL "")
    set(PYBIND11_LTO_LINKER_FLAGS "" CACHE INTERNAL "")
    set(module_name _core.pyd)
endif()

pybind11_add_module(${module_name} MODULE ${srcs})

# CUDA specific bindings.
if(${CUDA_FOUND})
    CUDA_ADD_LIBRARY(_core_cuda STATIC cuda/cuda_module.cc)
    target_link_libraries(_core_cuda pybind11::module)
    target_link_libraries(${module_name} PRIVATE _core_cuda)
endif()

target_link_libraries(${module_name} PRIVATE chainerx)

# Only python binding needs to be relocatable, it will load chainerx related shared library at runtime.
set_target_properties(${module_name}
    PROPERTIES
    SKIP_BUILD_RPATH FALSE
    BUILD_WITH_INSTALL_RPATH TRUE)

set_target_properties(${module_name}
    PROPERTIES
    PREFIX "${PYTHON_MODULE_PREFIX}"
    SUFFIX "${PYTHON_MODULE_SUFFIX}")
