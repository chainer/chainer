add_library(core_cuda STATIC
    hello.cc)
target_link_libraries(core_cuda
    PRIVATE
    "-Wl,-flto" pybind11::module
    "-Wl,-R,'$ORIGIN/.'" xchainer)
