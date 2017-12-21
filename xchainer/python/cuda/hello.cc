#include "xchainer/python/cuda/hello.h"

#include "xchainer/cuda/hello.h"

namespace xchainer {
namespace cuda {

namespace py = pybind11;  // standard convention

void InitXchainerCudaHello(pybind11::module& m) { m.def("hello", &Hello); }

}  // namespace cuda
}  // namespace xchainer
