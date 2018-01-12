#include "xchainer/python/cuda/hello.h"

#include "xchainer/cuda/hello.h"

namespace xchainer {
namespace cuda {

void InitXchainerCudaHello(pybind11::module& m) { m.def("hello", &Hello); }

}  // namespace cuda
}  // namespace xchainer
