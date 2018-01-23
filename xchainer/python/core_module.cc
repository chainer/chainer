#include <pybind11/pybind11.h>

#include "xchainer/array.h"
#include "xchainer/backprop.h"
#include "xchainer/python/array.h"
#include "xchainer/python/common.h"
#include "xchainer/python/device.h"
#include "xchainer/python/dtype.h"
#include "xchainer/python/error.h"
#include "xchainer/python/scalar.h"
#include "xchainer/python/shape.h"
#ifdef XCHAINER_ENABLE_CUDA
#include "xchainer/python/cuda/hello.h"
#endif  // XCHAINER_ENABLE_CUDA

namespace xchainer {

namespace py = pybind11;

namespace {

void InitXchainerModule(pybind11::module& m) {
    using ArrayBodyPtr = std::shared_ptr<internal::ArrayBody>;

    m.doc() = "xChainer";
    m.attr("__name__") = "xchainer";  // Show each member as "xchainer.*" instead of "xchainer.core.*"

    m.def("backward",
          [](const ArrayBodyPtr& body) {
              Array array{body};
              Backward(array);
          })
        .def("empty", [](const Shape& shape, Dtype dtype) { return Array::Empty(shape, dtype).move_body(); })
        .def("full", [](const Shape& shape, Scalar value, Dtype dtype) { return Array::Full(shape, value, dtype).move_body(); })
        .def("full", [](const Shape& shape, Scalar value) { return Array::Full(shape, value).move_body(); })
        .def("zeros", [](const Shape& shape, Dtype dtype) { return Array::Zeros(shape, dtype).move_body(); })
        .def("ones", [](const Shape& shape, Dtype dtype) { return Array::Ones(shape, dtype).move_body(); })
        .def("empty_like", [](const ArrayBodyPtr& other) { return Array::EmptyLike(Array{other}).move_body(); })
        .def("full_like", [](const ArrayBodyPtr& other, Scalar value) { return Array::FullLike(Array{other}, value).move_body(); })
        .def("zeros_like", [](const ArrayBodyPtr& other) { return Array::ZerosLike(Array{other}).move_body(); })
        .def("ones_like", [](const ArrayBodyPtr& other) { return Array::OnesLike(Array{other}).move_body(); });
}
}  // namespace
}  // namespace xchainer

PYBIND11_MODULE(_core, m) {  // NOLINT
    xchainer::InitXchainerModule(m);
    xchainer::InitXchainerDevice(m);
    xchainer::InitXchainerDtype(m);
    xchainer::InitXchainerError(m);
    xchainer::InitXchainerScalar(m);
    xchainer::InitXchainerShape(m);
    xchainer::InitXchainerArray(m);
#ifdef XCHAINER_ENABLE_CUDA
    xchainer::cuda::InitXchainerCudaHello(m);
#endif  // XCHAINER_ENABLE_CUDA
}
