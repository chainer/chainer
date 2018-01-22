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

namespace xchainer {

namespace py = pybind11;

namespace {

void InitXchainerModule(pybind11::module& m) {
    m.doc() = "xChainer";
    m.attr("__name__") = "xchainer";  // Show each member as "xchainer.*" instead of "xchainer.core.*"

    m.def("backward", &Backward)
        .def("empty", &Array::Empty)
        .def("full", py::overload_cast<const Shape&, Scalar, Dtype>(&Array::Full))
        .def("full", py::overload_cast<const Shape&, Scalar>(&Array::Full))
        .def("zeros", &Array::Zeros)
        .def("ones", &Array::Ones)
        .def("empty_like", &Array::EmptyLike)
        .def("full_like", &Array::FullLike)
        .def("zeros_like", &Array::ZerosLike)
        .def("ones_like", &Array::OnesLike);
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
}
