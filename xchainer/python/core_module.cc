#include <pybind11/pybind11.h>

#include <string>

#include <nonstd/optional.hpp>

#include "xchainer/array.h"
#include "xchainer/backprop.h"
#include "xchainer/constant.h"
#include "xchainer/context.h"
#include "xchainer/python/array.h"
#include "xchainer/python/backend.h"
#include "xchainer/python/common.h"
#include "xchainer/python/context.h"
#include "xchainer/python/device.h"
#include "xchainer/python/dtype.h"
#include "xchainer/python/error.h"
#include "xchainer/python/scalar.h"
#include "xchainer/python/shape.h"

namespace xchainer {

namespace py = pybind11;

namespace {

Device& GetDevice(const nonstd::optional<std::string>& device_id) {
    return device_id.has_value() ? GetDefaultContext().GetDevice(device_id.value()) : GetDefaultDevice();
}

void InitXchainerModule(pybind11::module& m) {
    using ArrayBodyPtr = std::shared_ptr<internal::ArrayBody>;

    m.doc() = "xChainer";
    m.attr("__name__") = "xchainer";  // Show each member as "xchainer.*" instead of "xchainer.core.*"

    m.attr("DEFAULT_GRAPH_ID") = kDefaultGraphId;

    m.def("backward",
          [](const ArrayBodyPtr& body, const GraphId& graph_id, bool enable_double_backprop) {
              Array array{body};
              auto double_backprop = enable_double_backprop ? DoubleBackpropOption::kEnable : DoubleBackpropOption::kDisable;
              Backward(array, graph_id, double_backprop);
          },
          py::arg().noconvert(), py::arg("graph_id") = kDefaultGraphId, py::arg("enable_double_backprop") = false)
        .def("empty",
             [](const Shape& shape, Dtype dtype, const nonstd::optional<std::string>& device_id) {
                 return Array::Empty(shape, dtype, GetDevice(device_id)).move_body();
             },
             py::arg("shape"), py::arg("dtype"), py::arg("device") = nullptr)
        .def("full",
             [](const Shape& shape, Scalar value, Dtype dtype, const nonstd::optional<std::string>& device_id) {
                 return Array::Full(shape, value, dtype, GetDevice(device_id)).move_body();
             },
             py::arg("shape"), py::arg("value"), py::arg("dtype"), py::arg("device") = nullptr)
        .def("full",
             [](const Shape& shape, Scalar value, const nonstd::optional<std::string>& device_id) {
                 return Array::Full(shape, value, GetDevice(device_id)).move_body();
             },
             py::arg("shape"), py::arg("value"), py::arg("device") = nullptr)
        .def("zeros",
             [](const Shape& shape, Dtype dtype, const nonstd::optional<std::string>& device_id) {
                 return Array::Zeros(shape, dtype, GetDevice(device_id)).move_body();
             },
             py::arg("shape"), py::arg("dtype"), py::arg("device") = nullptr)
        .def("ones",
             [](const Shape& shape, Dtype dtype, const nonstd::optional<std::string>& device_id) {
                 return Array::Ones(shape, dtype, GetDevice(device_id)).move_body();
             },
             py::arg("shape"), py::arg("dtype"), py::arg("device") = nullptr)
        .def("empty_like",
             [](const ArrayBodyPtr& other, const nonstd::optional<std::string>& device_id) {
                 return Array::EmptyLike(Array{other}, GetDevice(device_id)).move_body();
             },
             py::arg("other"), py::arg("device") = nullptr)
        .def("full_like",
             [](const ArrayBodyPtr& other, Scalar value, const nonstd::optional<std::string>& device_id) {
                 return Array::FullLike(Array{other}, value, GetDevice(device_id)).move_body();
             },
             py::arg("other"), py::arg("value"), py::arg("device") = nullptr)
        .def("zeros_like",
             [](const ArrayBodyPtr& other, const nonstd::optional<std::string>& device_id) {
                 return Array::ZerosLike(Array{other}, GetDevice(device_id)).move_body();
             },
             py::arg("other"), py::arg("device") = nullptr)
        .def("ones_like",
             [](const ArrayBodyPtr& other, const nonstd::optional<std::string>& device_id) {
                 return Array::OnesLike(Array{other}, GetDevice(device_id)).move_body();
             },
             py::arg("other"), py::arg("device") = nullptr);
}
}  // namespace
}  // namespace xchainer

PYBIND11_MODULE(_core, m) {  // NOLINT
    xchainer::InitXchainerModule(m);
    xchainer::InitXchainerContext(m);
    xchainer::InitXchainerBackend(m);
    xchainer::InitXchainerDevice(m);
    xchainer::InitXchainerDtype(m);
    xchainer::InitXchainerError(m);
    xchainer::InitXchainerScalar(m);
    xchainer::InitXchainerShape(m);
    xchainer::InitXchainerArray(m);
}
