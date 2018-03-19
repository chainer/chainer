#include "xchainer/python/array_index.h"

#include "xchainer/array_index.h"

#include "xchainer/python/slice.h"

namespace py = pybind11;

namespace xchainer {
namespace python {
namespace {

ArrayIndex MakeArrayIndex(py::handle handle) {
    if (handle.is_none()) {
        return ArrayIndex{NewAxis{}};
    }
    if (py::int_::check_(handle)) {
        return ArrayIndex{py::cast<int64_t>(handle)};
    }
    if (py::slice::check_(handle)) {
        return ArrayIndex{internal::MakeSlice(py::cast<py::slice>(handle))};
    }
    if (py::isinstance<Broadcastable>(handle)) {
        return ArrayIndex{py::cast<Broadcastable>(handle)};
    }
    throw py::index_error("only integers, slices (`:`), xchainer.newaxis (`None`), and xchainer.broadcastable are valid indices");
}

std::vector<ArrayIndex> MakeArrayIndicesFromTuple(py::tuple tup) {
    std::vector<ArrayIndex> indicies;
    for (auto& handle : tup) {
        indicies.emplace_back(MakeArrayIndex(handle));
    }
    return indicies;
}

}  // namespace

namespace internal {

std::vector<ArrayIndex> MakeArrayIndices(py::handle handle) {
    if (py::tuple::check_(handle)) {
        return MakeArrayIndicesFromTuple(py::cast<py::tuple>(handle));
    }
    return {MakeArrayIndex(handle)};
}

}  // namespace internal
}  // namespace python

void InitXchainerArrayIndex(py::module& m) {
    py::class_<Broadcastable>(m, "Broadcastable");
    m.attr("newaxis") = py::none();
    m.attr("broadcastable") = Broadcastable{};
}

}  // namespace xchainer
