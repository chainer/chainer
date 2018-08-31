#include "chainerx/python/array_index.h"

#include "chainerx/array_index.h"

#include "chainerx/python/slice.h"

namespace chainerx {
namespace python {
namespace python_internal {

namespace py = pybind11;

namespace {

ArrayIndex MakeArrayIndex(py::handle handle) {
    if (handle.is_none()) {
        return ArrayIndex{NewAxis{}};
    }
    if (py::int_::check_(handle)) {
        return ArrayIndex{py::cast<int64_t>(handle)};
    }
    if (py::slice::check_(handle)) {
        return ArrayIndex{MakeSlice(py::cast<py::slice>(handle))};
    }
    throw py::index_error{"only integers, slices (`:`), and chainerx.newaxis (`None`) are valid indices"};
}

std::vector<ArrayIndex> MakeArrayIndicesFromTuple(py::tuple tup) {
    std::vector<ArrayIndex> indicies;
    for (auto& handle : tup) {
        indicies.emplace_back(MakeArrayIndex(handle));
    }
    return indicies;
}

}  // namespace

std::vector<ArrayIndex> MakeArrayIndices(py::handle handle) {
    if (py::tuple::check_(handle)) {
        return MakeArrayIndicesFromTuple(py::cast<py::tuple>(handle));
    }
    return {MakeArrayIndex(handle)};
}

void InitChainerxArrayIndex(py::module& m) { m.attr("newaxis") = py::none(); }

}  // namespace python_internal
}  // namespace python
}  // namespace chainerx
