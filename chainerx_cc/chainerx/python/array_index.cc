#include "chainerx/python/array_index.h"

#include <type_traits>
#include <vector>

#include <pybind11/pybind11.h>

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
    if (py::isinstance<py::int_>(handle)) {
        return ArrayIndex{py::cast<int64_t>(handle)};
    }
    if (py::isinstance<py::slice>(handle)) {
        return ArrayIndex{MakeSlice(py::cast<py::slice>(handle))};
    }
    // NumPy integer scalar
    // numpy.integer is cached because it's time consuming to import each time.
    // (py::handle is trivially destructible)
    static py::handle numpy_integer = py::module::import("numpy").attr("integer");
    static_assert(std::is_trivially_destructible<py::handle>::value, "");
    if (py::isinstance(handle, numpy_integer)) {
        return ArrayIndex{py::cast<int64_t>(handle)};
    }
    throw py::index_error{"only integers, slices (`:`), and chainerx.newaxis (`None`) are valid indices"};
}

std::vector<ArrayIndex> MakeArrayIndicesFromTuple(py::tuple tup) {
    std::vector<ArrayIndex> indicies;
    for (auto handle : tup) {
        indicies.emplace_back(MakeArrayIndex(handle));
    }
    return indicies;
}

}  // namespace

std::vector<ArrayIndex> MakeArrayIndices(py::handle handle) {
    if (py::isinstance<py::tuple>(handle)) {
        return MakeArrayIndicesFromTuple(py::cast<py::tuple>(handle));
    }
    return {MakeArrayIndex(handle)};
}

void InitChainerxArrayIndex(py::module& m) { m.attr("newaxis") = py::none(); }

}  // namespace python_internal
}  // namespace python
}  // namespace chainerx
