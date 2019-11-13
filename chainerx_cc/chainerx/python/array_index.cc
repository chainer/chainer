#include "chainerx/python/common_export.h"

#include "chainerx/python/array_index.h"

#include <type_traits>
#include <vector>

#include <pybind11/pybind11.h>

#include "chainerx/array_index.h"

#include "chainerx/python/py_cached_objects.h"
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
    if (py::isinstance<py::ellipsis>(handle)) {
        return ArrayIndex{Ellipsis{}};
    }
    // NumPy integer scalar
    auto numpy_integer_type = GetCachedNumpyInteger();
    static_assert(std::is_trivially_destructible<py::handle>::value, "");
    if (py::isinstance(handle, numpy_integer_type)) {
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

}  // namespace python_internal
}  // namespace python
}  // namespace chainerx
