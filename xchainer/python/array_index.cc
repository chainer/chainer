#include "xchainer/python/array_index.h"

#include "xchainer/array_index.h"

#include "xchainer/python/slice.h"

namespace xchainer {
namespace python {

namespace py = pybind11;

namespace {

ArrayIndex MakeArrayIndex(py::handle handle) {
    PyObject* obj = handle.ptr();
    if (obj == Py_None) {
        return ArrayIndex(NewAxis{});
    }
    if (PYBIND11_LONG_CHECK(obj)) {
        return ArrayIndex(py::cast<int64_t>(handle));
    }
    if (PySlice_Check(obj)) {
        return ArrayIndex(internal::MakeSlice(py::cast<py::slice>(handle)));
    }
    throw XchainerError("only integers, slices (`:`), xchainer.newaxis (`None`) are valid indices");
}

std::vector<ArrayIndex> MakeArrayIndicesFromTuple(py::tuple tup) {
    std::vector<ArrayIndex> indicies;
    for (auto& handle : tup) {
        indicies.emplace_back(MakeArrayIndex(handle));
    }
    return indicies;
}

} // namespace

namespace internal {

std::vector<ArrayIndex> MakeArrayIndices(py::handle handle) {
    PyObject* obj = handle.ptr();
    if (PyTuple_Check(obj)) {
        return MakeArrayIndicesFromTuple(py::cast<py::tuple>(handle));
    }
    return {MakeArrayIndex(handle)};
}

}  // namespace internal
}  // namespace python
}  // namespace xchainer
