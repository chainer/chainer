#include "xchainer/python/array_index.h"

#include "xchainer/array_index.h"
#include "xchainer/slice.h"

#include "xchainer/python/common.h"

#include <iostream>

namespace xchainer {

namespace py = pybind11;

void InitXchainerArrayIndex(pybind11::module& m) {
    py::class_<ArrayIndex>{m, "ArrayIndex"}
            .def(py::init([](py::int_ index) { return ArrayIndex(py::cast<int64_t>(index)); }))
            .def(py::init([](const Slice& slice) { return ArrayIndex(slice); }));
}

}  // namespace xchainer
