#pragma once

#include <tuple>

#include <pybind11/pybind11.h>

namespace py = pybind11;

template <typename... Keys>
auto GetKwargs(const py::kwargs& kwargs, Keys... keys) {
    py::none None;

    auto lookup_func = [None, kwargs](auto key) -> py::handle {
        py::handle handle{None};
        if (kwargs.contains(key)) {
            handle = kwargs[key];
        }
        return handle;
    };

    // TODO(kshitij12345): Find if pybind support `pop` on a dict.
    // if (kwargs.size() > 0) {
    //     throw py::type_error{};
    // }

    return std::make_tuple<std::conditional_t<true, py::handle, Keys>...>(lookup_func(keys)...);
}
