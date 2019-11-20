#pragma once

#include <string>
#include <tuple>
#include <type_traits>

#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace chainerx {
namespace python {
namespace python_internal {

namespace meta {

template <typename T>
struct is_string
    : public std::integral_constant<
              bool,
              std::is_same<char*, typename std::decay<T>::type>::value || std::is_same<const char*, typename std::decay<T>::type>::value> {
};

template <>
struct is_string<std::string> : std::true_type {};

template <typename... Conds>
struct all : std::true_type {};

template <typename Cond, typename... Conds>
struct all<Cond, Conds...> : std::conditional<Cond::value, all<Conds...>, std::false_type>::type {};

template <typename... Keys>
using all_strings = all<is_string<Keys>...>;

}  // namespace meta

template <typename... Keys>
auto GetKwargs(const py::kwargs& kwargs, Keys... keys) {
    static_assert(meta::all_strings<Keys...>::value, "All keys should be of type string or char*");

    py::none None;

    auto get_valid_arg = [None, kwargs](auto key) -> py::object { return kwargs.attr("pop")(key, None); };

    auto kwargs_tuple = std::make_tuple<std::conditional_t<true, py::object, Keys>...>(get_valid_arg(keys)...);

    if (kwargs.size() > 0) {
        std::string err_msg = "Invalid Keyword Arguments : ";
        for (auto key_val : kwargs) {
            err_msg += py::cast<std::string>(key_val.first) + " ";
        }
        throw py::type_error{err_msg};
    }

    return kwargs_tuple;
}

}  // namespace python_internal
}  // namespace python
}  // namespace chainerx
