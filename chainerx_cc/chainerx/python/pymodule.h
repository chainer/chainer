#pragma once

#include <pybind11/pybind11.h>

#include "chainerx/macro.h"

namespace chainerx {
namespace python {
namespace python_internal {

namespace py = pybind11;

struct CHAINERX_VISIBILITY_HIDDEN numpy_module {
    py::handle module_ = py::none();

    py::handle& module() {
        if (module_.is_none()) {
            auto ret = py::module::import("numpy");
            module_ = ret;
        }
        return module_;
    }

    py::handle array_ = py::none();

    py::handle& array() {
        if (array_.is_none()) {
            auto ret = py::module::import("numpy").attr("array");
            array_ = ret;
        }
        return array_;
    }

    py::handle integer_ = py::none();

    py::handle& integer() {
        if (integer_.is_none()) {
            auto ret = py::module::import("numpy").attr("integer");
            integer_ = ret;
        }
        return integer_;
    }
};

inline PYBIND11_NOINLINE void load_numpy_module(numpy_module* &ptr) {
    ptr = &py::get_or_create_shared_data<numpy_module>("_numpy_module");
}

inline numpy_module& get_numpy_module() {
    static numpy_module* ptr = nullptr;
    if (!ptr) {
        load_numpy_module(ptr);
    }
    return *ptr;
}

struct CHAINERX_VISIBILITY_HIDDEN cupy_module {
    py::handle module_ = py::none();

    py::handle& module() {
        if (module_.is_none()) {
            auto ret = py::module::import("cupy");
            module_ = ret;
        }
        return module_;
    }

    py::handle ndarray_ = py::none();

    py::handle& ndarray() {
        if (ndarray_.is_none()) {
            auto ret = py::module::import("cupy").attr("ndarray");
            ndarray_ = ret;
        }
        return ndarray_;
    }

    py::handle cuda_memory_MemoryPointer_ = py::none();

    py::handle& cuda_memory_MemoryPointer() {
        if (cuda_memory_MemoryPointer_.is_none()) {
            auto ret = py::module::import("cupy").attr("cuda").attr("memory").attr("MemoryPointer");
            cuda_memory_MemoryPointer_ = ret;
        }
        return cuda_memory_MemoryPointer_;
    }

    py::handle cuda_memory_UnownedMemory_ = py::none();

    py::handle& cuda_memory_UnownedMemory() {
        if (cuda_memory_UnownedMemory_.is_none()) {
            auto ret = py::module::import("cupy").attr("cuda").attr("memory").attr("UnownedMemory");
            cuda_memory_UnownedMemory_ = ret;
        }
        return cuda_memory_UnownedMemory_;
    }
};

inline PYBIND11_NOINLINE void load_cupy_module(cupy_module* &ptr) {
    ptr = &py::get_or_create_shared_data<cupy_module>("_cupy_module");
}

inline cupy_module& get_cupy_module() {
    static cupy_module* ptr = nullptr;
    if (!ptr) {
        load_cupy_module(ptr);
    }
    return *ptr;
}

}  // namespace python_internal
}  // namespace python
}  // namespace chainerx
