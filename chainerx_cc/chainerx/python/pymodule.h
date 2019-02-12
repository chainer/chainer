#pragma once

#include <pybind11/pybind11.h>

namespace chainerx {
namespace python {
namespace python_internal {

namespace py = pybind11;

class numpy {
public:
    static py::module module() {
        // create an instance by lazy initialization
        static py::module ret = py::module::import("numpy");
        return ret;
    }

    static py::handle array() {
        // create an instance by lazy initialization
        static py::handle ret = py::module::import("numpy").attr("array");
        return ret;
    }

private:
    // hide the (de)constructors
    numpy();
    numpy(numpy const&);
    numpy& operator=(numpy const&);
    ~numpy();
};

class cupy {
public:
    static py::module module() {
        // create an instance by lazy initialization
        static py::module ret = py::module::import("cupy");
        return ret;
    }

    static py::handle ndarray() {
        // create an instance by lazy initialization
        static py::handle ret = py::module::import("cupy").attr("ndarray");
        return ret;
    }

    class cuda {
    public:
        class memory {
        public:
            static py::handle MemoryPointer() {
                // create an instance by lazy initialization
                static py::handle ret = py::module::import("cupy").attr("cuda").attr("memory").attr("MemoryPointer");
                return ret;
            }

            static py::handle UnownedMemory() {
                // create an instance by lazy initialization
                static py::handle ret = py::module::import("cupy").attr("cuda").attr("memory").attr("UnownedMemory");
                return ret;
            }

        private:
            // hide the (de)constructors
            memory();
            memory(memory const&);
            memory& operator=(memory const&);
            ~memory();
        };

    private:
        // hide the (de)constructors
        cuda();
        cuda(cuda const&);
        cuda& operator=(cuda const&);
        ~cuda();
    };

private:
    // hide the (de)constructors
    cupy();
    cupy(cupy const&);
    cupy& operator=(cupy const&);
    ~cupy();
};

}  // namespace python_internal
}  // namespace python
}  // namespace chainerx
