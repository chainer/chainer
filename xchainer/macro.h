#pragma once

#include <cassert>
#include <cstdlib>

#ifdef NDEBUG
#define XCHAINER_DEBUG false
#else  // NDEBUG
#define XCHAINER_DEBUG true
#endif  // NDEBUG

#define XCHAINER_ASSERT(...)                                   \
    do {                                                       \
        if (XCHAINER_DEBUG) {                                  \
            (void)(false && (__VA_ARGS__)); /* maybe unused */ \
            assert(__VA_ARGS__);                               \
        }                                                      \
    } while (false)

#ifndef XCHAINER_HOST_DEVICE
#ifdef __CUDACC__
#define XCHAINER_HOST_DEVICE __host__ __device__
#else  // __CUDA__
#define XCHAINER_HOST_DEVICE
#endif  // __CUDACC__
#endif  // XCHAINER_HOST_DEVICE

#ifndef XCHAINER_NEVER_REACH
#ifdef NDEBUG
#define XCHAINER_NEVER_REACH() (std::abort())
#else  // NDEBUG
#define XCHAINER_NEVER_REACH()                    \
    do {                                          \
        assert(false); /* NOLINT(cert-dcl03-c) */ \
        std::abort();                             \
    } while (false)
#endif  // NDEBUG
#endif  // XCHAINER_NEVER_REACH
