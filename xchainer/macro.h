#pragma once

#ifndef XCHAINER_HOST_DEVICE
#ifdef __CUDACC__
#define XCHAINER_HOST_DEVICE __host__ __device__
#else
#define XCHAINER_HOST_DEVICE
#endif  // __CUDACC__
#endif  // XCHAINER_HOST_DEVICE

#ifndef XCHAINER_NEVER_REACH
#ifdef NDEBUG
#include <cstdlib>
#define XCHAINER_NEVER_REACH() (std::abort())
#else
#include <cassert>
#define XCHAINER_NEVER_REACH()                    \
    do {                                          \
        assert(false); /* NOLINT(cert-dcl03-c) */ \
        std::abort();                             \
    } while (false)
#endif  // NDEBUG
#endif  // XCHAINER_NEVER_REACH
