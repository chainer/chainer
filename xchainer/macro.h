#pragma once

#include <cassert>
#include <cstdlib>

#ifdef NDEBUG
#define XCHAINER_DEBUG false
#else  // NDEBUG
#define XCHAINER_DEBUG true
#endif  // NDEBUG

#if XCHAINER_DEBUG
#define XCHAINER_ASSERT assert
#else
// We use a lambda call to bypass clant-tidy's dead-code analysis, which currently does not evaluate non-const expressions.
#define XCHAINER_ASSERT(...) (void)([] { return false; }() && (__VA_ARGS__))  // maybe unused
#endif  // XCHAINER_DEBUG

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
