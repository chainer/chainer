#pragma once

#include <cassert>
#include <cstdlib>

#ifdef NDEBUG
#define CHAINERX_DEBUG false
#else  // NDEBUG
#define CHAINERX_DEBUG true
#endif  // NDEBUG

#if CHAINERX_DEBUG
#define CHAINERX_ASSERT assert
#else  // CHAINERX_DEBUG
// This expression suppresses dead code and unused variable warnings of clang-tidy caused by "unused" __VA_ARGS__.
// We use a lambda call to bypass clant-tidy's dead-code analysis, which currently does not evaluate non-const expressions.
#define CHAINERX_ASSERT(...) (void)([] { return false; }() && (__VA_ARGS__))
#endif  // CHAINERX_DEBUG

#ifndef CHAINERX_HOST_DEVICE
#ifdef __CUDACC__
#define CHAINERX_HOST_DEVICE __host__ __device__
#else  // __CUDA__
#define CHAINERX_HOST_DEVICE
#endif  // __CUDACC__
#endif  // CHAINERX_HOST_DEVICE

#ifndef CHAINERX_NEVER_REACH
#ifdef NDEBUG
#define CHAINERX_NEVER_REACH() (std::abort())
#else  // NDEBUG
#define CHAINERX_NEVER_REACH()                                        \
    do {                                                              \
        assert(false); /* NOLINT(cert-dcl03-c, misc-static-assert) */ \
        std::abort();                                                 \
    } while (false)
#endif  // NDEBUG
#endif  // CHAINERX_NEVER_REACH

// Hides a symbol from the ChainerX shared object symbol table.
// Note that symbols are hidden by default on Windows, which is opposite from POSIX.
//
// CHAINERX_VISIBILITY_HIDDEN was introduced in order to allow defining classes holding pybind11 objects (which are hidden by default) as
// members.
// https://pybind11.readthedocs.io/en/stable/faq.html#someclass-declared-with-greater-visibility-than-the-type-of-its-field-someclass-member-wattributes
#ifndef CHAINERX_VISIBILITY_HIDDEN
#if defined(WIN32) || defined(_WIN32)
#define CHAINERX_VISIBILITY_HIDDEN
#else  // defined(WIN32) || defined(_WIN32)
#define CHAINERX_VISIBILITY_HIDDEN __attribute__((visibility("hidden")))
#endif  // defined(WIN32) || defined(_WIN32)
#endif  // CHAINERX_VISIBILITY_HIDDEN
