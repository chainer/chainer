// This header provides chainerx a consistent interface to CBLAS code.
// It is needed because not all providers of cblas provide cblas.h, e.g. MKL.

#pragma once

extern "C" {

#ifndef CBLAS_ENUM_DEFINED_H
#define CBLAS_ENUM_DEFINED_H
enum CBLAS_ORDER { CblasRowMajor = 101 };
enum CBLAS_TRANSPOSE { CblasNoTrans = 111, CblasTrans = 112 };
#endif

void cblas_sgemm(
        enum CBLAS_ORDER Order,
        enum CBLAS_TRANSPOSE TransA,
        enum CBLAS_TRANSPOSE TransB,
        int M,
        int N,
        int K,
        float alpha,
        const float* A,
        int lda,
        const float* B,
        int ldb,
        float beta,
        float* C,
        int ldc);

void cblas_dgemm(
        enum CBLAS_ORDER Order,
        enum CBLAS_TRANSPOSE TransA,
        enum CBLAS_TRANSPOSE TransB,
        int M,
        int N,
        int K,
        double alpha,
        const double* A,
        int lda,
        const double* B,
        int ldb,
        double beta,
        double* C,
        int ldc);

}  // extern "C"
