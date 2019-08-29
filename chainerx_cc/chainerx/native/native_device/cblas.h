#pragma once

extern "C" {

#ifndef CBLAS_ENUM_DEFINED_H
#define CBLAS_ENUM_DEFINED_H
enum CBLAS_ORDER { CblasRowMajor = 101 };
enum CBLAS_TRANSPOSE { CblasNoTrans = 111, CblasTrans = 112 };
#endif

void cblas_sgemm(
        const enum CBLAS_ORDER Order,
        const enum CBLAS_TRANSPOSE TransA,
        const enum CBLAS_TRANSPOSE TransB,
        const int M,
        const int N,
        const int K,
        const float alpha,
        const float* A,
        const int lda,
        const float* B,
        const int ldb,
        const float beta,
        float* C,
        const int ldc);

void cblas_dgemm(
        const enum CBLAS_ORDER Order,
        const enum CBLAS_TRANSPOSE TransA,
        const enum CBLAS_TRANSPOSE TransB,
        const int M,
        const int N,
        const int K,
        const double alpha,
        const double* A,
        const int lda,
        const double* B,
        const int ldb,
        const double beta,
        double* C,
        const int ldc);

}  // extern "C"
