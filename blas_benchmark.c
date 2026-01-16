#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cblas.h>

#define N 1000 // Matrix dimension, adjust for performance

int main() {
    double *A, *B, *C;
    int i;
    clock_t start, end;
    double cpu_time_used;

    // Allocate memory for matrices
    A = (double *)malloc(N * N * sizeof(double));
    B = (double *)malloc(N * N * sizeof(double));
    C = (double *)malloc(N * N * sizeof(double));

    // Initialize matrices with random values
    for (i = 0; i < N * N; i++) {
        A[i] = (double)rand() / RAND_MAX;
        B[i] = (double)rand() / RAND_MAX;
        C[i] = 0.0;
    }

    // Benchmark the matrix multiplication (C = A * B)
    start = clock();

    // Call dgemm (double precision general matrix multiplication) function
    // Parameters: Order, TransA, TransB, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, N, N, N, 1.0, A, N, B, N, 0.0, C, N);

    end = clock();
    cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;

    printf("Matrix multiplication of size %dx%d completed in %f seconds\n", N, N, cpu_time_used);
    printf("Performance: %f GFLOPS\n", (2.0 * N * N * N) / (cpu_time_used * 1e9));

    // Free allocated memory
    free(A);
    free(B);
    free(C);

    return 0;
}

