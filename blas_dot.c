// gcc -O3 blas_dot.c -o blas_benchmark -lblas -lpthread -lm -lgfortran
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <cblas.h>

int main() {
    int N = 100000000; // Size of the vectors (adjust as needed)
    double *x = (double*) malloc(N * sizeof(double));
    double *y = (double*) malloc(N * sizeof(double));
    double result;
    int incx = 1; // Increment for x array
    int incy = 1; // Increment for y array

    // Initialize vectors with some values
    for (int i = 0; i < N; i++) {
        x[i] = (double)i;
        y[i] = (double)(i * 2);
    }

    struct timeval start, end;
    
    // Start timing
    gettimeofday(&start, NULL);

    // Perform the dot product using CBLAS
    // Function signature: double cblas_ddot(const int N, const double *X, const int incX, const double *Y, const int incY)
    result = cblas_ddot(N, x, incx, y, incy);

    // Stop timing
    gettimeofday(&end, NULL);

    // Calculate time taken in seconds
    double time_taken = (end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec) / 1e6;

    printf("Dot product result: %f\n", result);
    printf("Time taken for ddot product with N=%d: %f seconds\n", N, time_taken);
    printf("Performance: %f GFLOPS\n", (2.0 * N) / (time_taken * 1e9)); // 2 FLOPs per element (mul and add)

    // Free allocated memory
    free(x);
    free(y);

    return 0;
}

