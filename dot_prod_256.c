#include <immintrin.h>
#include <stdio.h>
#include "dot_prod_256.h"

// Note: Requires compilation with flags like -mavx2 and -mfma (GCC/Clang)
// or /arch:AVX2 and /fp:fast (MSVC).

float dot_product_fma_avx2(const float* v1, const float* v2, int n) {
    // Ensure the array length is a multiple of 8 for optimal AVX2 processing
    if (n % 8 != 0) {
        // Handle remainder or assert/error
        printf("Warning: Vector length not a multiple of 8. Behavior undefined for simplicity.\n");
    }

    // Initialize 8 parallel accumulators to zero
    __m256 sum_vec = _mm256_setzero_ps(); 

    // Process the vectors in chunks of 8 floats
    for (int i = 0; i < n; i += 8) {
        // Load 8 floats from memory into 256-bit registers
        __m256 r1 = _mm256_loadu_ps(&v1[i]);
        __m256 r2 = _mm256_loadu_ps(&v2[i]);

        // Fused Multiply-Add: sum_vec = (r1 * r2) + sum_vec
        // This intrinsic maps directly to a single FMA instruction (e.g., VFMADD231PS)
        sum_vec = _mm256_fmadd_ps(r1, r2, sum_vec);
    }

    // Horizontal sum: Add the 8 elements in the accumulator vector
    float sum_arr[8];
    _mm256_storeu_ps(sum_arr, sum_vec); // Store the vector to a temporary array

    float final_sum = 0.0f;
    for (int i = 0; i < 8; ++i) {
        final_sum += sum_arr[i];
    }
    
    return final_sum;
}

#undef TEST
#ifdef TEST
// Example Usage:
int main() {
    float a[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    float b[] = {8.0f, 7.0f, 6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f};
    int len = 8;

    float result = dot_product_fma_avx2(a, b, len);

    printf("Dot product result: %f\n", result); // Expected: 120.000000

    return 0;
}

#endif // TEST
