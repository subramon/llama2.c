// gcc -O3 -mavx2 -march=native  -o softmax softmax.c -lm
#include <stdio.h>
#include <x86intrin.h>
#include <immintrin.h>
#include <stdint.h>
#include <stdalign.h>
#include <math.h>

// Fast vectorized exponential for AVX2.
// Works for all real numbers, uses the exp2(x * log2(e)) method with
// rounding to nearest integer and polynomial approximation of 2^f.
static inline __m256 _mm256_exp_ps(__m256 x) {
    // Constants
    const __m256 log2e = _mm256_set1_ps(1.442695041f);          // log2(e)
    const __m256 c1 = _mm256_set1_ps(0.6931471805599453f);      // ln(2)
    const __m256 c2 = _mm256_set1_ps(0.2402265069591007f);
    const __m256 c3 = _mm256_set1_ps(0.05550410866482158f);
    const __m256 c4 = _mm256_set1_ps(0.009618129107628477f);
    const __m256 one = _mm256_set1_ps(1.0f);
    const __m256 min_arg = _mm256_set1_ps(-30.0f);              // clamp for underflow

    // Protect against large negative arguments (exp(-30) ~ 9e-14, effectively zero)
    x = _mm256_max_ps(x, min_arg);

    // z = x * log2(e)
    __m256 z = _mm256_mul_ps(x, log2e);

    // Round to nearest integer (ties to even)
    __m256 zi = _mm256_round_ps(z, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    __m256i i = _mm256_cvtps_epi32(zi);       // integer part as 32-bit ints

    // Fractional part f = z - i
    __m256 f = _mm256_sub_ps(z, zi);

    // Polynomial approximation of 2^f in [0,1] : 2^f ≈ 1 + f*(c1 + f*(c2 + f*(c3 + f*c4)))
    __m256 p = _mm256_add_ps(one, _mm256_mul_ps(f,
        _mm256_add_ps(c1, _mm256_mul_ps(f,
            _mm256_add_ps(c2, _mm256_mul_ps(f,
                _mm256_add_ps(c3, _mm256_mul_ps(f, c4))))))));

    // Now compute p * 2^i by adjusting the exponent of p.
    // p is in [1,2), its exponent bias is 127. Adding i to the exponent field
    // yields multiplication by 2^i.
    __m256i p_bits = _mm256_castps_si256(p);
    __m256i i_shift = _mm256_slli_epi32(i, 23);   // i << 23
    __m256i exp_bits = _mm256_add_epi32(p_bits, i_shift);
    __m256 result = _mm256_castsi256_ps(exp_bits);

    // Handle underflow: if i is very negative, above addition yields zero/denormal.
    // The earlier clamp on x already prevents extreme underflow.
    return result;
}

// Compute softmax of array x into array y.
// Both x and y are assumed to be 64-byte aligned, length n.
void softmax_avx2(const float* x, float* y, int n) {
    // ---------- Step 1: find maximum element ----------
    __m256 max_vec = _mm256_set1_ps(-INFINITY);
    int i = 0;
    for (; i <= n - 8; i += 8) {
        __m256 x_vec = _mm256_load_ps(x + i);
        max_vec = _mm256_max_ps(max_vec, x_vec);
    }
    // Handle remainder (less than 8 elements)
    float max_scalar = -INFINITY;
    for (int j = i; j < n; ++j) {
        if (x[j] > max_scalar) max_scalar = x[j];
    }
    // Reduce vector max to scalar
    alignas(64) float max_arr[8];
    _mm256_store_ps(max_arr, max_vec);
    for (int k = 0; k < 8; ++k) {
        if (max_arr[k] > max_scalar) max_scalar = max_arr[k];
    }
    __m256 max_bcast = _mm256_set1_ps(max_scalar);

    // ---------- Step 2: compute exponentials and sum ----------
    __m256 sum_vec = _mm256_setzero_ps();
    // First pass: compute y = exp(x - max) and accumulate sum in vector
    for (i = 0; i <= n - 8; i += 8) {
        __m256 x_vec = _mm256_load_ps(x + i);
        __m256 x_minus_max = _mm256_sub_ps(x_vec, max_bcast);
        __m256 exp_vec = _mm256_exp_ps(x_minus_max);
        _mm256_store_ps(y + i, exp_vec);
        sum_vec = _mm256_add_ps(sum_vec, exp_vec);
    }
    // Handle remainder (scalar)
    float sum_scalar = 0.0f;
    for (int j = i; j < n; ++j) {
        float val = expf(x[j] - max_scalar);
        y[j] = val;
        sum_scalar += val;
    }
    // Reduce vector sum to scalar
    alignas(64) float sum_arr[8];
    _mm256_store_ps(sum_arr, sum_vec);
    for (int k = 0; k < 8; ++k) {
        sum_scalar += sum_arr[k];
    }
    float inv_sum = 1.0f / sum_scalar;

    // ---------- Step 3: normalize ----------
    __m256 inv_sum_vec = _mm256_set1_ps(inv_sum);
    for (i = 0; i <= n - 8; i += 8) {
        __m256 exp_vec = _mm256_load_ps(y + i);
        __m256 norm_vec = _mm256_mul_ps(exp_vec, inv_sum_vec);
        _mm256_store_ps(y + i, norm_vec);
    }
    for (int j = i; j < n; ++j) {
        y[j] *= inv_sum;
    }
}
void 
my_softmax(
    const float* x, 
    float* restrict y, 
    int size
    ) 
{
  // find max value (for numerical stability)
  register float max_val = x[0];
  for (int i = 1; i < size; i++) {
    if (x[i] > max_val) {
      max_val = x[i];
    }
  }
  // exp and sum
  for (int i = 0; i < size; i++) {
    y[i] = expf(x[i] - max_val);
  }
  register float sum = 0.0f;
  for (int i = 0; i < size; i++) {
    sum += y[i];
  }
  // normalize
  for (int i = 0; i < size; i++) {
    y[i] /= sum;
  }
}

#define TEST
#ifdef TEST
int
main(
    int argc,
    char **argv
    )
{
  int status = 0;
  uint64_t t1 = 0, t2 = 0, t3 = 0;
  int n = 65536;
  alignas(64) float x[n];
  alignas(64) float y[n];
  alignas(64) float z[n];
  for ( int i = 0; i < n; i++ ) { x[i] = drand48(); }
  t1 = __rdtsc();
  printf("start\n");
  softmax_avx2(x, y, n); 
  t2 = __rdtsc();
  printf("avx2\n");
  my_softmax(x, z, n); 
  printf("mine\n");
  t3 = __rdtsc();
  printf("avx2 = %lu \n", (t2-t1));
  printf("mine = %lu \n", (t3-t2));
  /*
  for ( int i = 0; i < 32; i++ ) {
    printf("%2d %f %f \n", i, y[i], z[i]);
  }
  */

BYE:
  
  return status;
}
#endif
