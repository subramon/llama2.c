#include <x86intrin.h> // for rdtsc
#include <stdint.h> // for uint64_t
extern uint64_t g_t_dot_prod; // for timing 
#include "dot_prod.h"
void 
dot_prod(
    const float * const x, 
    const float * const y, 
    int n,
    float * restrict ptr_rslt
    ) 
{
  uint64_t t = __rdtsc();
  register float sum = 0.0f;
  for ( int i = 0; i < n; i++ ) { 
    sum += x[i] * y[i];
  }
  *ptr_rslt = sum;
  g_t_dot_prod += (__rdtsc() - t);
}
