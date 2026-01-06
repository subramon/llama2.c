#include "mul_v_add_s.h"
// vec1 += vec2 * scalar 
void 
mul_v_add_s(
    float * restrict x, 
    float scalar,
    const float * const y, 
    int n
    ) 
{
  for ( int i = 0; i < n; i++ ) { 
    x[i] += scalar * y[i];
  }
}
