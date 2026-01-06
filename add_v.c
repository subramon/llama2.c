#include "add_v.h"
// vec1 += vec2
void 
add_v(
    float * restrict x,  // vector 
    const float * const y, // scalar
    int n
    ) 
{
  for ( int i = 0; i < n; i++ ) { 
    x[i] += y[i];
  }
}
