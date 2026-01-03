#include "add_to.h"
// x <= x + y
void 
mul_add_to(
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
