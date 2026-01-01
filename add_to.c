#include "add_to.h"
// x <= x + y
void 
add_to(
    float * restrict x, 
    const float * const y, 
    int n
    ) 
{
  for ( int i = 0; i < n; i++ ) { 
    x[i] += y[i];
  }
}
