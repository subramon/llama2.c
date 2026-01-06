#include "div_s.h"
// vec /= scalar
void 
div_s(
    float * restrict x,  // vector 
    float scalar, // scalar
    int n
    ) 
{
  for ( int i = 0; i < n; i++ ) { 
    x[i] /= scalar; 
  }
}
