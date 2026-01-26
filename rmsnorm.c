#include <math.h>
#include <assert.h>
#include "rmsnorm.h"
void 
rmsnorm(
    float*  restrict o,  // OUTPUT 
    const float* const x,  // INPUT 
    const float* const weight,  // INPUT 
    int n
    ) 
{
  // calculate sum of squares
  register float ss = 0.0f;
  for (int j = 0; j < n; j++) {
    ss += x[j] * x[j];
  }
  ss /= (float)n;
  ss += 1e-5f;
  ss = 1.0f / sqrtf(ss);
  // normalize and scale
  for ( register int j = 0; j < n; j++) {
    o[j] = weight[j] * (ss * x[j]);
  }
}
