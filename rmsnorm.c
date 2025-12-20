#include <math.h>
#include <assert.h>
#include "rmsnorm.h"
void 
rmsnorm(
    float*  restrict o, 
    const float* const x, 
    const float* const weight, 
    int n,
    int dummy // for parity with ISPC version
    ) 
{
  assert(dummy == 0);
  // calculate sum of squares
  float ss = 0.0f;
  for (int j = 0; j < n; j++) {
    ss += x[j] * x[j];
  }
  ss /= n;
  ss += 1e-5f;
  ss = 1.0f / sqrtf(ss);
  // normalize and scale
  for (int j = 0; j < n; j++) {
    o[j] = weight[j] * (ss * x[j]);
  }
}
