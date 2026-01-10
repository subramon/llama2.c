#include "matmul.h"
void 
matmul(
    float * restrict xout, 
    const float * const x, 
    float * const w, 
    int n, 
    int d
    ) 
{
  // W (d,n) @ x (n,) -> xout (d,)
  // by far the most amount of time is spent inside this little function
  int i;
  for (i = 0; i < d; i++) {
    register float val = 0.0f;
    register float *w_i = w + (i*n);
    for (int j = 0; j < n; j++) {
      val += w_i[j] * x[j];
    }
    xout[i] = val;
  }
}

