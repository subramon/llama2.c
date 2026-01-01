#include "swiglu.h"
void
swiglu(
    float * restrict x, // [n]
    const float * const y, // [n]
    int n
    )
{
  // SwiGLU non-linearity
  for (int i = 0; i < n; i++) {
    float val = x[i];
    // silu(x)=x*σ(x), where σ(x) is the logistic sigmoid
    val *= (1.0f / (1.0f + expf(-val)));
    // elementwise multiply with w3(x)
    val *= y[i];
    x[i] = val;
  }
}
