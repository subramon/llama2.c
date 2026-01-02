#include <math.h>
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
    // elementwise multiply with y
    val *= y[i];
    x[i] = val;
  }
}

/* 
 * In LLaMA 2, SwiGLU (Sigmoid-Weighted Linear Unit) replaces
 * traditional activation functions like ReLU in the Feed-Forward
 * Network (FFN) layers, combining the Swish function (smooth,
 * non-monotonic) with the Gated Linear Unit (GLU) structure, creating
 * a "gated" mechanism that dynamically controls information flow,
 * leading to better performance and gradient flow by allowing the
 * network to learn which features to pass through and which to
 * suppress, improving expressiveness and training stability over
 * simpler activations.

How it Works (The Formula)

SwiGLU uses a formula similar to: SwiGLU(x) = (x * sigmoid(W1x + b)) ⊗
(W2x + c)`.

Input Splitting: The input x is passed through two parallel linear
transformations (matrix multiplications).

Gate Path: One path applies the Swish (or SILU, which is x *
sigmoid(x)) activation to its output, creating a "gate".

Value Path: The other path remains a linear transformation (the
"value").

Element-wise Multiplication: The gate's output is multiplied
element-wise with the value's output, deciding how much of the value
to let through.

Down Projection: This result is then projected back down to the
original dimension.

*/
