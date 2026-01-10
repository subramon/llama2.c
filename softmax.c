#include <math.h>
#include "softmax.h"
void 
softmax(
    float* restrict x, 
    int size
    ) 
{
  // find max value (for numerical stability)
  register float max_val = x[0];
  for (int i = 1; i < size; i++) {
    if (x[i] > max_val) {
      max_val = x[i];
    }
  }
  // exp and sum
  for (int i = 0; i < size; i++) {
    x[i] = expf(x[i] - max_val);
  }
  register float sum = 0.0f;
  for (int i = 0; i < size; i++) {
    sum += x[i];
  }
  // normalize
  for (int i = 0; i < size; i++) {
    x[i] /= sum;
  }
}
