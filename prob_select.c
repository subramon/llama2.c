#include "prob_select.h"
int 
prob_select(
    const float* const probabilities,  // [n] 
    int n, 
    float coin
    ) 
{
  // sample index from probabilities (they must sum to 1!)
  // coin is a random number in [0, 1), usually from random_f32()
  register float cdf = 0.0f;
  for (int i = 0; i < n; i++) {
    cdf += probabilities[i];
    if (coin < cdf) {
      return i;
    }
  }
  return n - 1; // in case of rounding errors
}
