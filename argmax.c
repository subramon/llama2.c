#include "argmax.h"
int
argmax(
    float* X,  // [n]
    int n
    ) 
{
  // return the index, i, that has the highest value X[i]
  int max_i = 0;
  float max_p = X[0];
  for (int i = 1; i < n; i++) {
    if (X[i] > max_p) {
      max_i = i;
      max_p = X[i];
    }
  }
  return max_i;
}
