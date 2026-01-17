#include "consts.h"
__attribute__((external_only))
export uniform int 
argmax(
    __attribute__((aligned(BYTES_IN_REG))) uniform float *uniform X, 
    uniform int count
    ) 
{
  assume((count % programCount) == 0);
  assume(((uniform uint64)((void*)X) & (32 * TARGET_WIDTH)-1) == 0);
  // 1. Find the maximum value and its local index within each active program instance (lane)
  // Initialize with the first element for simplicity
  varying float value_local = X[programIndex];
  varying int index_local = programIndex;

  // Iterate over the uniform data range, processing 'programCount' elements at a time
  for (uniform int i = 0; i < count; i += programCount) {
    // Check if the current lane is active and within bounds
    if (i + programIndex < count) {
      varying float current_value = X[i + programIndex];
      if (current_value > value_local) {
        value_local = current_value;
        index_local = i + programIndex;
      }
    }
  }

  // 2. Reduce across all program instances (lanes) to find the global maximum index
  // The reduce_max() function finds the max value across all active lanes
  uniform float max_value = reduce_max(value_local);

  // Determine which lanes have the global maximum value
  // This creates a mask
  if (value_local == max_value) {
    // Use lanemask() to capture the mask of the currently active lanes
    uniform int mask = lanemask();
    // count_trailing_zeros() can be used to efficiently find the index of the first active lane in the mask,
    // which corresponds to the first occurrence of the maximum value
    return count_trailing_zeros(mask);
  }
  print("XXXXX\n");

  // This part is generally unreachable if 'count' > 0, but required for a return value
  return -1; 
}

