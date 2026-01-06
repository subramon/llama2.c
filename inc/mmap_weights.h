#include "weights_file_layout.h"
extern int
mmap_weights(
    const Config * const ptr_C,
    TransformerWeights * restrict ptr_w
    );
extern int
munmap_weights(
    TransformerWeights * restrict ptr_w
    );
