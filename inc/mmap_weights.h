#include "weights_file_layout.h"
extern int
mmap_weights(
    TransformerWeights * restrict ptr_w
    );
extern int
munmap_weights(
    TransformerWeights * restrict ptr_w
    );
