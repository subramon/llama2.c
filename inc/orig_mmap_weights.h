#include <stddef.h>
#include "weights_file_layout.h"
extern void 
memory_map_weights(
    TransformerWeights *w, 
    Config* p, 
    float* ptr, 
    int shared_weights
    );
