#include "weights_file_layout.h"
#ifndef __RUN_STATE_H
#define __RUN_STATE_H
typedef struct {
    // current wave of activations
    float *x; // activation at current time stamp (dim,)
    float *xb; // same, but inside a residual branch (dim,)
    float *xb2; // an additional buffer just for convenience (dim,)
    float *hb; // buffer for hidden dimension in the ffn (hidden_dim,)
    float *hb2; // buffer for hidden dimension in the ffn (hidden_dim,)
    float *q; // query (dim,)
// NOT MALLOC'd    float *k; // key (dim,)
// NOT MALLOC'd   float *v; // value (dim,)
    float *att; // buffer for scores/attention values (n_heads, seq_len)
    float *logits; // output logits
    // kv cache
    float* key_cache;   // (layer, seq_len, dim)
    float* val_cache; // (layer, seq_len, dim)
} RunState;
extern int
malloc_run_state(
    RunState* s, 
    Config* p
    );
extern void 
free_run_state(
    RunState* s
    );
#endif // __RUN_STATE_H
