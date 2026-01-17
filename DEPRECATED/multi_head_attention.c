/*
multi_head_attention(p->n_heads, pos, s->q, s->att, s->kc,
head_size, seq_len, kv_dim, kv_mul,
*/
// multihead attention. iterate over all heads in parallel
// TODO P3: Study taskloop in OpenMP
// TODO P3: Consider Collapse these 2 loops into one
    // CAUTION: Parallelizing this loop slows things down!
    // STRANGE: Slows it down for gcc but not for ISPC. Puzzling...
#include <math.h>
#include <stdlib.h>
#include "macros.h"
#include "dot_prod.h"
#include "multi_head_attention.h"
int
multi_head_attention(
    int nH,
    int pos,
    float *s_q, 
    float *s_att,
    float *s_kc,
    int head_size,
    int seq_len,
    int kv_dim,
    int kv_mul,
    int lidx // which layer
  )
{
  int status = 0; 
  size_t ispc_seq_len = mcr_round_up(seq_len);
  size_t ispc_head_size = mcr_round_up(head_size);
  size_t ispc_kv_dim = mcr_round_up(kv_dim);

  // #pragma omp parallel for 
  for ( int h = 0; h < nH; h++) {
    // get the query vector for this head
    const float* const q_h = mcr_2d_to_1d(s_q, (size_t)h, ispc_head_size); 
    // attention scores for this head
    float* att_h = mcr_2d_to_1d(s_att, (size_t)h, ispc_seq_len);
    // iterate over all timesteps, including the current one
    for (int t = 0; t <= pos; t++) {
      // get the key vector for this head and at this timestep
      float *keyptr = mcr_3d_to_1d(s_kc, (size_t)lidx, t, seq_len, ispc_kv_dim);
      keyptr += (h / kv_mul) * head_size;
#ifdef DEBUG
      int loff = l * seq_len * kv_dim; // kv cache layer offset for convenience
      float *old_k = s->kc + loff + t * kv_dim + (h / kv_mul) * head_size;
      if ( old_k != keyptr ) { status = -1; continue; }
#endif
      // calculate the attention score as the dot product of q and k
      float score;
      dot_prod(q_h, keyptr, head_size, &score); 
      score /= sqrtf((float)head_size);
      // save the score to the attention buffer
      att_h[t] = score;
    }
  }
  return status;
}
