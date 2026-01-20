#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include "macros.h"
#include "weights_file_layout.h"
#include "consts.h"
#include "macros.h"
#include "run_state.h"

#define ALIGN 512 // TODO P1 Get this value from ISPC 

int
malloc_run_state(
    RunState* s, 
    Config* p
    ) 
{
  int status = 0;
  size_t sz;
  int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
  int head_size = p->dim / p->n_heads;
  // STOP : Get dimensions of various vectors we will allocate now 
  // START: over-allocate to get stuff to align for ISPC
  size_t ispc_dim = mcr_round_up(p->dim);
  size_t ispc_hidden_dim = mcr_round_up(p->hidden_dim);
  size_t ispc_vocab_size = mcr_round_up(p->vocab_size);
  size_t ispc_seq_len = mcr_round_up(p->seq_len);
  size_t ispc_kv_dim = mcr_round_up(kv_dim);
  size_t ispc_head_size = mcr_round_up(head_size);
  // STOP: over-allocate to get stuff to align for ISPC

  status = posix_memalign((void **)&(s->x), ALIGN, 
      (ispc_dim * sizeof(float)));
  cBYE(status);
  memset(s->x, 0, (ispc_dim * sizeof(float)));

  status = posix_memalign((void **)&(s->xb), ALIGN, 
      (ispc_dim * sizeof(float)));
  cBYE(status);
  memset(s->xb, 0, (ispc_dim * sizeof(float)));

  status = posix_memalign((void **)&(s->xb2), ALIGN, 
      (ispc_dim * sizeof(float)));
  cBYE(status);
  memset(s->xb2, 0, (ispc_dim * sizeof(float)));


  sz = (ispc_hidden_dim * sizeof(float));
  status = posix_memalign((void **)&(s->hb), ALIGN, sz);
  cBYE(status);
  memset(s->hb, 0, sz);

  sz = (ispc_hidden_dim * sizeof(float));
  status = posix_memalign((void **)&(s->hb2), ALIGN, sz);
  cBYE(status);
  memset(s->hb2, 0, sz);

  sz = (((size_t)p->n_heads * ispc_head_size) * sizeof(float));
  status = posix_memalign((void **)&(s->q), ALIGN, sz);
  cBYE(status);
  memset(s->q, 0, sz);

  sz = (((size_t)p->n_layers * (size_t)p->seq_len * ispc_kv_dim) * sizeof(float));
  status = posix_memalign((void **)&(s->kc), ALIGN, sz);
  cBYE(status);
  memset(s->kc, 0, sz);

  sz = (((size_t)p->n_layers * (size_t)p->seq_len * ispc_kv_dim) * sizeof(float));
  status = posix_memalign((void **)&(s->vc), ALIGN, sz);
  cBYE(status);
  memset(s->vc, 0, sz);

  sz = (((size_t)p->n_heads * ispc_seq_len) * sizeof(float));
  status = posix_memalign((void **)&(s->att), ALIGN, sz); 
  cBYE(status);
  memset(s->att, 0, sz);

  sz = (ispc_vocab_size * sizeof(float));
  status = posix_memalign((void **)&(s->logits), ALIGN, sz);
  cBYE(status);
  memset(s->logits, 0, sz);

  // ensure all mallocs went fine
  if (!s->x || !s->xb || !s->xb2 || !s->hb || !s->hb2 || !s->q
      || !s->kc || !s->vc || !s->att || !s->logits) {
    fprintf(stderr, "malloc failed!\n");
    go_BYE(-1); 
  }
BYE:
  return status;
}

void 
free_run_state(
    RunState* s
    ) 
{
  free_if_non_null(s->x);
  free_if_non_null(s->xb);
  free_if_non_null(s->xb2);
  free_if_non_null(s->hb);
  free_if_non_null(s->hb2);
  free_if_non_null(s->q);
  free_if_non_null(s->att);
  free_if_non_null(s->logits);
  free_if_non_null(s->kc);
  free_if_non_null(s->vc);
}
