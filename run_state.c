#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include "macros.h"
#include "weights_file_layout.h"
#include "consts.h"
#include "macros.h"
#include "run_state.h"

int
malloc_run_state(
    RunState* s, 
    Config* p
    ) 
{
  int status = 0;
  // we calloc instead of malloc to keep valgrind happy
  int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
  int head_size = p->dim / p->n_heads;
  // STOP : Get dimensions of various vectors we will allocate now 
  // START: over-allocate to get stuff to align for ISPC
  size_t ispc_dim = mcr_round_up(p->dim);
  size_t ispc_hidden_dim = mcr_round_up(p->hidden_dim);
  size_t ispc_vocab_size = mcr_round_up(p->vocab_size);
  size_t ispc_seq_len = mcr_round_up(p->seq_len);
  size_t ispc_kv_dim = mcr_round_up(kv_dim);
  // STOP: over-allocate to get stuff to align for ISPC

  s->x   = calloc(ispc_dim, sizeof(float));
  s->xb  = calloc(ispc_dim, sizeof(float));
  s->xb2 = calloc(ispc_dim, sizeof(float));
  s->hb  = calloc(ispc_hidden_dim, sizeof(float));
  s->hb2 = calloc(ispc_hidden_dim, sizeof(float));
  s->q   = calloc((p->n_heads * head_size), sizeof(float));
  s->kc = calloc((p->n_layers * p->seq_len * ispc_kv_dim), sizeof(float));
  s->vc = calloc((p->n_layers * p->seq_len * ispc_kv_dim), sizeof(float));
  s->att    = calloc((p->n_heads * ispc_seq_len), sizeof(float));
  s->logits = calloc(ispc_vocab_size, sizeof(float));

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
