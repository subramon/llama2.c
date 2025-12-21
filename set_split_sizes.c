#include <stdio.h>
#include "q_macros.h"
#include "weights_file_layout.h"
#include "set_split_sizes.h"
int
set_split_sizes(
      Config *C,
      size_t * split_sizes
      )
{
  int status = 0;
  if ( C == NULL ) { go_BYE(-1); }
  for ( int i = 0; i <= sp_num; i++ ) { split_sizes[i] = 0; }
  split_sizes[sp_token_embedding_table] = C->vocab_size * C->dim;
  split_sizes[sp_rms_att_weight] = C->n_layers * C->dim;
  split_sizes[sp_wq] = C->n_layers * C->dim * C->dim;
  split_sizes[sp_wk] = C->n_layers * C->dim * C->dim;
  split_sizes[sp_wv] = C->n_layers * C->dim * C->dim;
  split_sizes[sp_wo] = C->n_layers * C->dim * C->dim;
  split_sizes[sp_rms_ffn_weight] = C->n_layers * C->dim;
  split_sizes[sp_w1] = C->n_layers * C->dim * C->hidden_dim;
  split_sizes[sp_w2] = C->n_layers * C->hidden_dim * C->dim;
  split_sizes[sp_w3] = C->n_layers * C->dim * C->hidden_dim;
  split_sizes[sp_rms_final_weight] = C->dim;
  split_sizes[sp_wcls] = 0; // TODO P1 
BYE:
  return status;
}
int
chk_split_sizes(
    size_t file_size,
    size_t * split_sizes
    )
{
  int status = 0;
  char *split_names[sp_num] = { 
  "sp_token_embedding_table",
  "sp_rms_att_weight",
  "sp_wq",
  "sp_wk",
  "sp_wv",
  "sp_wo",
  "sp_rms_ffn_weight",
  "sp_w1",
  "sp_w2",
  "sp_w3",
  "sp_rms_final_weight",
  "sp_wcls",
  };
  size_t sum_split_sizes = 0;
  for ( splits i = 0; i < sp_num; i++ ) { 
    sum_split_sizes += split_sizes[i];
    printf("S[%s] = %lu \n", split_names[i], split_sizes[i]);
  }
  printf("sum_split_sizes = %lu \n", sum_split_sizes);
  size_t exp_split_sizes = (file_size - sizeof(Config)) / sizeof(float);
  printf("exp_split_sizes = %lu \n", exp_split_sizes);
  // TODO Do some checking 
BYE:
  return status;
}
