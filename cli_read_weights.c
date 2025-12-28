#include <stdio.h>
#include "macros.h"
#include "rs_mmap.h"
#include "consts.h"
#include "weights_file_layout.h"
#include "read_weights.h"

int
main(
    int argc,
    char **argv
    )
{
  int status = 0;
  FILE *fp = NULL;
  size_t split_sizes[sp_num]; 
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
  char *X = NULL; size_t nX = 0; char *bak_X = NULL; size_t bak_nX = 0;
  Config *C = NULL;
  const char *wfile; // input weights fle 
  if ( argc != 2 ) { go_BYE(-1); }
  wfile = argv[1];
  status = rs_mmap(wfile, &X, &nX, 0); cBYE(status);
  C = (Config *)X; 
  printf("dim        = %d \n", C->dim );
  printf("hidden_dim = %d \n", C->hidden_dim );
  printf("n_layers   = %d \n", C->n_layers );
  printf("n_heads    = %d \n", C->n_heads );
  printf("n_kv_heads = %d \n", C->n_kv_heads);
  printf("vocab_size = %d \n", C->vocab_size );
  //-------------------------------
  status = set_split_sizes(C, split_sizes); cBYE(status);
  //---------------------------------------------------
  size_t sum_split_sizes = 0;
  for ( splits i = 0; i < sp_num; i++ ) { 
    sum_split_sizes += split_sizes[i];
    printf("S[%s] = %lu \n", split_names[i], split_sizes[i]);
  }
  printf("sum_split_sizes = %lu \n", sum_split_sizes);
  size_t exp_split_sizes = (nX - sizeof(Config)) / sizeof(float);
  printf("exp_split_sizes = %lu \n", exp_split_sizes);
  //---------------------------------------------------
  int padding = 0; float fzero = 0;
  // Create individual file for each "split"
  X += sizeof(Config); nX -= sizeof(Config);
  // create token_embedding_table
  char *Y = X; 
  fp = fopen("_token_embedding_table.bin", "wb");
  return_if_fopen_failed(fp, "_token_embedding_table.bin", "wb");

  padding = C->dim % FLOATS_IN_REG;
  if ( padding != 0 ) { padding = FLOATS_IN_REG - padding; }
  for ( int i = 0; i < C->vocab_size; i++ ) { 
    fwrite(X, sizeof(float), C->dim, fp); 
    for ( int p = 0; p < padding; p++ ) { 
      fwrite(&fzero, sizeof(float), 1, fp); 
    }
    X += C->dim; nX -= C->dim;
  }
  if ( ( X - Y ) != split_sizes[sp_token_embedding_table] ) { go_BYE(-1); }
  fclose_if_non_null(fp); 
  //-------------------------------------------------------
  // create rms_att_weight
  Y = X; 
  fp = fopen("_rms_att_weight.bin", "wb");
  return_if_fopen_failed(fp, "_rms_att_weight.bin", "wb");

  padding = C->dim % FLOATS_IN_REG;
  if ( padding != 0 ) { padding = FLOATS_IN_REG - padding; }
  for ( int i = 0; i < C->n_layers; i++ ) { 
    fwrite(X, sizeof(float), C->dim, fp); 
    for ( int p = 0; p < padding; p++ ) { 
      fwrite(&fzero, sizeof(float), 1, fp); 
    }
    X += C->dim; nX -= C->dim;
  }
  if ( ( X - Y ) != split_sizes[sp_rms_att_weight] ) { go_BYE(-1); }
  fclose_if_non_null(fp); 
  //-------------------------------------------------------


  



  printf("Split file %s \n", wfile);
BYE:
  fclose_if_non_null(fp); 
  mcr_rs_munmap(bak_X, bak_nX);
  return status;
}
