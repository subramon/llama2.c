#include <stdio.h>
#include "rs_mmap.h"
#include "consts.h"
#include "macros.h"
#include "weights_file_layout.h"
#include "set_split_sizes.h"
#include "read_config.h"
#include "mat_1d_to_bin_file.h"
#include "mat_2d_to_bin_file.h"
#include "mat_3d_to_bin_file.h"
#include "mmap_weights.h"

int
main(
    int argc,
    char **argv
    )
{
  int status = 0;
  FILE *fp = NULL;
  size_t split_sizes[sp_num]; 
  char *Z = NULL; size_t nZ = 0; 
  char *X = NULL; size_t nX = 0; char *bak_X = NULL; size_t bak_nX = 0;
  Config C; memset(&C, 0, sizeof(Config));
  TransformerWeights W; memset(&W, 0, sizeof(TransformerWeights));
  const char *wfile; // input weights fle 
  if ( argc != 2 ) { go_BYE(-1); }
  wfile = argv[1];
  //---------------------------------------------------
  status = read_config(wfile, &C); cBYE(status); 
  status = set_split_sizes(&C, split_sizes); cBYE(status);
  status = rs_mmap(wfile, &X, &nX, 0); cBYE(status);
  bak_X = X; bak_nX = nX; 
  status = chk_split_sizes(nX, split_sizes); cBYE(status);
  int head_size = C.dim / C.n_heads;
  //---------------------------------------------------
  int padding = 0; float fzero = 0;
  // Create individual file for each "split"
  X += sizeof(Config); nX -= sizeof(Config);
  // create token_embedding_table
  status = mat_2d_to_bin_file(&X, &nX, "_token_embedding_table.bin", 
      sizeof(float), C.vocab_size, C.dim);
  cBYE(status);
  //-------------------------------------------------------
  // create rms_att_weight
  status = mat_2d_to_bin_file(&X, &nX, "_rms_att_weight.bin", 
      sizeof(float), C.n_layers, C.dim);
  cBYE(status);
  //-------------------------------------------------------
  // create wq
  status = mat_3d_to_bin_file(&X, &nX, "_wq.bin", 
      sizeof(float), C.n_layers, C.dim, C.n_heads * head_size);
  cBYE(status);
  //-------------------------------------------------------
  // create wk
  status = mat_3d_to_bin_file(&X, &nX, "_wk.bin", 
      sizeof(float), C.n_layers, C.dim, C.n_kv_heads * head_size);
  cBYE(status);
  //-------------------------------------------------------
  // create wv
  status = mat_3d_to_bin_file(&X, &nX, "_wv.bin", 
      sizeof(float), C.n_layers, C.dim, C.n_kv_heads * head_size);
  cBYE(status);
  //-------------------------------------------------------
  // create wo
  status = mat_3d_to_bin_file(&X, &nX, "_wo.bin", 
      sizeof(float), C.n_layers, C.n_heads * head_size, C.dim);
  cBYE(status);
  //-------------------------------------------------------
  // create rms_ffn_weight
  status = mat_2d_to_bin_file(&X, &nX, "_rms_ffn_weight.bin", 
      sizeof(float), C.n_layers, C.dim);
  cBYE(status);
  //-------------------------------------------------------
  // create w1
  status = mat_3d_to_bin_file(&X, &nX, "_w1.bin", 
      sizeof(float), C.n_layers, C.dim, C.hidden_dim);
  cBYE(status);
  //-------------------------------------------------------
  // create w2
  status = mat_3d_to_bin_file(&X, &nX, "_w2.bin", 
      sizeof(float), C.n_layers, C.hidden_dim, C.dim);
  cBYE(status);
  //-------------------------------------------------------
  // create w3
  status = mat_3d_to_bin_file(&X, &nX, "_w3.bin", 
      sizeof(float), C.n_layers, C.dim, C.hidden_dim);
  cBYE(status);
  //-------------------------------------------------------
  // create rms_final_weight
  status = mat_1d_to_bin_file(&X, &nX, "_rms_final_weight.bin", 
      sizeof(float), C.dim);
  cBYE(status);
  // create wcls
  size_t off = C.seq_len * head_size / 2; // skip what used to be freq_cis_real (for RoPE)
  X += off; nX -= off;
  off =  C.seq_len * head_size / 2; // skip what used to be freq_cis_imag (for RoPE)
  X += off; nX -= off;
  // TODO P1 set shared_weights to true 
  status = mat_1d_to_bin_file(&X, &nX, "_wcls.bin", 
      sizeof(float), nX / sizeof(float)); // TODO P1 NOT SURE ABOUT THIS 
  cBYE(status);
  

  //-------------------------------------------------------
  // Now read them in (mmap)
  status = mmap_weights(&C, &W); cBYE(status);
  // some testing
  float *fptr = mcr_3d_to_1d(W.wo, 0, 0, C.n_layers, C.n_heads * head_size);
  printf("%f \n", *fptr);
  munmap_weights(&W);
BYE:
  fclose_if_non_null(fp); 
  mcr_rs_munmap(bak_X, bak_nX);
  mcr_rs_munmap(Z, nZ);
  return status;
}
