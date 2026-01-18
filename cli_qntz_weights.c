#include <stdio.h>
#include "rs_mmap.h"
#include "consts.h"
#include "macros.h"
#include "weights_file_layout.h"
#include "read_config.h"
#include "qntz_2d.h"
#include "qntz_3d.h"
#include "mmap_weights.h"

int
main(
    int argc,
    char **argv
    )
{
  int status = 0;
  Config C; memset(&C, 0, sizeof(Config));
  const char *wfile; // input weights fle 
  if ( argc != 2 ) { go_BYE(-1); }
  wfile = argv[1];
  //---------------------------------------------------
  status = read_config(wfile, &C); cBYE(status); 
  int head_size = C.dim / C.n_heads;
  //---------------------------------------------------
  int padding = 0; float fzero = 0;
  // Create individual file for each "split"
  X += sizeof(Config); nX -= sizeof(Config);
  // create token_embedding_table
  status = qntz_2d(
      "_token_embedding_table.bin", "_token_embedding_table.ui8",
      "_token_embedding_table.offset", "_token_embedding_table.delta", 
      C.vocab_size, C.dim);
  cBYE(status);
  //-------------------------------------------------------
  // create rms_att_weight
  status = qntz_2d(
      "_rms_att_weight.bin", "_rms_att_weight.ui8", 
      "_rms_att_weight.offset", "_rms_att_weight.delta", 
      C.n_layers, C.dim);
  cBYE(status);
  //-------------------------------------------------------
  // create wq
  status = qntz_3d(
      "_wq.bin", "_wq.ui8", 
      "_wq.offset", "_wq.delta", 
      C.n_layers, C.dim, C.n_heads * head_size);
  cBYE(status);
  //-------------------------------------------------------
#ifdef XXX
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
  // TODO ??? "_rms_final_weight.bin", 
  // TODO ??? "_wcls.bin", 
#endif
  //-------------------------------------------------------
BYE:
  return status;
}
