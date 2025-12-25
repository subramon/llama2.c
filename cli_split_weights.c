#include <stdio.h>
#include "q_macros.h"
#include "rs_mmap.h"
#include "consts.h"
#include "weights_file_layout.h"
#include "set_split_sizes.h"
#include "read_config.h"

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
  Config C; 
  const char *wfile; // input weights fle 
  if ( argc != 2 ) { go_BYE(-1); }
  wfile = argv[1];
  //---------------------------------------------------
  status = read_config(wfile, &C); cBYE(status); 
  status = set_split_sizes(&C, split_sizes); cBYE(status);
  status = rs_mmap(wfile, &X, &nX, 0); cBYE(status);
  bak_X = X; bak_nX = nX; 
  status = chk_split_sizes(nX, split_sizes); cBYE(status);
  //---------------------------------------------------
  int padding = 0; float fzero = 0;
  // Create individual file for each "split"
  X += sizeof(Config); nX -= sizeof(Config);
  // create token_embedding_table
  char *Y = X; 
  fp = fopen("_token_embedding_table.bin", "wb");
  return_if_fopen_failed(fp, "_token_embedding_table.bin", "wb");

  padding = C.dim % FLOATS_IN_REG;
  if ( padding != 0 ) { padding = FLOATS_IN_REG - padding; }
  for ( int i = 0; i < C.vocab_size; i++ ) { 
    fwrite(X, sizeof(float), C.dim, fp); 
    for ( int p = 0; p < padding; p++ ) { 
      fwrite(&fzero, sizeof(float), 1, fp); 
    }
    X += sizeof(float)*C.dim; nX -= sizeof(float)*C.dim;
  }
  if ( ( X - Y ) != sizeof(float)*split_sizes[sp_token_embedding_table] ){
    go_BYE(-1); 
  }
  fclose_if_non_null(fp); 
#ifdef DEBUG
  status = rs_mmap("_token_embedding_table.bin", &Z, &nZ, 0); cBYE(status);
  float *token_embedding_table = (float *)Z;
  mcr_rs_munmap(Z, nZ);
#endif

  //-------------------------------------------------------
  // create rms_att_weight
  Y = X; 
  fp = fopen("_rms_att_weight.bin", "wb");
  return_if_fopen_failed(fp, "_rms_att_weight.bin", "wb");

  padding = C.dim % FLOATS_IN_REG;
  if ( padding != 0 ) { padding = FLOATS_IN_REG - padding; }
  for ( int i = 0; i < C.n_layers; i++ ) { 
    fwrite(X, sizeof(float), C.dim, fp); 
    for ( int p = 0; p < padding; p++ ) { 
      fwrite(&fzero, sizeof(float), 1, fp); 
    }
    X += sizeof(float)*C.dim; nX -= sizeof(float)*C.dim;
  }
  if ( ( X - Y ) != sizeof(float)*split_sizes[sp_rms_att_weight] ) { 
    go_BYE(-1); 
  }
  fclose_if_non_null(fp); 
#ifdef DEBUG
  status = rs_mmap("_rms_att_weight.bin", &Z, &nZ, 0); cBYE(status);
  float *rms_att_weight = (float *)Z;
  mcr_rs_munmap(Z, nZ);
#endif
  //-------------------------------------------------------


  



  printf("Split file %s \n", wfile);
BYE:
  fclose_if_non_null(fp); 
  mcr_rs_munmap(bak_X, bak_nX);
  mcr_rs_munmap(Z, nZ);
  return status;
}
