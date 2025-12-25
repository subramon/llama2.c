#include <stdio.h>
#include "q_macros.h"
#include "rs_mmap.h"
#include "consts.h"
#include "rs_mmap.h"
#include "weights_file_layout.h"
#include "read_config.h"

int
read_config(
    const char * const infile,
    Config *ptr_C
    )
{
  int status = 0;
  char *X = NULL; size_t nX = 0; 
  status = rs_mmap(infile, &X, &nX, 0); cBYE(status);
  memcpy(ptr_C, X, sizeof(Config));
  //-------------------------------
  printf("dim        = %d \n", ptr_C->dim );
  printf("hidden_dim = %d \n", ptr_C->hidden_dim );
  printf("n_layers   = %d \n", ptr_C->n_layers );
  printf("n_heads    = %d \n", ptr_C->n_heads );
  printf("n_kv_heads = %d \n", ptr_C->n_kv_heads);
  printf("vocab_size = %d \n", ptr_C->vocab_size );
  printf("seq_len    = %d \n", ptr_C->seq_len );
  //-------------------------------
BYE:
  mcr_rs_munmap(X, nX);
  return status;
}
