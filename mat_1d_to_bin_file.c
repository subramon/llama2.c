#include <stdio.h>
#include "q_macros.h"
#include "rs_mmap.h"
#include "consts.h"
#include "weights_file_layout.h"
#include "mat_1d_to_bin_file.h"
int
mat_1d_to_bin_file(
    char **ptr_X,
    size_t *ptr_nX,
    const char * file_name,
    int sz,
    int nC
    )
{
  int status = 0;
  double dzero = 0;
  FILE *fp = NULL;
  char *X = *ptr_X; size_t nX = *ptr_nX;
  char *Y = X; 
  fp = fopen(file_name, "wb");
  return_if_fopen_failed(fp, file_name, "wb");

  int padding = nC % FLOATS_IN_REG;
  if ( padding != 0 ) { padding = FLOATS_IN_REG - padding; }
  fwrite(X, sz, nC, fp); 
  X += sz * nC; nX -= sz * nC; 
  for ( int p = 0; p < padding; p++ ) { 
    fwrite(&dzero, sz, 1, fp); 
  }
  fclose_if_non_null(fp); 
  *ptr_X = X;
  *ptr_nX = nX;
BYE:
  fclose_if_non_null(fp); 
  return status;
}
