#include <stdio.h>
#include "q_macros.h"
#include "rs_mmap.h"
#include "consts.h"
#include "weights_file_layout.h"
#include "mat_2d_to_bin_file.h"
int
mat_2d_to_bin_file(
    char **ptr_X,
    size_t *ptr_nX,
    const char * file_name,
    int sz,
    int nR,
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
  for ( int i = 0; i < nR; i++ ) { 
    fwrite(X, sz, nC, fp); 
    for ( int p = 0; p < padding; p++ ) { 
      fwrite(&dzero, sz, 1, fp); 
    }
    X += sz * nC; nX -= sz * nC; 
  }
  fclose_if_non_null(fp); 
#ifdef DEBUG
  char *Z = NULL; size_t nZ = 0;
  status = rs_mmap(file_name, &Z, &nZ, 0); cBYE(status);
  mcr_rs_munmap(Z, nZ);
#endif
  *ptr_X = X;
  *ptr_nX = nX;
BYE:
  fclose_if_non_null(fp); 
  return status;
}
