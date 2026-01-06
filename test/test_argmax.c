#include <stdio.h>
#include "q_macros.h"
#include "argmax.h"

#define N 32 
int
main()
{
  int status = 0;
  float X[N];
  for ( int i = 0; i < N; i++ ) { X[i] = i; }
  int pos = argmax(X, N);
  printf("pos = %d \n", pos);
  if ( pos != N-1 ) { go_BYE(-1); }

  for ( int i = 0; i < N; i++ ) { X[i] = N+1-i; }
  pos = argmax(X, N);
  printf("pos = %d \n", pos);
  if ( pos != 0 ) { go_BYE(-1); }
  printf("SUCCESS\n");
BYE:
  if ( status != 0 ) { printf("FAILURE\n"); }
  return status;
}
