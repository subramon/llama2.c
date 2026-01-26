#include <stdio.h>
#include <stdbool.h>
#include <omp.h>
#include <stdlib.h>
#include <stdint.h>
#include <inttypes.h>
#include <ctype.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <x86intrin.h> // for rdtsc
#include "macros.h"
#include "div_s.h"

int
main(
    int argc,
    char **argv
    )
{
  int status = 0;
  int n = 1024; 
  srand48(time(NULL));
  for ( int i = 0; i < 10; i++ ) { 
    float *x = NULL, *y = NULL, *z = NULL;

    status = posix_memalign((void **)&x, 32, (n * sizeof(float)));
    status = posix_memalign((void **)&y, 32, (n * sizeof(float)));
    status = posix_memalign((void **)&z, 32, (n * sizeof(float)));
    int iters = 100;
    uint64_t t = 0;
    for ( int j = 0; j < iters; j++ ) {
      // initialize
      for ( int k = 0; k < n; k++ ) { 
        x[k] = (float)drand48();
        y[k] = (float)drand48();
        z[k] = (float)drand48();
      }
      float s1 = (float)drand48();
      uint64_t t0 = __rdtsc();
      div_s(x, s1, n); 
      uint64_t t1 = __rdtsc();
      t += (t1 - t0);
    }
    printf("%8d %lf \n", n, ((double)t/(double)iters));
    free(x);
    free(y);
    free(z);
    n *= 2;
  }
  return status;
}
