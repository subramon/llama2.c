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

int
main()
{
  int status = 0;
  int n = 1024; 
  srand48(time(NULL));
  for ( int i = 0; i < 10; i++ ) { 
    uint64_t t0 = 0, t1 = 0, t = 0;
    float *x = NULL, *y = NULL, *z = NULL;

    status = posix_memalign(&x, 32, (n * sizeof(float)));
    status = posix_memalign(&y, 32, (n * sizeof(float)));
    status = posix_memalign(&z, 32, (n * sizeof(float)));
    for ( int j = 0; j < 20; j++ ) {
      // initialize
      for ( int i = 0; i < n; i++ ) { 
        x[i] = drand48();
        y[i] = drand48();
        z[i] = drand48();
      }
      uint64_t t0 = __rdtsc();
      swiglu(x, y, n); 
      uint64_t t1 = __rdtsc();
      t += (t1 - t0);
    }
    printf("%8d %" PRIu64 "\n", n, t);
    free(x);
    free(y);
    free(z);
    n *= 2;
  }
}
