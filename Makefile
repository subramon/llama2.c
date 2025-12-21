# choose your compiler, e.g. gcc/clang
# example override to clang: make run CC=clang
CC = gcc

# the most basic way of building that is most likely to work on most systems
all : run runq run_ispc cli_split_weights

CFLAGS := -g -O0
# CFLAGS := -O3
CFLAGS += -DDEBUG
CFLAGS += -flto # for Link Time Optimization 
# CFLAGS += -msse4.1 # TODO 

INCS := -I./inc/
INCS += -I${RSUTILS_SRC_ROOT}/inc/

.c.o :
	$(CC) -c -o $@ $< $(CFLAGS)  $(INCS)

ISPC_SRCS += rmsnorm_ispc.ispc 
ISPC_SRCS += softmax_ispc.ispc 
ISPC_SRCS += dot_prod_ispc.ispc 

SRCS += rmsnorm.c 
SRCS += matmul.c 
SRCS += softmax.c 

OBJS  = $(SRCS:.c=.o)

ISPC_OBJS  = $(ISPC_SRCS:.ispc=.o)

rmsnorm_ispc.o : 
	  ispc rmsnorm_ispc.ispc -o rmsnorm_ispc.o 

softmax_ispc.o : 
	ispc softmax_ispc.ispc -o softmax_ispc.o 

dot_prod_ispc.o : 
	ispc dot_prod_ispc.ispc -o dot_prod_ispc.o 

run: run.o  ${OBJS}
	$(CC) -o run run.o ${OBJS}  -lm -lgomp

run_ispc: run.o  ${ISPC_OBJS} matmul_ispc_wrap.o 
	$(CC) -o run_ispc run.o ${ISPC_OBJS} matmul_ispc_wrap.o -lm -lgomp

runq: runq.o  ${OBJS}
	$(CC) -o runq runq.o ${OBJS}  -lm -lgomp

cli_split_weights : cli_split_weights.o 
	$(CC) -o cli_split_weights cli_split_weights.o \
		${RSUTILS_SRC_ROOT}/src/librsutils.so

  # https://gcc.gnu.org/onlinedocs/gcc/Optimize-Options.html
  # https://simonbyrne.github.io/notes/fastmath/
  # -Ofast enables all -O3 optimizations.
  # Disregards strict standards compliance.
  # It also enables optimizations that are not valid for all standard-compliant programs.
  # It turns on -ffast-math, -fallow-store-data-races and the Fortran-specific
  # -fstack-arrays, unless -fmax-stack-var-size is specified, and -fno-protect-parens.
  # It turns off -fsemantic-interposition.
  # In our specific application this is *probably* okay to use

  # run all tests
  .PHONY: test
  test:
	  pytest

  # run only tests for run.c C implementation (is a bit faster if only C code changed)
  .PHONY: testc
  testc:
	  pytest -k runc

  # run the C tests, without touching pytest / python
  # to increase verbosity level run e.g. as `make testcc VERBOSITY=1`
  VERBOSITY ?= 0
  .PHONY: testcc
testcc:
	$(CC) -DVERBOSITY=$(VERBOSITY) -O3 -o testc test.c -lm
	./testc

.PHONY: clean
clean:
	rm -f run
	rm -f runq
	rm -f *.o
