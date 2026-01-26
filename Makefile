# choose your compiler, e.g. gcc/clang
# example override to clang: make run CC=clang
CC = gcc

all : run run_ispc \
	cli_split_weights cli_qntz_weights \
	bench_rmsnorm_ispc bench_rmsnorm_gcc \
	bench_swiglu_ispc bench_swiglu_gcc \
	bench_mul_v_add_s_ispc bench_mul_v_add_s_gcc \
	bench_div_s_gcc bench_div_s_ispc \
	bench_add_v_gcc bench_add_v_ispc \
	bench_dot_prod_gcc bench_dot_prod_ispc \
	bench_softmax_ispc bench_softmax_gcc \


#	bench_prob_select_ispc bench_prob_select_gcc \
#	bench_argmax_ispc bench_argmax_gcc

LNK_FLAGS := -flto
# LNK_FLAGS += -pg 

# CFLAGS := -g -O0
CFLAGS := -O3 -Ofast -march=native
# CFLAGS += -pg # for profiling
# CFLAGS += -DDEBUG
CFLAGS += -flto # for Link Time Optimization 
# CFLAGS += -msse4.1 # TODO 
CFLAGS += -fopenmp
CFLAGS += -fstrict-aliasing 
CFLAGS += -std=gnu99 -Wall -fPIC -W -Waggregate-return -Wcast-align 
CFLAGS += -Wstrict-aliasing
CFLAGS += -Wmissing-prototypes -Wnested-externs -Wshadow -Wwrite-strings 
CFLAGS += -Wunused-variable -Wunused-parameter 
CFLAGS += -Wstrict-prototypes -Wmissing-prototypes -Wpointer-arith 
CFLAGS += -Wmissing-declarations -Wredundant-decls -Wnested-externs 
CFLAGS += -Wshadow -Wcast-qual -Wcast-align -Wwrite-strings 
CFLAGS += -Wold-style-definition 
CFLAGS += -Wattributes
CFLAGS += -Wunused-label 
CFLAGS += -Wconversion
CFLAGS += -Wpedantic  # ??? conflicts with stdatomic.h
CFLAGS += -Wuninitialized

INCS := -I./inc/
INCS += -I${RSUTILS_SRC_ROOT}/inc/

.c.o :
	$(CC) -c -o $@ $< $(CFLAGS)  $(INCS)

ISPC_SRCS += ./ispc/rmsnorm.ispc 
ISPC_SRCS += ./ispc/softmax.ispc 
ISPC_SRCS += ./ispc/dot_prod.ispc 
ISPC_SRCS += ./ispc/dot_prod_qnt.ispc 
ISPC_SRCS += ./ispc/add_v.ispc 
ISPC_SRCS += ./ispc/div_s.ispc 
ISPC_SRCS += ./ispc/mul_v_add_s.ispc 
ISPC_SRCS += ./ispc/swiglu.ispc 
ISPC_SRCS += ./ispc/target_width.ispc 
# TODO  ISPC_SRCS += ./ispc/argmax.ispc 
# TODO  ISPC_SRCS += ./ispc/prob_select.ispc 

SRCS += rmsnorm.c 
SRCS += matmul.c 
SRCS += matmul3.c 
SRCS += matmul2.c 
SRCS += matmul_qnt.c 
SRCS += softmax.c 
# TODO SRCS += orig_mmap_weights.c 
SRCS += mmap_weights.c 
SRCS += qnt_mmap_weights.c 
SRCS += run_state.c 
SRCS += rope.c 
SRCS += dot_prod.c 
SRCS += add_v.c 
SRCS += prob_select.c 
SRCS += div_s.c 
SRCS += mul_v_add_s.c 
SRCS += swiglu.c 
SRCS += argmax.c 
SRCS += read_config.c 
SRCS += target_width.c 

OBJS  = $(SRCS:.c=.o)

ISPC_OBJS  = $(ISPC_SRCS:.ispc=.o)

ISPC_FLAGS := --addressing=32
ISPC_FLAGS += --opt=fast-math --math-lib=fast --opt=force-aligned-memory
# ISPC_FLAGS += --vec-report=5
ISPC_FLAGS += -O3

dot_prod_256.o : dot_prod_256.c
	gcc -c ${CFLAGS} ${INCS} dot_prod_256.c -mfma -mavx2 -o dot_prod_256.o

ispc/argmax.o : 
	  ispc ${INCS} ${ISPC_FLAGS} ispc/argmax.ispc -o ispc/argmax.o 

ispc/target_width.o : 
	  ispc ${INCS} ${ISPC_FLAGS} ispc/target_width.ispc -o ispc/target_width.o 

ispc/mul_v_add_s.o : 
	  ispc ${INCS} ${ISPC_FLAGS} ispc/mul_v_add_s.ispc -o ispc/mul_v_add_s.o 

ispc/div_s.o : 
	  ispc ${INCS} ${ISPC_FLAGS} ispc/div_s.ispc -o ispc/div_s.o 

ispc/add_v.o : 
	  ispc ${INCS} ${ISPC_FLAGS} ispc/add_v.ispc -o ispc/add_v.o 

ispc/dot_prod_qnt.o : 
	ispc ${INCS} ${ISPC_FLAGS} ispc/dot_prod_qnt.ispc -o ispc/dot_prod_qnt.o 

ispc/dot_prod.o : 
	ispc ${INCS} ${ISPC_FLAGS} ispc/dot_prod.ispc -o ispc/dot_prod.o 

ispc/rmsnorm.o : 
	  ispc ${INCS} ${ISPC_FLAGS} ispc/rmsnorm.ispc -o ispc/rmsnorm.o 

ispc/softmax.o : 
	ispc ${INCS} ${ISPC_FLAGS} ispc/softmax.ispc -o ispc/softmax.o 

ispc/swiglu.o : 
	  ispc ${INCS} ${ISPC_FLAGS} ispc/swiglu.ispc -o ispc/swiglu.o 

run: run.o  ${OBJS} matmul_prefetch.o 
	$(CC) -o run run.o ${OBJS}   matmul_prefetch.o \
	${RSUTILS_SRC_ROOT}/src/librsutils.so \
	-lm -lgomp

#TODO Delete argmax below 
#TODO Delete prob_select below 
run_ispc: run.o  ${ISPC_OBJS} \
	matmul_ispc_wrap.o \
	matmul_prefetch.o \
	matmul3.o \
	matmul2.o \
	matmul_qnt_ispc_wrap.o \
	dot_prod_256.o \
	mmap_weights.o \
	qnt_mmap_weights.o \
	rope.o \
	read_config.o \
	argmax.o \
	prob_select.o \
	run_state.o 
	$(CC) ${LNK_FLAGS} -o run_ispc run.o ${ISPC_OBJS} \
	matmul_ispc_wrap.o \
	matmul_prefetch.o \
	matmul3.o \
	matmul2.o \
	matmul_qnt_ispc_wrap.o \
	dot_prod_256.o \
	mmap_weights.o \
	qnt_mmap_weights.o \
	rope.o \
	read_config.o \
	argmax.o \
	prob_select.o \
	run_state.o  \
	${RSUTILS_SRC_ROOT}/src/librsutils.so \
	-lm -lgomp

runq: runq.o  ${OBJS}
	$(CC) -o runq runq.o \
	rmsnorm.o \
	softmax.o \
	matmul.o \
	${RSUTILS_SRC_ROOT}/src/librsutils.so \
	-lm -lgomp

cli_qntz_weights : cli_qntz_weights.o \
	read_config.o \
	mmap_weights.o \
	qntz_2d.o \
	qntz_3d.o 
	$(CC) -o cli_qntz_weights cli_qntz_weights.o  \
	read_config.o \
	mmap_weights.o \
	qntz_2d.o \
	qntz_3d.o \
	${RSUTILS_SRC_ROOT}/src/librsutils.so

cli_split_weights : cli_split_weights.o \
	set_split_sizes.o \
	read_config.o \
	mmap_weights.o \
	mat_1d_to_bin_file.o \
	mat_2d_to_bin_file.o \
	mat_3d_to_bin_file.o 
	$(CC) -o cli_split_weights cli_split_weights.o  \
	set_split_sizes.o \
	read_config.o \
	mmap_weights.o \
	mat_1d_to_bin_file.o \
	mat_2d_to_bin_file.o \
	mat_3d_to_bin_file.o \
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

bench_rmsnorm_ispc : ispc/rmsnorm.o bench_rmsnorm.o 
	$(CC) -o bench_rmsnorm_ispc \
	bench_rmsnorm.o \
	ispc/rmsnorm.o \
	${RSUTILS_SRC_ROOT}/src/librsutils.so -lm 

bench_rmsnorm_gcc : rmsnorm.o bench_rmsnorm.o 
	$(CC) -o bench_rmsnorm_gcc \
	bench_rmsnorm.o \
	rmsnorm.o \
	${RSUTILS_SRC_ROOT}/src/librsutils.so -lm 

bench_swiglu_ispc : ispc/swiglu.o bench_swiglu.o 
	$(CC) -o bench_swiglu_ispc \
	bench_swiglu.o \
	ispc/swiglu.o \
	${RSUTILS_SRC_ROOT}/src/librsutils.so -lm 

bench_swiglu_gcc : swiglu.o bench_swiglu.o 
	$(CC) -o bench_swiglu_gcc \
	bench_swiglu.o \
	swiglu.o \
	${RSUTILS_SRC_ROOT}/src/librsutils.so -lm 

bench_argmax_ispc : ispc/argmax.o bench_argmax.o 
	$(CC) -o bench_argmax_ispc \
	bench_argmax.o \
	ispc/argmax.o \
	${RSUTILS_SRC_ROOT}/src/librsutils.so -lm 

bench_argmax_gcc : argmax.o bench_argmax.o 
	$(CC) -o bench_argmax_gcc \
	bench_argmax.o \
	argmax.o \
	${RSUTILS_SRC_ROOT}/src/librsutils.so -lm 

bench_mul_v_add_s_ispc : ispc/mul_v_add_s.o bench_mul_v_add_s.o 
	$(CC) -o bench_mul_v_add_s_ispc \
 	bench_mul_v_add_s.o \
 	ispc/mul_v_add_s.o \
	${RSUTILS_SRC_ROOT}/src/librsutils.so -lm 

bench_mul_v_add_s_gcc : mul_v_add_s.o bench_mul_v_add_s.o 
	$(CC) -o bench_mul_v_add_s_gcc \
	bench_mul_v_add_s.o \
	mul_v_add_s.o \
	${RSUTILS_SRC_ROOT}/src/librsutils.so -lm 

bench_div_s_ispc : ispc/div_s.o bench_div_s.o 
	$(CC) -o bench_div_s_ispc \
 	bench_div_s.o \
 	ispc/div_s.o \
	${RSUTILS_SRC_ROOT}/src/librsutils.so -lm 

bench_div_s_gcc : div_s.o bench_div_s.o 
	$(CC) -o bench_div_s_gcc \
	bench_div_s.o \
	div_s.o \
	${RSUTILS_SRC_ROOT}/src/librsutils.so -lm 

bench_dot_prod_ispc : ispc/dot_prod.o bench_dot_prod.o 
	$(CC) -o bench_dot_prod_ispc \
 	bench_dot_prod.o \
 	ispc/dot_prod.o \
	${RSUTILS_SRC_ROOT}/src/librsutils.so -lm 

bench_dot_prod_gcc : dot_prod.o bench_dot_prod.o 
	$(CC) -o bench_dot_prod_gcc \
	bench_dot_prod.o \
	dot_prod.o \
	${RSUTILS_SRC_ROOT}/src/librsutils.so -lm 

bench_softmax_ispc : ispc/softmax.o bench_softmax.o 
	$(CC) -o bench_softmax_ispc \
 	bench_softmax.o \
 	ispc/softmax.o \
	${RSUTILS_SRC_ROOT}/src/librsutils.so -lm 

bench_softmax_gcc : softmax.o bench_softmax.o 
	$(CC) -o bench_softmax_gcc \
	bench_softmax.o \
	softmax.o \
	${RSUTILS_SRC_ROOT}/src/librsutils.so -lm 

bench_prob_select_ispc : ispc/prob_select.o bench_prob_select.o 
	$(CC) -o bench_prob_select_ispc \
 	bench_prob_select.o \
 	ispc/prob_select.o \
	${RSUTILS_SRC_ROOT}/src/librsutils.so -lm 

bench_prob_select_gcc : prob_select.o bench_prob_select.o 
	$(CC) -o bench_prob_select_gcc \
	bench_prob_select.o \
	prob_select.o \
	${RSUTILS_SRC_ROOT}/src/librsutils.so -lm 

bench_add_v_ispc : ispc/add_v.o bench_add_v.o 
	$(CC) -o bench_add_v_ispc \
 	bench_add_v.o \
 	ispc/add_v.o \
	${RSUTILS_SRC_ROOT}/src/librsutils.so -lm 

bench_add_v_gcc : add_v.o bench_add_v.o 
	$(CC) -o bench_add_v_gcc \
	bench_add_v.o \
	add_v.o \
	${RSUTILS_SRC_ROOT}/src/librsutils.so -lm 

.PHONY: clean
clean:
	rm -f run
	rm -f runq
	rm -f *.o
	rm -f ispc/*.o
