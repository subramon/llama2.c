#!/bin/bashj
set -e 
gcc -c -fPIC test_argmax.c -I../inc/ -I/home/subramon/RSUTILS/inc/
ispc --pic ../ispc/argmax.ispc -o argmax.o 
gcc test_argmax.o ../ispc/argmax.o  -o test_argmax
./test_argmax
