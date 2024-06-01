# Active Matter - CUDA

This implementation takes 256 threads per block. It makes the output IO operation parallel with the device computation.

**Usage:** 
``` shell

nvcc ./activeMatterCUDA.cu -o activeMatterCUDA.out
./activeMatterCUDA.out 3000 # takes one argument as the bird number

```

The output can be enabled and disabled as in the serial code. 



