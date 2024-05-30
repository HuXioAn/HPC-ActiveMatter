# Active Matter - OpenMP

## Compile on dardel 

```shell
cc -fopenmp activeMatterHPC.cpp -o a.out

srun -n 1 ./a.out [birdNum]
```

Adjust thread number by the following command:

```shell
export OMP_NUM_THREADS=[thredNum]
```

## Display

Use:

```c++
constexpr bool OUTPUT_TO_FILE = true;
```

to enable or disable data output. Default output file: `output.plot`.

Then

```shell
python3 plot.py
```

to display the simulation result.

You will find optimization codes in openmp branch.
