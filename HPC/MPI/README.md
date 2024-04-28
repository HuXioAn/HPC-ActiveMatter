# Active Matter - MPI

## Compile

```shell
g++ -o a.exe activeMatter_MPI.cpp -l msmpi -L "C:\Program Files (x86)\Microsoft SDKs\MPI\Lib\x64" -I "C:\Program Files (x86)\Microsoft SDKs\MPI\Include"

mpiexec -n 8 a.exe
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

to display the simulation resualt.
