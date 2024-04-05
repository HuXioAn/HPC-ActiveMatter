# Active Matter - CPP

> Cpp version, single threaded

## Usage

``` shell
g++ -std=c++11 activeMatter.cpp -o a.out
./a.out # default 500 birds
./a.out 600 # 600 birds
```

## Display

Use: 
``` c++
constexpr bool OUTPUT_TO_FILE = true;
```
to enable or disable data output. Default output file: `output.plot`.

Then 

``` shell
python3 plot.py 
```
to display the simulation resualt.