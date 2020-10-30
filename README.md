# Local-Volatility Calibration
White paper for the final exam of the "Programming Massively Parallel Hardware" course at the University of Copenhagen. This paper has been presented in October 2020.

## Abstract
The aim of this project is to explain how it was possible to parallelize the code that implements a simplified version of volatility calibration, which uses Crank-Nicolson finite difference method. In particular, there are two sections, in which are defined both parallelizations: CPU via OpenMP and GPU via CUDA. 

## Results
Execution time (in µS) of different datasets with parallel GPU version, CPU with optimizations (W/) and CPU without optimizations (W/O).
 Dataset| GPU | CPU W/ | CPU W/O
 --- | --- | --- | --- |
Small | 313337 | 234362 | 2429592
Medium | 411741 | 513036 | 5240587
Large | 7211866 | 16026744 | 232179276 

## Report
[Report.pdf](https://github.com/francescodone/Local-Volatility-Calibration/tree/master/Report/PDF/Report.pdf)

## Usage
```console
make clean
make
make run_small
make run_medium
make run_large
```

## Structure
```console
Code
 ├─OrigImpl (contains the original implementation)
 │  ├─ProjectMain.cpp (contains the main function)
 │  ├─ProjCoreOrig.cpp (contains the core functions)
 │  └─ProjHelperFun.cpp (contains the functions that compute the input parameters)
 ├─include
 │  ├─ParserC.h (implements a simple parser)
 │  ├─ParseInput.h (reads the input/output data and provides validation)
 │  ├─OpenmpUtil.h (some OpenMP-related helpers)
 │  ├─Constants.h (currently only sets up REAL to either double or float based on the compile-time  parameter WITH_FLOATS)
 │  ├─CudaUtilProj.cu.h (provides stubs for calling transposition and inclusive (segmented) scan)
 │  └─TestCudaUtil.cu (a simple tester for  and scan)
 └─ParTridagCuda (contains code that demonstrates how TRIDAG can be parallelized by intra-block scans, i.e., it assumes that NUM_X, NUM_Y are a power of 2 less than or equal to 1024)

Report
 ├─LaTex
 └─PDF
    └─Report.pdf
```

## Authors
[Justinas Antanavičius](https://github.com/Justinas256)\
[Francesco Done'](https://github.com/francescodone)\
[Valdemar Erk](https://github.com/Erk-)