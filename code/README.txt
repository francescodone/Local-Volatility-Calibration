To compile and run:
    $ make clean; make; make run_small
                                _medium
                                _large


Folder `CudaImpl' contains the CUDA implementation:
    -- `ProjectMain.cu'    contains the main function
    -- `ProjCoreOrig.cu'   contains the run_OrigCPU function which executes CUDA kernels
    -- `CudaKernels.cu.h'  contains CUDA kernels

Folder `CpuTransformedImpl' contains the transformed CPU program which resembles the CUDA program
    -- `ProjectMain.cpp'   contains the main function
    -- `ProjCoreOrig.cpp'  contains the transformed core functions 
    -- `ProjHelperFun.cpp' contains the functions that compute the input parameters

Folder `CpuParImpl' contains the parallelized CPU program
    -- `ProjectMain.cpp'   contains the main function
    -- `ProjCoreOrig.cpp'  contains the core functions with parallelized run_OrigCPU function
    -- `ProjHelperFun.cpp' contains the functions that compute the input parameters