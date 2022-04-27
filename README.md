# WARPSPEED

## Summary

Supply a description of a GPU kernel like this

``` python
kernel = WarpspeedKernel(
    {"A": ["tidx*3 + 0", "tidx*3 + 1", "tidx*3 + 2"], "B": ["tidx"]},
    {"C": ["tidx*3 + 0", "tidx*3 + 1", "tidx*3 + 2"], "D": ["tidx"]},
    (5000000, 1, 1),
    32,
)
```

and get an estimate of the data volumes throughout the memory hierachy plus a performance estimate:

``` 
TLBpages        :     8        blockL2Load    :  32.0 kB     waveValidCells       :  220736        
L1Cycles        :  16.0        blockL2Store   :    80 kB     waveMemLoadOverlap[0]:     0.0 kB     
blockL1LoadAlloc:  32.0 kB     waveMemLoadNew :  6898 kB     waveMemLoadOverlap[1]:     0.0 kB     
blockL1Load     :    80 kB     waveMemStoreNew:  6898 kB     waveMemOld[1]        :     0.0 kB     

L1Load      :    80 B/Lup  memLoadOverlap[0]:   0.0 B/Lup  basic.waveMemOld[0]:  13.5 MB     L1Cycles :  16.0 cyc    
smL1Alloc   :    64 kB     memLoadOverlap[1]:   0.0 B/Lup  basic.waveMemOld[1]:   0.0 MB     perfMemV3:  20.0 GLups/s
L1LoadEvicts:   0.6 B/Lup  memLoadV1        :  32.0 B/Lup  L2Store            :    80 B/Lup  perfL2V2 :  40.0 GLups/s
L2LoadV1    :  32.0 B/Lup  memLoadV2        :  32.0 B/Lup  memStoreV1         :  32.0 B/Lup  perfL1   :   305 GLups/s
L2LoadV2    :  32.6 B/Lup  memLoadV3        :  35.1 B/Lup  memStoreV2         :  35.1 B/Lup  perfV3   :  20.0 GLups/s
```



## Usage

Dependencies:
 - sympy
 - islpy
 
Import the module `warpspeedkernel` to get access to the classes

- `WarpspeedKernel`, requires on construction:
   - a dictionary of field names to lists of load 1D linear address expression strings
   - a dictionary of field names to lists of store 1D linear address expression strings
   - a domain size tuple
   - the register count of the kernel


- `WarpspeedGridKernel`, requires on construction:
   - a dictionary of field names to lists of loaded 3D address expression tuples
   - a dictionary of field names to lists of stored 3D address expression tuples
   - a domain size tuple
   - the register count of the kernel
   - grid alignment
  

Import the module `predict_metrics` to get access to the three classes 
- `LaunchConfig`: collects data about the launch configuration of the kernel
- `BasicMetrics`: computes basic volumes and metrics from the kernel and launch configuration
- `DerivedMetrics`: computes derived metrics from the basic values 

## Samples
### 1D sample on a A100: 

This example (see sample1D.py) contains the memory transfers for a kernel that loads a 3D vector from `A`, normalizes and scales it by `B`, and saves the normalized vector and its previous lengths to `C` and `D`. 

``` 
TLBpages        :     8        blockL2Load    :  32.0 kB     waveValidCells       :  220736        
L1Cycles        :  16.0        blockL2Store   :    80 kB     waveMemLoadOverlap[0]:     0.0 kB     
blockL1LoadAlloc:  32.0 kB     waveMemLoadNew :  6898 kB     waveMemLoadOverlap[1]:     0.0 kB     
blockL1Load     :    80 kB     waveMemStoreNew:  6898 kB     waveMemOld[1]        :     0.0 kB     

L1Load      :    80 B/Lup  memLoadOverlap[0]:   0.0 B/Lup  basic.waveMemOld[0]:  13.5 MB     L1Cycles :  16.0 cyc    
smL1Alloc   :    64 kB     memLoadOverlap[1]:   0.0 B/Lup  basic.waveMemOld[1]:   0.0 MB     perfMemV3:  20.0 GLups/s
L1LoadEvicts:   0.6 B/Lup  memLoadV1        :  32.0 B/Lup  L2Store            :    80 B/Lup  perfL2V2 :  40.0 GLups/s
L2LoadV1    :  32.0 B/Lup  memLoadV2        :  32.0 B/Lup  memStoreV1         :  32.0 B/Lup  perfL1   :   305 GLups/s
L2LoadV2    :  32.6 B/Lup  memLoadV3        :  35.1 B/Lup  memStoreV2         :  35.1 B/Lup  perfV3   :  20.0 GLups/s
```

Analysis: the predicted performance of 20 GLups/s is memory bandwidth bound.

### 2D 5-point stencil sample on a A100: 

Output of sample2D5pt.py, which analyzes a range 1, 2D 5-point stencil using a (256,2,1) thread block size for a (15000, 15000) domain:

```
TLBpages        :     3        blockL2Load    :   6.1 kB     waveValidCells       :  240000        
L1Cycles        :  10.0        blockL2Store   :   4.5 kB     waveMemLoadOverlap[0]:     234 kB     
blockL1LoadAlloc:   6.4 kB     waveMemLoadNew :  2110 kB     waveMemLoadOverlap[1]:     0.0 kB     
blockL1Load     :  17.5 kB     waveMemStoreNew:  1875 kB     waveMemOld[1]        :     0.0 kB     

L1Load      :  35.0 B/Lup  memLoadOverlap[0]:  1.0 B/Lup  basic.waveMemOld[0]:  3.9 MB     L1Cycles :  10.0 cyc    
smL1Alloc   :  25.5 kB     memLoadOverlap[1]:  0.0 B/Lup  basic.waveMemOld[1]:  0.0 MB     perfMemV3:    87 GLups/s
L1LoadEvicts:   0.2 B/Lup  memLoadV1        :  9.0 B/Lup  L2Store            :  9.0 B/Lup  perfL2V2 :   210 GLups/s
L2LoadV1    :  12.2 B/Lup  memLoadV2        :  8.1 B/Lup  memStoreV1         :  8.0 B/Lup  perfL1   :   487 GLups/s
L2LoadV2    :  12.4 B/Lup  memLoadV3        :  8.1 B/Lup  memStoreV2         :  8.0 B/Lup  perfV3   :    87 GLups/s

```

Analysis: the predicted performance of 87 GLups/s is memory bandwidth bound.

### 3D 25-point stencil sample on a A100

Output of sample3D25pt.py, which analyzes a range 4 star, 3D 25-point stencil using a (256,1,2) thread block size for a (640, 512, 512) domain:

```
TLBpages        :    11        blockL2Load    :  38.7 kB     waveValidCells       :  221440        
L1Cycles        :    56        blockL2Store   :   4.5 kB     waveMemLoadOverlap[0]:      60 kB     
blockL1LoadAlloc:  40.6 kB     waveMemLoadNew :  7002 kB     waveMemLoadOverlap[1]:    5160 kB     
blockL1Load     :   112 kB     waveMemStoreNew:  1730 kB     waveMemOld[1]        :   25724 kB     

L1Load      :  223 B/Lup  memLoadOverlap[0]:   0.3 B/Lup  basic.waveMemOld[0]:   8.5 MB     L1Cycles :  56 cyc    
smL1Alloc   :  162 kB     memLoadOverlap[1]:  23.9 B/Lup  basic.waveMemOld[1]:  25.1 MB     perfMemV3:  54 GLups/s
L1LoadEvicts:  3.4 B/Lup  memLoadV1        :  32.4 B/Lup  L2Store            :   9.0 B/Lup  perfL2V2 :  50 GLups/s
L2LoadV1    :   77 B/Lup  memLoadV2        :  18.0 B/Lup  memStoreV1         :   8.0 B/Lup  perfL1   :  87 GLups/s
L2LoadV2    :   81 B/Lup  memLoadV3        :  18.1 B/Lup  memStoreV2         :   8.1 B/Lup  perfV3   :  50 GLups/s
```

Analysis: the predicted performance of 50 GLups/s is L2 cache bandwidth bound (hint, a different thread block size would change that).
