# WARPSPEED


## 2D 5-point sample on a A100: 

Output of sample2D5pt.py, which analyzes a range 1, 2D 5-point stencil using a (256,2,1) thread block size for a (15000, 15000) domain:

```
TLBpages        :     3        blockL2Load    :   6.1 kB     waveValidCells       :  240000        
L1Cycles        :  10.0        blockL2Store   :   4.5 kB     waveMemLoadOverlap[0]:     234 kB     
blockL1LoadAlloc:   6.4 kB     waveMemLoadNew :  2110 kB     waveMemLoadOverlap[1]:     0.0 kB     
blockL1Load     :  17.5 kB     waveMemStoreNew:  1875 kB     waveMemOld[1]        :     0.0 kB     

L1Load      :  35.0 B/Lup  memLoadOverlap[0]:  1.0 B/Lup  basic.waveMemOld[0]:  3.9 MB     L1Cycles :  10.0 cyc    
smL1Alloc   :  25.5 kB     memLoadOverlap[1]:  0.0 B/Lup  basic.waveMemOld[1]:  0.0 MB     perfMemV3:    87 GFlop/s
L1LoadEvicts:   0.2 B/Lup  memLoadV1        :  9.0 B/Lup  L2Store            :  9.0 B/Lup  perfL2V2 :   210 GFlop/s
L2LoadV1    :  12.2 B/Lup  memLoadV2        :  8.1 B/Lup  memStoreV1         :  8.0 B/Lup  perfL1   :   487 GFlop/s
L2LoadV2    :  12.4 B/Lup  memLoadV3        :  8.1 B/Lup  memStoreV2         :  8.0 B/Lup  perfV3   :    87 GFlop/s

```

## 3D 25-point sample on a A100

Output of sample3D25pt.py, which analyzes a range 4 star, 3D 25-point stencil using a (256,2,2) thread block size for a (640, 512, 512) domain:

```
TLBpages        :   11        blockL2Load    :  48.9 kB     waveValidCells       :  222720        
L1Cycles        :   56        blockL2Store   :   9.0 kB     waveMemLoadOverlap[0]:      60 kB     
blockL1LoadAlloc:   52 kB     waveMemLoadNew :  7042 kB     waveMemLoadOverlap[1]:    5190 kB     
blockL1Load     :  223 kB     waveMemStoreNew:  1740 kB     waveMemOld[1]        :   25724 kB     

L1Load      :   223 B/Lup  memLoadOverlap[0]:   0.3 B/Lup  basic.waveMemOld[0]:   8.6 MB     L1Cycles :  56 cyc    
smL1Alloc   :   103 kB     memLoadOverlap[1]:  23.9 B/Lup  basic.waveMemOld[1]:  25.1 MB     perfMemV3:  54 GFlop/s
L1LoadEvicts:   2.8 B/Lup  memLoadV1        :  32.4 B/Lup  L2Store            :   9.0 B/Lup  perfL2V2 :  74 GFlop/s
L2LoadV1    :  48.9 B/Lup  memLoadV2        :  18.0 B/Lup  memStoreV1         :   8.0 B/Lup  perfL1   :  87 GFlop/s
L2LoadV2    :    52 B/Lup  memLoadV3        :  18.1 B/Lup  memStoreV2         :   8.1 B/Lup  perfV3   :  54 GFlop/s

```
