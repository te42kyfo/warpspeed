* WARPSPEED


2D 5-point sample on a A100: 

```
LBpages        :     3       blockL2Load    :       6  kB   waveValidCells     :       240000       
L1Cycles        :    10       blockL2Store   :       4  kB   waveMemLoadOverlap : (234.375, 0.0) kB   
blockL1LoadAlloc:     6  kB   waveMemLoadNew :    2110  kB   waveMemStoreOverlap: (0.0, 0.0) kB   
blockL1Load     :    18  kB   waveMemStoreNew:    1875  kB   waveMemOld         : (3985.375, 0.0) kB   

L1Cycles    :  10.0                              waveL2Alloc       :    3985kB perfMemV3:  86.6   
L1Load      :  35.0   memLoadEvicts    :   0.0   L2Oversubscription:     0.2   perfL2V2 : 210.4   
smL1Alloc   :    26kB memLoadV1        :   9.0   memStoreEvicts    :     0.0   perfL1   : 487.3   
L1LoadEvicts:   0.2   memLoadV2        :   8.1   L2Store           :     9.0   perfV3   :  86.6   
L2LoadV1    :  12.2   memLoadV3        :   8.1   memStoreV1        :     8.0                      
L2LoadV2    :  12.4   memLoadV4        :   8.1   memStoreV2        :     8.0   
```
