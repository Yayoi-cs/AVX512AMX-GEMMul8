# benchmark definision

## HOST Environment

CPU Only, No GPU. Hosted on GCP.

CPU: Intel(R) Xeon(R) Platinum 8481C CPU @ 2.70GHz

Important feature: AVX, AVX2, AVX-512, AMX, SSE

```
tsuneki@amx-test:~$ lscpu
Architecture:                x86_64
  CPU op-mode(s):            32-bit, 64-bit
  Address sizes:             52 bits physical, 57 bits virtual
  Byte Order:                Little Endian
CPU(s):                      4
  On-line CPU(s) list:       0-3
Vendor ID:                   GenuineIntel
  Model name:                Intel(R) Xeon(R) Platinum 8481C CPU @ 2.70GHz
    CPU family:              6
    Model:                   143
    Thread(s) per core:      2
    Core(s) per socket:      2
    Socket(s):               1
    Stepping:                8
    BogoMIPS:                5399.99
    Flags:                   fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush mmx fxsr sse sse2 ss ht 
                             syscall nx pdpe1gb rdtscp lm constant_tsc rep_good nopl xtopology nonstop_tsc cpuid tsc_known_freq pni p
                             clmulqdq ssse3 fma cx16 pcid sse4_1 sse4_2 x2apic movbe popcnt aes xsave avx f16c rdrand hypervisor lahf
                             _lm abm 3dnowprefetch ssbd ibrs ibpb stibp ibrs_enhanced fsgsbase tsc_adjust bmi1 avx2 smep bmi2 erms in
                             vpcid rtm avx512f avx512dq rdseed adx smap avx512ifma clflushopt clwb avx512cd sha_ni avx512bw avx512vl 
                             xsaveopt xsavec xgetbv1 xsaves avx_vnni avx512_bf16 arat avx512vbmi umip avx512_vbmi2 gfni vaes vpclmulq
                             dq avx512_vnni avx512_bitalg avx512_vpopcntdq la57 rdpid cldemote movdiri movdir64b fsrm md_clear serial
                             ize tsxldtrk amx_bf16 avx512_fp16 amx_tile amx_int8 arch_capabilities
Virtualization features:     
  Hypervisor vendor:         KVM
  Virtualization type:       full
Caches (sum of all):         
  L1d:                       96 KiB (2 instances)
  L1i:                       64 KiB (2 instances)
  L2:                        4 MiB (2 instances)
  L3:                        105 MiB (1 instance)
NUMA:                        
  NUMA node(s):              1
  NUMA node0 CPU(s):         0-3
Vulnerabilities:             
  Gather data sampling:      Not affected
  Ghostwrite:                Not affected
  Indirect target selection: Not affected
  Itlb multihit:             Not affected
  L1tf:                      Not affected
  Mds:                       Not affected
  Meltdown:                  Not affected
  Mmio stale data:           Not affected
  Old microcode:             Not affected
  Reg file data sampling:    Not affected
  Retbleed:                  Not affected
  Spec rstack overflow:      Not affected
  Spec store bypass:         Mitigation; Speculative Store Bypass disabled via prctl
  Spectre v1:                Mitigation; usercopy/swapgs barriers and __user pointer sanitization
  Spectre v2:                Mitigation; Enhanced / Automatic IBRS; IBPB conditional; PBRSB-eIBRS SW sequence; BHI BHI_DIS_S
  Srbds:                     Not affected
  Tsa:                       Not affected
  Tsx async abort:           Not affected
  Vmscape:                   Not affected
tsuneki@amx-test:~$ exit
```

## What we do in benchmark

### Dataset

Uniform randomized value with seed=1337.
alpha, beta is also randomized value.

Test size: 64x64~4096x4096

Do not generate randomized value when measuring.
Initializing the dataset at first.
Make copy before measuring.

### Computation

Do GEMM ($C={\alpha}AB+{\beta}C$) for 0x10000 times.

C will be updated everytime, compiler cannot omit the loop.

Use std::chrono to measure the execution time.

After the measuring, print out the result to suggest compiler do not omit the computation.

### Target kernel

- ../avx-512
- ../amx
- Intel MKL

../GEMMul8 which is the GPU implementation is not the target.


