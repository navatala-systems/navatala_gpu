# ROCm Vendor Benchmark Report

- Generated at: `2026-06-24T23:02:11Z`
- Source JSON: `rocm_vendor_benchmark.json`
- Schema: `navatala_gpu.rocm_vendor_benchmark.v1`
- Timing mode: `back_to_back_throughput_mean_per_launch`
- Quick mode: `false`
- Matrix set: `edge-tails`
- Iterations: `30`
- Warmup: `8`
- Device: `AMD Instinct MI300X VF`
- GCN arch: `gfx942:sramecc+:xnack-`
- Global memory: `196288 MiB`
- hipSPARSELt available: `true`
- hipSPARSELt mode: `capability_reporting_only`

## Results

| Operation | Shape | Generated path | Vendor baseline | Kernel class | Tuning path | Correctness | Generated mean ms | Vendor mean ms | Ratio | Max abs error |
| --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| GEMM_F16_MFMA_CTA64_SHARED_EDGE | m=1023,n=1024,k=2003,compute=F32,edge=true | Navatala HIP MFMA kernel navatala_transformer_tiled_gemm_f16_mfma_cta64_shared_edge | rocBLAS rocblas_gemm_ex F16 inputs/F32 accumulation | mfma_f16 | hip_mfma_gfx942_64x64x8_f16_f32_cta64_shared_edge | pass | 0.188990 | 0.093039 | 2.031x | 2.86102e-06 |
| GEMM_F16_MFMA_CTA128_EDGE | m=1023,n=1024,k=2003,compute=F32,edge=true | Navatala HIP MFMA kernel navatala_transformer_tiled_gemm_f16_mfma_cta128_edge | rocBLAS rocblas_gemm_ex F16 inputs/F32 accumulation | mfma_f16 | hip_mfma_gfx942_128x128x32_f16_f32_cta128_edge | pass | 0.354167 | 0.092999 | 3.808x | 2.86102e-06 |
| GEMM_F16_F32_WRAPPER_MFMA_TAIL_M1023_N1024_K2003 | m=1023,n=1024,k=2003,transA=N,transB=N,alpha=1,beta=0,batch=1,output=F32,compute=F32,wrapper=mfma | public C ABI navatala_gpu_gemm_f16_f32 with NAVATALA_GPU_GEMM_IMPL=mfma,NAVATALA_GPU_GEMM_VENDOR_MODE=auto,NAVATALA_GPU_GEMM_MFMA_MODE=auto | rocBLAS rocblas_gemm_ex F16 input/F32 output/F32 accumulation | mfma_f16 | hip_mfma_gfx942_wrapper_dispatch | pass | 0.199329 | 0.093647 | 2.129x | 2.86102e-06 |
| GEMM_F16_F32_WRAPPER_MFMA_EDGE_SPLIT_TAIL_M1023_N1024_K2003 | m=1023,n=1024,k=2003,transA=N,transB=N,alpha=1,beta=0,batch=1,output=F32,compute=F32,wrapper=mfma | public C ABI navatala_gpu_gemm_f16_f32 with NAVATALA_GPU_GEMM_IMPL=mfma,NAVATALA_GPU_GEMM_VENDOR_MODE=auto,NAVATALA_GPU_GEMM_MFMA_MODE=split | rocBLAS rocblas_gemm_ex F16 input/F32 output/F32 accumulation | mfma_f16 | hip_mfma_gfx942_wrapper_dispatch_edge_split | pass | 0.324070 | 0.093694 | 3.459x | 2.86102e-06 |
| GEMM_F16_F32_WRAPPER_AUTO_TAIL_M1023_N1024_K2003 | m=1023,n=1024,k=2003,transA=N,transB=N,alpha=1,beta=0,batch=1,output=F32,compute=F32,wrapper=auto | public C ABI navatala_gpu_gemm_f16_f32 with NAVATALA_GPU_GEMM_IMPL=auto,NAVATALA_GPU_GEMM_VENDOR_MODE=auto,NAVATALA_GPU_GEMM_MFMA_MODE=auto | rocBLAS rocblas_gemm_ex F16 input/F32 output/F32 accumulation | vendor_library | vendor_dispatch | pass | 0.093157 | 0.092911 | 1.003x | 0 |
| GEMM_F16_F32_WRAPPER_VENDOR_TAIL_M1023_N1024_K2003 | m=1023,n=1024,k=2003,transA=N,transB=N,alpha=1,beta=0,batch=1,output=F32,compute=F32,wrapper=vendor | public C ABI navatala_gpu_gemm_f16_f32 with NAVATALA_GPU_GEMM_IMPL=vendor,NAVATALA_GPU_GEMM_VENDOR_MODE=auto,NAVATALA_GPU_GEMM_MFMA_MODE=auto | rocBLAS rocblas_gemm_ex F16 input/F32 output/F32 accumulation | vendor_library | vendor_dispatch | pass | 0.093041 | 0.092870 | 1.002x | 0 |
| GEMM_F16_MFMA_CTA64_SHARED_EDGE | m=1024,n=1023,k=2047,compute=F32,edge=true | Navatala HIP MFMA kernel navatala_transformer_tiled_gemm_f16_mfma_cta64_shared_edge | rocBLAS rocblas_gemm_ex F16 inputs/F32 accumulation | mfma_f16 | hip_mfma_gfx942_64x64x8_f16_f32_cta64_shared_edge | pass | 0.196723 | 0.108432 | 1.814x | 1.54972e-06 |
| GEMM_F16_MFMA_CTA128_EDGE | m=1024,n=1023,k=2047,compute=F32,edge=true | Navatala HIP MFMA kernel navatala_transformer_tiled_gemm_f16_mfma_cta128_edge | rocBLAS rocblas_gemm_ex F16 inputs/F32 accumulation | mfma_f16 | hip_mfma_gfx942_128x128x32_f16_f32_cta128_edge | pass | 0.378499 | 0.108554 | 3.487x | 1.54972e-06 |
| GEMM_F16_F32_WRAPPER_MFMA_TAIL_M1024_N1023_K2047 | m=1024,n=1023,k=2047,transA=N,transB=N,alpha=1,beta=0,batch=1,output=F32,compute=F32,wrapper=mfma | public C ABI navatala_gpu_gemm_f16_f32 with NAVATALA_GPU_GEMM_IMPL=mfma,NAVATALA_GPU_GEMM_VENDOR_MODE=auto,NAVATALA_GPU_GEMM_MFMA_MODE=auto | rocBLAS rocblas_gemm_ex F16 input/F32 output/F32 accumulation | mfma_f16 | hip_mfma_gfx942_wrapper_dispatch | pass | 0.194358 | 0.108661 | 1.789x | 1.54972e-06 |
| GEMM_F16_F32_WRAPPER_MFMA_EDGE_SPLIT_TAIL_M1024_N1023_K2047 | m=1024,n=1023,k=2047,transA=N,transB=N,alpha=1,beta=0,batch=1,output=F32,compute=F32,wrapper=mfma | public C ABI navatala_gpu_gemm_f16_f32 with NAVATALA_GPU_GEMM_IMPL=mfma,NAVATALA_GPU_GEMM_VENDOR_MODE=auto,NAVATALA_GPU_GEMM_MFMA_MODE=split | rocBLAS rocblas_gemm_ex F16 input/F32 output/F32 accumulation | mfma_f16 | hip_mfma_gfx942_wrapper_dispatch_edge_split | pass | 0.319091 | 0.108666 | 2.936x | 1.54972e-06 |
| GEMM_F16_F32_WRAPPER_AUTO_TAIL_M1024_N1023_K2047 | m=1024,n=1023,k=2047,transA=N,transB=N,alpha=1,beta=0,batch=1,output=F32,compute=F32,wrapper=auto | public C ABI navatala_gpu_gemm_f16_f32 with NAVATALA_GPU_GEMM_IMPL=auto,NAVATALA_GPU_GEMM_VENDOR_MODE=auto,NAVATALA_GPU_GEMM_MFMA_MODE=auto | rocBLAS rocblas_gemm_ex F16 input/F32 output/F32 accumulation | vendor_library | vendor_dispatch | pass | 0.108424 | 0.108279 | 1.001x | 0 |
| GEMM_F16_F32_WRAPPER_VENDOR_TAIL_M1024_N1023_K2047 | m=1024,n=1023,k=2047,transA=N,transB=N,alpha=1,beta=0,batch=1,output=F32,compute=F32,wrapper=vendor | public C ABI navatala_gpu_gemm_f16_f32 with NAVATALA_GPU_GEMM_IMPL=vendor,NAVATALA_GPU_GEMM_VENDOR_MODE=auto,NAVATALA_GPU_GEMM_MFMA_MODE=auto | rocBLAS rocblas_gemm_ex F16 input/F32 output/F32 accumulation | vendor_library | vendor_dispatch | pass | 0.108346 | 0.108086 | 1.002x | 0 |
| GEMM_F16_MFMA_CTA64_SHARED_EDGE | m=1025,n=513,k=1024,compute=F32,edge=true | Navatala HIP MFMA kernel navatala_transformer_tiled_gemm_f16_mfma_cta64_shared_edge | rocBLAS rocblas_gemm_ex F16 inputs/F32 accumulation | mfma_f16 | hip_mfma_gfx942_64x64x8_f16_f32_cta64_shared_edge | pass | 0.072570 | 0.059945 | 1.211x | 4.76837e-07 |
| GEMM_F16_MFMA_CTA128_EDGE | m=1025,n=513,k=1024,compute=F32,edge=true | Navatala HIP MFMA kernel navatala_transformer_tiled_gemm_f16_mfma_cta128_edge | rocBLAS rocblas_gemm_ex F16 inputs/F32 accumulation | mfma_f16 | hip_mfma_gfx942_128x128x32_f16_f32_cta128_edge | pass | 0.127921 | 0.059957 | 2.134x | 4.76837e-07 |
| GEMM_F16_F32_WRAPPER_MFMA_TAIL_M1025_N513_K1024 | m=1025,n=513,k=1024,transA=N,transB=N,alpha=1,beta=0,batch=1,output=F32,compute=F32,wrapper=mfma | public C ABI navatala_gpu_gemm_f16_f32 with NAVATALA_GPU_GEMM_IMPL=mfma,NAVATALA_GPU_GEMM_VENDOR_MODE=auto,NAVATALA_GPU_GEMM_MFMA_MODE=auto | rocBLAS rocblas_gemm_ex F16 input/F32 output/F32 accumulation | mfma_f16 | hip_mfma_gfx942_wrapper_dispatch | pass | 0.071668 | 0.059831 | 1.198x | 4.76837e-07 |
| GEMM_F16_F32_WRAPPER_MFMA_EDGE_SPLIT_TAIL_M1025_N513_K1024 | m=1025,n=513,k=1024,transA=N,transB=N,alpha=1,beta=0,batch=1,output=F32,compute=F32,wrapper=mfma | public C ABI navatala_gpu_gemm_f16_f32 with NAVATALA_GPU_GEMM_IMPL=mfma,NAVATALA_GPU_GEMM_VENDOR_MODE=auto,NAVATALA_GPU_GEMM_MFMA_MODE=split | rocBLAS rocblas_gemm_ex F16 input/F32 output/F32 accumulation | mfma_f16 | hip_mfma_gfx942_wrapper_dispatch_edge_split | pass | 0.169462 | 0.059798 | 2.834x | 4.76837e-07 |
| GEMM_F16_F32_WRAPPER_AUTO_TAIL_M1025_N513_K1024 | m=1025,n=513,k=1024,transA=N,transB=N,alpha=1,beta=0,batch=1,output=F32,compute=F32,wrapper=auto | public C ABI navatala_gpu_gemm_f16_f32 with NAVATALA_GPU_GEMM_IMPL=auto,NAVATALA_GPU_GEMM_VENDOR_MODE=auto,NAVATALA_GPU_GEMM_MFMA_MODE=auto | rocBLAS rocblas_gemm_ex F16 input/F32 output/F32 accumulation | vendor_library | vendor_dispatch | pass | 0.059961 | 0.059839 | 1.002x | 0 |
| GEMM_F16_F32_WRAPPER_VENDOR_TAIL_M1025_N513_K1024 | m=1025,n=513,k=1024,transA=N,transB=N,alpha=1,beta=0,batch=1,output=F32,compute=F32,wrapper=vendor | public C ABI navatala_gpu_gemm_f16_f32 with NAVATALA_GPU_GEMM_IMPL=vendor,NAVATALA_GPU_GEMM_VENDOR_MODE=auto,NAVATALA_GPU_GEMM_MFMA_MODE=auto | rocBLAS rocblas_gemm_ex F16 input/F32 output/F32 accumulation | vendor_library | vendor_dispatch | pass | 0.059963 | 0.059815 | 1.002x | 0 |

## Interpretation Notes

- `generatedMeanMs` and `vendorMeanMs` are back-to-back throughput means per launch, not isolated launch latency.
- Correctness is measured against the vendor baseline in the benchmark harness for the listed shape.
- This report covers only the operations listed above; it is not a full HIP backend certification.
- hipSPARSELt is reported as a capability only; no hipSPARSELt performance baseline is claimed.

## Wrapper Timing Diagnostics

| Operation | Shape | Stream-event mean ms | Host submit mean ms | Host wall mean ms | Submit/event | Wall/event |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| GEMM_F16_F32_WRAPPER_MFMA_TAIL_M1023_N1024_K2003 | m=1023,n=1024,k=2003,transA=N,transB=N,alpha=1,beta=0,batch=1,output=F32,compute=F32,wrapper=mfma | 0.199329 | 0.006603 | 0.199521 | 0.033x | 1.001x |
| GEMM_F16_F32_WRAPPER_MFMA_EDGE_SPLIT_TAIL_M1023_N1024_K2003 | m=1023,n=1024,k=2003,transA=N,transB=N,alpha=1,beta=0,batch=1,output=F32,compute=F32,wrapper=mfma | 0.324070 | 0.017893 | 0.324328 | 0.055x | 1.001x |
| GEMM_F16_F32_WRAPPER_AUTO_TAIL_M1023_N1024_K2003 | m=1023,n=1024,k=2003,transA=N,transB=N,alpha=1,beta=0,batch=1,output=F32,compute=F32,wrapper=auto | 0.093157 | 0.007954 | 0.093390 | 0.085x | 1.002x |
| GEMM_F16_F32_WRAPPER_VENDOR_TAIL_M1023_N1024_K2003 | m=1023,n=1024,k=2003,transA=N,transB=N,alpha=1,beta=0,batch=1,output=F32,compute=F32,wrapper=vendor | 0.093041 | 0.007857 | 0.093256 | 0.084x | 1.002x |
| GEMM_F16_F32_WRAPPER_MFMA_TAIL_M1024_N1023_K2047 | m=1024,n=1023,k=2047,transA=N,transB=N,alpha=1,beta=0,batch=1,output=F32,compute=F32,wrapper=mfma | 0.194358 | 0.002683 | 0.194643 | 0.014x | 1.001x |
| GEMM_F16_F32_WRAPPER_MFMA_EDGE_SPLIT_TAIL_M1024_N1023_K2047 | m=1024,n=1023,k=2047,transA=N,transB=N,alpha=1,beta=0,batch=1,output=F32,compute=F32,wrapper=mfma | 0.319091 | 0.008389 | 0.319336 | 0.026x | 1.001x |
| GEMM_F16_F32_WRAPPER_AUTO_TAIL_M1024_N1023_K2047 | m=1024,n=1023,k=2047,transA=N,transB=N,alpha=1,beta=0,batch=1,output=F32,compute=F32,wrapper=auto | 0.108424 | 0.005821 | 0.108646 | 0.054x | 1.002x |
| GEMM_F16_F32_WRAPPER_VENDOR_TAIL_M1024_N1023_K2047 | m=1024,n=1023,k=2047,transA=N,transB=N,alpha=1,beta=0,batch=1,output=F32,compute=F32,wrapper=vendor | 0.108346 | 0.005832 | 0.108576 | 0.054x | 1.002x |
| GEMM_F16_F32_WRAPPER_MFMA_TAIL_M1025_N513_K1024 | m=1025,n=513,k=1024,transA=N,transB=N,alpha=1,beta=0,batch=1,output=F32,compute=F32,wrapper=mfma | 0.071668 | 0.002934 | 0.071946 | 0.041x | 1.004x |
| GEMM_F16_F32_WRAPPER_MFMA_EDGE_SPLIT_TAIL_M1025_N513_K1024 | m=1025,n=513,k=1024,transA=N,transB=N,alpha=1,beta=0,batch=1,output=F32,compute=F32,wrapper=mfma | 0.169462 | 0.007627 | 0.169689 | 0.045x | 1.001x |
| GEMM_F16_F32_WRAPPER_AUTO_TAIL_M1025_N513_K1024 | m=1025,n=513,k=1024,transA=N,transB=N,alpha=1,beta=0,batch=1,output=F32,compute=F32,wrapper=auto | 0.059961 | 0.006288 | 0.060183 | 0.105x | 1.004x |
| GEMM_F16_F32_WRAPPER_VENDOR_TAIL_M1025_N513_K1024 | m=1025,n=513,k=1024,transA=N,transB=N,alpha=1,beta=0,batch=1,output=F32,compute=F32,wrapper=vendor | 0.059963 | 0.006157 | 0.060189 | 0.103x | 1.004x |

- `hostSubmitMeanMs` is CPU wall time to call the wrapper repeatedly and enqueue work, excluding the final event wait.
- `hostWallMeanMs` includes the final event wait. Compare both with the stream-event mean to separate host submission overhead from device work.

## Validation Warnings

- hipSPARSELt is capability reporting only; no hipSPARSELt benchmark row is claimed
