# ROCm Vendor Benchmark Report

- Generated at: `2026-06-24T13:27:21Z`
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
| GEMM_F16_MFMA_CTA64_SHARED_EDGE | m=1023,n=1024,k=2003,compute=F32,edge=true | Navatala HIP MFMA kernel navatala_transformer_tiled_gemm_f16_mfma_cta64_shared_edge | rocBLAS rocblas_gemm_ex F16 inputs/F32 accumulation | mfma_f16 | hip_mfma_gfx942_64x64x8_f16_f32_cta64_shared_edge | pass | 0.187162 | 0.092749 | 2.018x | 2.86102e-06 |
| GEMM_F16_MFMA_CTA128_EDGE | m=1023,n=1024,k=2003,compute=F32,edge=true | Navatala HIP MFMA kernel navatala_transformer_tiled_gemm_f16_mfma_cta128_edge | rocBLAS rocblas_gemm_ex F16 inputs/F32 accumulation | mfma_f16 | hip_mfma_gfx942_128x128x32_f16_f32_cta128_edge | pass | 0.354455 | 0.092947 | 3.814x | 2.86102e-06 |
| GEMM_F16_F32_WRAPPER_MFMA_TAIL_M1023_N1024_K2003 | m=1023,n=1024,k=2003,transA=N,transB=N,alpha=1,beta=0,batch=1,output=F32,compute=F32,wrapper=mfma | public C ABI navatala_gpu_gemm_f16_f32 with NAVATALA_GPU_GEMM_IMPL=mfma,NAVATALA_GPU_GEMM_VENDOR_MODE=auto,NAVATALA_GPU_GEMM_MFMA_MODE=auto | rocBLAS rocblas_gemm_ex F16 input/F32 output/F32 accumulation | mfma_f16 | hip_mfma_gfx942_wrapper_dispatch | pass | 0.202441 | 0.093795 | 2.158x | 2.86102e-06 |
| GEMM_F16_F32_WRAPPER_AUTO_TAIL_M1023_N1024_K2003 | m=1023,n=1024,k=2003,transA=N,transB=N,alpha=1,beta=0,batch=1,output=F32,compute=F32,wrapper=auto | public C ABI navatala_gpu_gemm_f16_f32 with NAVATALA_GPU_GEMM_IMPL=auto,NAVATALA_GPU_GEMM_VENDOR_MODE=auto,NAVATALA_GPU_GEMM_MFMA_MODE=auto | rocBLAS rocblas_gemm_ex F16 input/F32 output/F32 accumulation | vendor_library | vendor_dispatch | pass | 0.093014 | 0.093081 | 0.999x | 0 |
| GEMM_F16_F32_WRAPPER_VENDOR_TAIL_M1023_N1024_K2003 | m=1023,n=1024,k=2003,transA=N,transB=N,alpha=1,beta=0,batch=1,output=F32,compute=F32,wrapper=vendor | public C ABI navatala_gpu_gemm_f16_f32 with NAVATALA_GPU_GEMM_IMPL=vendor,NAVATALA_GPU_GEMM_VENDOR_MODE=auto,NAVATALA_GPU_GEMM_MFMA_MODE=auto | rocBLAS rocblas_gemm_ex F16 input/F32 output/F32 accumulation | vendor_library | vendor_dispatch | pass | 0.092878 | 0.092971 | 0.999x | 0 |
| GEMM_F16_MFMA_CTA64_SHARED_EDGE | m=1024,n=1023,k=2047,compute=F32,edge=true | Navatala HIP MFMA kernel navatala_transformer_tiled_gemm_f16_mfma_cta64_shared_edge | rocBLAS rocblas_gemm_ex F16 inputs/F32 accumulation | mfma_f16 | hip_mfma_gfx942_64x64x8_f16_f32_cta64_shared_edge | pass | 0.195389 | 0.108331 | 1.804x | 1.54972e-06 |
| GEMM_F16_MFMA_CTA128_EDGE | m=1024,n=1023,k=2047,compute=F32,edge=true | Navatala HIP MFMA kernel navatala_transformer_tiled_gemm_f16_mfma_cta128_edge | rocBLAS rocblas_gemm_ex F16 inputs/F32 accumulation | mfma_f16 | hip_mfma_gfx942_128x128x32_f16_f32_cta128_edge | pass | 0.378784 | 0.108510 | 3.491x | 1.54972e-06 |
| GEMM_F16_F32_WRAPPER_MFMA_TAIL_M1024_N1023_K2047 | m=1024,n=1023,k=2047,transA=N,transB=N,alpha=1,beta=0,batch=1,output=F32,compute=F32,wrapper=mfma | public C ABI navatala_gpu_gemm_f16_f32 with NAVATALA_GPU_GEMM_IMPL=mfma,NAVATALA_GPU_GEMM_VENDOR_MODE=auto,NAVATALA_GPU_GEMM_MFMA_MODE=auto | rocBLAS rocblas_gemm_ex F16 input/F32 output/F32 accumulation | mfma_f16 | hip_mfma_gfx942_wrapper_dispatch | pass | 0.197441 | 0.108482 | 1.820x | 1.54972e-06 |
| GEMM_F16_F32_WRAPPER_AUTO_TAIL_M1024_N1023_K2047 | m=1024,n=1023,k=2047,transA=N,transB=N,alpha=1,beta=0,batch=1,output=F32,compute=F32,wrapper=auto | public C ABI navatala_gpu_gemm_f16_f32 with NAVATALA_GPU_GEMM_IMPL=auto,NAVATALA_GPU_GEMM_VENDOR_MODE=auto,NAVATALA_GPU_GEMM_MFMA_MODE=auto | rocBLAS rocblas_gemm_ex F16 input/F32 output/F32 accumulation | vendor_library | vendor_dispatch | pass | 0.108547 | 0.108140 | 1.004x | 0 |
| GEMM_F16_F32_WRAPPER_VENDOR_TAIL_M1024_N1023_K2047 | m=1024,n=1023,k=2047,transA=N,transB=N,alpha=1,beta=0,batch=1,output=F32,compute=F32,wrapper=vendor | public C ABI navatala_gpu_gemm_f16_f32 with NAVATALA_GPU_GEMM_IMPL=vendor,NAVATALA_GPU_GEMM_VENDOR_MODE=auto,NAVATALA_GPU_GEMM_MFMA_MODE=auto | rocBLAS rocblas_gemm_ex F16 input/F32 output/F32 accumulation | vendor_library | vendor_dispatch | pass | 0.108461 | 0.107891 | 1.005x | 0 |
| GEMM_F16_MFMA_CTA64_SHARED_EDGE | m=1025,n=513,k=1024,compute=F32,edge=true | Navatala HIP MFMA kernel navatala_transformer_tiled_gemm_f16_mfma_cta64_shared_edge | rocBLAS rocblas_gemm_ex F16 inputs/F32 accumulation | mfma_f16 | hip_mfma_gfx942_64x64x8_f16_f32_cta64_shared_edge | pass | 0.071904 | 0.060427 | 1.190x | 4.76837e-07 |
| GEMM_F16_MFMA_CTA128_EDGE | m=1025,n=513,k=1024,compute=F32,edge=true | Navatala HIP MFMA kernel navatala_transformer_tiled_gemm_f16_mfma_cta128_edge | rocBLAS rocblas_gemm_ex F16 inputs/F32 accumulation | mfma_f16 | hip_mfma_gfx942_128x128x32_f16_f32_cta128_edge | pass | 0.128257 | 0.060246 | 2.129x | 4.76837e-07 |
| GEMM_F16_F32_WRAPPER_MFMA_TAIL_M1025_N513_K1024 | m=1025,n=513,k=1024,transA=N,transB=N,alpha=1,beta=0,batch=1,output=F32,compute=F32,wrapper=mfma | public C ABI navatala_gpu_gemm_f16_f32 with NAVATALA_GPU_GEMM_IMPL=mfma,NAVATALA_GPU_GEMM_VENDOR_MODE=auto,NAVATALA_GPU_GEMM_MFMA_MODE=auto | rocBLAS rocblas_gemm_ex F16 input/F32 output/F32 accumulation | mfma_f16 | hip_mfma_gfx942_wrapper_dispatch | pass | 0.073283 | 0.059854 | 1.224x | 4.76837e-07 |
| GEMM_F16_F32_WRAPPER_AUTO_TAIL_M1025_N513_K1024 | m=1025,n=513,k=1024,transA=N,transB=N,alpha=1,beta=0,batch=1,output=F32,compute=F32,wrapper=auto | public C ABI navatala_gpu_gemm_f16_f32 with NAVATALA_GPU_GEMM_IMPL=auto,NAVATALA_GPU_GEMM_VENDOR_MODE=auto,NAVATALA_GPU_GEMM_MFMA_MODE=auto | rocBLAS rocblas_gemm_ex F16 input/F32 output/F32 accumulation | vendor_library | vendor_dispatch | pass | 0.059872 | 0.059853 | 1.000x | 0 |
| GEMM_F16_F32_WRAPPER_VENDOR_TAIL_M1025_N513_K1024 | m=1025,n=513,k=1024,transA=N,transB=N,alpha=1,beta=0,batch=1,output=F32,compute=F32,wrapper=vendor | public C ABI navatala_gpu_gemm_f16_f32 with NAVATALA_GPU_GEMM_IMPL=vendor,NAVATALA_GPU_GEMM_VENDOR_MODE=auto,NAVATALA_GPU_GEMM_MFMA_MODE=auto | rocBLAS rocblas_gemm_ex F16 input/F32 output/F32 accumulation | vendor_library | vendor_dispatch | pass | 0.059876 | 0.059881 | 1.000x | 0 |

## Interpretation Notes

- `generatedMeanMs` and `vendorMeanMs` are back-to-back throughput means per launch, not isolated launch latency.
- Correctness is measured against the vendor baseline in the benchmark harness for the listed shape.
- This report covers only the operations listed above; it is not a full HIP backend certification.
- hipSPARSELt is reported as a capability only; no hipSPARSELt performance baseline is claimed.

## Wrapper Timing Diagnostics

| Operation | Shape | Stream-event mean ms | Host submit mean ms | Host wall mean ms | Submit/event | Wall/event |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| GEMM_F16_F32_WRAPPER_MFMA_TAIL_M1023_N1024_K2003 | m=1023,n=1024,k=2003,transA=N,transB=N,alpha=1,beta=0,batch=1,output=F32,compute=F32,wrapper=mfma | 0.202441 | 0.006088 | 0.202642 | 0.030x | 1.001x |
| GEMM_F16_F32_WRAPPER_AUTO_TAIL_M1023_N1024_K2003 | m=1023,n=1024,k=2003,transA=N,transB=N,alpha=1,beta=0,batch=1,output=F32,compute=F32,wrapper=auto | 0.093014 | 0.009022 | 0.093244 | 0.097x | 1.002x |
| GEMM_F16_F32_WRAPPER_VENDOR_TAIL_M1023_N1024_K2003 | m=1023,n=1024,k=2003,transA=N,transB=N,alpha=1,beta=0,batch=1,output=F32,compute=F32,wrapper=vendor | 0.092878 | 0.007425 | 0.093113 | 0.080x | 1.003x |
| GEMM_F16_F32_WRAPPER_MFMA_TAIL_M1024_N1023_K2047 | m=1024,n=1023,k=2047,transA=N,transB=N,alpha=1,beta=0,batch=1,output=F32,compute=F32,wrapper=mfma | 0.197441 | 0.002297 | 0.197718 | 0.012x | 1.001x |
| GEMM_F16_F32_WRAPPER_AUTO_TAIL_M1024_N1023_K2047 | m=1024,n=1023,k=2047,transA=N,transB=N,alpha=1,beta=0,batch=1,output=F32,compute=F32,wrapper=auto | 0.108547 | 0.006379 | 0.108772 | 0.059x | 1.002x |
| GEMM_F16_F32_WRAPPER_VENDOR_TAIL_M1024_N1023_K2047 | m=1024,n=1023,k=2047,transA=N,transB=N,alpha=1,beta=0,batch=1,output=F32,compute=F32,wrapper=vendor | 0.108461 | 0.005979 | 0.108687 | 0.055x | 1.002x |
| GEMM_F16_F32_WRAPPER_MFMA_TAIL_M1025_N513_K1024 | m=1025,n=513,k=1024,transA=N,transB=N,alpha=1,beta=0,batch=1,output=F32,compute=F32,wrapper=mfma | 0.073283 | 0.002620 | 0.073551 | 0.036x | 1.004x |
| GEMM_F16_F32_WRAPPER_AUTO_TAIL_M1025_N513_K1024 | m=1025,n=513,k=1024,transA=N,transB=N,alpha=1,beta=0,batch=1,output=F32,compute=F32,wrapper=auto | 0.059872 | 0.005806 | 0.060098 | 0.097x | 1.004x |
| GEMM_F16_F32_WRAPPER_VENDOR_TAIL_M1025_N513_K1024 | m=1025,n=513,k=1024,transA=N,transB=N,alpha=1,beta=0,batch=1,output=F32,compute=F32,wrapper=vendor | 0.059876 | 0.006459 | 0.060105 | 0.108x | 1.004x |

- `hostSubmitMeanMs` is CPU wall time to call the wrapper repeatedly and enqueue work, excluding the final event wait.
- `hostWallMeanMs` includes the final event wait. Compare both with the stream-event mean to separate host submission overhead from device work.

## Validation Warnings

- hipSPARSELt is capability reporting only; no hipSPARSELt benchmark row is claimed
