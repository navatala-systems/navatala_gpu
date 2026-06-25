# ROCm Vendor Benchmark Report

- Generated at: `2026-06-24T17:36:17Z`
- Source JSON: `rocm_vendor_benchmark.json`
- Schema: `navatala_gpu.rocm_vendor_benchmark.v1`
- Timing mode: `back_to_back_throughput_mean_per_launch`
- Quick mode: `false`
- Matrix set: `edge-tails`
- Iterations: `5`
- Warmup: `1`
- Device: `AMD Instinct MI300X VF`
- GCN arch: `gfx942:sramecc+:xnack-`
- Global memory: `196288 MiB`
- hipSPARSELt available: `true`
- hipSPARSELt mode: `capability_reporting_only`

## Results

| Operation | Shape | Generated path | Vendor baseline | Kernel class | Tuning path | Correctness | Generated mean ms | Vendor mean ms | Ratio | Max abs error |
| --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| GEMM_F16_MFMA_CTA64_SHARED_EDGE | m=1023,n=1024,k=2003,compute=F32,edge=true | Navatala HIP MFMA kernel navatala_transformer_tiled_gemm_f16_mfma_cta64_shared_edge | rocBLAS rocblas_gemm_ex F16 inputs/F32 accumulation | mfma_f16 | hip_mfma_gfx942_64x64x8_f16_f32_cta64_shared_edge | pass | 0.185196 | 0.101342 | 1.827x | 2.86102e-06 |
| GEMM_F16_MFMA_CTA128_EDGE | m=1023,n=1024,k=2003,compute=F32,edge=true | Navatala HIP MFMA kernel navatala_transformer_tiled_gemm_f16_mfma_cta128_edge | rocBLAS rocblas_gemm_ex F16 inputs/F32 accumulation | mfma_f16 | hip_mfma_gfx942_128x128x32_f16_f32_cta128_edge | pass | 0.358225 | 0.096508 | 3.712x | 2.86102e-06 |
| GEMM_F16_F32_WRAPPER_MFMA_TAIL_M1023_N1024_K2003 | m=1023,n=1024,k=2003,transA=N,transB=N,alpha=1,beta=0,batch=1,output=F32,compute=F32,wrapper=mfma | public C ABI navatala_gpu_gemm_f16_f32 with NAVATALA_GPU_GEMM_IMPL=mfma,NAVATALA_GPU_GEMM_VENDOR_MODE=auto,NAVATALA_GPU_GEMM_MFMA_MODE=auto | rocBLAS rocblas_gemm_ex F16 input/F32 output/F32 accumulation | mfma_f16 | hip_mfma_gfx942_wrapper_dispatch | pass | 0.187108 | 0.096749 | 1.934x | 2.86102e-06 |
| GEMM_F16_F32_WRAPPER_AUTO_TAIL_M1023_N1024_K2003 | m=1023,n=1024,k=2003,transA=N,transB=N,alpha=1,beta=0,batch=1,output=F32,compute=F32,wrapper=auto | public C ABI navatala_gpu_gemm_f16_f32 with NAVATALA_GPU_GEMM_IMPL=auto,NAVATALA_GPU_GEMM_VENDOR_MODE=auto,NAVATALA_GPU_GEMM_MFMA_MODE=auto | rocBLAS rocblas_gemm_ex F16 input/F32 output/F32 accumulation | vendor_library | vendor_dispatch | pass | 0.097110 | 0.095875 | 1.013x | 0 |
| GEMM_F16_F32_WRAPPER_VENDOR_TAIL_M1023_N1024_K2003 | m=1023,n=1024,k=2003,transA=N,transB=N,alpha=1,beta=0,batch=1,output=F32,compute=F32,wrapper=vendor | public C ABI navatala_gpu_gemm_f16_f32 with NAVATALA_GPU_GEMM_IMPL=vendor,NAVATALA_GPU_GEMM_VENDOR_MODE=auto,NAVATALA_GPU_GEMM_MFMA_MODE=auto | rocBLAS rocblas_gemm_ex F16 input/F32 output/F32 accumulation | vendor_library | vendor_dispatch | pass | 0.096469 | 0.095948 | 1.005x | 0 |
| GEMM_F16_MFMA_CTA64_SHARED_EDGE | m=1024,n=1023,k=2047,compute=F32,edge=true | Navatala HIP MFMA kernel navatala_transformer_tiled_gemm_f16_mfma_cta64_shared_edge | rocBLAS rocblas_gemm_ex F16 inputs/F32 accumulation | mfma_f16 | hip_mfma_gfx942_64x64x8_f16_f32_cta64_shared_edge | pass | 0.196434 | 0.112546 | 1.745x | 1.54972e-06 |
| GEMM_F16_MFMA_CTA128_EDGE | m=1024,n=1023,k=2047,compute=F32,edge=true | Navatala HIP MFMA kernel navatala_transformer_tiled_gemm_f16_mfma_cta128_edge | rocBLAS rocblas_gemm_ex F16 inputs/F32 accumulation | mfma_f16 | hip_mfma_gfx942_128x128x32_f16_f32_cta128_edge | pass | 0.381451 | 0.111520 | 3.420x | 1.54972e-06 |
| GEMM_F16_F32_WRAPPER_MFMA_TAIL_M1024_N1023_K2047 | m=1024,n=1023,k=2047,transA=N,transB=N,alpha=1,beta=0,batch=1,output=F32,compute=F32,wrapper=mfma | public C ABI navatala_gpu_gemm_f16_f32 with NAVATALA_GPU_GEMM_IMPL=mfma,NAVATALA_GPU_GEMM_VENDOR_MODE=auto,NAVATALA_GPU_GEMM_MFMA_MODE=auto | rocBLAS rocblas_gemm_ex F16 input/F32 output/F32 accumulation | mfma_f16 | hip_mfma_gfx942_wrapper_dispatch | pass | 0.199794 | 0.111857 | 1.786x | 1.54972e-06 |
| GEMM_F16_F32_WRAPPER_AUTO_TAIL_M1024_N1023_K2047 | m=1024,n=1023,k=2047,transA=N,transB=N,alpha=1,beta=0,batch=1,output=F32,compute=F32,wrapper=auto | public C ABI navatala_gpu_gemm_f16_f32 with NAVATALA_GPU_GEMM_IMPL=auto,NAVATALA_GPU_GEMM_VENDOR_MODE=auto,NAVATALA_GPU_GEMM_MFMA_MODE=auto | rocBLAS rocblas_gemm_ex F16 input/F32 output/F32 accumulation | vendor_library | vendor_dispatch | pass | 0.112562 | 0.112306 | 1.002x | 0 |
| GEMM_F16_F32_WRAPPER_VENDOR_TAIL_M1024_N1023_K2047 | m=1024,n=1023,k=2047,transA=N,transB=N,alpha=1,beta=0,batch=1,output=F32,compute=F32,wrapper=vendor | public C ABI navatala_gpu_gemm_f16_f32 with NAVATALA_GPU_GEMM_IMPL=vendor,NAVATALA_GPU_GEMM_VENDOR_MODE=auto,NAVATALA_GPU_GEMM_MFMA_MODE=auto | rocBLAS rocblas_gemm_ex F16 input/F32 output/F32 accumulation | vendor_library | vendor_dispatch | pass | 0.112546 | 0.111824 | 1.006x | 0 |
| GEMM_F16_MFMA_CTA64_SHARED_EDGE | m=1025,n=513,k=1024,compute=F32,edge=true | Navatala HIP MFMA kernel navatala_transformer_tiled_gemm_f16_mfma_cta64_shared_edge | rocBLAS rocblas_gemm_ex F16 inputs/F32 accumulation | mfma_f16 | hip_mfma_gfx942_64x64x8_f16_f32_cta64_shared_edge | pass | 0.073705 | 0.062896 | 1.172x | 4.76837e-07 |
| GEMM_F16_MFMA_CTA128_EDGE | m=1025,n=513,k=1024,compute=F32,edge=true | Navatala HIP MFMA kernel navatala_transformer_tiled_gemm_f16_mfma_cta128_edge | rocBLAS rocblas_gemm_ex F16 inputs/F32 accumulation | mfma_f16 | hip_mfma_gfx942_128x128x32_f16_f32_cta128_edge | pass | 0.129144 | 0.062768 | 2.057x | 4.76837e-07 |
| GEMM_F16_F32_WRAPPER_MFMA_TAIL_M1025_N513_K1024 | m=1025,n=513,k=1024,transA=N,transB=N,alpha=1,beta=0,batch=1,output=F32,compute=F32,wrapper=mfma | public C ABI navatala_gpu_gemm_f16_f32 with NAVATALA_GPU_GEMM_IMPL=mfma,NAVATALA_GPU_GEMM_VENDOR_MODE=auto,NAVATALA_GPU_GEMM_MFMA_MODE=auto | rocBLAS rocblas_gemm_ex F16 input/F32 output/F32 accumulation | mfma_f16 | hip_mfma_gfx942_wrapper_dispatch | pass | 0.075485 | 0.063105 | 1.196x | 4.76837e-07 |
| GEMM_F16_F32_WRAPPER_AUTO_TAIL_M1025_N513_K1024 | m=1025,n=513,k=1024,transA=N,transB=N,alpha=1,beta=0,batch=1,output=F32,compute=F32,wrapper=auto | public C ABI navatala_gpu_gemm_f16_f32 with NAVATALA_GPU_GEMM_IMPL=auto,NAVATALA_GPU_GEMM_VENDOR_MODE=auto,NAVATALA_GPU_GEMM_MFMA_MODE=auto | rocBLAS rocblas_gemm_ex F16 input/F32 output/F32 accumulation | vendor_library | vendor_dispatch | pass | 0.063129 | 0.062832 | 1.005x | 0 |
| GEMM_F16_F32_WRAPPER_VENDOR_TAIL_M1025_N513_K1024 | m=1025,n=513,k=1024,transA=N,transB=N,alpha=1,beta=0,batch=1,output=F32,compute=F32,wrapper=vendor | public C ABI navatala_gpu_gemm_f16_f32 with NAVATALA_GPU_GEMM_IMPL=vendor,NAVATALA_GPU_GEMM_VENDOR_MODE=auto,NAVATALA_GPU_GEMM_MFMA_MODE=auto | rocBLAS rocblas_gemm_ex F16 input/F32 output/F32 accumulation | vendor_library | vendor_dispatch | pass | 0.063001 | 0.062712 | 1.005x | 0 |

## Interpretation Notes

- `generatedMeanMs` and `vendorMeanMs` are back-to-back throughput means per launch, not isolated launch latency.
- Correctness is measured against the vendor baseline in the benchmark harness for the listed shape.
- This report covers only the operations listed above; it is not a full HIP backend certification.
- hipSPARSELt is reported as a capability only; no hipSPARSELt performance baseline is claimed.

## Wrapper Timing Diagnostics

| Operation | Shape | Stream-event mean ms | Host submit mean ms | Host wall mean ms | Submit/event | Wall/event |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| GEMM_F16_F32_WRAPPER_MFMA_TAIL_M1023_N1024_K2003 | m=1023,n=1024,k=2003,transA=N,transB=N,alpha=1,beta=0,batch=1,output=F32,compute=F32,wrapper=mfma | 0.187108 | 0.007217 | 0.188348 | 0.039x | 1.007x |
| GEMM_F16_F32_WRAPPER_AUTO_TAIL_M1023_N1024_K2003 | m=1023,n=1024,k=2003,transA=N,transB=N,alpha=1,beta=0,batch=1,output=F32,compute=F32,wrapper=auto | 0.097110 | 0.012873 | 0.098391 | 0.133x | 1.013x |
| GEMM_F16_F32_WRAPPER_VENDOR_TAIL_M1023_N1024_K2003 | m=1023,n=1024,k=2003,transA=N,transB=N,alpha=1,beta=0,batch=1,output=F32,compute=F32,wrapper=vendor | 0.096469 | 0.011634 | 0.098031 | 0.121x | 1.016x |
| GEMM_F16_F32_WRAPPER_MFMA_TAIL_M1024_N1023_K2047 | m=1024,n=1023,k=2047,transA=N,transB=N,alpha=1,beta=0,batch=1,output=F32,compute=F32,wrapper=mfma | 0.199794 | 0.006634 | 0.201180 | 0.033x | 1.007x |
| GEMM_F16_F32_WRAPPER_AUTO_TAIL_M1024_N1023_K2047 | m=1024,n=1023,k=2047,transA=N,transB=N,alpha=1,beta=0,batch=1,output=F32,compute=F32,wrapper=auto | 0.112562 | 0.012075 | 0.114103 | 0.107x | 1.014x |
| GEMM_F16_F32_WRAPPER_VENDOR_TAIL_M1024_N1023_K2047 | m=1024,n=1023,k=2047,transA=N,transB=N,alpha=1,beta=0,batch=1,output=F32,compute=F32,wrapper=vendor | 0.112546 | 0.012118 | 0.113903 | 0.108x | 1.012x |
| GEMM_F16_F32_WRAPPER_MFMA_TAIL_M1025_N513_K1024 | m=1025,n=513,k=1024,transA=N,transB=N,alpha=1,beta=0,batch=1,output=F32,compute=F32,wrapper=mfma | 0.075485 | 0.006957 | 0.076798 | 0.092x | 1.017x |
| GEMM_F16_F32_WRAPPER_AUTO_TAIL_M1025_N513_K1024 | m=1025,n=513,k=1024,transA=N,transB=N,alpha=1,beta=0,batch=1,output=F32,compute=F32,wrapper=auto | 0.063129 | 0.011982 | 0.064571 | 0.190x | 1.023x |
| GEMM_F16_F32_WRAPPER_VENDOR_TAIL_M1025_N513_K1024 | m=1025,n=513,k=1024,transA=N,transB=N,alpha=1,beta=0,batch=1,output=F32,compute=F32,wrapper=vendor | 0.063001 | 0.013221 | 0.064297 | 0.210x | 1.021x |

- `hostSubmitMeanMs` is CPU wall time to call the wrapper repeatedly and enqueue work, excluding the final event wait.
- `hostWallMeanMs` includes the final event wait. Compare both with the stream-event mean to separate host submission overhead from device work.

## Validation Warnings

- hipSPARSELt is capability reporting only; no hipSPARSELt benchmark row is claimed
- iterations<10 with quickMode=false; report may be too noisy for publication
