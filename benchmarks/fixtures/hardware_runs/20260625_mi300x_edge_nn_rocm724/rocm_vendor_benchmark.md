# ROCm Vendor Benchmark Report

- Generated at: `2026-06-25T00:19:59Z`
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
| GEMM_F16_MFMA_CTA64_SHARED_EDGE | m=1023,n=1024,k=2003,compute=F32,edge=true | Navatala HIP MFMA kernel navatala_transformer_tiled_gemm_f16_mfma_cta64_shared_edge | rocBLAS rocblas_gemm_ex F16 inputs/F32 accumulation | mfma_f16 | hip_mfma_gfx942_64x64x8_f16_f32_cta64_shared_edge | pass | 0.183973 | 0.093447 | 1.969x | 2.86102e-06 |
| GEMM_F16_MFMA_CTA64_SHARED_EDGE_NN | m=1023,n=1024,k=2003,compute=F32,edge=true,nnFast=true | Navatala HIP MFMA kernel navatala_transformer_tiled_gemm_f16_mfma_cta64_shared_edge_nn | rocBLAS rocblas_gemm_ex F16 inputs/F32 accumulation | mfma_f16 | hip_mfma_gfx942_64x64x8_f16_f32_cta64_shared_edge_nn | pass | 0.178986 | 0.092891 | 1.927x | 2.86102e-06 |
| GEMM_F16_MFMA_CTA128_EDGE | m=1023,n=1024,k=2003,compute=F32,edge=true | Navatala HIP MFMA kernel navatala_transformer_tiled_gemm_f16_mfma_cta128_edge | rocBLAS rocblas_gemm_ex F16 inputs/F32 accumulation | mfma_f16 | hip_mfma_gfx942_128x128x32_f16_f32_cta128_edge | pass | 0.353011 | 0.092886 | 3.800x | 2.86102e-06 |
| GEMM_F16_F32_WRAPPER_MFMA_TAIL_M1023_N1024_K2003 | m=1023,n=1024,k=2003,transA=N,transB=N,alpha=1,beta=0,batch=1,output=F32,compute=F32,wrapper=mfma | public C ABI navatala_gpu_gemm_f16_f32 with NAVATALA_GPU_GEMM_IMPL=mfma,NAVATALA_GPU_GEMM_VENDOR_MODE=auto,NAVATALA_GPU_GEMM_MFMA_MODE=auto | rocBLAS rocblas_gemm_ex F16 input/F32 output/F32 accumulation | mfma_f16 | hip_mfma_gfx942_wrapper_dispatch | pass | 0.178730 | 0.092831 | 1.925x | 2.86102e-06 |
| GEMM_F16_F32_WRAPPER_MFMA_EDGE_SPLIT_TAIL_M1023_N1024_K2003 | m=1023,n=1024,k=2003,transA=N,transB=N,alpha=1,beta=0,batch=1,output=F32,compute=F32,wrapper=mfma | public C ABI navatala_gpu_gemm_f16_f32 with NAVATALA_GPU_GEMM_IMPL=mfma,NAVATALA_GPU_GEMM_VENDOR_MODE=auto,NAVATALA_GPU_GEMM_MFMA_MODE=split | rocBLAS rocblas_gemm_ex F16 input/F32 output/F32 accumulation | mfma_f16 | hip_mfma_gfx942_wrapper_dispatch_edge_split | pass | 0.309942 | 0.092814 | 3.339x | 2.86102e-06 |
| GEMM_F16_F32_WRAPPER_AUTO_TAIL_M1023_N1024_K2003 | m=1023,n=1024,k=2003,transA=N,transB=N,alpha=1,beta=0,batch=1,output=F32,compute=F32,wrapper=auto | public C ABI navatala_gpu_gemm_f16_f32 with NAVATALA_GPU_GEMM_IMPL=auto,NAVATALA_GPU_GEMM_VENDOR_MODE=auto,NAVATALA_GPU_GEMM_MFMA_MODE=auto | rocBLAS rocblas_gemm_ex F16 input/F32 output/F32 accumulation | vendor_library | vendor_dispatch | pass | 0.092904 | 0.092558 | 1.004x | 0 |
| GEMM_F16_F32_WRAPPER_VENDOR_TAIL_M1023_N1024_K2003 | m=1023,n=1024,k=2003,transA=N,transB=N,alpha=1,beta=0,batch=1,output=F32,compute=F32,wrapper=vendor | public C ABI navatala_gpu_gemm_f16_f32 with NAVATALA_GPU_GEMM_IMPL=vendor,NAVATALA_GPU_GEMM_VENDOR_MODE=auto,NAVATALA_GPU_GEMM_MFMA_MODE=auto | rocBLAS rocblas_gemm_ex F16 input/F32 output/F32 accumulation | vendor_library | vendor_dispatch | pass | 0.092882 | 0.092709 | 1.002x | 0 |
| GEMM_F16_MFMA_CTA64_SHARED_EDGE | m=1024,n=1023,k=2047,compute=F32,edge=true | Navatala HIP MFMA kernel navatala_transformer_tiled_gemm_f16_mfma_cta64_shared_edge | rocBLAS rocblas_gemm_ex F16 inputs/F32 accumulation | mfma_f16 | hip_mfma_gfx942_64x64x8_f16_f32_cta64_shared_edge | pass | 0.196521 | 0.108273 | 1.815x | 1.54972e-06 |
| GEMM_F16_MFMA_CTA64_SHARED_EDGE_NN | m=1024,n=1023,k=2047,compute=F32,edge=true,nnFast=true | Navatala HIP MFMA kernel navatala_transformer_tiled_gemm_f16_mfma_cta64_shared_edge_nn | rocBLAS rocblas_gemm_ex F16 inputs/F32 accumulation | mfma_f16 | hip_mfma_gfx942_64x64x8_f16_f32_cta64_shared_edge_nn | pass | 0.192181 | 0.108462 | 1.772x | 1.54972e-06 |
| GEMM_F16_MFMA_CTA128_EDGE | m=1024,n=1023,k=2047,compute=F32,edge=true | Navatala HIP MFMA kernel navatala_transformer_tiled_gemm_f16_mfma_cta128_edge | rocBLAS rocblas_gemm_ex F16 inputs/F32 accumulation | mfma_f16 | hip_mfma_gfx942_128x128x32_f16_f32_cta128_edge | pass | 0.378344 | 0.108189 | 3.497x | 1.54972e-06 |
| GEMM_F16_F32_WRAPPER_MFMA_TAIL_M1024_N1023_K2047 | m=1024,n=1023,k=2047,transA=N,transB=N,alpha=1,beta=0,batch=1,output=F32,compute=F32,wrapper=mfma | public C ABI navatala_gpu_gemm_f16_f32 with NAVATALA_GPU_GEMM_IMPL=mfma,NAVATALA_GPU_GEMM_VENDOR_MODE=auto,NAVATALA_GPU_GEMM_MFMA_MODE=auto | rocBLAS rocblas_gemm_ex F16 input/F32 output/F32 accumulation | mfma_f16 | hip_mfma_gfx942_wrapper_dispatch | pass | 0.191120 | 0.108129 | 1.768x | 1.54972e-06 |
| GEMM_F16_F32_WRAPPER_MFMA_EDGE_SPLIT_TAIL_M1024_N1023_K2047 | m=1024,n=1023,k=2047,transA=N,transB=N,alpha=1,beta=0,batch=1,output=F32,compute=F32,wrapper=mfma | public C ABI navatala_gpu_gemm_f16_f32 with NAVATALA_GPU_GEMM_IMPL=mfma,NAVATALA_GPU_GEMM_VENDOR_MODE=auto,NAVATALA_GPU_GEMM_MFMA_MODE=split | rocBLAS rocblas_gemm_ex F16 input/F32 output/F32 accumulation | mfma_f16 | hip_mfma_gfx942_wrapper_dispatch_edge_split | pass | 0.319368 | 0.108398 | 2.946x | 1.54972e-06 |
| GEMM_F16_F32_WRAPPER_AUTO_TAIL_M1024_N1023_K2047 | m=1024,n=1023,k=2047,transA=N,transB=N,alpha=1,beta=0,batch=1,output=F32,compute=F32,wrapper=auto | public C ABI navatala_gpu_gemm_f16_f32 with NAVATALA_GPU_GEMM_IMPL=auto,NAVATALA_GPU_GEMM_VENDOR_MODE=auto,NAVATALA_GPU_GEMM_MFMA_MODE=auto | rocBLAS rocblas_gemm_ex F16 input/F32 output/F32 accumulation | vendor_library | vendor_dispatch | pass | 0.108736 | 0.107877 | 1.008x | 0 |
| GEMM_F16_F32_WRAPPER_VENDOR_TAIL_M1024_N1023_K2047 | m=1024,n=1023,k=2047,transA=N,transB=N,alpha=1,beta=0,batch=1,output=F32,compute=F32,wrapper=vendor | public C ABI navatala_gpu_gemm_f16_f32 with NAVATALA_GPU_GEMM_IMPL=vendor,NAVATALA_GPU_GEMM_VENDOR_MODE=auto,NAVATALA_GPU_GEMM_MFMA_MODE=auto | rocBLAS rocblas_gemm_ex F16 input/F32 output/F32 accumulation | vendor_library | vendor_dispatch | pass | 0.108613 | 0.108160 | 1.004x | 0 |
| GEMM_F16_MFMA_CTA64_SHARED_EDGE | m=1025,n=513,k=1024,compute=F32,edge=true | Navatala HIP MFMA kernel navatala_transformer_tiled_gemm_f16_mfma_cta64_shared_edge | rocBLAS rocblas_gemm_ex F16 inputs/F32 accumulation | mfma_f16 | hip_mfma_gfx942_64x64x8_f16_f32_cta64_shared_edge | pass | 0.072629 | 0.059975 | 1.211x | 4.76837e-07 |
| GEMM_F16_MFMA_CTA64_SHARED_EDGE_NN | m=1025,n=513,k=1024,compute=F32,edge=true,nnFast=true | Navatala HIP MFMA kernel navatala_transformer_tiled_gemm_f16_mfma_cta64_shared_edge_nn | rocBLAS rocblas_gemm_ex F16 inputs/F32 accumulation | mfma_f16 | hip_mfma_gfx942_64x64x8_f16_f32_cta64_shared_edge_nn | pass | 0.069763 | 0.059862 | 1.165x | 4.76837e-07 |
| GEMM_F16_MFMA_CTA128_EDGE | m=1025,n=513,k=1024,compute=F32,edge=true | Navatala HIP MFMA kernel navatala_transformer_tiled_gemm_f16_mfma_cta128_edge | rocBLAS rocblas_gemm_ex F16 inputs/F32 accumulation | mfma_f16 | hip_mfma_gfx942_128x128x32_f16_f32_cta128_edge | pass | 0.127696 | 0.059948 | 2.130x | 4.76837e-07 |
| GEMM_F16_F32_WRAPPER_MFMA_TAIL_M1025_N513_K1024 | m=1025,n=513,k=1024,transA=N,transB=N,alpha=1,beta=0,batch=1,output=F32,compute=F32,wrapper=mfma | public C ABI navatala_gpu_gemm_f16_f32 with NAVATALA_GPU_GEMM_IMPL=mfma,NAVATALA_GPU_GEMM_VENDOR_MODE=auto,NAVATALA_GPU_GEMM_MFMA_MODE=auto | rocBLAS rocblas_gemm_ex F16 input/F32 output/F32 accumulation | mfma_f16 | hip_mfma_gfx942_wrapper_dispatch | pass | 0.069478 | 0.059897 | 1.160x | 4.76837e-07 |
| GEMM_F16_F32_WRAPPER_MFMA_EDGE_SPLIT_TAIL_M1025_N513_K1024 | m=1025,n=513,k=1024,transA=N,transB=N,alpha=1,beta=0,batch=1,output=F32,compute=F32,wrapper=mfma | public C ABI navatala_gpu_gemm_f16_f32 with NAVATALA_GPU_GEMM_IMPL=mfma,NAVATALA_GPU_GEMM_VENDOR_MODE=auto,NAVATALA_GPU_GEMM_MFMA_MODE=split | rocBLAS rocblas_gemm_ex F16 input/F32 output/F32 accumulation | mfma_f16 | hip_mfma_gfx942_wrapper_dispatch_edge_split | pass | 0.168327 | 0.059883 | 2.811x | 4.76837e-07 |
| GEMM_F16_F32_WRAPPER_AUTO_TAIL_M1025_N513_K1024 | m=1025,n=513,k=1024,transA=N,transB=N,alpha=1,beta=0,batch=1,output=F32,compute=F32,wrapper=auto | public C ABI navatala_gpu_gemm_f16_f32 with NAVATALA_GPU_GEMM_IMPL=auto,NAVATALA_GPU_GEMM_VENDOR_MODE=auto,NAVATALA_GPU_GEMM_MFMA_MODE=auto | rocBLAS rocblas_gemm_ex F16 input/F32 output/F32 accumulation | vendor_library | vendor_dispatch | pass | 0.059781 | 0.059782 | 1.000x | 0 |
| GEMM_F16_F32_WRAPPER_VENDOR_TAIL_M1025_N513_K1024 | m=1025,n=513,k=1024,transA=N,transB=N,alpha=1,beta=0,batch=1,output=F32,compute=F32,wrapper=vendor | public C ABI navatala_gpu_gemm_f16_f32 with NAVATALA_GPU_GEMM_IMPL=vendor,NAVATALA_GPU_GEMM_VENDOR_MODE=auto,NAVATALA_GPU_GEMM_MFMA_MODE=auto | rocBLAS rocblas_gemm_ex F16 input/F32 output/F32 accumulation | vendor_library | vendor_dispatch | pass | 0.059824 | 0.059765 | 1.001x | 0 |

## Interpretation Notes

- `generatedMeanMs` and `vendorMeanMs` are back-to-back throughput means per launch, not isolated launch latency.
- Correctness is measured against the vendor baseline in the benchmark harness for the listed shape.
- This report covers only the operations listed above; it is not a full HIP backend certification.
- hipSPARSELt is reported as a capability only; no hipSPARSELt performance baseline is claimed.

## Wrapper Timing Diagnostics

| Operation | Shape | Stream-event mean ms | Host submit mean ms | Host wall mean ms | Submit/event | Wall/event |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| GEMM_F16_F32_WRAPPER_MFMA_TAIL_M1023_N1024_K2003 | m=1023,n=1024,k=2003,transA=N,transB=N,alpha=1,beta=0,batch=1,output=F32,compute=F32,wrapper=mfma | 0.178730 | 0.002527 | 0.178906 | 0.014x | 1.001x |
| GEMM_F16_F32_WRAPPER_MFMA_EDGE_SPLIT_TAIL_M1023_N1024_K2003 | m=1023,n=1024,k=2003,transA=N,transB=N,alpha=1,beta=0,batch=1,output=F32,compute=F32,wrapper=mfma | 0.309942 | 0.007507 | 0.310196 | 0.024x | 1.001x |
| GEMM_F16_F32_WRAPPER_AUTO_TAIL_M1023_N1024_K2003 | m=1023,n=1024,k=2003,transA=N,transB=N,alpha=1,beta=0,batch=1,output=F32,compute=F32,wrapper=auto | 0.092904 | 0.005603 | 0.093125 | 0.060x | 1.002x |
| GEMM_F16_F32_WRAPPER_VENDOR_TAIL_M1023_N1024_K2003 | m=1023,n=1024,k=2003,transA=N,transB=N,alpha=1,beta=0,batch=1,output=F32,compute=F32,wrapper=vendor | 0.092882 | 0.005878 | 0.093106 | 0.063x | 1.002x |
| GEMM_F16_F32_WRAPPER_MFMA_TAIL_M1024_N1023_K2047 | m=1024,n=1023,k=2047,transA=N,transB=N,alpha=1,beta=0,batch=1,output=F32,compute=F32,wrapper=mfma | 0.191120 | 0.002414 | 0.191385 | 0.013x | 1.001x |
| GEMM_F16_F32_WRAPPER_MFMA_EDGE_SPLIT_TAIL_M1024_N1023_K2047 | m=1024,n=1023,k=2047,transA=N,transB=N,alpha=1,beta=0,batch=1,output=F32,compute=F32,wrapper=mfma | 0.319368 | 0.008084 | 0.319593 | 0.025x | 1.001x |
| GEMM_F16_F32_WRAPPER_AUTO_TAIL_M1024_N1023_K2047 | m=1024,n=1023,k=2047,transA=N,transB=N,alpha=1,beta=0,batch=1,output=F32,compute=F32,wrapper=auto | 0.108736 | 0.005761 | 0.108954 | 0.053x | 1.002x |
| GEMM_F16_F32_WRAPPER_VENDOR_TAIL_M1024_N1023_K2047 | m=1024,n=1023,k=2047,transA=N,transB=N,alpha=1,beta=0,batch=1,output=F32,compute=F32,wrapper=vendor | 0.108613 | 0.006015 | 0.108836 | 0.055x | 1.002x |
| GEMM_F16_F32_WRAPPER_MFMA_TAIL_M1025_N513_K1024 | m=1025,n=513,k=1024,transA=N,transB=N,alpha=1,beta=0,batch=1,output=F32,compute=F32,wrapper=mfma | 0.069478 | 0.002459 | 0.069741 | 0.035x | 1.004x |
| GEMM_F16_F32_WRAPPER_MFMA_EDGE_SPLIT_TAIL_M1025_N513_K1024 | m=1025,n=513,k=1024,transA=N,transB=N,alpha=1,beta=0,batch=1,output=F32,compute=F32,wrapper=mfma | 0.168327 | 0.007623 | 0.168550 | 0.045x | 1.001x |
| GEMM_F16_F32_WRAPPER_AUTO_TAIL_M1025_N513_K1024 | m=1025,n=513,k=1024,transA=N,transB=N,alpha=1,beta=0,batch=1,output=F32,compute=F32,wrapper=auto | 0.059781 | 0.006137 | 0.060006 | 0.103x | 1.004x |
| GEMM_F16_F32_WRAPPER_VENDOR_TAIL_M1025_N513_K1024 | m=1025,n=513,k=1024,transA=N,transB=N,alpha=1,beta=0,batch=1,output=F32,compute=F32,wrapper=vendor | 0.059824 | 0.005888 | 0.060048 | 0.098x | 1.004x |

- `hostSubmitMeanMs` is CPU wall time to call the wrapper repeatedly and enqueue work, excluding the final event wait.
- `hostWallMeanMs` includes the final event wait. Compare both with the stream-event mean to separate host submission overhead from device work.

## Validation Warnings

- hipSPARSELt is capability reporting only; no hipSPARSELt benchmark row is claimed
