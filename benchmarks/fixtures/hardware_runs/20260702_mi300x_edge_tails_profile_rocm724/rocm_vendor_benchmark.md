# ROCm Vendor Benchmark Report

- Generated at: `2026-07-02T07:56:18Z`
- Source JSON: `rocm_vendor_benchmark.json`
- Schema: `navatala_gpu.rocm_vendor_benchmark.v1`
- Timing mode: `back_to_back_throughput_mean_per_launch`
- Quick mode: `false`
- Matrix set: `edge-tails`
- Iterations: `50`
- Warmup: `10`
- Device: `AMD Instinct MI300X VF`
- GCN arch: `gfx942:sramecc+:xnack-`
- Global memory: `196288 MiB`
- hipSPARSELt available: `true`
- hipSPARSELt mode: `capability_reporting_only`

## Results

| Operation | Shape | Generated path | Vendor baseline | Kernel class | Tuning path | Correctness | Generated mean ms | Vendor mean ms | Ratio | Max abs error |
| --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| GEMM_F16_MFMA_CTA64_SHARED_EDGE | m=1023,n=1024,k=2003,compute=F32,edge=true | Navatala HIP MFMA kernel navatala_transformer_tiled_gemm_f16_mfma_cta64_shared_edge | rocBLAS rocblas_gemm_ex F16 inputs/F32 accumulation | mfma_f16 | hip_mfma_gfx942_64x64x8_f16_f32_cta64_shared_edge | pass | 0.196656 | 0.092717 | 2.121x | 2.86102e-06 |
| GEMM_F16_MFMA_CTA64_SHARED_EDGE_NN | m=1023,n=1024,k=2003,compute=F32,edge=true,nnFast=true | Navatala HIP MFMA kernel navatala_transformer_tiled_gemm_f16_mfma_cta64_shared_edge_nn | rocBLAS rocblas_gemm_ex F16 inputs/F32 accumulation | mfma_f16 | hip_mfma_gfx942_64x64x8_f16_f32_cta64_shared_edge_nn | pass | 0.185662 | 0.093950 | 1.976x | 2.86102e-06 |
| GEMM_F16_MFMA_CTA128_EDGE | m=1023,n=1024,k=2003,compute=F32,edge=true | Navatala HIP MFMA kernel navatala_transformer_tiled_gemm_f16_mfma_cta128_edge | rocBLAS rocblas_gemm_ex F16 inputs/F32 accumulation | mfma_f16 | hip_mfma_gfx942_128x128x32_f16_f32_cta128_edge | pass | 0.366111 | 0.129987 | 2.817x | 2.86102e-06 |
| GEMM_F16_F32_WRAPPER_MFMA_TAIL_M1023_N1024_K2003 | m=1023,n=1024,k=2003,transA=N,transB=N,alpha=1,beta=0,batch=1,output=F32,compute=F32,wrapper=mfma | public C ABI navatala_gpu_gemm_f16_f32 with NAVATALA_GPU_GEMM_IMPL=mfma,NAVATALA_GPU_GEMM_VENDOR_MODE=auto,NAVATALA_GPU_GEMM_MFMA_MODE=auto | rocBLAS rocblas_gemm_ex F16 input/F32 output/F32 accumulation | mfma_f16 | hip_mfma_gfx942_wrapper_dispatch | pass | 0.212141 | 0.092992 | 2.281x | 2.86102e-06 |
| GEMM_F16_F32_WRAPPER_MFMA_EDGE_SPLIT_TAIL_M1023_N1024_K2003 | m=1023,n=1024,k=2003,transA=N,transB=N,alpha=1,beta=0,batch=1,output=F32,compute=F32,wrapper=mfma | public C ABI navatala_gpu_gemm_f16_f32 with NAVATALA_GPU_GEMM_IMPL=mfma,NAVATALA_GPU_GEMM_VENDOR_MODE=auto,NAVATALA_GPU_GEMM_MFMA_MODE=split | rocBLAS rocblas_gemm_ex F16 input/F32 output/F32 accumulation | mfma_f16 | hip_mfma_gfx942_wrapper_dispatch_edge_split | pass | 0.374674 | 0.168316 | 2.226x | 2.86102e-06 |
| GEMM_F16_F32_WRAPPER_AUTO_TAIL_M1023_N1024_K2003 | m=1023,n=1024,k=2003,transA=N,transB=N,alpha=1,beta=0,batch=1,output=F32,compute=F32,wrapper=auto | public C ABI navatala_gpu_gemm_f16_f32 with NAVATALA_GPU_GEMM_IMPL=auto,NAVATALA_GPU_GEMM_VENDOR_MODE=auto,NAVATALA_GPU_GEMM_MFMA_MODE=auto | rocBLAS rocblas_gemm_ex F16 input/F32 output/F32 accumulation | vendor_library | vendor_dispatch | pass | 0.114661 | 0.168926 | 0.679x | 0 |
| GEMM_F16_F32_WRAPPER_VENDOR_TAIL_M1023_N1024_K2003 | m=1023,n=1024,k=2003,transA=N,transB=N,alpha=1,beta=0,batch=1,output=F32,compute=F32,wrapper=vendor | public C ABI navatala_gpu_gemm_f16_f32 with NAVATALA_GPU_GEMM_IMPL=vendor,NAVATALA_GPU_GEMM_VENDOR_MODE=auto,NAVATALA_GPU_GEMM_MFMA_MODE=auto | rocBLAS rocblas_gemm_ex F16 input/F32 output/F32 accumulation | vendor_library | vendor_dispatch | pass | 0.114862 | 0.131303 | 0.875x | 0 |
| GEMM_F16_MFMA_CTA64_SHARED_EDGE | m=1024,n=1023,k=2047,compute=F32,edge=true | Navatala HIP MFMA kernel navatala_transformer_tiled_gemm_f16_mfma_cta64_shared_edge | rocBLAS rocblas_gemm_ex F16 inputs/F32 accumulation | mfma_f16 | hip_mfma_gfx942_64x64x8_f16_f32_cta64_shared_edge | pass | 0.201387 | 0.108519 | 1.856x | 1.54972e-06 |
| GEMM_F16_MFMA_CTA64_SHARED_EDGE_NN | m=1024,n=1023,k=2047,compute=F32,edge=true,nnFast=true | Navatala HIP MFMA kernel navatala_transformer_tiled_gemm_f16_mfma_cta64_shared_edge_nn | rocBLAS rocblas_gemm_ex F16 inputs/F32 accumulation | mfma_f16 | hip_mfma_gfx942_64x64x8_f16_f32_cta64_shared_edge_nn | pass | 0.197675 | 0.145382 | 1.360x | 1.54972e-06 |
| GEMM_F16_MFMA_CTA128_EDGE | m=1024,n=1023,k=2047,compute=F32,edge=true | Navatala HIP MFMA kernel navatala_transformer_tiled_gemm_f16_mfma_cta128_edge | rocBLAS rocblas_gemm_ex F16 inputs/F32 accumulation | mfma_f16 | hip_mfma_gfx942_128x128x32_f16_f32_cta128_edge | pass | 0.491811 | 0.108351 | 4.539x | 1.54972e-06 |
| GEMM_F16_F32_WRAPPER_MFMA_TAIL_M1024_N1023_K2047 | m=1024,n=1023,k=2047,transA=N,transB=N,alpha=1,beta=0,batch=1,output=F32,compute=F32,wrapper=mfma | public C ABI navatala_gpu_gemm_f16_f32 with NAVATALA_GPU_GEMM_IMPL=mfma,NAVATALA_GPU_GEMM_VENDOR_MODE=auto,NAVATALA_GPU_GEMM_MFMA_MODE=auto | rocBLAS rocblas_gemm_ex F16 input/F32 output/F32 accumulation | mfma_f16 | hip_mfma_gfx942_wrapper_dispatch | pass | 0.245196 | 0.107970 | 2.271x | 1.54972e-06 |
| GEMM_F16_F32_WRAPPER_MFMA_EDGE_SPLIT_TAIL_M1024_N1023_K2047 | m=1024,n=1023,k=2047,transA=N,transB=N,alpha=1,beta=0,batch=1,output=F32,compute=F32,wrapper=mfma | public C ABI navatala_gpu_gemm_f16_f32 with NAVATALA_GPU_GEMM_IMPL=mfma,NAVATALA_GPU_GEMM_VENDOR_MODE=auto,NAVATALA_GPU_GEMM_MFMA_MODE=split | rocBLAS rocblas_gemm_ex F16 input/F32 output/F32 accumulation | mfma_f16 | hip_mfma_gfx942_wrapper_dispatch_edge_split | pass | 0.361392 | 0.186065 | 1.942x | 1.54972e-06 |
| GEMM_F16_F32_WRAPPER_AUTO_TAIL_M1024_N1023_K2047 | m=1024,n=1023,k=2047,transA=N,transB=N,alpha=1,beta=0,batch=1,output=F32,compute=F32,wrapper=auto | public C ABI navatala_gpu_gemm_f16_f32 with NAVATALA_GPU_GEMM_IMPL=auto,NAVATALA_GPU_GEMM_VENDOR_MODE=auto,NAVATALA_GPU_GEMM_MFMA_MODE=auto | rocBLAS rocblas_gemm_ex F16 input/F32 output/F32 accumulation | vendor_library | vendor_dispatch | pass | 0.108116 | 0.108922 | 0.993x | 0 |
| GEMM_F16_F32_WRAPPER_VENDOR_TAIL_M1024_N1023_K2047 | m=1024,n=1023,k=2047,transA=N,transB=N,alpha=1,beta=0,batch=1,output=F32,compute=F32,wrapper=vendor | public C ABI navatala_gpu_gemm_f16_f32 with NAVATALA_GPU_GEMM_IMPL=vendor,NAVATALA_GPU_GEMM_VENDOR_MODE=auto,NAVATALA_GPU_GEMM_MFMA_MODE=auto | rocBLAS rocblas_gemm_ex F16 input/F32 output/F32 accumulation | vendor_library | vendor_dispatch | pass | 0.137006 | 0.108591 | 1.262x | 0 |
| GEMM_F16_MFMA_CTA64_SHARED_EDGE | m=1025,n=513,k=1024,compute=F32,edge=true | Navatala HIP MFMA kernel navatala_transformer_tiled_gemm_f16_mfma_cta64_shared_edge | rocBLAS rocblas_gemm_ex F16 inputs/F32 accumulation | mfma_f16 | hip_mfma_gfx942_64x64x8_f16_f32_cta64_shared_edge | pass | 0.098424 | 0.060388 | 1.630x | 4.76837e-07 |
| GEMM_F16_MFMA_CTA64_SHARED_EDGE_NN | m=1025,n=513,k=1024,compute=F32,edge=true,nnFast=true | Navatala HIP MFMA kernel navatala_transformer_tiled_gemm_f16_mfma_cta64_shared_edge_nn | rocBLAS rocblas_gemm_ex F16 inputs/F32 accumulation | mfma_f16 | hip_mfma_gfx942_64x64x8_f16_f32_cta64_shared_edge_nn | pass | 0.070179 | 0.059981 | 1.170x | 4.76837e-07 |
| GEMM_F16_MFMA_CTA128_EDGE | m=1025,n=513,k=1024,compute=F32,edge=true | Navatala HIP MFMA kernel navatala_transformer_tiled_gemm_f16_mfma_cta128_edge | rocBLAS rocblas_gemm_ex F16 inputs/F32 accumulation | mfma_f16 | hip_mfma_gfx942_128x128x32_f16_f32_cta128_edge | pass | 0.212404 | 0.059869 | 3.548x | 4.76837e-07 |
| GEMM_F16_F32_WRAPPER_MFMA_TAIL_M1025_N513_K1024 | m=1025,n=513,k=1024,transA=N,transB=N,alpha=1,beta=0,batch=1,output=F32,compute=F32,wrapper=mfma | public C ABI navatala_gpu_gemm_f16_f32 with NAVATALA_GPU_GEMM_IMPL=mfma,NAVATALA_GPU_GEMM_VENDOR_MODE=auto,NAVATALA_GPU_GEMM_MFMA_MODE=auto | rocBLAS rocblas_gemm_ex F16 input/F32 output/F32 accumulation | mfma_f16 | hip_mfma_gfx942_wrapper_dispatch | pass | 0.108198 | 0.097276 | 1.112x | 4.76837e-07 |
| GEMM_F16_F32_WRAPPER_MFMA_EDGE_SPLIT_TAIL_M1025_N513_K1024 | m=1025,n=513,k=1024,transA=N,transB=N,alpha=1,beta=0,batch=1,output=F32,compute=F32,wrapper=mfma | public C ABI navatala_gpu_gemm_f16_f32 with NAVATALA_GPU_GEMM_IMPL=mfma,NAVATALA_GPU_GEMM_VENDOR_MODE=auto,NAVATALA_GPU_GEMM_MFMA_MODE=split | rocBLAS rocblas_gemm_ex F16 input/F32 output/F32 accumulation | mfma_f16 | hip_mfma_gfx942_wrapper_dispatch_edge_split | pass | 0.170094 | 0.059810 | 2.844x | 4.76837e-07 |
| GEMM_F16_F32_WRAPPER_AUTO_TAIL_M1025_N513_K1024 | m=1025,n=513,k=1024,transA=N,transB=N,alpha=1,beta=0,batch=1,output=F32,compute=F32,wrapper=auto | public C ABI navatala_gpu_gemm_f16_f32 with NAVATALA_GPU_GEMM_IMPL=auto,NAVATALA_GPU_GEMM_VENDOR_MODE=auto,NAVATALA_GPU_GEMM_MFMA_MODE=auto | rocBLAS rocblas_gemm_ex F16 input/F32 output/F32 accumulation | vendor_library | vendor_dispatch | pass | 0.059744 | 0.059700 | 1.001x | 0 |
| GEMM_F16_F32_WRAPPER_VENDOR_TAIL_M1025_N513_K1024 | m=1025,n=513,k=1024,transA=N,transB=N,alpha=1,beta=0,batch=1,output=F32,compute=F32,wrapper=vendor | public C ABI navatala_gpu_gemm_f16_f32 with NAVATALA_GPU_GEMM_IMPL=vendor,NAVATALA_GPU_GEMM_VENDOR_MODE=auto,NAVATALA_GPU_GEMM_MFMA_MODE=auto | rocBLAS rocblas_gemm_ex F16 input/F32 output/F32 accumulation | vendor_library | vendor_dispatch | pass | 0.059812 | 0.059734 | 1.001x | 0 |

## Interpretation Notes

- `generatedMeanMs` and `vendorMeanMs` are back-to-back throughput means per launch, not isolated launch latency.
- Correctness is measured against the vendor baseline in the benchmark harness for the listed shape.
- This report covers only the operations listed above; it is not a full HIP backend certification.
- hipSPARSELt is reported as a capability only; no hipSPARSELt performance baseline is claimed.

## Wrapper Timing Diagnostics

| Operation | Shape | Stream-event mean ms | Host submit mean ms | Host wall mean ms | Submit/event | Wall/event |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| GEMM_F16_F32_WRAPPER_MFMA_TAIL_M1023_N1024_K2003 | m=1023,n=1024,k=2003,transA=N,transB=N,alpha=1,beta=0,batch=1,output=F32,compute=F32,wrapper=mfma | 0.212141 | 0.002333 | 0.212512 | 0.011x | 1.002x |
| GEMM_F16_F32_WRAPPER_MFMA_EDGE_SPLIT_TAIL_M1023_N1024_K2003 | m=1023,n=1024,k=2003,transA=N,transB=N,alpha=1,beta=0,batch=1,output=F32,compute=F32,wrapper=mfma | 0.374674 | 0.007564 | 0.374821 | 0.020x | 1.000x |
| GEMM_F16_F32_WRAPPER_AUTO_TAIL_M1023_N1024_K2003 | m=1023,n=1024,k=2003,transA=N,transB=N,alpha=1,beta=0,batch=1,output=F32,compute=F32,wrapper=auto | 0.114661 | 0.005375 | 0.115071 | 0.047x | 1.004x |
| GEMM_F16_F32_WRAPPER_VENDOR_TAIL_M1023_N1024_K2003 | m=1023,n=1024,k=2003,transA=N,transB=N,alpha=1,beta=0,batch=1,output=F32,compute=F32,wrapper=vendor | 0.114862 | 0.005374 | 0.115372 | 0.047x | 1.004x |
| GEMM_F16_F32_WRAPPER_MFMA_TAIL_M1024_N1023_K2047 | m=1024,n=1023,k=2047,transA=N,transB=N,alpha=1,beta=0,batch=1,output=F32,compute=F32,wrapper=mfma | 0.245196 | 0.003478 | 0.245368 | 0.014x | 1.001x |
| GEMM_F16_F32_WRAPPER_MFMA_EDGE_SPLIT_TAIL_M1024_N1023_K2047 | m=1024,n=1023,k=2047,transA=N,transB=N,alpha=1,beta=0,batch=1,output=F32,compute=F32,wrapper=mfma | 0.361392 | 0.007795 | 0.361568 | 0.022x | 1.000x |
| GEMM_F16_F32_WRAPPER_AUTO_TAIL_M1024_N1023_K2047 | m=1024,n=1023,k=2047,transA=N,transB=N,alpha=1,beta=0,batch=1,output=F32,compute=F32,wrapper=auto | 0.108116 | 0.006852 | 0.108254 | 0.063x | 1.001x |
| GEMM_F16_F32_WRAPPER_VENDOR_TAIL_M1024_N1023_K2047 | m=1024,n=1023,k=2047,transA=N,transB=N,alpha=1,beta=0,batch=1,output=F32,compute=F32,wrapper=vendor | 0.137006 | 0.005516 | 0.137522 | 0.040x | 1.004x |
| GEMM_F16_F32_WRAPPER_MFMA_TAIL_M1025_N513_K1024 | m=1025,n=513,k=1024,transA=N,transB=N,alpha=1,beta=0,batch=1,output=F32,compute=F32,wrapper=mfma | 0.108198 | 0.002482 | 0.108355 | 0.023x | 1.001x |
| GEMM_F16_F32_WRAPPER_MFMA_EDGE_SPLIT_TAIL_M1025_N513_K1024 | m=1025,n=513,k=1024,transA=N,transB=N,alpha=1,beta=0,batch=1,output=F32,compute=F32,wrapper=mfma | 0.170094 | 0.008275 | 0.170227 | 0.049x | 1.001x |
| GEMM_F16_F32_WRAPPER_AUTO_TAIL_M1025_N513_K1024 | m=1025,n=513,k=1024,transA=N,transB=N,alpha=1,beta=0,batch=1,output=F32,compute=F32,wrapper=auto | 0.059744 | 0.005621 | 0.059876 | 0.094x | 1.002x |
| GEMM_F16_F32_WRAPPER_VENDOR_TAIL_M1025_N513_K1024 | m=1025,n=513,k=1024,transA=N,transB=N,alpha=1,beta=0,batch=1,output=F32,compute=F32,wrapper=vendor | 0.059812 | 0.005619 | 0.059962 | 0.094x | 1.003x |

- `hostSubmitMeanMs` is CPU wall time to call the wrapper repeatedly and enqueue work, excluding the final event wait.
- `hostWallMeanMs` includes the final event wait. Compare both with the stream-event mean to separate host submission overhead from device work.

## Validation Warnings

- hipSPARSELt is capability reporting only; no hipSPARSELt benchmark row is claimed
