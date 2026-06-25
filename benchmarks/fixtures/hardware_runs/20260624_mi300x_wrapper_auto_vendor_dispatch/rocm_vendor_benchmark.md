# ROCm Vendor Benchmark Report

- Generated at: `2026-06-24T12:45:50Z`
- Source JSON: `rocm_vendor_benchmark.json`
- Schema: `navatala_gpu.rocm_vendor_benchmark.v1`
- Timing mode: `back_to_back_throughput_mean_per_launch`
- Quick mode: `false`
- Matrix set: `wrapper-semantics`
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
| GEMM_F16_F32_WRAPPER_MFMA_TAIL_SMALL | m=513,n=511,k=257,transA=N,transB=N,alpha=1,beta=0,batch=1,output=F32,compute=F32,wrapper=mfma | public C ABI navatala_gpu_gemm_f16_f32 with NAVATALA_GPU_GEMM_IMPL=mfma,NAVATALA_GPU_GEMM_VENDOR_MODE=auto,NAVATALA_GPU_GEMM_MFMA_MODE=auto | rocBLAS rocblas_gemm_ex F16 input/F32 output/F32 accumulation | mfma_f16 | hip_mfma_gfx942_wrapper_dispatch | pass | 0.019065 | 0.038707 | 0.493x | 7.15256e-07 |
| GEMM_F16_F32_WRAPPER_MFMA_TAIL_LARGE | m=1023,n=1024,k=2003,transA=N,transB=N,alpha=1,beta=0,batch=1,output=F32,compute=F32,wrapper=mfma | public C ABI navatala_gpu_gemm_f16_f32 with NAVATALA_GPU_GEMM_IMPL=mfma,NAVATALA_GPU_GEMM_VENDOR_MODE=auto,NAVATALA_GPU_GEMM_MFMA_MODE=auto | rocBLAS rocblas_gemm_ex F16 input/F32 output/F32 accumulation | mfma_f16 | hip_mfma_gfx942_wrapper_dispatch | pass | 0.185078 | 0.092755 | 1.995x | 2.86102e-06 |
| GEMM_F16_F32_WRAPPER_AUTO_TAIL_LARGE | m=1023,n=1024,k=2003,transA=N,transB=N,alpha=1,beta=0,batch=1,output=F32,compute=F32,wrapper=auto | public C ABI navatala_gpu_gemm_f16_f32 with NAVATALA_GPU_GEMM_IMPL=auto,NAVATALA_GPU_GEMM_VENDOR_MODE=auto,NAVATALA_GPU_GEMM_MFMA_MODE=auto | rocBLAS rocblas_gemm_ex F16 input/F32 output/F32 accumulation | vendor_library | vendor_dispatch | pass | 0.092827 | 0.092533 | 1.003x | 0 |
| GEMM_F16_F32_WRAPPER_VENDOR_TAIL_LARGE | m=1023,n=1024,k=2003,transA=N,transB=N,alpha=1,beta=0,batch=1,output=F32,compute=F32,wrapper=vendor | public C ABI navatala_gpu_gemm_f16_f32 with NAVATALA_GPU_GEMM_IMPL=vendor,NAVATALA_GPU_GEMM_VENDOR_MODE=auto,NAVATALA_GPU_GEMM_MFMA_MODE=auto | rocBLAS rocblas_gemm_ex F16 input/F32 output/F32 accumulation | vendor_library | vendor_dispatch | pass | 0.092984 | 0.092664 | 1.003x | 0 |
| GEMM_F16_MFMA_CTA64_SHARED_SEMANTIC_ALPHA_BETA | m=512,n=512,k=512,transA=N,transB=N,alpha=0.750000,beta=0.250000,compute=F32,semantic_full_tile=true | Navatala HIP MFMA kernel navatala_transformer_tiled_gemm_f16_mfma_cta64_shared_semantic | rocBLAS rocblas_gemm_ex F16 inputs/F32 accumulation | mfma_f16 | hip_mfma_gfx942_64x64x8_f16_f32_cta64_shared_semantic | pass | 0.025505 | 0.033057 | 0.772x | 1.04308e-06 |
| GEMM_F16_F32_WRAPPER_MFMA_ALPHA_BETA | m=512,n=512,k=512,transA=N,transB=N,alpha=0.75,beta=0.25,batch=1,output=F32,compute=F32,wrapper=mfma | public C ABI navatala_gpu_gemm_f16_f32 with NAVATALA_GPU_GEMM_IMPL=mfma,NAVATALA_GPU_GEMM_VENDOR_MODE=auto,NAVATALA_GPU_GEMM_MFMA_MODE=auto | rocBLAS rocblas_gemm_ex F16 input/F32 output/F32 accumulation | mfma_f16 | hip_mfma_gfx942_wrapper_dispatch | pass | 0.025474 | 0.032973 | 0.773x | 1.04308e-06 |
| GEMM_F16_MFMA_CTA64_SHARED_SEMANTIC_TRANSPOSE | m=384,n=512,k=256,transA=T,transB=T,alpha=1.000000,beta=0.000000,compute=F32,semantic_full_tile=true | Navatala HIP MFMA kernel navatala_transformer_tiled_gemm_f16_mfma_cta64_shared_semantic | rocBLAS rocblas_gemm_ex F16 inputs/F32 accumulation | mfma_f16 | hip_mfma_gfx942_64x64x8_f16_f32_cta64_shared_semantic | pass | 0.023347 | 0.020675 | 1.129x | 3.57628e-07 |
| GEMM_F16_F32_WRAPPER_MFMA_TRANSPOSE | m=384,n=512,k=256,transA=T,transB=T,alpha=1,beta=0,batch=1,output=F32,compute=F32,wrapper=mfma | public C ABI navatala_gpu_gemm_f16_f32_ex with NAVATALA_GPU_GEMM_IMPL=mfma,NAVATALA_GPU_GEMM_VENDOR_MODE=auto,NAVATALA_GPU_GEMM_MFMA_MODE=auto | rocBLAS rocblas_gemm_ex F16 input/F32 output/F32 accumulation | mfma_f16 | hip_mfma_gfx942_wrapper_dispatch | pass | 0.023343 | 0.020450 | 1.141x | 3.57628e-07 |
| GEMM_F16_F32_WRAPPER_AUTO_TRANSPOSE | m=384,n=512,k=256,transA=T,transB=T,alpha=1,beta=0,batch=1,output=F32,compute=F32,wrapper=auto | public C ABI navatala_gpu_gemm_f16_f32_ex with NAVATALA_GPU_GEMM_IMPL=auto,NAVATALA_GPU_GEMM_VENDOR_MODE=auto,NAVATALA_GPU_GEMM_MFMA_MODE=auto | rocBLAS rocblas_gemm_ex F16 input/F32 output/F32 accumulation | vendor_library | vendor_dispatch | pass | 0.020444 | 0.020056 | 1.019x | 0 |
| GEMM_F16_F32_WRAPPER_VENDOR_TRANSPOSE | m=384,n=512,k=256,transA=T,transB=T,alpha=1,beta=0,batch=1,output=F32,compute=F32,wrapper=vendor | public C ABI navatala_gpu_gemm_f16_f32_ex with NAVATALA_GPU_GEMM_IMPL=vendor,NAVATALA_GPU_GEMM_VENDOR_MODE=auto,NAVATALA_GPU_GEMM_MFMA_MODE=auto | rocBLAS rocblas_gemm_ex F16 input/F32 output/F32 accumulation | vendor_library | vendor_dispatch | pass | 0.020130 | 0.020031 | 1.005x | 0 |
| GEMM_F16_F32_WRAPPER_MFMA_BATCHED | m=256,n=256,k=256,transA=N,transB=N,alpha=1,beta=0,batch=3,output=F32,compute=F32,wrapper=mfma | public C ABI navatala_gpu_gemm_f16_f32_strided_batched with NAVATALA_GPU_GEMM_IMPL=mfma,NAVATALA_GPU_GEMM_VENDOR_MODE=auto,NAVATALA_GPU_GEMM_MFMA_MODE=auto | rocBLAS rocblas_gemm_strided_batched_ex F16 input/F32 output/F32 accumulation | mfma_f16 | hip_mfma_gfx942_wrapper_dispatch | pass | 0.018235 | 0.017365 | 1.050x | 3.57628e-07 |
| GEMM_F16_F32_WRAPPER_AUTO_BATCHED | m=256,n=256,k=256,transA=N,transB=N,alpha=1,beta=0,batch=3,output=F32,compute=F32,wrapper=auto | public C ABI navatala_gpu_gemm_f16_f32_strided_batched with NAVATALA_GPU_GEMM_IMPL=auto,NAVATALA_GPU_GEMM_VENDOR_MODE=auto,NAVATALA_GPU_GEMM_MFMA_MODE=auto | rocBLAS rocblas_gemm_strided_batched_ex F16 input/F32 output/F32 accumulation | vendor_library | vendor_dispatch | pass | 0.017437 | 0.017239 | 1.011x | 0 |
| GEMM_F16_F32_WRAPPER_VENDOR_BATCHED | m=256,n=256,k=256,transA=N,transB=N,alpha=1,beta=0,batch=3,output=F32,compute=F32,wrapper=vendor | public C ABI navatala_gpu_gemm_f16_f32_strided_batched with NAVATALA_GPU_GEMM_IMPL=vendor,NAVATALA_GPU_GEMM_VENDOR_MODE=auto,NAVATALA_GPU_GEMM_MFMA_MODE=auto | rocBLAS rocblas_gemm_strided_batched_ex F16 input/F32 output/F32 accumulation | vendor_library | vendor_dispatch | pass | 0.017356 | 0.017260 | 1.006x | 0 |

## Interpretation Notes

- `generatedMeanMs` and `vendorMeanMs` are back-to-back throughput means per launch, not isolated launch latency.
- Correctness is measured against the vendor baseline in the benchmark harness for the listed shape.
- This report covers only the operations listed above; it is not a full HIP backend certification.
- hipSPARSELt is reported as a capability only; no hipSPARSELt performance baseline is claimed.

## Wrapper Timing Diagnostics

| Operation | Shape | Stream-event mean ms | Host submit mean ms | Host wall mean ms | Submit/event | Wall/event |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| GEMM_F16_F32_WRAPPER_MFMA_TAIL_SMALL | m=513,n=511,k=257,transA=N,transB=N,alpha=1,beta=0,batch=1,output=F32,compute=F32,wrapper=mfma | 0.019065 | 0.002662 | 0.019266 | 0.140x | 1.011x |
| GEMM_F16_F32_WRAPPER_MFMA_TAIL_LARGE | m=1023,n=1024,k=2003,transA=N,transB=N,alpha=1,beta=0,batch=1,output=F32,compute=F32,wrapper=mfma | 0.185078 | 0.002659 | 0.185299 | 0.014x | 1.001x |
| GEMM_F16_F32_WRAPPER_AUTO_TAIL_LARGE | m=1023,n=1024,k=2003,transA=N,transB=N,alpha=1,beta=0,batch=1,output=F32,compute=F32,wrapper=auto | 0.092827 | 0.005916 | 0.093054 | 0.064x | 1.002x |
| GEMM_F16_F32_WRAPPER_VENDOR_TAIL_LARGE | m=1023,n=1024,k=2003,transA=N,transB=N,alpha=1,beta=0,batch=1,output=F32,compute=F32,wrapper=vendor | 0.092984 | 0.005822 | 0.093209 | 0.063x | 1.002x |
| GEMM_F16_F32_WRAPPER_MFMA_ALPHA_BETA | m=512,n=512,k=512,transA=N,transB=N,alpha=0.75,beta=0.25,batch=1,output=F32,compute=F32,wrapper=mfma | 0.025474 | 0.002392 | 0.025733 | 0.094x | 1.010x |
| GEMM_F16_F32_WRAPPER_MFMA_TRANSPOSE | m=384,n=512,k=256,transA=T,transB=T,alpha=1,beta=0,batch=1,output=F32,compute=F32,wrapper=mfma | 0.023343 | 0.002488 | 0.023596 | 0.107x | 1.011x |
| GEMM_F16_F32_WRAPPER_AUTO_TRANSPOSE | m=384,n=512,k=256,transA=T,transB=T,alpha=1,beta=0,batch=1,output=F32,compute=F32,wrapper=auto | 0.020444 | 0.006088 | 0.020686 | 0.298x | 1.012x |
| GEMM_F16_F32_WRAPPER_VENDOR_TRANSPOSE | m=384,n=512,k=256,transA=T,transB=T,alpha=1,beta=0,batch=1,output=F32,compute=F32,wrapper=vendor | 0.020130 | 0.006104 | 0.020349 | 0.303x | 1.011x |
| GEMM_F16_F32_WRAPPER_MFMA_BATCHED | m=256,n=256,k=256,transA=N,transB=N,alpha=1,beta=0,batch=3,output=F32,compute=F32,wrapper=mfma | 0.018235 | 0.002530 | 0.018492 | 0.139x | 1.014x |
| GEMM_F16_F32_WRAPPER_AUTO_BATCHED | m=256,n=256,k=256,transA=N,transB=N,alpha=1,beta=0,batch=3,output=F32,compute=F32,wrapper=auto | 0.017437 | 0.005938 | 0.017651 | 0.341x | 1.012x |
| GEMM_F16_F32_WRAPPER_VENDOR_BATCHED | m=256,n=256,k=256,transA=N,transB=N,alpha=1,beta=0,batch=3,output=F32,compute=F32,wrapper=vendor | 0.017356 | 0.005975 | 0.017579 | 0.344x | 1.013x |

- `hostSubmitMeanMs` is CPU wall time to call the wrapper repeatedly and enqueue work, excluding the final event wait.
- `hostWallMeanMs` includes the final event wait. Compare both with the stream-event mean to separate host submission overhead from device work.

## Validation Warnings

- hipSPARSELt is capability reporting only; no hipSPARSELt benchmark row is claimed
