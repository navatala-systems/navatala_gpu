# ROCm Vendor Benchmark Report

- Generated at: `2026-07-02T07:54:32Z`
- Source JSON: `rocm_vendor_benchmark.json`
- Schema: `navatala_gpu.rocm_vendor_benchmark.v1`
- Timing mode: `back_to_back_throughput_mean_per_launch`
- Quick mode: `false`
- Matrix set: `wrapper-semantics`
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
| GEMM_F16_F32_WRAPPER_MFMA_TAIL_SMALL | m=513,n=511,k=257,transA=N,transB=N,alpha=1,beta=0,batch=1,output=F32,compute=F32,wrapper=mfma | public C ABI navatala_gpu_gemm_f16_f32 with NAVATALA_GPU_GEMM_IMPL=mfma,NAVATALA_GPU_GEMM_VENDOR_MODE=auto,NAVATALA_GPU_GEMM_MFMA_MODE=auto | rocBLAS rocblas_gemm_ex F16 input/F32 output/F32 accumulation | mfma_f16 | hip_mfma_gfx942_wrapper_dispatch | pass | 0.017850 | 0.038386 | 0.465x | 7.15256e-07 |
| GEMM_F16_F32_WRAPPER_MFMA_TAIL_LARGE | m=1023,n=1024,k=2003,transA=N,transB=N,alpha=1,beta=0,batch=1,output=F32,compute=F32,wrapper=mfma | public C ABI navatala_gpu_gemm_f16_f32 with NAVATALA_GPU_GEMM_IMPL=mfma,NAVATALA_GPU_GEMM_VENDOR_MODE=auto,NAVATALA_GPU_GEMM_MFMA_MODE=auto | rocBLAS rocblas_gemm_ex F16 input/F32 output/F32 accumulation | mfma_f16 | hip_mfma_gfx942_wrapper_dispatch | pass | 0.184595 | 0.092764 | 1.990x | 2.86102e-06 |
| GEMM_F16_F32_WRAPPER_MFMA_EDGE_SPLIT_TAIL_LARGE | m=1023,n=1024,k=2003,transA=N,transB=N,alpha=1,beta=0,batch=1,output=F32,compute=F32,wrapper=mfma | public C ABI navatala_gpu_gemm_f16_f32 with NAVATALA_GPU_GEMM_IMPL=mfma,NAVATALA_GPU_GEMM_VENDOR_MODE=auto,NAVATALA_GPU_GEMM_MFMA_MODE=split | rocBLAS rocblas_gemm_ex F16 input/F32 output/F32 accumulation | mfma_f16 | hip_mfma_gfx942_wrapper_dispatch_edge_split | pass | 0.314792 | 0.092759 | 3.394x | 2.86102e-06 |
| GEMM_F16_F32_WRAPPER_AUTO_TAIL_LARGE | m=1023,n=1024,k=2003,transA=N,transB=N,alpha=1,beta=0,batch=1,output=F32,compute=F32,wrapper=auto | public C ABI navatala_gpu_gemm_f16_f32 with NAVATALA_GPU_GEMM_IMPL=auto,NAVATALA_GPU_GEMM_VENDOR_MODE=auto,NAVATALA_GPU_GEMM_MFMA_MODE=auto | rocBLAS rocblas_gemm_ex F16 input/F32 output/F32 accumulation | vendor_library | vendor_dispatch | pass | 0.092545 | 0.092676 | 0.999x | 0 |
| GEMM_F16_F32_WRAPPER_VENDOR_TAIL_LARGE | m=1023,n=1024,k=2003,transA=N,transB=N,alpha=1,beta=0,batch=1,output=F32,compute=F32,wrapper=vendor | public C ABI navatala_gpu_gemm_f16_f32 with NAVATALA_GPU_GEMM_IMPL=vendor,NAVATALA_GPU_GEMM_VENDOR_MODE=auto,NAVATALA_GPU_GEMM_MFMA_MODE=auto | rocBLAS rocblas_gemm_ex F16 input/F32 output/F32 accumulation | vendor_library | vendor_dispatch | pass | 0.092693 | 0.092712 | 1.000x | 0 |
| GEMM_F16_MFMA_CTA64_SHARED_SEMANTIC_ALPHA_BETA | m=512,n=512,k=512,transA=N,transB=N,alpha=0.750000,beta=0.250000,compute=F32,semantic_full_tile=true | Navatala HIP MFMA kernel navatala_transformer_tiled_gemm_f16_mfma_cta64_shared_semantic | rocBLAS rocblas_gemm_ex F16 inputs/F32 accumulation | mfma_f16 | hip_mfma_gfx942_64x64x8_f16_f32_cta64_shared_semantic | pass | 0.025506 | 0.032946 | 0.774x | 1.04308e-06 |
| GEMM_F16_F32_WRAPPER_MFMA_ALPHA_BETA | m=512,n=512,k=512,transA=N,transB=N,alpha=0.75,beta=0.25,batch=1,output=F32,compute=F32,wrapper=mfma | public C ABI navatala_gpu_gemm_f16_f32 with NAVATALA_GPU_GEMM_IMPL=mfma,NAVATALA_GPU_GEMM_VENDOR_MODE=auto,NAVATALA_GPU_GEMM_MFMA_MODE=auto | rocBLAS rocblas_gemm_ex F16 input/F32 output/F32 accumulation | mfma_f16 | hip_mfma_gfx942_wrapper_dispatch | pass | 0.025480 | 0.032783 | 0.777x | 1.04308e-06 |
| GEMM_F16_MFMA_CTA64_SHARED_SEMANTIC_TRANSPOSE | m=384,n=512,k=256,transA=T,transB=T,alpha=1.000000,beta=0.000000,compute=F32,semantic_full_tile=true | Navatala HIP MFMA kernel navatala_transformer_tiled_gemm_f16_mfma_cta64_shared_semantic | rocBLAS rocblas_gemm_ex F16 inputs/F32 accumulation | mfma_f16 | hip_mfma_gfx942_64x64x8_f16_f32_cta64_shared_semantic | pass | 0.023187 | 0.020331 | 1.140x | 3.57628e-07 |
| GEMM_F16_F32_WRAPPER_MFMA_TRANSPOSE | m=384,n=512,k=256,transA=T,transB=T,alpha=1,beta=0,batch=1,output=F32,compute=F32,wrapper=mfma | public C ABI navatala_gpu_gemm_f16_f32_ex with NAVATALA_GPU_GEMM_IMPL=mfma,NAVATALA_GPU_GEMM_VENDOR_MODE=auto,NAVATALA_GPU_GEMM_MFMA_MODE=auto | rocBLAS rocblas_gemm_ex F16 input/F32 output/F32 accumulation | mfma_f16 | hip_mfma_gfx942_wrapper_dispatch | pass | 0.023193 | 0.020234 | 1.146x | 3.57628e-07 |
| GEMM_F16_F32_WRAPPER_AUTO_TRANSPOSE | m=384,n=512,k=256,transA=T,transB=T,alpha=1,beta=0,batch=1,output=F32,compute=F32,wrapper=auto | public C ABI navatala_gpu_gemm_f16_f32_ex with NAVATALA_GPU_GEMM_IMPL=auto,NAVATALA_GPU_GEMM_VENDOR_MODE=auto,NAVATALA_GPU_GEMM_MFMA_MODE=auto | rocBLAS rocblas_gemm_ex F16 input/F32 output/F32 accumulation | vendor_library | vendor_dispatch | pass | 0.020138 | 0.020012 | 1.006x | 0 |
| GEMM_F16_F32_WRAPPER_VENDOR_TRANSPOSE | m=384,n=512,k=256,transA=T,transB=T,alpha=1,beta=0,batch=1,output=F32,compute=F32,wrapper=vendor | public C ABI navatala_gpu_gemm_f16_f32_ex with NAVATALA_GPU_GEMM_IMPL=vendor,NAVATALA_GPU_GEMM_VENDOR_MODE=auto,NAVATALA_GPU_GEMM_MFMA_MODE=auto | rocBLAS rocblas_gemm_ex F16 input/F32 output/F32 accumulation | vendor_library | vendor_dispatch | pass | 0.019991 | 0.020030 | 0.998x | 0 |
| GEMM_F16_F32_WRAPPER_MFMA_BATCHED | m=256,n=256,k=256,transA=N,transB=N,alpha=1,beta=0,batch=3,output=F32,compute=F32,wrapper=mfma | public C ABI navatala_gpu_gemm_f16_f32_strided_batched with NAVATALA_GPU_GEMM_IMPL=mfma,NAVATALA_GPU_GEMM_VENDOR_MODE=auto,NAVATALA_GPU_GEMM_MFMA_MODE=auto | rocBLAS rocblas_gemm_strided_batched_ex F16 input/F32 output/F32 accumulation | mfma_f16 | hip_mfma_gfx942_wrapper_dispatch | pass | 0.017740 | 0.017248 | 1.029x | 3.57628e-07 |
| GEMM_F16_F32_WRAPPER_AUTO_BATCHED | m=256,n=256,k=256,transA=N,transB=N,alpha=1,beta=0,batch=3,output=F32,compute=F32,wrapper=auto | public C ABI navatala_gpu_gemm_f16_f32_strided_batched with NAVATALA_GPU_GEMM_IMPL=auto,NAVATALA_GPU_GEMM_VENDOR_MODE=auto,NAVATALA_GPU_GEMM_MFMA_MODE=auto | rocBLAS rocblas_gemm_strided_batched_ex F16 input/F32 output/F32 accumulation | vendor_library | vendor_dispatch | pass | 0.017301 | 0.017152 | 1.009x | 0 |
| GEMM_F16_F32_WRAPPER_VENDOR_BATCHED | m=256,n=256,k=256,transA=N,transB=N,alpha=1,beta=0,batch=3,output=F32,compute=F32,wrapper=vendor | public C ABI navatala_gpu_gemm_f16_f32_strided_batched with NAVATALA_GPU_GEMM_IMPL=vendor,NAVATALA_GPU_GEMM_VENDOR_MODE=auto,NAVATALA_GPU_GEMM_MFMA_MODE=auto | rocBLAS rocblas_gemm_strided_batched_ex F16 input/F32 output/F32 accumulation | vendor_library | vendor_dispatch | pass | 0.017269 | 0.017152 | 1.007x | 0 |

## Interpretation Notes

- `generatedMeanMs` and `vendorMeanMs` are back-to-back throughput means per launch, not isolated launch latency.
- Correctness is measured against the vendor baseline in the benchmark harness for the listed shape.
- This report covers only the operations listed above; it is not a full HIP backend certification.
- hipSPARSELt is reported as a capability only; no hipSPARSELt performance baseline is claimed.

## Wrapper Timing Diagnostics

| Operation | Shape | Stream-event mean ms | Host submit mean ms | Host wall mean ms | Submit/event | Wall/event |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| GEMM_F16_F32_WRAPPER_MFMA_TAIL_SMALL | m=513,n=511,k=257,transA=N,transB=N,alpha=1,beta=0,batch=1,output=F32,compute=F32,wrapper=mfma | 0.017850 | 0.002710 | 0.017969 | 0.152x | 1.007x |
| GEMM_F16_F32_WRAPPER_MFMA_TAIL_LARGE | m=1023,n=1024,k=2003,transA=N,transB=N,alpha=1,beta=0,batch=1,output=F32,compute=F32,wrapper=mfma | 0.184595 | 0.002812 | 0.184752 | 0.015x | 1.001x |
| GEMM_F16_F32_WRAPPER_MFMA_EDGE_SPLIT_TAIL_LARGE | m=1023,n=1024,k=2003,transA=N,transB=N,alpha=1,beta=0,batch=1,output=F32,compute=F32,wrapper=mfma | 0.314792 | 0.007938 | 0.314930 | 0.025x | 1.000x |
| GEMM_F16_F32_WRAPPER_AUTO_TAIL_LARGE | m=1023,n=1024,k=2003,transA=N,transB=N,alpha=1,beta=0,batch=1,output=F32,compute=F32,wrapper=auto | 0.092545 | 0.005398 | 0.092701 | 0.058x | 1.002x |
| GEMM_F16_F32_WRAPPER_VENDOR_TAIL_LARGE | m=1023,n=1024,k=2003,transA=N,transB=N,alpha=1,beta=0,batch=1,output=F32,compute=F32,wrapper=vendor | 0.092693 | 0.005576 | 0.092830 | 0.060x | 1.001x |
| GEMM_F16_F32_WRAPPER_MFMA_ALPHA_BETA | m=512,n=512,k=512,transA=N,transB=N,alpha=0.75,beta=0.25,batch=1,output=F32,compute=F32,wrapper=mfma | 0.025480 | 0.003330 | 0.025642 | 0.131x | 1.006x |
| GEMM_F16_F32_WRAPPER_MFMA_TRANSPOSE | m=384,n=512,k=256,transA=T,transB=T,alpha=1,beta=0,batch=1,output=F32,compute=F32,wrapper=mfma | 0.023193 | 0.002568 | 0.023361 | 0.111x | 1.007x |
| GEMM_F16_F32_WRAPPER_AUTO_TRANSPOSE | m=384,n=512,k=256,transA=T,transB=T,alpha=1,beta=0,batch=1,output=F32,compute=F32,wrapper=auto | 0.020138 | 0.005499 | 0.020273 | 0.273x | 1.007x |
| GEMM_F16_F32_WRAPPER_VENDOR_TRANSPOSE | m=384,n=512,k=256,transA=T,transB=T,alpha=1,beta=0,batch=1,output=F32,compute=F32,wrapper=vendor | 0.019991 | 0.005454 | 0.020130 | 0.273x | 1.007x |
| GEMM_F16_F32_WRAPPER_MFMA_BATCHED | m=256,n=256,k=256,transA=N,transB=N,alpha=1,beta=0,batch=3,output=F32,compute=F32,wrapper=mfma | 0.017740 | 0.002584 | 0.017897 | 0.146x | 1.009x |
| GEMM_F16_F32_WRAPPER_AUTO_BATCHED | m=256,n=256,k=256,transA=N,transB=N,alpha=1,beta=0,batch=3,output=F32,compute=F32,wrapper=auto | 0.017301 | 0.005817 | 0.017433 | 0.336x | 1.008x |
| GEMM_F16_F32_WRAPPER_VENDOR_BATCHED | m=256,n=256,k=256,transA=N,transB=N,alpha=1,beta=0,batch=3,output=F32,compute=F32,wrapper=vendor | 0.017269 | 0.005607 | 0.017403 | 0.325x | 1.008x |

- `hostSubmitMeanMs` is CPU wall time to call the wrapper repeatedly and enqueue work, excluding the final event wait.
- `hostWallMeanMs` includes the final event wait. Compare both with the stream-event mean to separate host submission overhead from device work.

## Validation Warnings

- hipSPARSELt is capability reporting only; no hipSPARSELt benchmark row is claimed
