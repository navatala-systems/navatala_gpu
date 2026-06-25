# ROCm Vendor Benchmark Report

- Generated at: `2026-06-23T20:49:42Z`
- Source JSON: `rocm_vendor_benchmark.json`
- Schema: `navatala_gpu.rocm_vendor_benchmark.v1`
- Timing mode: `back_to_back_throughput_mean_per_launch`
- Quick mode: `false`
- Matrix set: `standard`
- Iterations: `10`
- Warmup: `3`
- Device: `AMD Instinct MI300X VF`
- GCN arch: `gfx942:sramecc+:xnack-`
- Global memory: `196288 MiB`
- hipSPARSELt available: `true`
- hipSPARSELt mode: `capability_reporting_only`

## Results

| Operation | Shape | Generated path | Vendor baseline | Kernel class | Tuning path | Correctness | Generated mean ms | Vendor mean ms | Ratio | Max abs error |
| --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| AXPY_F32 | n=4194304 | Navatala HIP kernel navatala_sparse_axpy_f32 | rocBLAS rocblas_saxpy | scalar | portable_kernel | pass | 0.017315 | 0.018357 | 0.943x | 0 |
| GEMM_F32 | m=512,n=512,k=512 | Navatala HIP kernel navatala_transformer_tiled_gemm_f32 | rocBLAS rocblas_sgemm | scalar | portable_kernel | pass | 0.027658 | 0.008900 | 3.108x | 8.04663e-07 |
| GEMM_F16_PORTABLE_F32OUT | m=512,n=512,k=512,output=F32,compute=F32 | Navatala HIP kernel navatala_transformer_tiled_gemm_f16_f32_out | rocBLAS rocblas_gemm_ex F16 input/F32 output/F32 accumulation | scalar | portable_f16_f32out_tiled | pass | 0.027281 | 0.026327 | 1.036x | 2.02656e-06 |
| GEMM_F16_MFMA | m=512,n=512,k=512,compute=F32 | Navatala HIP MFMA kernel navatala_transformer_tiled_gemm_f16_mfma_k_loop | rocBLAS rocblas_gemm_ex F16 inputs/F32 accumulation | mfma_f16 | hip_mfma_gfx942_32x32x8_f16_f32_k_loop | pass | 0.019528 | 0.026051 | 0.750x | 1.01328e-06 |
| GEMM_F16_MFMA_CTA64_DIRECT | m=512,n=512,k=512,compute=F32,cta=64x64x8,direct_load=true | Navatala HIP MFMA kernel navatala_transformer_tiled_gemm_f16_mfma_cta64_direct | rocBLAS rocblas_gemm_ex F16 inputs/F32 accumulation | mfma_f16 | hip_mfma_gfx942_64x64x8_f16_f32_cta64_direct | pass | 0.044396 | 0.026079 | 1.702x | 1.01328e-06 |
| GEMM_F16_MFMA_CTA64_SHARED | m=512,n=512,k=512,compute=F32,cta=64x64x8,lds_staged=true | Navatala HIP MFMA kernel navatala_transformer_tiled_gemm_f16_mfma_cta64_shared | rocBLAS rocblas_gemm_ex F16 inputs/F32 accumulation | mfma_f16 | hip_mfma_gfx942_64x64x8_f16_f32_cta64_shared | pass | 0.020727 | 0.026135 | 0.793x | 1.01328e-06 |
| GEMM_F16_MFMA_CTA64_SHARED_EARLY_BARRIER | m=512,n=512,k=512,compute=F32,cta=64x64x8,lds_staged=true,early_barrier=true | Navatala HIP MFMA kernel navatala_transformer_tiled_gemm_f16_mfma_cta64_shared_early_barrier | rocBLAS rocblas_gemm_ex F16 inputs/F32 accumulation | mfma_f16 | hip_mfma_gfx942_64x64x8_f16_f32_cta64_shared_early_barrier | pass | 0.020703 | 0.025971 | 0.797x | 1.01328e-06 |
| GEMM_F16_MFMA_CTA64_SHARED_PADDED | m=512,n=512,k=512,compute=F32,cta=64x64x8,lds_staged=true,lds_padded=true | Navatala HIP MFMA kernel navatala_transformer_tiled_gemm_f16_mfma_cta64_shared_padded | rocBLAS rocblas_gemm_ex F16 inputs/F32 accumulation | mfma_f16 | hip_mfma_gfx942_64x64x8_f16_f32_cta64_shared_padded | pass | 0.027490 | 0.025979 | 1.058x | 1.01328e-06 |
| GEMM_F16_MFMA_CTA64_PIPELINED | m=512,n=512,k=512,compute=F32,cta=64x64x8,lds_staged=true,two_slot=true,r6_panel_copy=true | Navatala HIP MFMA kernel navatala_transformer_tiled_gemm_f16_mfma_cta64_pipelined | rocBLAS rocblas_gemm_ex F16 inputs/F32 accumulation | mfma_f16 | hip_mfma_gfx942_64x64x8_f16_f32_cta64_pipelined | pass | 0.022900 | 0.026067 | 0.879x | 1.01328e-06 |
| GEMM_F16_MFMA_CTA128 | m=512,n=512,k=512,compute=F32,cta=128x128x32 | Navatala HIP MFMA kernel navatala_transformer_tiled_gemm_f16_mfma_cta128 | rocBLAS rocblas_gemm_ex F16 inputs/F32 accumulation | mfma_f16 | hip_mfma_gfx942_128x128x32_f16_f32_cta128 | pass | 0.026492 | 0.025979 | 1.020x | 1.01328e-06 |
| GEMM_F32_WRAPPER_VENDOR | m=512,n=512,k=512 | public C ABI navatala_gpu_gemm_f32 with NAVATALA_GPU_GEMM_VENDOR_MODE=vendor | rocBLAS rocblas_sgemm | vendor_library | vendor_dispatch | pass | 0.008768 | 0.008732 | 1.004x | 0 |
| GEMM_F16_F32_WRAPPER_MFMA | m=512,n=512,k=512,output=F32,compute=F32,wrapper=mfma | public C ABI navatala_gpu_gemm_f16_f32 with NAVATALA_GPU_GEMM_IMPL=mfma,NAVATALA_GPU_GEMM_VENDOR_MODE=auto,NAVATALA_GPU_GEMM_MFMA_MODE=auto | rocBLAS rocblas_gemm_ex F16 input/F32 output/F32 accumulation | mfma_f16 | hip_mfma_gfx942_wrapper_dispatch | pass | 0.051632 | 0.026255 | 1.967x | 1.01328e-06 |
| GEMM_F16_F32_WRAPPER_MFMA | m=513,n=511,k=257,output=F32,compute=F32,wrapper=mfma | public C ABI navatala_gpu_gemm_f16_f32 with NAVATALA_GPU_GEMM_IMPL=mfma,NAVATALA_GPU_GEMM_VENDOR_MODE=auto,NAVATALA_GPU_GEMM_MFMA_MODE=auto | rocBLAS rocblas_gemm_ex F16 input/F32 output/F32 accumulation | mfma_f16 | hip_mfma_gfx942_wrapper_dispatch | pass | 0.033700 | 0.039212 | 0.859x | 7.15256e-07 |
| CSR_SPMV_F32 | rows=1048576,rowNnz=7,nnz=7340032 | Navatala HIP kernel navatala_graph_spmv_weighted_f32 | rocSPARSE rocsparse_v2_spmv | scalar | thread_per_row | pass | 0.052013 | 0.019584 | 2.656x | 1.49012e-08 |

## Interpretation Notes

- `generatedMeanMs` and `vendorMeanMs` are back-to-back throughput means per launch, not isolated launch latency.
- Correctness is measured against the vendor baseline in the benchmark harness for the listed shape.
- This report covers only the operations listed above; it is not a full HIP backend certification.
- hipSPARSELt is reported as a capability only; no hipSPARSELt performance baseline is claimed.

## Validation Warnings

- hipSPARSELt is capability reporting only; no hipSPARSELt benchmark row is claimed
