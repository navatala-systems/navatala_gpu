# ROCm Vendor Benchmark Report

- Generated at: `2026-06-23T10:34:23Z`
- Source JSON: `rocm_vendor_benchmark.json`
- Schema: `navatala_gpu.rocm_vendor_benchmark.v1`
- Timing mode: `back_to_back_throughput_mean_per_launch`
- Quick mode: `true`
- Matrix set: `quick`
- Iterations: `10`
- Warmup: `3`
- Device: `AMD Instinct MI300X VF`
- GCN arch: `gfx942:sramecc+:xnack-`
- Global memory: `196288 MiB`
- hipSPARSELt available: `true`
- hipSPARSELt mode: `vendor_benchmark`

> Quick mode is a smoke check only. Do not quote quick-mode numbers as performance claims.

## Results

| Operation | Shape | Generated path | Vendor baseline | Kernel class | Tuning path | Correctness | Generated mean ms | Vendor mean ms | Ratio | Max abs error |
| --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| AXPY_F32 | n=65536 | Navatala HIP kernel navatala_sparse_axpy_f32 | rocBLAS rocblas_saxpy | scalar | portable_kernel | pass | 0.002698 | 0.002706 | 0.997x | 0 |
| GEMM_F32 | m=128,n=128,k=128 | Navatala HIP kernel navatala_transformer_tiled_gemm_f32 | rocBLAS rocblas_sgemm | scalar | portable_kernel | pass | 0.005548 | 0.007340 | 0.756x | 3.42727e-07 |
| GEMM_F16_PORTABLE_F32OUT | m=128,n=128,k=128,output=F32,compute=F32 | Navatala HIP kernel navatala_transformer_tiled_gemm_f16_f32_out | rocBLAS rocblas_gemm_ex F16 input/F32 output/F32 accumulation | scalar | portable_f16_f32out_tiled | pass | 0.006198 | 0.024780 | 0.250x | 2.38419e-07 |
| GEMM_F16_MFMA | m=128,n=128,k=128,compute=F32 | Navatala HIP MFMA kernel navatala_transformer_tiled_gemm_f16_mfma_k_loop | rocBLAS rocblas_gemm_ex F16 inputs/F32 accumulation | mfma_f16 | hip_mfma_gfx942_32x32x8_f16_f32_k_loop | pass | 0.006286 | 0.024367 | 0.258x | 1.19209e-07 |
| GEMM_F16_MFMA_CTA64_DIRECT | m=128,n=128,k=128,compute=F32,cta=64x64x8,direct_load=true | Navatala HIP MFMA kernel navatala_transformer_tiled_gemm_f16_mfma_cta64_direct | rocBLAS rocblas_gemm_ex F16 inputs/F32 accumulation | mfma_f16 | hip_mfma_gfx942_64x64x8_f16_f32_cta64_direct | pass | 0.012099 | 0.023681 | 0.511x | 1.19209e-07 |
| GEMM_F16_MFMA_CTA64_SHARED | m=128,n=128,k=128,compute=F32,cta=64x64x8,lds_staged=true | Navatala HIP MFMA kernel navatala_transformer_tiled_gemm_f16_mfma_cta64_shared | rocBLAS rocblas_gemm_ex F16 inputs/F32 accumulation | mfma_f16 | hip_mfma_gfx942_64x64x8_f16_f32_cta64_shared | pass | 0.006338 | 0.024319 | 0.261x | 1.19209e-07 |
| GEMM_F16_MFMA_CTA64_SHARED_EARLY_BARRIER | m=128,n=128,k=128,compute=F32,cta=64x64x8,lds_staged=true,early_barrier=true | Navatala HIP MFMA kernel navatala_transformer_tiled_gemm_f16_mfma_cta64_shared_early_barrier | rocBLAS rocblas_gemm_ex F16 inputs/F32 accumulation | mfma_f16 | hip_mfma_gfx942_64x64x8_f16_f32_cta64_shared_early_barrier | pass | 0.006330 | 0.024078 | 0.263x | 1.19209e-07 |
| GEMM_F16_MFMA_CTA64_SHARED_PADDED | m=128,n=128,k=128,compute=F32,cta=64x64x8,lds_staged=true,lds_padded=true | Navatala HIP MFMA kernel navatala_transformer_tiled_gemm_f16_mfma_cta64_shared_padded | rocBLAS rocblas_gemm_ex F16 inputs/F32 accumulation | mfma_f16 | hip_mfma_gfx942_64x64x8_f16_f32_cta64_shared_padded | pass | 0.011141 | 0.023994 | 0.464x | 1.19209e-07 |
| GEMM_F16_MFMA_CTA64_PIPELINED | m=128,n=128,k=128,compute=F32,cta=64x64x8,lds_staged=true,two_slot=true,r6_panel_copy=true | Navatala HIP MFMA kernel navatala_transformer_tiled_gemm_f16_mfma_cta64_pipelined | rocBLAS rocblas_gemm_ex F16 inputs/F32 accumulation | mfma_f16 | hip_mfma_gfx942_64x64x8_f16_f32_cta64_pipelined | pass | 0.008659 | 0.024082 | 0.360x | 1.19209e-07 |
| GEMM_F16_MFMA_CTA128 | m=128,n=128,k=128,compute=F32,cta=128x128x32 | Navatala HIP MFMA kernel navatala_transformer_tiled_gemm_f16_mfma_cta128 | rocBLAS rocblas_gemm_ex F16 inputs/F32 accumulation | mfma_f16 | hip_mfma_gfx942_128x128x32_f16_f32_cta128 | pass | 0.014581 | 0.024066 | 0.606x | 1.19209e-07 |
| CSR_SPMV_F32 | rows=16384,rowNnz=7,nnz=114688 | Navatala HIP kernel navatala_graph_spmv_weighted_f32 | rocSPARSE rocsparse_spmv | scalar | thread_per_row | pass | 0.003516 | 0.002927 | 1.201x | 1.49012e-08 |
| HIPSPARSELT_STRUCTURED_GEMM_F16 | m=128,n=128,k=128,sparsity=50%,compute=F32 | hipSPARSELt prune/compress/matmul path for SparseLt_StructuredMatmul | rocBLAS rocblas_hgemm dense pruned-A reference | vendor_library | vendor_dispatch | pass | 0.011939 | 0.006988 | 1.709x | 0 |

## Interpretation Notes

- `generatedMeanMs` and `vendorMeanMs` are back-to-back throughput means per launch, not isolated launch latency.
- Correctness is measured against the vendor baseline in the benchmark harness for the listed shape.
- This report covers only the operations listed above; it is not a full HIP backend certification.
- hipSPARSELt structured rows compute 50% structured sparse matmul; dense MFMA rows compute ordinary dense matmul. Do not compare their ratios directly.

## Validation Warnings

- quickMode=true: smoke-only report; do not use as a performance claim
