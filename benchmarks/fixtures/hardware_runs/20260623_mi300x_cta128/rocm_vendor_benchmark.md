# ROCm Vendor Benchmark Report

- Generated at: `2026-06-23T05:14:31Z`
- Source JSON: `rocm_vendor_benchmark.json`
- Schema: `navatala_gpu.rocm_vendor_benchmark.v1`
- Timing mode: `back_to_back_throughput_mean_per_launch`
- Quick mode: `false`
- Matrix set: `broad`
- Iterations: `30`
- Warmup: `10`
- Device: `AMD Instinct MI300X VF`
- GCN arch: `gfx942:sramecc+:xnack-`
- Global memory: `196288 MiB`
- hipSPARSELt available: `true`
- hipSPARSELt mode: `vendor_benchmark`

## Results

| Operation | Shape | Generated path | Vendor baseline | Kernel class | Tuning path | Correctness | Generated mean ms | Vendor mean ms | Ratio | Max abs error |
| --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| AXPY_F32 | n=65536 | Navatala HIP kernel navatala_sparse_axpy_f32 | rocBLAS rocblas_saxpy | scalar | portable_kernel | pass | 0.002623 | 0.003267 | 0.803x | 0 |
| AXPY_F32 | n=1048576 | Navatala HIP kernel navatala_sparse_axpy_f32 | rocBLAS rocblas_saxpy | scalar | portable_kernel | pass | 0.005899 | 0.005469 | 1.079x | 0 |
| AXPY_F32 | n=4194304 | Navatala HIP kernel navatala_sparse_axpy_f32 | rocBLAS rocblas_saxpy | scalar | portable_kernel | pass | 0.018021 | 0.023006 | 0.783x | 0 |
| GEMM_F32 | m=128,n=128,k=128 | Navatala HIP kernel navatala_transformer_tiled_gemm_f32 | rocBLAS rocblas_sgemm | scalar | portable_kernel | pass | 0.005412 | 0.008255 | 0.656x | 3.42727e-07 |
| GEMM_F32 | m=512,n=512,k=512 | Navatala HIP kernel navatala_transformer_tiled_gemm_f32 | rocBLAS rocblas_sgemm | scalar | portable_kernel | pass | 0.027247 | 0.009928 | 2.745x | 8.04663e-07 |
| GEMM_F32 | m=1024,n=1024,k=1024 | Navatala HIP kernel navatala_transformer_tiled_gemm_f32 | rocBLAS rocblas_sgemm | scalar | portable_kernel | pass | 0.181811 | 0.026037 | 6.983x | 1.84774e-06 |
| GEMM_F16_PORTABLE_F32OUT | m=128,n=128,k=128,output=F32,compute=F32 | Navatala HIP kernel navatala_transformer_tiled_gemm_f16_f32_out | rocBLAS rocblas_gemm_ex F16 input/F32 output/F32 accumulation | scalar | portable_f16_f32out_tiled | pass | 0.006010 | 0.024224 | 0.248x | 2.38419e-07 |
| GEMM_F16_PORTABLE_F32OUT | m=512,n=512,k=512,output=F32,compute=F32 | Navatala HIP kernel navatala_transformer_tiled_gemm_f16_f32_out | rocBLAS rocblas_gemm_ex F16 input/F32 output/F32 accumulation | scalar | portable_f16_f32out_tiled | pass | 0.027271 | 0.026837 | 1.016x | 2.02656e-06 |
| GEMM_F16_PORTABLE_F32OUT | m=1024,n=1024,k=1024,output=F32,compute=F32 | Navatala HIP kernel navatala_transformer_tiled_gemm_f16_f32_out | rocBLAS rocblas_gemm_ex F16 input/F32 output/F32 accumulation | scalar | portable_f16_f32out_tiled | pass | 0.179148 | 0.047758 | 3.751x | 2.86102e-06 |
| GEMM_F16_MFMA | m=128,n=128,k=128,compute=F32 | Navatala HIP MFMA kernel navatala_transformer_tiled_gemm_f16_mfma_k_loop | rocBLAS rocblas_gemm_ex F16 inputs/F32 accumulation | mfma_f16 | hip_mfma_gfx942_32x32x8_f16_f32_k_loop | pass | 0.006075 | 0.023734 | 0.256x | 1.19209e-07 |
| GEMM_F16_MFMA | m=512,n=512,k=512,compute=F32 | Navatala HIP MFMA kernel navatala_transformer_tiled_gemm_f16_mfma_k_loop | rocBLAS rocblas_gemm_ex F16 inputs/F32 accumulation | mfma_f16 | hip_mfma_gfx942_32x32x8_f16_f32_k_loop | pass | 0.019507 | 0.025622 | 0.761x | 1.01328e-06 |
| GEMM_F16_MFMA | m=1024,n=1024,k=1024,compute=F32 | Navatala HIP MFMA kernel navatala_transformer_tiled_gemm_f16_mfma_k_loop | rocBLAS rocblas_gemm_ex F16 inputs/F32 accumulation | mfma_f16 | hip_mfma_gfx942_32x32x8_f16_f32_k_loop | pass | 0.107612 | 0.042801 | 2.514x | 2.14577e-06 |
| GEMM_F16_MFMA_CTA128 | m=128,n=128,k=128,compute=F32,cta=128x128x32 | Navatala HIP MFMA kernel navatala_transformer_tiled_gemm_f16_mfma_cta128 | rocBLAS rocblas_gemm_ex F16 inputs/F32 accumulation | mfma_f16 | hip_mfma_gfx942_128x128x32_f16_f32_cta128 | pass | 0.014631 | 0.023763 | 0.616x | 1.19209e-07 |
| GEMM_F16_MFMA_CTA128 | m=512,n=512,k=512,compute=F32,cta=128x128x32 | Navatala HIP MFMA kernel navatala_transformer_tiled_gemm_f16_mfma_cta128 | rocBLAS rocblas_gemm_ex F16 inputs/F32 accumulation | mfma_f16 | hip_mfma_gfx942_128x128x32_f16_f32_cta128 | pass | 0.051251 | 0.025904 | 1.978x | 1.01328e-06 |
| GEMM_F16_MFMA_CTA128 | m=1024,n=1024,k=1024,compute=F32,cta=128x128x32 | Navatala HIP MFMA kernel navatala_transformer_tiled_gemm_f16_mfma_cta128 | rocBLAS rocblas_gemm_ex F16 inputs/F32 accumulation | mfma_f16 | hip_mfma_gfx942_128x128x32_f16_f32_cta128 | pass | 0.146791 | 0.042785 | 3.431x | 2.14577e-06 |
| GEMM_F32_WRAPPER_VENDOR | m=128,n=128,k=128 | public C ABI navatala_gpu_gemm_f32 with NAVATALA_GPU_GEMM_VENDOR_MODE=vendor | rocBLAS rocblas_sgemm | vendor_library | vendor_dispatch | pass | 0.006793 | 0.006831 | 0.994x | 0 |
| GEMM_F32_WRAPPER_VENDOR | m=512,n=512,k=512 | public C ABI navatala_gpu_gemm_f32 with NAVATALA_GPU_GEMM_VENDOR_MODE=vendor | rocBLAS rocblas_sgemm | vendor_library | vendor_dispatch | pass | 0.008355 | 0.008313 | 1.005x | 0 |
| GEMM_F32_WRAPPER_VENDOR | m=1024,n=1024,k=1024 | public C ABI navatala_gpu_gemm_f32 with NAVATALA_GPU_GEMM_VENDOR_MODE=vendor | rocBLAS rocblas_sgemm | vendor_library | vendor_dispatch | pass | 0.031374 | 0.027489 | 1.141x | 0 |
| CSR_SPMV_F32 | rows=16384,rowNnz=7,nnz=114688 | Navatala HIP kernel navatala_graph_spmv_weighted_f32 | rocSPARSE rocsparse_spmv | scalar | thread_per_row | pass | 0.002920 | 0.003770 | 0.775x | 1.49012e-08 |
| CSR_SPMV_F32 | rows=262144,rowNnz=7,nnz=1835008 | Navatala HIP kernel navatala_graph_spmv_weighted_f32 | rocSPARSE rocsparse_spmv | scalar | thread_per_row | pass | 0.006440 | 0.005243 | 1.228x | 2.98023e-08 |
| CSR_SPMV_F32 | rows=1048576,rowNnz=7,nnz=7340032 | Navatala HIP kernel navatala_graph_spmv_weighted_f32 | rocSPARSE rocsparse_spmv | scalar | thread_per_row | pass | 0.049144 | 0.021724 | 2.262x | 1.49012e-08 |
| CSR_SPMV_F32 | rows=262144,rowNnz=15,nnz=3932160 | Navatala HIP kernel navatala_graph_spmv_weighted_f32 | rocSPARSE rocsparse_spmv | scalar | thread_per_row | pass | 0.024053 | 0.010972 | 2.192x | 4.47035e-08 |
| CSR_SPMV_F32 | rows=262144,rowNnz=27,nnz=7077888 | Navatala HIP kernel navatala_graph_spmv_weighted_subgroup_f32 | rocSPARSE rocsparse_spmv | scalar | subgroup_per_row | pass | 0.045523 | 0.020747 | 2.194x | 5.96046e-08 |
| HIPSPARSELT_STRUCTURED_GEMM_F16 | m=128,n=128,k=128,sparsity=50%,compute=F32 | hipSPARSELt prune/compress/matmul path for SparseLt_StructuredMatmul | rocBLAS rocblas_hgemm dense pruned-A reference | vendor_library | vendor_dispatch | pass | 0.011898 | 0.009578 | 1.242x | 0 |
| HIPSPARSELT_STRUCTURED_GEMM_F16 | m=512,n=512,k=512,sparsity=50%,compute=F32 | hipSPARSELt prune/compress/matmul path for SparseLt_StructuredMatmul | rocBLAS rocblas_hgemm dense pruned-A reference | vendor_library | vendor_dispatch | pass | 0.017375 | 0.022944 | 0.757x | 0 |

## Interpretation Notes

- `generatedMeanMs` and `vendorMeanMs` are back-to-back throughput means per launch, not isolated launch latency.
- Correctness is measured against the vendor baseline in the benchmark harness for the listed shape.
- This report covers only the operations listed above; it is not a full HIP backend certification.
- hipSPARSELt structured rows compute 50% structured sparse matmul; dense MFMA rows compute ordinary dense matmul. Do not compare their ratios directly.
