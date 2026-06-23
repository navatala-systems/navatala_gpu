# ROCm Vendor Benchmark Report

- Generated at: `2026-06-23T09:56:04Z`
- Source JSON: `rocm_vendor_benchmark.json`
- Schema: `navatala_gpu.rocm_vendor_benchmark.v1`
- Timing mode: `back_to_back_throughput_mean_per_launch`
- Quick mode: `false`
- Matrix set: `broad`
- Iterations: `30`
- Warmup: `8`
- Device: `AMD Instinct MI300X VF`
- GCN arch: `gfx942:sramecc+:xnack-`
- Global memory: `196288 MiB`
- hipSPARSELt available: `true`
- hipSPARSELt mode: `vendor_benchmark`

## Results

| Operation | Shape | Generated path | Vendor baseline | Kernel class | Tuning path | Correctness | Generated mean ms | Vendor mean ms | Ratio | Max abs error |
| --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| AXPY_F32 | n=65536 | generated HIP kernel navatala_sparse_axpy_f32 | rocBLAS rocblas_saxpy | scalar | portable_kernel | pass | 0.002617 | 0.003205 | 0.817x | 0 |
| AXPY_F32 | n=1048576 | generated HIP kernel navatala_sparse_axpy_f32 | rocBLAS rocblas_saxpy | scalar | portable_kernel | pass | 0.005764 | 0.005583 | 1.032x | 0 |
| AXPY_F32 | n=4194304 | generated HIP kernel navatala_sparse_axpy_f32 | rocBLAS rocblas_saxpy | scalar | portable_kernel | pass | 0.018339 | 0.018474 | 0.993x | 0 |
| GEMM_F32 | m=128,n=128,k=128 | generated HIP kernel navatala_transformer_tiled_gemm_f32 | rocBLAS rocblas_sgemm | scalar | portable_kernel | pass | 0.005363 | 0.008376 | 0.640x | 3.42727e-07 |
| GEMM_F32 | m=512,n=512,k=512 | generated HIP kernel navatala_transformer_tiled_gemm_f32 | rocBLAS rocblas_sgemm | scalar | portable_kernel | pass | 0.026703 | 0.009499 | 2.811x | 8.04663e-07 |
| GEMM_F32 | m=1024,n=1024,k=1024 | generated HIP kernel navatala_transformer_tiled_gemm_f32 | rocBLAS rocblas_sgemm | scalar | portable_kernel | pass | 0.178133 | 0.026550 | 6.709x | 1.84774e-06 |
| GEMM_F16_PORTABLE_F32OUT | m=128,n=128,k=128,output=F32,compute=F32 | generated HIP kernel navatala_transformer_tiled_gemm_f16_f32_out | rocBLAS rocblas_gemm_ex F16 input/F32 output/F32 accumulation | scalar | portable_f16_f32out_tiled | pass | 0.006103 | 0.024222 | 0.252x | 2.38419e-07 |
| GEMM_F16_PORTABLE_F32OUT | m=512,n=512,k=512,output=F32,compute=F32 | generated HIP kernel navatala_transformer_tiled_gemm_f16_f32_out | rocBLAS rocblas_gemm_ex F16 input/F32 output/F32 accumulation | scalar | portable_f16_f32out_tiled | pass | 0.027147 | 0.026053 | 1.042x | 2.02656e-06 |
| GEMM_F16_PORTABLE_F32OUT | m=1024,n=1024,k=1024,output=F32,compute=F32 | generated HIP kernel navatala_transformer_tiled_gemm_f16_f32_out | rocBLAS rocblas_gemm_ex F16 input/F32 output/F32 accumulation | scalar | portable_f16_f32out_tiled | pass | 0.179738 | 0.046907 | 3.832x | 2.86102e-06 |
| GEMM_F16_MFMA | m=128,n=128,k=128,compute=F32 | generated HIP MFMA kernel navatala_transformer_tiled_gemm_f16_mfma_k_loop | rocBLAS rocblas_gemm_ex F16 inputs/F32 accumulation | mfma_f16 | hip_mfma_gfx942_32x32x8_f16_f32_k_loop | pass | 0.006270 | 0.023747 | 0.264x | 1.19209e-07 |
| GEMM_F16_MFMA | m=512,n=512,k=512,compute=F32 | generated HIP MFMA kernel navatala_transformer_tiled_gemm_f16_mfma_k_loop | rocBLAS rocblas_gemm_ex F16 inputs/F32 accumulation | mfma_f16 | hip_mfma_gfx942_32x32x8_f16_f32_k_loop | pass | 0.019482 | 0.025686 | 0.758x | 1.01328e-06 |
| GEMM_F16_MFMA | m=1024,n=1024,k=1024,compute=F32 | generated HIP MFMA kernel navatala_transformer_tiled_gemm_f16_mfma_k_loop | rocBLAS rocblas_gemm_ex F16 inputs/F32 accumulation | mfma_f16 | hip_mfma_gfx942_32x32x8_f16_f32_k_loop | pass | 0.107628 | 0.042781 | 2.516x | 2.14577e-06 |
| GEMM_F16_MFMA_CTA64_DIRECT | m=128,n=128,k=128,compute=F32,cta=64x64x8,direct_load=true | generated HIP MFMA kernel navatala_transformer_tiled_gemm_f16_mfma_cta64_direct | rocBLAS rocblas_gemm_ex F16 inputs/F32 accumulation | mfma_f16 | hip_mfma_gfx942_64x64x8_f16_f32_cta64_direct | pass | 0.012185 | 0.023818 | 0.512x | 1.19209e-07 |
| GEMM_F16_MFMA_CTA64_DIRECT | m=512,n=512,k=512,compute=F32,cta=64x64x8,direct_load=true | generated HIP MFMA kernel navatala_transformer_tiled_gemm_f16_mfma_cta64_direct | rocBLAS rocblas_gemm_ex F16 inputs/F32 accumulation | mfma_f16 | hip_mfma_gfx942_64x64x8_f16_f32_cta64_direct | pass | 0.044196 | 0.025617 | 1.725x | 1.01328e-06 |
| GEMM_F16_MFMA_CTA64_DIRECT | m=1024,n=1024,k=1024,compute=F32,cta=64x64x8,direct_load=true | generated HIP MFMA kernel navatala_transformer_tiled_gemm_f16_mfma_cta64_direct | rocBLAS rocblas_gemm_ex F16 inputs/F32 accumulation | mfma_f16 | hip_mfma_gfx942_64x64x8_f16_f32_cta64_direct | pass | 0.108800 | 0.042733 | 2.546x | 2.14577e-06 |
| GEMM_F16_MFMA_CTA64_SHARED | m=128,n=128,k=128,compute=F32,cta=64x64x8,lds_staged=true | generated HIP MFMA kernel navatala_transformer_tiled_gemm_f16_mfma_cta64_shared | rocBLAS rocblas_gemm_ex F16 inputs/F32 accumulation | mfma_f16 | hip_mfma_gfx942_64x64x8_f16_f32_cta64_shared | pass | 0.006306 | 0.023723 | 0.266x | 1.19209e-07 |
| GEMM_F16_MFMA_CTA64_SHARED | m=512,n=512,k=512,compute=F32,cta=64x64x8,lds_staged=true | generated HIP MFMA kernel navatala_transformer_tiled_gemm_f16_mfma_cta64_shared | rocBLAS rocblas_gemm_ex F16 inputs/F32 accumulation | mfma_f16 | hip_mfma_gfx942_64x64x8_f16_f32_cta64_shared | pass | 0.020569 | 0.025712 | 0.800x | 1.01328e-06 |
| GEMM_F16_MFMA_CTA64_SHARED | m=1024,n=1024,k=1024,compute=F32,cta=64x64x8,lds_staged=true | generated HIP MFMA kernel navatala_transformer_tiled_gemm_f16_mfma_cta64_shared | rocBLAS rocblas_gemm_ex F16 inputs/F32 accumulation | mfma_f16 | hip_mfma_gfx942_64x64x8_f16_f32_cta64_shared | pass | 0.059997 | 0.042869 | 1.400x | 2.14577e-06 |
| GEMM_F16_MFMA_CTA64_SHARED_EARLY_BARRIER | m=128,n=128,k=128,compute=F32,cta=64x64x8,lds_staged=true,early_barrier=true | generated HIP MFMA kernel navatala_transformer_tiled_gemm_f16_mfma_cta64_shared_early_barrier | rocBLAS rocblas_gemm_ex F16 inputs/F32 accumulation | mfma_f16 | hip_mfma_gfx942_64x64x8_f16_f32_cta64_shared_early_barrier | pass | 0.006314 | 0.023730 | 0.266x | 1.19209e-07 |
| GEMM_F16_MFMA_CTA64_SHARED_EARLY_BARRIER | m=512,n=512,k=512,compute=F32,cta=64x64x8,lds_staged=true,early_barrier=true | generated HIP MFMA kernel navatala_transformer_tiled_gemm_f16_mfma_cta64_shared_early_barrier | rocBLAS rocblas_gemm_ex F16 inputs/F32 accumulation | mfma_f16 | hip_mfma_gfx942_64x64x8_f16_f32_cta64_shared_early_barrier | pass | 0.020548 | 0.025616 | 0.802x | 1.01328e-06 |
| GEMM_F16_MFMA_CTA64_SHARED_EARLY_BARRIER | m=1024,n=1024,k=1024,compute=F32,cta=64x64x8,lds_staged=true,early_barrier=true | generated HIP MFMA kernel navatala_transformer_tiled_gemm_f16_mfma_cta64_shared_early_barrier | rocBLAS rocblas_gemm_ex F16 inputs/F32 accumulation | mfma_f16 | hip_mfma_gfx942_64x64x8_f16_f32_cta64_shared_early_barrier | pass | 0.059885 | 0.042931 | 1.395x | 2.14577e-06 |
| GEMM_F16_MFMA_CTA64_SHARED_PADDED | m=128,n=128,k=128,compute=F32,cta=64x64x8,lds_staged=true,lds_padded=true | generated HIP MFMA kernel navatala_transformer_tiled_gemm_f16_mfma_cta64_shared_padded | rocBLAS rocblas_gemm_ex F16 inputs/F32 accumulation | mfma_f16 | hip_mfma_gfx942_64x64x8_f16_f32_cta64_shared_padded | pass | 0.008092 | 0.023761 | 0.341x | 1.19209e-07 |
| GEMM_F16_MFMA_CTA64_SHARED_PADDED | m=512,n=512,k=512,compute=F32,cta=64x64x8,lds_staged=true,lds_padded=true | generated HIP MFMA kernel navatala_transformer_tiled_gemm_f16_mfma_cta64_shared_padded | rocBLAS rocblas_gemm_ex F16 inputs/F32 accumulation | mfma_f16 | hip_mfma_gfx942_64x64x8_f16_f32_cta64_shared_padded | pass | 0.027301 | 0.025690 | 1.063x | 1.01328e-06 |
| GEMM_F16_MFMA_CTA64_SHARED_PADDED | m=1024,n=1024,k=1024,compute=F32,cta=64x64x8,lds_staged=true,lds_padded=true | generated HIP MFMA kernel navatala_transformer_tiled_gemm_f16_mfma_cta64_shared_padded | rocBLAS rocblas_gemm_ex F16 inputs/F32 accumulation | mfma_f16 | hip_mfma_gfx942_64x64x8_f16_f32_cta64_shared_padded | pass | 0.073313 | 0.042912 | 1.708x | 2.14577e-06 |
| GEMM_F16_MFMA_CTA64_PIPELINED | m=128,n=128,k=128,compute=F32,cta=64x64x8,lds_staged=true,two_slot=true,r6_panel_copy=true | generated HIP MFMA kernel navatala_transformer_tiled_gemm_f16_mfma_cta64_pipelined | rocBLAS rocblas_gemm_ex F16 inputs/F32 accumulation | mfma_f16 | hip_mfma_gfx942_64x64x8_f16_f32_cta64_pipelined | pass | 0.009071 | 0.023759 | 0.382x | 1.19209e-07 |
| GEMM_F16_MFMA_CTA64_PIPELINED | m=512,n=512,k=512,compute=F32,cta=64x64x8,lds_staged=true,two_slot=true,r6_panel_copy=true | generated HIP MFMA kernel navatala_transformer_tiled_gemm_f16_mfma_cta64_pipelined | rocBLAS rocblas_gemm_ex F16 inputs/F32 accumulation | mfma_f16 | hip_mfma_gfx942_64x64x8_f16_f32_cta64_pipelined | pass | 0.031763 | 0.025713 | 1.235x | 1.01328e-06 |
| GEMM_F16_MFMA_CTA64_PIPELINED | m=1024,n=1024,k=1024,compute=F32,cta=64x64x8,lds_staged=true,two_slot=true,r6_panel_copy=true | generated HIP MFMA kernel navatala_transformer_tiled_gemm_f16_mfma_cta64_pipelined | rocBLAS rocblas_gemm_ex F16 inputs/F32 accumulation | mfma_f16 | hip_mfma_gfx942_64x64x8_f16_f32_cta64_pipelined | pass | 0.084913 | 0.042828 | 1.983x | 2.14577e-06 |
| GEMM_F16_MFMA_CTA128 | m=128,n=128,k=128,compute=F32,cta=128x128x32 | generated HIP MFMA kernel navatala_transformer_tiled_gemm_f16_mfma_cta128 | rocBLAS rocblas_gemm_ex F16 inputs/F32 accumulation | mfma_f16 | hip_mfma_gfx942_128x128x32_f16_f32_cta128 | pass | 0.014708 | 0.023680 | 0.621x | 1.19209e-07 |
| GEMM_F16_MFMA_CTA128 | m=512,n=512,k=512,compute=F32,cta=128x128x32 | generated HIP MFMA kernel navatala_transformer_tiled_gemm_f16_mfma_cta128 | rocBLAS rocblas_gemm_ex F16 inputs/F32 accumulation | mfma_f16 | hip_mfma_gfx942_128x128x32_f16_f32_cta128 | pass | 0.051199 | 0.025708 | 1.992x | 1.01328e-06 |
| GEMM_F16_MFMA_CTA128 | m=1024,n=1024,k=1024,compute=F32,cta=128x128x32 | generated HIP MFMA kernel navatala_transformer_tiled_gemm_f16_mfma_cta128 | rocBLAS rocblas_gemm_ex F16 inputs/F32 accumulation | mfma_f16 | hip_mfma_gfx942_128x128x32_f16_f32_cta128 | pass | 0.146690 | 0.042709 | 3.435x | 2.14577e-06 |
| CSR_SPMV_F32 | rows=16384,rowNnz=7,nnz=114688 | generated HIP kernel navatala_graph_spmv_weighted_f32 | rocSPARSE rocsparse_spmv | scalar | thread_per_row | pass | 0.003135 | 0.004183 | 0.750x | 1.49012e-08 |
| CSR_SPMV_F32 | rows=262144,rowNnz=7,nnz=1835008 | generated HIP kernel navatala_graph_spmv_weighted_f32 | rocSPARSE rocsparse_spmv | scalar | thread_per_row | pass | 0.006456 | 0.005213 | 1.238x | 2.98023e-08 |
| CSR_SPMV_F32 | rows=1048576,rowNnz=7,nnz=7340032 | generated HIP kernel navatala_graph_spmv_weighted_f32 | rocSPARSE rocsparse_spmv | scalar | thread_per_row | pass | 0.049892 | 0.021656 | 2.304x | 1.49012e-08 |
| CSR_SPMV_F32 | rows=262144,rowNnz=15,nnz=3932160 | generated HIP kernel navatala_graph_spmv_weighted_f32 | rocSPARSE rocsparse_spmv | scalar | thread_per_row | pass | 0.024021 | 0.011006 | 2.182x | 4.47035e-08 |
| CSR_SPMV_F32 | rows=262144,rowNnz=27,nnz=7077888 | generated HIP kernel navatala_graph_spmv_weighted_subgroup_f32 | rocSPARSE rocsparse_spmv | scalar | subgroup_per_row | pass | 0.044208 | 0.020369 | 2.170x | 5.96046e-08 |
| HIPSPARSELT_STRUCTURED_GEMM_F16 | m=128,n=128,k=128,sparsity=50%,compute=F32 | hipSPARSELt prune/compress/matmul path for SparseLt_StructuredMatmul | rocBLAS rocblas_hgemm dense pruned-A reference | vendor_library | vendor_dispatch | pass | 0.012365 | 0.009892 | 1.250x | 0 |
| HIPSPARSELT_STRUCTURED_GEMM_F16 | m=512,n=512,k=512,sparsity=50%,compute=F32 | hipSPARSELt prune/compress/matmul path for SparseLt_StructuredMatmul | rocBLAS rocblas_hgemm dense pruned-A reference | vendor_library | vendor_dispatch | pass | 0.012082 | 0.022758 | 0.531x | 0 |

## Interpretation Notes

- `generatedMeanMs` and `vendorMeanMs` are back-to-back throughput means per launch, not isolated launch latency.
- Correctness is measured against the vendor baseline in the benchmark harness for the listed shape.
- This report covers only the operations listed above; it is not a full HIP backend certification.
- hipSPARSELt structured rows compute 50% structured sparse matmul; dense MFMA rows compute ordinary dense matmul. Do not compare their ratios directly.
