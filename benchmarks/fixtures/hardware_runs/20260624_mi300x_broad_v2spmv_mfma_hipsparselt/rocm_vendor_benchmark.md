# ROCm Vendor Benchmark Report

- Generated at: `2026-06-23T20:12:42Z`
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
| AXPY_F32 | n=65536 | Navatala HIP kernel navatala_sparse_axpy_f32 | rocBLAS rocblas_saxpy | scalar | portable_kernel | pass | 0.002933 | 0.003181 | 0.922x | 0 |
| AXPY_F32 | n=1048576 | Navatala HIP kernel navatala_sparse_axpy_f32 | rocBLAS rocblas_saxpy | scalar | portable_kernel | pass | 0.005716 | 0.005384 | 1.062x | 0 |
| AXPY_F32 | n=4194304 | Navatala HIP kernel navatala_sparse_axpy_f32 | rocBLAS rocblas_saxpy | scalar | portable_kernel | pass | 0.018003 | 0.021586 | 0.834x | 0 |
| GEMM_F32 | m=128,n=128,k=128 | Navatala HIP kernel navatala_transformer_tiled_gemm_f32 | rocBLAS rocblas_sgemm | scalar | portable_kernel | pass | 0.005471 | 0.009363 | 0.584x | 3.42727e-07 |
| GEMM_F32 | m=512,n=512,k=512 | Navatala HIP kernel navatala_transformer_tiled_gemm_f32 | rocBLAS rocblas_sgemm | scalar | portable_kernel | pass | 0.027199 | 0.009566 | 2.843x | 8.04663e-07 |
| GEMM_F32 | m=1024,n=1024,k=1024 | Navatala HIP kernel navatala_transformer_tiled_gemm_f32 | rocBLAS rocblas_sgemm | scalar | portable_kernel | pass | 0.184931 | 0.026186 | 7.062x | 1.84774e-06 |
| GEMM_F16_PORTABLE_F32OUT | m=128,n=128,k=128,output=F32,compute=F32 | Navatala HIP kernel navatala_transformer_tiled_gemm_f16_f32_out | rocBLAS rocblas_gemm_ex F16 input/F32 output/F32 accumulation | scalar | portable_f16_f32out_tiled | pass | 0.006146 | 0.024112 | 0.255x | 2.38419e-07 |
| GEMM_F16_PORTABLE_F32OUT | m=512,n=512,k=512,output=F32,compute=F32 | Navatala HIP kernel navatala_transformer_tiled_gemm_f16_f32_out | rocBLAS rocblas_gemm_ex F16 input/F32 output/F32 accumulation | scalar | portable_f16_f32out_tiled | pass | 0.027265 | 0.027652 | 0.986x | 2.02656e-06 |
| GEMM_F16_PORTABLE_F32OUT | m=1024,n=1024,k=1024,output=F32,compute=F32 | Navatala HIP kernel navatala_transformer_tiled_gemm_f16_f32_out | rocBLAS rocblas_gemm_ex F16 input/F32 output/F32 accumulation | scalar | portable_f16_f32out_tiled | pass | 0.185571 | 0.046312 | 4.007x | 2.86102e-06 |
| GEMM_F16_MFMA | m=128,n=128,k=128,compute=F32 | Navatala HIP MFMA kernel navatala_transformer_tiled_gemm_f16_mfma_k_loop | rocBLAS rocblas_gemm_ex F16 inputs/F32 accumulation | mfma_f16 | hip_mfma_gfx942_32x32x8_f16_f32_k_loop | pass | 0.006111 | 0.023718 | 0.258x | 1.19209e-07 |
| GEMM_F16_MFMA | m=512,n=512,k=512,compute=F32 | Navatala HIP MFMA kernel navatala_transformer_tiled_gemm_f16_mfma_k_loop | rocBLAS rocblas_gemm_ex F16 inputs/F32 accumulation | mfma_f16 | hip_mfma_gfx942_32x32x8_f16_f32_k_loop | pass | 0.019416 | 0.025779 | 0.753x | 1.01328e-06 |
| GEMM_F16_MFMA | m=1024,n=1024,k=1024,compute=F32 | Navatala HIP MFMA kernel navatala_transformer_tiled_gemm_f16_mfma_k_loop | rocBLAS rocblas_gemm_ex F16 inputs/F32 accumulation | mfma_f16 | hip_mfma_gfx942_32x32x8_f16_f32_k_loop | pass | 0.107539 | 0.042743 | 2.516x | 2.14577e-06 |
| GEMM_F16_MFMA_CTA64_DIRECT | m=128,n=128,k=128,compute=F32,cta=64x64x8,direct_load=true | Navatala HIP MFMA kernel navatala_transformer_tiled_gemm_f16_mfma_cta64_direct | rocBLAS rocblas_gemm_ex F16 inputs/F32 accumulation | mfma_f16 | hip_mfma_gfx942_64x64x8_f16_f32_cta64_direct | pass | 0.012141 | 0.023722 | 0.512x | 1.19209e-07 |
| GEMM_F16_MFMA_CTA64_DIRECT | m=512,n=512,k=512,compute=F32,cta=64x64x8,direct_load=true | Navatala HIP MFMA kernel navatala_transformer_tiled_gemm_f16_mfma_cta64_direct | rocBLAS rocblas_gemm_ex F16 inputs/F32 accumulation | mfma_f16 | hip_mfma_gfx942_64x64x8_f16_f32_cta64_direct | pass | 0.044229 | 0.025689 | 1.722x | 1.01328e-06 |
| GEMM_F16_MFMA_CTA64_DIRECT | m=1024,n=1024,k=1024,compute=F32,cta=64x64x8,direct_load=true | Navatala HIP MFMA kernel navatala_transformer_tiled_gemm_f16_mfma_cta64_direct | rocBLAS rocblas_gemm_ex F16 inputs/F32 accumulation | mfma_f16 | hip_mfma_gfx942_64x64x8_f16_f32_cta64_direct | pass | 0.108795 | 0.042731 | 2.546x | 2.14577e-06 |
| GEMM_F16_MFMA_CTA64_SHARED | m=128,n=128,k=128,compute=F32,cta=64x64x8,lds_staged=true | Navatala HIP MFMA kernel navatala_transformer_tiled_gemm_f16_mfma_cta64_shared | rocBLAS rocblas_gemm_ex F16 inputs/F32 accumulation | mfma_f16 | hip_mfma_gfx942_64x64x8_f16_f32_cta64_shared | pass | 0.006334 | 0.023753 | 0.267x | 1.19209e-07 |
| GEMM_F16_MFMA_CTA64_SHARED | m=512,n=512,k=512,compute=F32,cta=64x64x8,lds_staged=true | Navatala HIP MFMA kernel navatala_transformer_tiled_gemm_f16_mfma_cta64_shared | rocBLAS rocblas_gemm_ex F16 inputs/F32 accumulation | mfma_f16 | hip_mfma_gfx942_64x64x8_f16_f32_cta64_shared | pass | 0.020580 | 0.025724 | 0.800x | 1.01328e-06 |
| GEMM_F16_MFMA_CTA64_SHARED | m=1024,n=1024,k=1024,compute=F32,cta=64x64x8,lds_staged=true | Navatala HIP MFMA kernel navatala_transformer_tiled_gemm_f16_mfma_cta64_shared | rocBLAS rocblas_gemm_ex F16 inputs/F32 accumulation | mfma_f16 | hip_mfma_gfx942_64x64x8_f16_f32_cta64_shared | pass | 0.059883 | 0.042806 | 1.399x | 2.14577e-06 |
| GEMM_F16_MFMA_CTA64_SHARED_EARLY_BARRIER | m=128,n=128,k=128,compute=F32,cta=64x64x8,lds_staged=true,early_barrier=true | Navatala HIP MFMA kernel navatala_transformer_tiled_gemm_f16_mfma_cta64_shared_early_barrier | rocBLAS rocblas_gemm_ex F16 inputs/F32 accumulation | mfma_f16 | hip_mfma_gfx942_64x64x8_f16_f32_cta64_shared_early_barrier | pass | 0.006413 | 0.023916 | 0.268x | 1.19209e-07 |
| GEMM_F16_MFMA_CTA64_SHARED_EARLY_BARRIER | m=512,n=512,k=512,compute=F32,cta=64x64x8,lds_staged=true,early_barrier=true | Navatala HIP MFMA kernel navatala_transformer_tiled_gemm_f16_mfma_cta64_shared_early_barrier | rocBLAS rocblas_gemm_ex F16 inputs/F32 accumulation | mfma_f16 | hip_mfma_gfx942_64x64x8_f16_f32_cta64_shared_early_barrier | pass | 0.020579 | 0.025761 | 0.799x | 1.01328e-06 |
| GEMM_F16_MFMA_CTA64_SHARED_EARLY_BARRIER | m=1024,n=1024,k=1024,compute=F32,cta=64x64x8,lds_staged=true,early_barrier=true | Navatala HIP MFMA kernel navatala_transformer_tiled_gemm_f16_mfma_cta64_shared_early_barrier | rocBLAS rocblas_gemm_ex F16 inputs/F32 accumulation | mfma_f16 | hip_mfma_gfx942_64x64x8_f16_f32_cta64_shared_early_barrier | pass | 0.059911 | 0.042825 | 1.399x | 2.14577e-06 |
| GEMM_F16_MFMA_CTA64_SHARED_PADDED | m=128,n=128,k=128,compute=F32,cta=64x64x8,lds_staged=true,lds_padded=true | Navatala HIP MFMA kernel navatala_transformer_tiled_gemm_f16_mfma_cta64_shared_padded | rocBLAS rocblas_gemm_ex F16 inputs/F32 accumulation | mfma_f16 | hip_mfma_gfx942_64x64x8_f16_f32_cta64_shared_padded | pass | 0.008002 | 0.023886 | 0.335x | 1.19209e-07 |
| GEMM_F16_MFMA_CTA64_SHARED_PADDED | m=512,n=512,k=512,compute=F32,cta=64x64x8,lds_staged=true,lds_padded=true | Navatala HIP MFMA kernel navatala_transformer_tiled_gemm_f16_mfma_cta64_shared_padded | rocBLAS rocblas_gemm_ex F16 inputs/F32 accumulation | mfma_f16 | hip_mfma_gfx942_64x64x8_f16_f32_cta64_shared_padded | pass | 0.027342 | 0.025795 | 1.060x | 1.01328e-06 |
| GEMM_F16_MFMA_CTA64_SHARED_PADDED | m=1024,n=1024,k=1024,compute=F32,cta=64x64x8,lds_staged=true,lds_padded=true | Navatala HIP MFMA kernel navatala_transformer_tiled_gemm_f16_mfma_cta64_shared_padded | rocBLAS rocblas_gemm_ex F16 inputs/F32 accumulation | mfma_f16 | hip_mfma_gfx942_64x64x8_f16_f32_cta64_shared_padded | pass | 0.073241 | 0.042777 | 1.712x | 2.14577e-06 |
| GEMM_F16_MFMA_CTA64_PIPELINED | m=128,n=128,k=128,compute=F32,cta=64x64x8,lds_staged=true,two_slot=true,r6_panel_copy=true | Navatala HIP MFMA kernel navatala_transformer_tiled_gemm_f16_mfma_cta64_pipelined | rocBLAS rocblas_gemm_ex F16 inputs/F32 accumulation | mfma_f16 | hip_mfma_gfx942_64x64x8_f16_f32_cta64_pipelined | pass | 0.007163 | 0.023880 | 0.300x | 1.19209e-07 |
| GEMM_F16_MFMA_CTA64_PIPELINED | m=512,n=512,k=512,compute=F32,cta=64x64x8,lds_staged=true,two_slot=true,r6_panel_copy=true | Navatala HIP MFMA kernel navatala_transformer_tiled_gemm_f16_mfma_cta64_pipelined | rocBLAS rocblas_gemm_ex F16 inputs/F32 accumulation | mfma_f16 | hip_mfma_gfx942_64x64x8_f16_f32_cta64_pipelined | pass | 0.022757 | 0.025829 | 0.881x | 1.01328e-06 |
| GEMM_F16_MFMA_CTA64_PIPELINED | m=1024,n=1024,k=1024,compute=F32,cta=64x64x8,lds_staged=true,two_slot=true,r6_panel_copy=true | Navatala HIP MFMA kernel navatala_transformer_tiled_gemm_f16_mfma_cta64_pipelined | rocBLAS rocblas_gemm_ex F16 inputs/F32 accumulation | mfma_f16 | hip_mfma_gfx942_64x64x8_f16_f32_cta64_pipelined | pass | 0.062435 | 0.042727 | 1.461x | 2.14577e-06 |
| GEMM_F16_MFMA_CTA128 | m=128,n=128,k=128,compute=F32,cta=128x128x32 | Navatala HIP MFMA kernel navatala_transformer_tiled_gemm_f16_mfma_cta128 | rocBLAS rocblas_gemm_ex F16 inputs/F32 accumulation | mfma_f16 | hip_mfma_gfx942_128x128x32_f16_f32_cta128 | pass | 0.008561 | 0.023924 | 0.358x | 1.19209e-07 |
| GEMM_F16_MFMA_CTA128 | m=512,n=512,k=512,compute=F32,cta=128x128x32 | Navatala HIP MFMA kernel navatala_transformer_tiled_gemm_f16_mfma_cta128 | rocBLAS rocblas_gemm_ex F16 inputs/F32 accumulation | mfma_f16 | hip_mfma_gfx942_128x128x32_f16_f32_cta128 | pass | 0.026300 | 0.025776 | 1.020x | 1.01328e-06 |
| GEMM_F16_MFMA_CTA128 | m=1024,n=1024,k=1024,compute=F32,cta=128x128x32 | Navatala HIP MFMA kernel navatala_transformer_tiled_gemm_f16_mfma_cta128 | rocBLAS rocblas_gemm_ex F16 inputs/F32 accumulation | mfma_f16 | hip_mfma_gfx942_128x128x32_f16_f32_cta128 | pass | 0.053409 | 0.042726 | 1.250x | 2.14577e-06 |
| CSR_SPMV_F32 | rows=16384,rowNnz=7,nnz=114688 | Navatala HIP kernel navatala_graph_spmv_weighted_f32 | rocSPARSE rocsparse_v2_spmv | scalar | thread_per_row | pass | 0.002992 | 0.002726 | 1.098x | 1.49012e-08 |
| CSR_SPMV_F32 | rows=262144,rowNnz=7,nnz=1835008 | Navatala HIP kernel navatala_graph_spmv_weighted_f32 | rocSPARSE rocsparse_v2_spmv | scalar | thread_per_row | pass | 0.006483 | 0.004437 | 1.461x | 2.98023e-08 |
| CSR_SPMV_F32 | rows=1048576,rowNnz=7,nnz=7340032 | Navatala HIP kernel navatala_graph_spmv_weighted_f32 | rocSPARSE rocsparse_v2_spmv | scalar | thread_per_row | pass | 0.049713 | 0.020497 | 2.425x | 1.49012e-08 |
| CSR_SPMV_F32 | rows=262144,rowNnz=15,nnz=3932160 | Navatala HIP kernel navatala_graph_spmv_weighted_f32 | rocSPARSE rocsparse_v2_spmv | scalar | thread_per_row | pass | 0.024228 | 0.010192 | 2.377x | 5.96046e-08 |
| CSR_SPMV_F32 | rows=262144,rowNnz=27,nnz=7077888 | Navatala HIP kernel navatala_graph_spmv_weighted_subgroup_f32 | rocSPARSE rocsparse_v2_spmv | scalar | subgroup_per_row | pass | 0.044558 | 0.019642 | 2.269x | 5.96046e-08 |
| HIPSPARSELT_STRUCTURED_GEMM_F16 | m=128,n=128,k=128,sparsity=50%,compute=F32 | hipSPARSELt prune/compress/matmul path for SparseLt_StructuredMatmul | rocBLAS rocblas_hgemm dense pruned-A reference | vendor_library | vendor_dispatch | pass | 0.012071 | 0.009587 | 1.259x | 0 |
| HIPSPARSELT_STRUCTURED_GEMM_F16 | m=512,n=512,k=512,sparsity=50%,compute=F32 | hipSPARSELt prune/compress/matmul path for SparseLt_StructuredMatmul | rocBLAS rocblas_hgemm dense pruned-A reference | vendor_library | vendor_dispatch | pass | 0.012455 | 0.019395 | 0.642x | 0 |

## Interpretation Notes

- `generatedMeanMs` and `vendorMeanMs` are back-to-back throughput means per launch, not isolated launch latency.
- Correctness is measured against the vendor baseline in the benchmark harness for the listed shape.
- This report covers only the operations listed above; it is not a full HIP backend certification.
- hipSPARSELt structured rows compute 50% structured sparse matmul; dense MFMA rows compute ordinary dense matmul. Do not compare their ratios directly.
