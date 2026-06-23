# ROCm Vendor Benchmark Report

- Generated at: `2026-06-23T20:00:31Z`
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
| AXPY_F32 | n=65536 | generated HIP kernel navatala_sparse_axpy_f32 | rocBLAS rocblas_saxpy | scalar | portable_kernel | pass | 0.003182 | 0.003309 | 0.962x | 0 |
| AXPY_F32 | n=1048576 | generated HIP kernel navatala_sparse_axpy_f32 | rocBLAS rocblas_saxpy | scalar | portable_kernel | pass | 0.005728 | 0.005294 | 1.082x | 0 |
| AXPY_F32 | n=4194304 | generated HIP kernel navatala_sparse_axpy_f32 | rocBLAS rocblas_saxpy | scalar | portable_kernel | pass | 0.018342 | 0.022221 | 0.825x | 0 |
| GEMM_F32 | m=128,n=128,k=128 | generated HIP kernel navatala_transformer_tiled_gemm_f32 | rocBLAS rocblas_sgemm | scalar | portable_kernel | pass | 0.005388 | 0.008679 | 0.621x | 3.42727e-07 |
| GEMM_F32 | m=512,n=512,k=512 | generated HIP kernel navatala_transformer_tiled_gemm_f32 | rocBLAS rocblas_sgemm | scalar | portable_kernel | pass | 0.026548 | 0.009661 | 2.748x | 8.04663e-07 |
| GEMM_F32 | m=1024,n=1024,k=1024 | generated HIP kernel navatala_transformer_tiled_gemm_f32 | rocBLAS rocblas_sgemm | scalar | portable_kernel | pass | 0.183389 | 0.026499 | 6.920x | 1.84774e-06 |
| GEMM_F16_PORTABLE_F32OUT | m=128,n=128,k=128,output=F32,compute=F32 | generated HIP kernel navatala_transformer_tiled_gemm_f16_f32_out | rocBLAS rocblas_gemm_ex F16 input/F32 output/F32 accumulation | scalar | portable_f16_f32out_tiled | pass | 0.006143 | 0.024197 | 0.254x | 2.38419e-07 |
| GEMM_F16_PORTABLE_F32OUT | m=512,n=512,k=512,output=F32,compute=F32 | generated HIP kernel navatala_transformer_tiled_gemm_f16_f32_out | rocBLAS rocblas_gemm_ex F16 input/F32 output/F32 accumulation | scalar | portable_f16_f32out_tiled | pass | 0.027197 | 0.026022 | 1.045x | 2.02656e-06 |
| GEMM_F16_PORTABLE_F32OUT | m=1024,n=1024,k=1024,output=F32,compute=F32 | generated HIP kernel navatala_transformer_tiled_gemm_f16_f32_out | rocBLAS rocblas_gemm_ex F16 input/F32 output/F32 accumulation | scalar | portable_f16_f32out_tiled | pass | 0.180157 | 0.047827 | 3.767x | 2.86102e-06 |
| GEMM_F16_MFMA | m=128,n=128,k=128,compute=F32 | generated HIP MFMA kernel navatala_transformer_tiled_gemm_f16_mfma_k_loop | rocBLAS rocblas_gemm_ex F16 inputs/F32 accumulation | mfma_f16 | hip_mfma_gfx942_32x32x8_f16_f32_k_loop | pass | 0.006089 | 0.023875 | 0.255x | 1.19209e-07 |
| GEMM_F16_MFMA | m=512,n=512,k=512,compute=F32 | generated HIP MFMA kernel navatala_transformer_tiled_gemm_f16_mfma_k_loop | rocBLAS rocblas_gemm_ex F16 inputs/F32 accumulation | mfma_f16 | hip_mfma_gfx942_32x32x8_f16_f32_k_loop | pass | 0.019391 | 0.025669 | 0.755x | 1.01328e-06 |
| GEMM_F16_MFMA | m=1024,n=1024,k=1024,compute=F32 | generated HIP MFMA kernel navatala_transformer_tiled_gemm_f16_mfma_k_loop | rocBLAS rocblas_gemm_ex F16 inputs/F32 accumulation | mfma_f16 | hip_mfma_gfx942_32x32x8_f16_f32_k_loop | pass | 0.107334 | 0.042882 | 2.503x | 2.14577e-06 |
| GEMM_F16_MFMA_CTA64_DIRECT | m=128,n=128,k=128,compute=F32,cta=64x64x8,direct_load=true | generated HIP MFMA kernel navatala_transformer_tiled_gemm_f16_mfma_cta64_direct | rocBLAS rocblas_gemm_ex F16 inputs/F32 accumulation | mfma_f16 | hip_mfma_gfx942_64x64x8_f16_f32_cta64_direct | pass | 0.012070 | 0.023865 | 0.506x | 1.19209e-07 |
| GEMM_F16_MFMA_CTA64_DIRECT | m=512,n=512,k=512,compute=F32,cta=64x64x8,direct_load=true | generated HIP MFMA kernel navatala_transformer_tiled_gemm_f16_mfma_cta64_direct | rocBLAS rocblas_gemm_ex F16 inputs/F32 accumulation | mfma_f16 | hip_mfma_gfx942_64x64x8_f16_f32_cta64_direct | pass | 0.044258 | 0.025675 | 1.724x | 1.01328e-06 |
| GEMM_F16_MFMA_CTA64_DIRECT | m=1024,n=1024,k=1024,compute=F32,cta=64x64x8,direct_load=true | generated HIP MFMA kernel navatala_transformer_tiled_gemm_f16_mfma_cta64_direct | rocBLAS rocblas_gemm_ex F16 inputs/F32 accumulation | mfma_f16 | hip_mfma_gfx942_64x64x8_f16_f32_cta64_direct | pass | 0.108759 | 0.042839 | 2.539x | 2.14577e-06 |
| GEMM_F16_MFMA_CTA64_SHARED | m=128,n=128,k=128,compute=F32,cta=64x64x8,lds_staged=true | generated HIP MFMA kernel navatala_transformer_tiled_gemm_f16_mfma_cta64_shared | rocBLAS rocblas_gemm_ex F16 inputs/F32 accumulation | mfma_f16 | hip_mfma_gfx942_64x64x8_f16_f32_cta64_shared | pass | 0.006248 | 0.023853 | 0.262x | 1.19209e-07 |
| GEMM_F16_MFMA_CTA64_SHARED | m=512,n=512,k=512,compute=F32,cta=64x64x8,lds_staged=true | generated HIP MFMA kernel navatala_transformer_tiled_gemm_f16_mfma_cta64_shared | rocBLAS rocblas_gemm_ex F16 inputs/F32 accumulation | mfma_f16 | hip_mfma_gfx942_64x64x8_f16_f32_cta64_shared | pass | 0.020594 | 0.025680 | 0.802x | 1.01328e-06 |
| GEMM_F16_MFMA_CTA64_SHARED | m=1024,n=1024,k=1024,compute=F32,cta=64x64x8,lds_staged=true | generated HIP MFMA kernel navatala_transformer_tiled_gemm_f16_mfma_cta64_shared | rocBLAS rocblas_gemm_ex F16 inputs/F32 accumulation | mfma_f16 | hip_mfma_gfx942_64x64x8_f16_f32_cta64_shared | pass | 0.059771 | 0.042855 | 1.395x | 2.14577e-06 |
| GEMM_F16_MFMA_CTA64_SHARED_EARLY_BARRIER | m=128,n=128,k=128,compute=F32,cta=64x64x8,lds_staged=true,early_barrier=true | generated HIP MFMA kernel navatala_transformer_tiled_gemm_f16_mfma_cta64_shared_early_barrier | rocBLAS rocblas_gemm_ex F16 inputs/F32 accumulation | mfma_f16 | hip_mfma_gfx942_64x64x8_f16_f32_cta64_shared_early_barrier | pass | 0.006325 | 0.024052 | 0.263x | 1.19209e-07 |
| GEMM_F16_MFMA_CTA64_SHARED_EARLY_BARRIER | m=512,n=512,k=512,compute=F32,cta=64x64x8,lds_staged=true,early_barrier=true | generated HIP MFMA kernel navatala_transformer_tiled_gemm_f16_mfma_cta64_shared_early_barrier | rocBLAS rocblas_gemm_ex F16 inputs/F32 accumulation | mfma_f16 | hip_mfma_gfx942_64x64x8_f16_f32_cta64_shared_early_barrier | pass | 0.020586 | 0.025799 | 0.798x | 1.01328e-06 |
| GEMM_F16_MFMA_CTA64_SHARED_EARLY_BARRIER | m=1024,n=1024,k=1024,compute=F32,cta=64x64x8,lds_staged=true,early_barrier=true | generated HIP MFMA kernel navatala_transformer_tiled_gemm_f16_mfma_cta64_shared_early_barrier | rocBLAS rocblas_gemm_ex F16 inputs/F32 accumulation | mfma_f16 | hip_mfma_gfx942_64x64x8_f16_f32_cta64_shared_early_barrier | pass | 0.059834 | 0.042885 | 1.395x | 2.14577e-06 |
| GEMM_F16_MFMA_CTA64_SHARED_PADDED | m=128,n=128,k=128,compute=F32,cta=64x64x8,lds_staged=true,lds_padded=true | generated HIP MFMA kernel navatala_transformer_tiled_gemm_f16_mfma_cta64_shared_padded | rocBLAS rocblas_gemm_ex F16 inputs/F32 accumulation | mfma_f16 | hip_mfma_gfx942_64x64x8_f16_f32_cta64_shared_padded | pass | 0.007933 | 0.023853 | 0.333x | 1.19209e-07 |
| GEMM_F16_MFMA_CTA64_SHARED_PADDED | m=512,n=512,k=512,compute=F32,cta=64x64x8,lds_staged=true,lds_padded=true | generated HIP MFMA kernel navatala_transformer_tiled_gemm_f16_mfma_cta64_shared_padded | rocBLAS rocblas_gemm_ex F16 inputs/F32 accumulation | mfma_f16 | hip_mfma_gfx942_64x64x8_f16_f32_cta64_shared_padded | pass | 0.027324 | 0.025752 | 1.061x | 1.01328e-06 |
| GEMM_F16_MFMA_CTA64_SHARED_PADDED | m=1024,n=1024,k=1024,compute=F32,cta=64x64x8,lds_staged=true,lds_padded=true | generated HIP MFMA kernel navatala_transformer_tiled_gemm_f16_mfma_cta64_shared_padded | rocBLAS rocblas_gemm_ex F16 inputs/F32 accumulation | mfma_f16 | hip_mfma_gfx942_64x64x8_f16_f32_cta64_shared_padded | pass | 0.073286 | 0.042721 | 1.715x | 2.14577e-06 |
| GEMM_F16_MFMA_CTA64_PIPELINED | m=128,n=128,k=128,compute=F32,cta=64x64x8,lds_staged=true,two_slot=true,r6_panel_copy=true | generated HIP MFMA kernel navatala_transformer_tiled_gemm_f16_mfma_cta64_pipelined | rocBLAS rocblas_gemm_ex F16 inputs/F32 accumulation | mfma_f16 | hip_mfma_gfx942_64x64x8_f16_f32_cta64_pipelined | pass | 0.007524 | 0.023842 | 0.316x | 1.19209e-07 |
| GEMM_F16_MFMA_CTA64_PIPELINED | m=512,n=512,k=512,compute=F32,cta=64x64x8,lds_staged=true,two_slot=true,r6_panel_copy=true | generated HIP MFMA kernel navatala_transformer_tiled_gemm_f16_mfma_cta64_pipelined | rocBLAS rocblas_gemm_ex F16 inputs/F32 accumulation | mfma_f16 | hip_mfma_gfx942_64x64x8_f16_f32_cta64_pipelined | pass | 0.022767 | 0.025784 | 0.883x | 1.01328e-06 |
| GEMM_F16_MFMA_CTA64_PIPELINED | m=1024,n=1024,k=1024,compute=F32,cta=64x64x8,lds_staged=true,two_slot=true,r6_panel_copy=true | generated HIP MFMA kernel navatala_transformer_tiled_gemm_f16_mfma_cta64_pipelined | rocBLAS rocblas_gemm_ex F16 inputs/F32 accumulation | mfma_f16 | hip_mfma_gfx942_64x64x8_f16_f32_cta64_pipelined | pass | 0.061313 | 0.042930 | 1.428x | 2.14577e-06 |
| GEMM_F16_MFMA_CTA128 | m=128,n=128,k=128,compute=F32,cta=128x128x32 | generated HIP MFMA kernel navatala_transformer_tiled_gemm_f16_mfma_cta128 | rocBLAS rocblas_gemm_ex F16 inputs/F32 accumulation | mfma_f16 | hip_mfma_gfx942_128x128x32_f16_f32_cta128 | pass | 0.008604 | 0.023841 | 0.361x | 1.19209e-07 |
| GEMM_F16_MFMA_CTA128 | m=512,n=512,k=512,compute=F32,cta=128x128x32 | generated HIP MFMA kernel navatala_transformer_tiled_gemm_f16_mfma_cta128 | rocBLAS rocblas_gemm_ex F16 inputs/F32 accumulation | mfma_f16 | hip_mfma_gfx942_128x128x32_f16_f32_cta128 | pass | 0.026327 | 0.026231 | 1.004x | 1.01328e-06 |
| GEMM_F16_MFMA_CTA128 | m=1024,n=1024,k=1024,compute=F32,cta=128x128x32 | generated HIP MFMA kernel navatala_transformer_tiled_gemm_f16_mfma_cta128 | rocBLAS rocblas_gemm_ex F16 inputs/F32 accumulation | mfma_f16 | hip_mfma_gfx942_128x128x32_f16_f32_cta128 | pass | 0.053358 | 0.042992 | 1.241x | 2.14577e-06 |
| CSR_SPMV_F32 | rows=16384,rowNnz=7,nnz=114688 | generated HIP kernel navatala_graph_spmv_weighted_f32 | rocSPARSE rocsparse_spmv | scalar | thread_per_row | pass | 0.002963 | 0.004031 | 0.735x | 1.49012e-08 |
| CSR_SPMV_F32 | rows=262144,rowNnz=7,nnz=1835008 | generated HIP kernel navatala_graph_spmv_weighted_f32 | rocSPARSE rocsparse_spmv | scalar | thread_per_row | pass | 0.006491 | 0.005309 | 1.223x | 2.98023e-08 |
| CSR_SPMV_F32 | rows=1048576,rowNnz=7,nnz=7340032 | generated HIP kernel navatala_graph_spmv_weighted_f32 | rocSPARSE rocsparse_spmv | scalar | thread_per_row | pass | 0.050800 | 0.021462 | 2.367x | 1.49012e-08 |
| CSR_SPMV_F32 | rows=262144,rowNnz=15,nnz=3932160 | generated HIP kernel navatala_graph_spmv_weighted_f32 | rocSPARSE rocsparse_spmv | scalar | thread_per_row | pass | 0.024185 | 0.011355 | 2.130x | 4.47035e-08 |
| CSR_SPMV_F32 | rows=262144,rowNnz=27,nnz=7077888 | generated HIP kernel navatala_graph_spmv_weighted_subgroup_f32 | rocSPARSE rocsparse_spmv | scalar | subgroup_per_row | pass | 0.043721 | 0.020461 | 2.137x | 5.96046e-08 |
| HIPSPARSELT_STRUCTURED_GEMM_F16 | m=128,n=128,k=128,sparsity=50%,compute=F32 | hipSPARSELt prune/compress/matmul path for SparseLt_StructuredMatmul | rocBLAS rocblas_hgemm dense pruned-A reference | vendor_library | vendor_dispatch | pass | 0.011660 | 0.010350 | 1.127x | 0 |
| HIPSPARSELT_STRUCTURED_GEMM_F16 | m=512,n=512,k=512,sparsity=50%,compute=F32 | hipSPARSELt prune/compress/matmul path for SparseLt_StructuredMatmul | rocBLAS rocblas_hgemm dense pruned-A reference | vendor_library | vendor_dispatch | pass | 0.012567 | 0.022986 | 0.547x | 0 |

## Interpretation Notes

- `generatedMeanMs` and `vendorMeanMs` are back-to-back throughput means per launch, not isolated launch latency.
- Correctness is measured against the vendor baseline in the benchmark harness for the listed shape.
- This report covers only the operations listed above; it is not a full HIP backend certification.
- hipSPARSELt structured rows compute 50% structured sparse matmul; dense MFMA rows compute ordinary dense matmul. Do not compare their ratios directly.
