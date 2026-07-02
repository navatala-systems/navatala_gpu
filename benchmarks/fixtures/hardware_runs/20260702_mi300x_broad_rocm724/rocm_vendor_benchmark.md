# ROCm Vendor Benchmark Report

- Generated at: `2026-07-02T07:53:34Z`
- Source JSON: `rocm_vendor_benchmark.json`
- Schema: `navatala_gpu.rocm_vendor_benchmark.v1`
- Timing mode: `back_to_back_throughput_mean_per_launch`
- Quick mode: `false`
- Matrix set: `broad`
- Iterations: `50`
- Warmup: `10`
- Device: `AMD Instinct MI300X VF`
- GCN arch: `gfx942:sramecc+:xnack-`
- Global memory: `196288 MiB`
- hipSPARSELt available: `true`
- hipSPARSELt mode: `vendor_benchmark`

## Results

| Operation | Shape | Generated path | Vendor baseline | Kernel class | Tuning path | Correctness | Generated mean ms | Vendor mean ms | Ratio | Max abs error |
| --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| AXPY_F32 | n=65536 | Navatala HIP kernel navatala_sparse_axpy_f32 | rocBLAS rocblas_saxpy | scalar | portable_kernel | pass | 0.002529 | 0.002466 | 1.025x | 0 |
| AXPY_F32 | n=1048576 | Navatala HIP kernel navatala_sparse_axpy_f32 | rocBLAS rocblas_saxpy | scalar | portable_kernel | pass | 0.005671 | 0.005600 | 1.013x | 0 |
| AXPY_F32 | n=4194304 | Navatala HIP kernel navatala_sparse_axpy_f32 | rocBLAS rocblas_saxpy | scalar | portable_kernel | pass | 0.018002 | 0.018354 | 0.981x | 0 |
| GEMM_F32 | m=128,n=128,k=128 | Navatala HIP kernel navatala_transformer_tiled_gemm_f32 | rocBLAS rocblas_sgemm | scalar | portable_kernel | pass | 0.005340 | 0.007072 | 0.755x | 3.42727e-07 |
| GEMM_F32 | m=512,n=512,k=512 | Navatala HIP kernel navatala_transformer_tiled_gemm_f32 | rocBLAS rocblas_sgemm | scalar | portable_kernel | pass | 0.026261 | 0.009374 | 2.802x | 8.04663e-07 |
| GEMM_F32 | m=1024,n=1024,k=1024 | Navatala HIP kernel navatala_transformer_tiled_gemm_f32 | rocBLAS rocblas_sgemm | scalar | portable_kernel | pass | 0.172018 | 0.024812 | 6.933x | 1.84774e-06 |
| GEMM_F16_PORTABLE_F32OUT | m=128,n=128,k=128,output=F32,compute=F32 | Navatala HIP kernel navatala_transformer_tiled_gemm_f16_f32_out | rocBLAS rocblas_gemm_ex F16 input/F32 output/F32 accumulation | scalar | portable_f16_f32out_tiled | pass | 0.005985 | 0.023987 | 0.250x | 2.38419e-07 |
| GEMM_F16_PORTABLE_F32OUT | m=512,n=512,k=512,output=F32,compute=F32 | Navatala HIP kernel navatala_transformer_tiled_gemm_f16_f32_out | rocBLAS rocblas_gemm_ex F16 input/F32 output/F32 accumulation | scalar | portable_f16_f32out_tiled | pass | 0.027224 | 0.026125 | 1.042x | 2.02656e-06 |
| GEMM_F16_PORTABLE_F32OUT | m=1024,n=1024,k=1024,output=F32,compute=F32 | Navatala HIP kernel navatala_transformer_tiled_gemm_f16_f32_out | rocBLAS rocblas_gemm_ex F16 input/F32 output/F32 accumulation | scalar | portable_f16_f32out_tiled | pass | 0.170832 | 0.043874 | 3.894x | 2.86102e-06 |
| GEMM_F16_MFMA | m=128,n=128,k=128,compute=F32 | Navatala HIP MFMA kernel navatala_transformer_tiled_gemm_f16_mfma_k_loop | rocBLAS rocblas_gemm_ex F16 inputs/F32 accumulation | mfma_f16 | hip_mfma_gfx942_32x32x8_f16_f32_k_loop | pass | 0.006011 | 0.023964 | 0.251x | 1.19209e-07 |
| GEMM_F16_MFMA | m=512,n=512,k=512,compute=F32 | Navatala HIP MFMA kernel navatala_transformer_tiled_gemm_f16_mfma_k_loop | rocBLAS rocblas_gemm_ex F16 inputs/F32 accumulation | mfma_f16 | hip_mfma_gfx942_32x32x8_f16_f32_k_loop | pass | 0.019332 | 0.025616 | 0.755x | 1.01328e-06 |
| GEMM_F16_MFMA | m=1024,n=1024,k=1024,compute=F32 | Navatala HIP MFMA kernel navatala_transformer_tiled_gemm_f16_mfma_k_loop | rocBLAS rocblas_gemm_ex F16 inputs/F32 accumulation | mfma_f16 | hip_mfma_gfx942_32x32x8_f16_f32_k_loop | pass | 0.107438 | 0.042597 | 2.522x | 2.14577e-06 |
| GEMM_F16_MFMA_CTA64_DIRECT | m=128,n=128,k=128,compute=F32,cta=64x64x8,direct_load=true | Navatala HIP MFMA kernel navatala_transformer_tiled_gemm_f16_mfma_cta64_direct | rocBLAS rocblas_gemm_ex F16 inputs/F32 accumulation | mfma_f16 | hip_mfma_gfx942_64x64x8_f16_f32_cta64_direct | pass | 0.012005 | 0.023945 | 0.501x | 1.19209e-07 |
| GEMM_F16_MFMA_CTA64_DIRECT | m=512,n=512,k=512,compute=F32,cta=64x64x8,direct_load=true | Navatala HIP MFMA kernel navatala_transformer_tiled_gemm_f16_mfma_cta64_direct | rocBLAS rocblas_gemm_ex F16 inputs/F32 accumulation | mfma_f16 | hip_mfma_gfx942_64x64x8_f16_f32_cta64_direct | pass | 0.044148 | 0.025639 | 1.722x | 1.01328e-06 |
| GEMM_F16_MFMA_CTA64_DIRECT | m=1024,n=1024,k=1024,compute=F32,cta=64x64x8,direct_load=true | Navatala HIP MFMA kernel navatala_transformer_tiled_gemm_f16_mfma_cta64_direct | rocBLAS rocblas_gemm_ex F16 inputs/F32 accumulation | mfma_f16 | hip_mfma_gfx942_64x64x8_f16_f32_cta64_direct | pass | 0.108031 | 0.042596 | 2.536x | 2.14577e-06 |
| GEMM_F16_MFMA_CTA64_SHARED | m=128,n=128,k=128,compute=F32,cta=64x64x8,lds_staged=true | Navatala HIP MFMA kernel navatala_transformer_tiled_gemm_f16_mfma_cta64_shared | rocBLAS rocblas_gemm_ex F16 inputs/F32 accumulation | mfma_f16 | hip_mfma_gfx942_64x64x8_f16_f32_cta64_shared | pass | 0.006084 | 0.023798 | 0.256x | 1.19209e-07 |
| GEMM_F16_MFMA_CTA64_SHARED | m=512,n=512,k=512,compute=F32,cta=64x64x8,lds_staged=true | Navatala HIP MFMA kernel navatala_transformer_tiled_gemm_f16_mfma_cta64_shared | rocBLAS rocblas_gemm_ex F16 inputs/F32 accumulation | mfma_f16 | hip_mfma_gfx942_64x64x8_f16_f32_cta64_shared | pass | 0.020521 | 0.025609 | 0.801x | 1.01328e-06 |
| GEMM_F16_MFMA_CTA64_SHARED | m=1024,n=1024,k=1024,compute=F32,cta=64x64x8,lds_staged=true | Navatala HIP MFMA kernel navatala_transformer_tiled_gemm_f16_mfma_cta64_shared | rocBLAS rocblas_gemm_ex F16 inputs/F32 accumulation | mfma_f16 | hip_mfma_gfx942_64x64x8_f16_f32_cta64_shared | pass | 0.060055 | 0.042751 | 1.405x | 2.14577e-06 |
| GEMM_F16_MFMA_CTA64_SHARED_EARLY_BARRIER | m=128,n=128,k=128,compute=F32,cta=64x64x8,lds_staged=true,early_barrier=true | Navatala HIP MFMA kernel navatala_transformer_tiled_gemm_f16_mfma_cta64_shared_early_barrier | rocBLAS rocblas_gemm_ex F16 inputs/F32 accumulation | mfma_f16 | hip_mfma_gfx942_64x64x8_f16_f32_cta64_shared_early_barrier | pass | 0.006243 | 0.023818 | 0.262x | 1.19209e-07 |
| GEMM_F16_MFMA_CTA64_SHARED_EARLY_BARRIER | m=512,n=512,k=512,compute=F32,cta=64x64x8,lds_staged=true,early_barrier=true | Navatala HIP MFMA kernel navatala_transformer_tiled_gemm_f16_mfma_cta64_shared_early_barrier | rocBLAS rocblas_gemm_ex F16 inputs/F32 accumulation | mfma_f16 | hip_mfma_gfx942_64x64x8_f16_f32_cta64_shared_early_barrier | pass | 0.020511 | 0.025659 | 0.799x | 1.01328e-06 |
| GEMM_F16_MFMA_CTA64_SHARED_EARLY_BARRIER | m=1024,n=1024,k=1024,compute=F32,cta=64x64x8,lds_staged=true,early_barrier=true | Navatala HIP MFMA kernel navatala_transformer_tiled_gemm_f16_mfma_cta64_shared_early_barrier | rocBLAS rocblas_gemm_ex F16 inputs/F32 accumulation | mfma_f16 | hip_mfma_gfx942_64x64x8_f16_f32_cta64_shared_early_barrier | pass | 0.059718 | 0.042838 | 1.394x | 2.14577e-06 |
| GEMM_F16_MFMA_CTA64_SHARED_PADDED | m=128,n=128,k=128,compute=F32,cta=64x64x8,lds_staged=true,lds_padded=true | Navatala HIP MFMA kernel navatala_transformer_tiled_gemm_f16_mfma_cta64_shared_padded | rocBLAS rocblas_gemm_ex F16 inputs/F32 accumulation | mfma_f16 | hip_mfma_gfx942_64x64x8_f16_f32_cta64_shared_padded | pass | 0.007939 | 0.023827 | 0.333x | 1.19209e-07 |
| GEMM_F16_MFMA_CTA64_SHARED_PADDED | m=512,n=512,k=512,compute=F32,cta=64x64x8,lds_staged=true,lds_padded=true | Navatala HIP MFMA kernel navatala_transformer_tiled_gemm_f16_mfma_cta64_shared_padded | rocBLAS rocblas_gemm_ex F16 inputs/F32 accumulation | mfma_f16 | hip_mfma_gfx942_64x64x8_f16_f32_cta64_shared_padded | pass | 0.027303 | 0.025718 | 1.062x | 1.01328e-06 |
| GEMM_F16_MFMA_CTA64_SHARED_PADDED | m=1024,n=1024,k=1024,compute=F32,cta=64x64x8,lds_staged=true,lds_padded=true | Navatala HIP MFMA kernel navatala_transformer_tiled_gemm_f16_mfma_cta64_shared_padded | rocBLAS rocblas_gemm_ex F16 inputs/F32 accumulation | mfma_f16 | hip_mfma_gfx942_64x64x8_f16_f32_cta64_shared_padded | pass | 0.073640 | 0.042921 | 1.716x | 2.14577e-06 |
| GEMM_F16_MFMA_CTA64_PIPELINED | m=128,n=128,k=128,compute=F32,cta=64x64x8,lds_staged=true,two_slot=true,r6_panel_copy=true | Navatala HIP MFMA kernel navatala_transformer_tiled_gemm_f16_mfma_cta64_pipelined | rocBLAS rocblas_gemm_ex F16 inputs/F32 accumulation | mfma_f16 | hip_mfma_gfx942_64x64x8_f16_f32_cta64_pipelined | pass | 0.007051 | 0.023774 | 0.297x | 1.19209e-07 |
| GEMM_F16_MFMA_CTA64_PIPELINED | m=512,n=512,k=512,compute=F32,cta=64x64x8,lds_staged=true,two_slot=true,r6_panel_copy=true | Navatala HIP MFMA kernel navatala_transformer_tiled_gemm_f16_mfma_cta64_pipelined | rocBLAS rocblas_gemm_ex F16 inputs/F32 accumulation | mfma_f16 | hip_mfma_gfx942_64x64x8_f16_f32_cta64_pipelined | pass | 0.022703 | 0.025603 | 0.887x | 1.01328e-06 |
| GEMM_F16_MFMA_CTA64_PIPELINED | m=1024,n=1024,k=1024,compute=F32,cta=64x64x8,lds_staged=true,two_slot=true,r6_panel_copy=true | Navatala HIP MFMA kernel navatala_transformer_tiled_gemm_f16_mfma_cta64_pipelined | rocBLAS rocblas_gemm_ex F16 inputs/F32 accumulation | mfma_f16 | hip_mfma_gfx942_64x64x8_f16_f32_cta64_pipelined | pass | 0.061287 | 0.042789 | 1.432x | 2.14577e-06 |
| GEMM_F16_MFMA_CTA128 | m=128,n=128,k=128,compute=F32,cta=128x128x32 | Navatala HIP MFMA kernel navatala_transformer_tiled_gemm_f16_mfma_cta128 | rocBLAS rocblas_gemm_ex F16 inputs/F32 accumulation | mfma_f16 | hip_mfma_gfx942_128x128x32_f16_f32_cta128 | pass | 0.008406 | 0.023807 | 0.353x | 1.19209e-07 |
| GEMM_F16_MFMA_CTA128 | m=512,n=512,k=512,compute=F32,cta=128x128x32 | Navatala HIP MFMA kernel navatala_transformer_tiled_gemm_f16_mfma_cta128 | rocBLAS rocblas_gemm_ex F16 inputs/F32 accumulation | mfma_f16 | hip_mfma_gfx942_128x128x32_f16_f32_cta128 | pass | 0.026049 | 0.025631 | 1.016x | 1.01328e-06 |
| GEMM_F16_MFMA_CTA128 | m=1024,n=1024,k=1024,compute=F32,cta=128x128x32 | Navatala HIP MFMA kernel navatala_transformer_tiled_gemm_f16_mfma_cta128 | rocBLAS rocblas_gemm_ex F16 inputs/F32 accumulation | mfma_f16 | hip_mfma_gfx942_128x128x32_f16_f32_cta128 | pass | 0.053407 | 0.042657 | 1.252x | 2.14577e-06 |
| GEMM_F32_WRAPPER_VENDOR | m=128,n=128,k=128 | public C ABI navatala_gpu_gemm_f32 with NAVATALA_GPU_GEMM_VENDOR_MODE=vendor | rocBLAS rocblas_sgemm | vendor_library | vendor_dispatch | pass | 0.006891 | 0.006619 | 1.041x | 0 |
| GEMM_F32_WRAPPER_VENDOR | m=512,n=512,k=512 | public C ABI navatala_gpu_gemm_f32 with NAVATALA_GPU_GEMM_VENDOR_MODE=vendor | rocBLAS rocblas_sgemm | vendor_library | vendor_dispatch | pass | 0.008236 | 0.008228 | 1.001x | 0 |
| GEMM_F32_WRAPPER_VENDOR | m=1024,n=1024,k=1024 | public C ABI navatala_gpu_gemm_f32 with NAVATALA_GPU_GEMM_VENDOR_MODE=vendor | rocBLAS rocblas_sgemm | vendor_library | vendor_dispatch | pass | 0.027568 | 0.026130 | 1.055x | 0 |
| GEMM_F16_F32_WRAPPER_MFMA | m=127,n=129,k=65,transA=N,transB=N,alpha=1,beta=0,batch=1,output=F32,compute=F32,wrapper=mfma | public C ABI navatala_gpu_gemm_f16_f32 with NAVATALA_GPU_GEMM_IMPL=mfma,NAVATALA_GPU_GEMM_VENDOR_MODE=auto,NAVATALA_GPU_GEMM_MFMA_MODE=auto | rocBLAS rocblas_gemm_ex F16 input/F32 output/F32 accumulation | mfma_f16 | hip_mfma_gfx942_wrapper_dispatch | pass | 0.006345 | 0.028194 | 0.225x | 1.19209e-07 |
| GEMM_F16_F32_WRAPPER_MFMA | m=512,n=512,k=512,transA=N,transB=N,alpha=1,beta=0,batch=1,output=F32,compute=F32,wrapper=mfma | public C ABI navatala_gpu_gemm_f16_f32 with NAVATALA_GPU_GEMM_IMPL=mfma,NAVATALA_GPU_GEMM_VENDOR_MODE=auto,NAVATALA_GPU_GEMM_MFMA_MODE=auto | rocBLAS rocblas_gemm_ex F16 input/F32 output/F32 accumulation | mfma_f16 | hip_mfma_gfx942_wrapper_dispatch | pass | 0.020866 | 0.025889 | 0.806x | 1.01328e-06 |
| GEMM_F16_F32_WRAPPER_MFMA | m=1025,n=1023,k=257,transA=N,transB=N,alpha=1,beta=0,batch=1,output=F32,compute=F32,wrapper=mfma | public C ABI navatala_gpu_gemm_f16_f32 with NAVATALA_GPU_GEMM_IMPL=mfma,NAVATALA_GPU_GEMM_VENDOR_MODE=auto,NAVATALA_GPU_GEMM_MFMA_MODE=auto | rocBLAS rocblas_gemm_ex F16 input/F32 output/F32 accumulation | mfma_f16 | hip_mfma_gfx942_wrapper_dispatch | pass | 0.018631 | 0.038772 | 0.481x | 1.56462e-07 |
| GEMM_F16_F32_WRAPPER_MFMA | m=1024,n=1024,k=1024,transA=N,transB=N,alpha=1,beta=0,batch=1,output=F32,compute=F32,wrapper=mfma | public C ABI navatala_gpu_gemm_f16_f32 with NAVATALA_GPU_GEMM_IMPL=mfma,NAVATALA_GPU_GEMM_VENDOR_MODE=auto,NAVATALA_GPU_GEMM_MFMA_MODE=auto | rocBLAS rocblas_gemm_ex F16 input/F32 output/F32 accumulation | mfma_f16 | hip_mfma_gfx942_wrapper_dispatch | pass | 0.057392 | 0.042836 | 1.340x | 2.14577e-06 |
| GEMM_F16_F32_WRAPPER_MFMA_ALPHA_BETA | m=512,n=512,k=512,transA=N,transB=N,alpha=0.75,beta=0.25,batch=1,output=F32,compute=F32,wrapper=mfma | public C ABI navatala_gpu_gemm_f16_f32 with NAVATALA_GPU_GEMM_IMPL=mfma,NAVATALA_GPU_GEMM_VENDOR_MODE=auto,NAVATALA_GPU_GEMM_MFMA_MODE=auto | rocBLAS rocblas_gemm_ex F16 input/F32 output/F32 accumulation | mfma_f16 | hip_mfma_gfx942_wrapper_dispatch | pass | 0.026572 | 0.035128 | 0.756x | 1.04308e-06 |
| GEMM_F16_F32_WRAPPER_MFMA_TRANSPOSE | m=384,n=512,k=256,transA=T,transB=T,alpha=1,beta=0,batch=1,output=F32,compute=F32,wrapper=mfma | public C ABI navatala_gpu_gemm_f16_f32_ex with NAVATALA_GPU_GEMM_IMPL=mfma,NAVATALA_GPU_GEMM_VENDOR_MODE=auto,NAVATALA_GPU_GEMM_MFMA_MODE=auto | rocBLAS rocblas_gemm_ex F16 input/F32 output/F32 accumulation | mfma_f16 | hip_mfma_gfx942_wrapper_dispatch | pass | 0.023205 | 0.020272 | 1.145x | 3.57628e-07 |
| GEMM_F16_F32_WRAPPER_MFMA_BATCHED | m=256,n=256,k=256,transA=N,transB=N,alpha=1,beta=0,batch=3,output=F32,compute=F32,wrapper=mfma | public C ABI navatala_gpu_gemm_f16_f32_strided_batched with NAVATALA_GPU_GEMM_IMPL=mfma,NAVATALA_GPU_GEMM_VENDOR_MODE=auto,NAVATALA_GPU_GEMM_MFMA_MODE=auto | rocBLAS rocblas_gemm_strided_batched_ex F16 input/F32 output/F32 accumulation | mfma_f16 | hip_mfma_gfx942_wrapper_dispatch | pass | 0.018027 | 0.017515 | 1.029x | 3.57628e-07 |
| CSR_SPMV_F32 | rows=16384,rowNnz=7,nnz=114688 | Navatala HIP kernel navatala_graph_spmv_weighted_f32 | rocSPARSE rocsparse_v2_spmv | scalar | thread_per_row | pass | 0.003246 | 0.002673 | 1.214x | 1.49012e-08 |
| CSR_SPMV_F32 | rows=262144,rowNnz=7,nnz=1835008 | Navatala HIP kernel navatala_graph_spmv_weighted_f32 | rocSPARSE rocsparse_v2_spmv | scalar | thread_per_row | pass | 0.006473 | 0.004414 | 1.466x | 2.98023e-08 |
| CSR_SPMV_F32 | rows=1048576,rowNnz=7,nnz=7340032 | Navatala HIP kernel navatala_graph_spmv_weighted_f32 | rocSPARSE rocsparse_v2_spmv | scalar | thread_per_row | pass | 0.050479 | 0.020085 | 2.513x | 1.49012e-08 |
| CSR_SPMV_F32 | rows=262144,rowNnz=15,nnz=3932160 | Navatala HIP kernel navatala_graph_spmv_weighted_f32 | rocSPARSE rocsparse_v2_spmv | scalar | thread_per_row | pass | 0.023996 | 0.010369 | 2.314x | 5.96046e-08 |
| CSR_SPMV_F32 | rows=262144,rowNnz=27,nnz=7077888 | Navatala HIP kernel navatala_graph_spmv_weighted_subgroup_f32 | rocSPARSE rocsparse_v2_spmv | scalar | subgroup_per_row | pass | 0.045313 | 0.019299 | 2.348x | 5.96046e-08 |
| HIPSPARSELT_STRUCTURED_GEMM_F16 | m=128,n=128,k=128,sparsity=50%,compute=F32 | hipSPARSELt prune/compress/matmul path for SparseLt_StructuredMatmul | rocBLAS rocblas_hgemm dense pruned-A reference | vendor_library | vendor_dispatch | pass | 0.010865 | 0.006892 | 1.576x | 0 |
| HIPSPARSELT_STRUCTURED_GEMM_F16 | m=512,n=512,k=512,sparsity=50%,compute=F32 | hipSPARSELt prune/compress/matmul path for SparseLt_StructuredMatmul | rocBLAS rocblas_hgemm dense pruned-A reference | vendor_library | vendor_dispatch | pass | 0.011683 | 0.019957 | 0.585x | 0 |

## Interpretation Notes

- `generatedMeanMs` and `vendorMeanMs` are back-to-back throughput means per launch, not isolated launch latency.
- Correctness is measured against the vendor baseline in the benchmark harness for the listed shape.
- This report covers only the operations listed above; it is not a full HIP backend certification.

## Wrapper Timing Diagnostics

| Operation | Shape | Stream-event mean ms | Host submit mean ms | Host wall mean ms | Submit/event | Wall/event |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| GEMM_F32_WRAPPER_VENDOR | m=128,n=128,k=128 | 0.006891 | 0.006094 | 0.007010 | 0.884x | 1.017x |
| GEMM_F32_WRAPPER_VENDOR | m=512,n=512,k=512 | 0.008236 | 0.005436 | 0.008368 | 0.660x | 1.016x |
| GEMM_F32_WRAPPER_VENDOR | m=1024,n=1024,k=1024 | 0.027568 | 0.005713 | 0.027707 | 0.207x | 1.005x |
| GEMM_F16_F32_WRAPPER_MFMA | m=127,n=129,k=65,transA=N,transB=N,alpha=1,beta=0,batch=1,output=F32,compute=F32,wrapper=mfma | 0.006345 | 0.002620 | 0.006498 | 0.413x | 1.024x |
| GEMM_F16_F32_WRAPPER_MFMA | m=512,n=512,k=512,transA=N,transB=N,alpha=1,beta=0,batch=1,output=F32,compute=F32,wrapper=mfma | 0.020866 | 0.002733 | 0.021010 | 0.131x | 1.007x |
| GEMM_F16_F32_WRAPPER_MFMA | m=1025,n=1023,k=257,transA=N,transB=N,alpha=1,beta=0,batch=1,output=F32,compute=F32,wrapper=mfma | 0.018631 | 0.003579 | 0.018798 | 0.192x | 1.009x |
| GEMM_F16_F32_WRAPPER_MFMA | m=1024,n=1024,k=1024,transA=N,transB=N,alpha=1,beta=0,batch=1,output=F32,compute=F32,wrapper=mfma | 0.057392 | 0.002637 | 0.057553 | 0.046x | 1.003x |
| GEMM_F16_F32_WRAPPER_MFMA_ALPHA_BETA | m=512,n=512,k=512,transA=N,transB=N,alpha=0.75,beta=0.25,batch=1,output=F32,compute=F32,wrapper=mfma | 0.026572 | 0.002865 | 0.026753 | 0.108x | 1.007x |
| GEMM_F16_F32_WRAPPER_MFMA_TRANSPOSE | m=384,n=512,k=256,transA=T,transB=T,alpha=1,beta=0,batch=1,output=F32,compute=F32,wrapper=mfma | 0.023205 | 0.002856 | 0.023361 | 0.123x | 1.007x |
| GEMM_F16_F32_WRAPPER_MFMA_BATCHED | m=256,n=256,k=256,transA=N,transB=N,alpha=1,beta=0,batch=3,output=F32,compute=F32,wrapper=mfma | 0.018027 | 0.002783 | 0.018196 | 0.154x | 1.009x |

- `hostSubmitMeanMs` is CPU wall time to call the wrapper repeatedly and enqueue work, excluding the final event wait.
- `hostWallMeanMs` includes the final event wait. Compare both with the stream-event mean to separate host submission overhead from device work.
- hipSPARSELt structured rows compute 50% structured sparse matmul; dense MFMA rows compute ordinary dense matmul. Do not compare their ratios directly.
