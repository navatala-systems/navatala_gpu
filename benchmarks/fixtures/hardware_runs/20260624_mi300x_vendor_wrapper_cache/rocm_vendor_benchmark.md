# ROCm Vendor Benchmark Report

- Generated at: `2026-06-24T11:25:16Z`
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
- hipSPARSELt mode: `capability_reporting_only`

## Results

| Operation | Shape | Generated path | Vendor baseline | Kernel class | Tuning path | Correctness | Generated mean ms | Vendor mean ms | Ratio | Max abs error |
| --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| AXPY_F32 | n=65536 | Navatala HIP kernel navatala_sparse_axpy_f32 | rocBLAS rocblas_saxpy | scalar | portable_kernel | pass | 0.006610 | 0.002360 | 2.801x | 0 |
| AXPY_F32 | n=1048576 | Navatala HIP kernel navatala_sparse_axpy_f32 | rocBLAS rocblas_saxpy | scalar | portable_kernel | pass | 0.005659 | 0.005542 | 1.021x | 0 |
| AXPY_F32 | n=4194304 | Navatala HIP kernel navatala_sparse_axpy_f32 | rocBLAS rocblas_saxpy | scalar | portable_kernel | pass | 0.018317 | 0.022246 | 0.823x | 0 |
| GEMM_F32 | m=128,n=128,k=128 | Navatala HIP kernel navatala_transformer_tiled_gemm_f32 | rocBLAS rocblas_sgemm | scalar | portable_kernel | pass | 0.005382 | 0.008617 | 0.625x | 3.42727e-07 |
| GEMM_F32 | m=512,n=512,k=512 | Navatala HIP kernel navatala_transformer_tiled_gemm_f32 | rocBLAS rocblas_sgemm | scalar | portable_kernel | pass | 0.026558 | 0.009391 | 2.828x | 8.04663e-07 |
| GEMM_F32 | m=1024,n=1024,k=1024 | Navatala HIP kernel navatala_transformer_tiled_gemm_f32 | rocBLAS rocblas_sgemm | scalar | portable_kernel | pass | 0.184815 | 0.025857 | 7.148x | 1.84774e-06 |
| GEMM_F16_PORTABLE_F32OUT | m=128,n=128,k=128,output=F32,compute=F32 | Navatala HIP kernel navatala_transformer_tiled_gemm_f16_f32_out | rocBLAS rocblas_gemm_ex F16 input/F32 output/F32 accumulation | scalar | portable_f16_f32out_tiled | pass | 0.006019 | 0.024200 | 0.249x | 2.38419e-07 |
| GEMM_F16_PORTABLE_F32OUT | m=512,n=512,k=512,output=F32,compute=F32 | Navatala HIP kernel navatala_transformer_tiled_gemm_f16_f32_out | rocBLAS rocblas_gemm_ex F16 input/F32 output/F32 accumulation | scalar | portable_f16_f32out_tiled | pass | 0.027180 | 0.025875 | 1.050x | 2.02656e-06 |
| GEMM_F16_PORTABLE_F32OUT | m=1024,n=1024,k=1024,output=F32,compute=F32 | Navatala HIP kernel navatala_transformer_tiled_gemm_f16_f32_out | rocBLAS rocblas_gemm_ex F16 input/F32 output/F32 accumulation | scalar | portable_f16_f32out_tiled | pass | 0.181667 | 0.047651 | 3.812x | 2.86102e-06 |
| GEMM_F32_WRAPPER_VENDOR | m=128,n=128,k=128 | public C ABI navatala_gpu_gemm_f32 with NAVATALA_GPU_GEMM_VENDOR_MODE=vendor | rocBLAS rocblas_sgemm | vendor_library | vendor_dispatch | pass | 0.006858 | 0.006797 | 1.009x | 0 |
| GEMM_F32_WRAPPER_VENDOR | m=512,n=512,k=512 | public C ABI navatala_gpu_gemm_f32 with NAVATALA_GPU_GEMM_VENDOR_MODE=vendor | rocBLAS rocblas_sgemm | vendor_library | vendor_dispatch | pass | 0.008297 | 0.008317 | 0.998x | 0 |
| GEMM_F32_WRAPPER_VENDOR | m=1024,n=1024,k=1024 | public C ABI navatala_gpu_gemm_f32 with NAVATALA_GPU_GEMM_VENDOR_MODE=vendor | rocBLAS rocblas_sgemm | vendor_library | vendor_dispatch | pass | 0.030552 | 0.025679 | 1.190x | 0 |
| GEMM_F16_F32_WRAPPER_MFMA | m=127,n=129,k=65,transA=N,transB=N,alpha=1,beta=0,batch=1,output=F32,compute=F32,wrapper=mfma | public C ABI navatala_gpu_gemm_f16_f32 with NAVATALA_GPU_GEMM_IMPL=mfma,NAVATALA_GPU_GEMM_VENDOR_MODE=auto,NAVATALA_GPU_GEMM_MFMA_MODE=auto | rocBLAS rocblas_gemm_ex F16 input/F32 output/F32 accumulation | mfma_f16 | hip_mfma_gfx942_wrapper_dispatch | pass | 0.007085 | 0.029138 | 0.243x | 1.19209e-07 |
| GEMM_F16_F32_WRAPPER_MFMA | m=512,n=512,k=512,transA=N,transB=N,alpha=1,beta=0,batch=1,output=F32,compute=F32,wrapper=mfma | public C ABI navatala_gpu_gemm_f16_f32 with NAVATALA_GPU_GEMM_IMPL=mfma,NAVATALA_GPU_GEMM_VENDOR_MODE=auto,NAVATALA_GPU_GEMM_MFMA_MODE=auto | rocBLAS rocblas_gemm_ex F16 input/F32 output/F32 accumulation | mfma_f16 | hip_mfma_gfx942_wrapper_dispatch | pass | 0.021085 | 0.026753 | 0.788x | 1.01328e-06 |
| GEMM_F16_F32_WRAPPER_MFMA | m=1025,n=1023,k=257,transA=N,transB=N,alpha=1,beta=0,batch=1,output=F32,compute=F32,wrapper=mfma | public C ABI navatala_gpu_gemm_f16_f32 with NAVATALA_GPU_GEMM_IMPL=mfma,NAVATALA_GPU_GEMM_VENDOR_MODE=auto,NAVATALA_GPU_GEMM_MFMA_MODE=auto | rocBLAS rocblas_gemm_ex F16 input/F32 output/F32 accumulation | mfma_f16 | hip_mfma_gfx942_wrapper_dispatch | pass | 0.019947 | 0.039615 | 0.504x | 1.56462e-07 |
| GEMM_F16_F32_WRAPPER_MFMA | m=1024,n=1024,k=1024,transA=N,transB=N,alpha=1,beta=0,batch=1,output=F32,compute=F32,wrapper=mfma | public C ABI navatala_gpu_gemm_f16_f32 with NAVATALA_GPU_GEMM_IMPL=mfma,NAVATALA_GPU_GEMM_VENDOR_MODE=auto,NAVATALA_GPU_GEMM_MFMA_MODE=auto | rocBLAS rocblas_gemm_ex F16 input/F32 output/F32 accumulation | mfma_f16 | hip_mfma_gfx942_wrapper_dispatch | pass | 0.057701 | 0.047498 | 1.215x | 2.14577e-06 |
| GEMM_F16_F32_WRAPPER_MFMA_ALPHA_BETA | m=512,n=512,k=512,transA=N,transB=N,alpha=0.75,beta=0.25,batch=1,output=F32,compute=F32,wrapper=mfma | public C ABI navatala_gpu_gemm_f16_f32 with NAVATALA_GPU_GEMM_IMPL=mfma,NAVATALA_GPU_GEMM_VENDOR_MODE=auto,NAVATALA_GPU_GEMM_MFMA_MODE=auto | rocBLAS rocblas_gemm_ex F16 input/F32 output/F32 accumulation | mfma_f16 | hip_mfma_gfx942_wrapper_dispatch | pass | 0.038734 | 0.033428 | 1.159x | 1.04308e-06 |
| GEMM_F16_F32_WRAPPER_MFMA_TRANSPOSE | m=384,n=512,k=256,transA=T,transB=T,alpha=1,beta=0,batch=1,output=F32,compute=F32,wrapper=mfma | public C ABI navatala_gpu_gemm_f16_f32_ex with NAVATALA_GPU_GEMM_IMPL=mfma,NAVATALA_GPU_GEMM_VENDOR_MODE=auto,NAVATALA_GPU_GEMM_MFMA_MODE=auto | rocBLAS rocblas_gemm_ex F16 input/F32 output/F32 accumulation | mfma_f16 | hip_mfma_gfx942_wrapper_dispatch | pass | 0.025670 | 0.021425 | 1.198x | 3.57628e-07 |
| GEMM_F16_F32_WRAPPER_MFMA_BATCHED | m=256,n=256,k=256,transA=N,transB=N,alpha=1,beta=0,batch=3,output=F32,compute=F32,wrapper=mfma | public C ABI navatala_gpu_gemm_f16_f32_strided_batched with NAVATALA_GPU_GEMM_IMPL=mfma,NAVATALA_GPU_GEMM_VENDOR_MODE=auto,NAVATALA_GPU_GEMM_MFMA_MODE=auto | rocBLAS rocblas_gemm_strided_batched_ex F16 input/F32 output/F32 accumulation | mfma_f16 | hip_mfma_gfx942_wrapper_dispatch | pass | 0.018338 | 0.017707 | 1.036x | 3.57628e-07 |
| CSR_SPMV_F32 | rows=16384,rowNnz=7,nnz=114688 | Navatala HIP kernel navatala_graph_spmv_weighted_f32 | rocSPARSE rocsparse_v2_spmv | scalar | thread_per_row | pass | 0.004037 | 0.002813 | 1.435x | 1.49012e-08 |
| CSR_SPMV_F32 | rows=262144,rowNnz=7,nnz=1835008 | Navatala HIP kernel navatala_graph_spmv_weighted_f32 | rocSPARSE rocsparse_v2_spmv | scalar | thread_per_row | pass | 0.006475 | 0.004417 | 1.466x | 2.98023e-08 |
| CSR_SPMV_F32 | rows=1048576,rowNnz=7,nnz=7340032 | Navatala HIP kernel navatala_graph_spmv_weighted_f32 | rocSPARSE rocsparse_v2_spmv | scalar | thread_per_row | pass | 0.049933 | 0.020764 | 2.405x | 1.49012e-08 |
| CSR_SPMV_F32 | rows=262144,rowNnz=15,nnz=3932160 | Navatala HIP kernel navatala_graph_spmv_weighted_f32 | rocSPARSE rocsparse_v2_spmv | scalar | thread_per_row | pass | 0.024270 | 0.014573 | 1.665x | 5.96046e-08 |
| CSR_SPMV_F32 | rows=262144,rowNnz=27,nnz=7077888 | Navatala HIP kernel navatala_graph_spmv_weighted_subgroup_f32 | rocSPARSE rocsparse_v2_spmv | scalar | subgroup_per_row | pass | 0.045173 | 0.019494 | 2.317x | 5.96046e-08 |

## Interpretation Notes

- `generatedMeanMs` and `vendorMeanMs` are back-to-back throughput means per launch, not isolated launch latency.
- Correctness is measured against the vendor baseline in the benchmark harness for the listed shape.
- This report covers only the operations listed above; it is not a full HIP backend certification.
- hipSPARSELt is reported as a capability only; no hipSPARSELt performance baseline is claimed.

## Validation Warnings

- hipSPARSELt is capability reporting only; no hipSPARSELt benchmark row is claimed
