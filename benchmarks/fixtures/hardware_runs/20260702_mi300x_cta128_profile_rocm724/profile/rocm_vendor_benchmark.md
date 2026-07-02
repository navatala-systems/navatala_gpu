# ROCm Vendor Benchmark Report

- Generated at: `2026-07-02T07:56:19Z`
- Source JSON: `rocm_vendor_benchmark.json`
- Schema: `navatala_gpu.rocm_vendor_benchmark.v1`
- Timing mode: `back_to_back_throughput_mean_per_launch`
- Quick mode: `false`
- Matrix set: `cta128-evidence`
- Iterations: `5`
- Warmup: `1`
- Device: `AMD Instinct MI300X VF`
- GCN arch: `gfx942:sramecc+:xnack-`
- Global memory: `196288 MiB`
- hipSPARSELt available: `true`
- hipSPARSELt mode: `capability_reporting_only`

## Results

| Operation | Shape | Generated path | Vendor baseline | Kernel class | Tuning path | Correctness | Generated mean ms | Vendor mean ms | Ratio | Max abs error |
| --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| GEMM_F16_PORTABLE_F32OUT | m=128,n=128,k=128,output=F32,compute=F32 | Navatala HIP kernel navatala_transformer_tiled_gemm_f16_f32_out | rocBLAS rocblas_gemm_ex F16 input/F32 output/F32 accumulation | scalar | portable_f16_f32out_tiled | pass | 0.008075 | 0.029845 | 0.271x | 2.38419e-07 |
| GEMM_F16_MFMA_CTA64_SHARED | m=128,n=128,k=128,compute=F32,cta=64x64x8,lds_staged=true | Navatala HIP MFMA kernel navatala_transformer_tiled_gemm_f16_mfma_cta64_shared | rocBLAS rocblas_gemm_ex F16 inputs/F32 accumulation | mfma_f16 | hip_mfma_gfx942_64x64x8_f16_f32_cta64_shared | pass | 0.007858 | 0.026806 | 0.293x | 1.19209e-07 |
| GEMM_F16_MFMA_CTA64_PIPELINED | m=128,n=128,k=128,compute=F32,cta=64x64x8,lds_staged=true,two_slot=true,r6_panel_copy=true | Navatala HIP MFMA kernel navatala_transformer_tiled_gemm_f16_mfma_cta64_pipelined | rocBLAS rocblas_gemm_ex F16 inputs/F32 accumulation | mfma_f16 | hip_mfma_gfx942_64x64x8_f16_f32_cta64_pipelined | pass | 0.009582 | 0.026685 | 0.359x | 1.19209e-07 |
| GEMM_F16_MFMA_CTA128 | m=128,n=128,k=128,compute=F32,cta=128x128x32 | Navatala HIP MFMA kernel navatala_transformer_tiled_gemm_f16_mfma_cta128 | rocBLAS rocblas_gemm_ex F16 inputs/F32 accumulation | mfma_f16 | hip_mfma_gfx942_128x128x32_f16_f32_cta128 | pass | 0.009927 | 0.025723 | 0.386x | 1.19209e-07 |
| GEMM_F16_MFMA_CTA64_SHARED_EDGE | m=128,n=128,k=128,compute=F32,edge=true | Navatala HIP MFMA kernel navatala_transformer_tiled_gemm_f16_mfma_cta64_shared_edge | rocBLAS rocblas_gemm_ex F16 inputs/F32 accumulation | mfma_f16 | hip_mfma_gfx942_64x64x8_f16_f32_cta64_shared_edge | pass | 0.011458 | 0.025860 | 0.443x | 1.19209e-07 |
| GEMM_F16_MFMA_CTA128_EDGE | m=128,n=128,k=128,compute=F32,edge=true | Navatala HIP MFMA kernel navatala_transformer_tiled_gemm_f16_mfma_cta128_edge | rocBLAS rocblas_gemm_ex F16 inputs/F32 accumulation | mfma_f16 | hip_mfma_gfx942_128x128x32_f16_f32_cta128_edge | pass | 0.021409 | 0.025924 | 0.826x | 1.19209e-07 |
| GEMM_F16_PORTABLE_F32OUT | m=512,n=512,k=512,output=F32,compute=F32 | Navatala HIP kernel navatala_transformer_tiled_gemm_f16_f32_out | rocBLAS rocblas_gemm_ex F16 input/F32 output/F32 accumulation | scalar | portable_f16_f32out_tiled | pass | 0.029276 | 0.028209 | 1.038x | 2.02656e-06 |
| GEMM_F16_MFMA_CTA64_SHARED | m=512,n=512,k=512,compute=F32,cta=64x64x8,lds_staged=true | Navatala HIP MFMA kernel navatala_transformer_tiled_gemm_f16_mfma_cta64_shared | rocBLAS rocblas_gemm_ex F16 inputs/F32 accumulation | mfma_f16 | hip_mfma_gfx942_64x64x8_f16_f32_cta64_shared | pass | 0.022059 | 0.028041 | 0.787x | 1.01328e-06 |
| GEMM_F16_MFMA_CTA64_PIPELINED | m=512,n=512,k=512,compute=F32,cta=64x64x8,lds_staged=true,two_slot=true,r6_panel_copy=true | Navatala HIP MFMA kernel navatala_transformer_tiled_gemm_f16_mfma_cta64_pipelined | rocBLAS rocblas_gemm_ex F16 inputs/F32 accumulation | mfma_f16 | hip_mfma_gfx942_64x64x8_f16_f32_cta64_pipelined | pass | 0.024288 | 0.027912 | 0.870x | 1.01328e-06 |
| GEMM_F16_MFMA_CTA128 | m=512,n=512,k=512,compute=F32,cta=128x128x32 | Navatala HIP MFMA kernel navatala_transformer_tiled_gemm_f16_mfma_cta128 | rocBLAS rocblas_gemm_ex F16 inputs/F32 accumulation | mfma_f16 | hip_mfma_gfx942_128x128x32_f16_f32_cta128 | pass | 0.027696 | 0.028113 | 0.985x | 1.01328e-06 |
| GEMM_F16_MFMA_CTA64_SHARED_EDGE | m=512,n=512,k=512,compute=F32,edge=true | Navatala HIP MFMA kernel navatala_transformer_tiled_gemm_f16_mfma_cta64_shared_edge | rocBLAS rocblas_gemm_ex F16 inputs/F32 accumulation | mfma_f16 | hip_mfma_gfx942_64x64x8_f16_f32_cta64_shared_edge | pass | 0.035362 | 0.027864 | 1.269x | 1.01328e-06 |
| GEMM_F16_MFMA_CTA128_EDGE | m=512,n=512,k=512,compute=F32,edge=true | Navatala HIP MFMA kernel navatala_transformer_tiled_gemm_f16_mfma_cta128_edge | rocBLAS rocblas_gemm_ex F16 inputs/F32 accumulation | mfma_f16 | hip_mfma_gfx942_128x128x32_f16_f32_cta128_edge | pass | 0.067724 | 0.027503 | 2.462x | 1.01328e-06 |
| GEMM_F16_PORTABLE_F32OUT | m=1024,n=1024,k=1024,output=F32,compute=F32 | Navatala HIP kernel navatala_transformer_tiled_gemm_f16_f32_out | rocBLAS rocblas_gemm_ex F16 input/F32 output/F32 accumulation | scalar | portable_f16_f32out_tiled | pass | 0.153363 | 0.045297 | 3.386x | 2.86102e-06 |
| GEMM_F16_MFMA_CTA64_SHARED | m=1024,n=1024,k=1024,compute=F32,cta=64x64x8,lds_staged=true | Navatala HIP MFMA kernel navatala_transformer_tiled_gemm_f16_mfma_cta64_shared | rocBLAS rocblas_gemm_ex F16 inputs/F32 accumulation | mfma_f16 | hip_mfma_gfx942_64x64x8_f16_f32_cta64_shared | pass | 0.061807 | 0.045714 | 1.352x | 2.14577e-06 |
| GEMM_F16_MFMA_CTA64_PIPELINED | m=1024,n=1024,k=1024,compute=F32,cta=64x64x8,lds_staged=true,two_slot=true,r6_panel_copy=true | Navatala HIP MFMA kernel navatala_transformer_tiled_gemm_f16_mfma_cta64_pipelined | rocBLAS rocblas_gemm_ex F16 inputs/F32 accumulation | mfma_f16 | hip_mfma_gfx942_64x64x8_f16_f32_cta64_pipelined | pass | 0.064902 | 0.045112 | 1.439x | 2.14577e-06 |
| GEMM_F16_MFMA_CTA128 | m=1024,n=1024,k=1024,compute=F32,cta=128x128x32 | Navatala HIP MFMA kernel navatala_transformer_tiled_gemm_f16_mfma_cta128 | rocBLAS rocblas_gemm_ex F16 inputs/F32 accumulation | mfma_f16 | hip_mfma_gfx942_128x128x32_f16_f32_cta128 | pass | 0.055151 | 0.045128 | 1.222x | 2.14577e-06 |
| GEMM_F16_MFMA_CTA64_SHARED_EDGE | m=1024,n=1024,k=1024,compute=F32,edge=true | Navatala HIP MFMA kernel navatala_transformer_tiled_gemm_f16_mfma_cta64_shared_edge | rocBLAS rocblas_gemm_ex F16 inputs/F32 accumulation | mfma_f16 | hip_mfma_gfx942_64x64x8_f16_f32_cta64_shared_edge | pass | 0.472003 | 0.045297 | 10.420x | 2.14577e-06 |
| GEMM_F16_MFMA_CTA128_EDGE | m=1024,n=1024,k=1024,compute=F32,edge=true | Navatala HIP MFMA kernel navatala_transformer_tiled_gemm_f16_mfma_cta128_edge | rocBLAS rocblas_gemm_ex F16 inputs/F32 accumulation | mfma_f16 | hip_mfma_gfx942_128x128x32_f16_f32_cta128_edge | pass | 0.182333 | 0.045265 | 4.028x | 2.14577e-06 |
| GEMM_F16_PORTABLE_F32OUT | m=1024,n=512,k=1024,output=F32,compute=F32 | Navatala HIP kernel navatala_transformer_tiled_gemm_f16_f32_out | rocBLAS rocblas_gemm_ex F16 input/F32 output/F32 accumulation | scalar | portable_f16_f32out_tiled | pass | 0.083866 | 0.045192 | 1.856x | 2.5034e-06 |
| GEMM_F16_MFMA_CTA64_SHARED | m=1024,n=512,k=1024,compute=F32,cta=64x64x8,lds_staged=true | Navatala HIP MFMA kernel navatala_transformer_tiled_gemm_f16_mfma_cta64_shared | rocBLAS rocblas_gemm_ex F16 inputs/F32 accumulation | mfma_f16 | hip_mfma_gfx942_64x64x8_f16_f32_cta64_shared | pass | 0.054077 | 0.043829 | 1.234x | 1.3113e-06 |
| GEMM_F16_MFMA_CTA64_PIPELINED | m=1024,n=512,k=1024,compute=F32,cta=64x64x8,lds_staged=true,two_slot=true,r6_panel_copy=true | Navatala HIP MFMA kernel navatala_transformer_tiled_gemm_f16_mfma_cta64_pipelined | rocBLAS rocblas_gemm_ex F16 inputs/F32 accumulation | mfma_f16 | hip_mfma_gfx942_64x64x8_f16_f32_cta64_pipelined | pass | 0.057132 | 0.044736 | 1.277x | 1.3113e-06 |
| GEMM_F16_MFMA_CTA128 | m=1024,n=512,k=1024,compute=F32,cta=128x128x32 | Navatala HIP MFMA kernel navatala_transformer_tiled_gemm_f16_mfma_cta128 | rocBLAS rocblas_gemm_ex F16 inputs/F32 accumulation | mfma_f16 | hip_mfma_gfx942_128x128x32_f16_f32_cta128 | pass | 0.051527 | 0.044936 | 1.147x | 1.3113e-06 |
| GEMM_F16_MFMA_CTA64_SHARED_EDGE | m=1024,n=512,k=1024,compute=F32,edge=true | Navatala HIP MFMA kernel navatala_transformer_tiled_gemm_f16_mfma_cta64_shared_edge | rocBLAS rocblas_gemm_ex F16 inputs/F32 accumulation | mfma_f16 | hip_mfma_gfx942_64x64x8_f16_f32_cta64_shared_edge | pass | 0.090843 | 0.045521 | 1.996x | 1.3113e-06 |
| GEMM_F16_MFMA_CTA128_EDGE | m=1024,n=512,k=1024,compute=F32,edge=true | Navatala HIP MFMA kernel navatala_transformer_tiled_gemm_f16_mfma_cta128_edge | rocBLAS rocblas_gemm_ex F16 inputs/F32 accumulation | mfma_f16 | hip_mfma_gfx942_128x128x32_f16_f32_cta128_edge | pass | 0.130862 | 0.045441 | 2.880x | 1.3113e-06 |
| GEMM_F16_PORTABLE_F32OUT | m=512,n=1024,k=1024,output=F32,compute=F32 | Navatala HIP kernel navatala_transformer_tiled_gemm_f16_f32_out | rocBLAS rocblas_gemm_ex F16 input/F32 output/F32 accumulation | scalar | portable_f16_f32out_tiled | pass | 0.084972 | 0.045305 | 1.876x | 2.86102e-06 |
| GEMM_F16_MFMA_CTA64_SHARED | m=512,n=1024,k=1024,compute=F32,cta=64x64x8,lds_staged=true | Navatala HIP MFMA kernel navatala_transformer_tiled_gemm_f16_mfma_cta64_shared | rocBLAS rocblas_gemm_ex F16 inputs/F32 accumulation | mfma_f16 | hip_mfma_gfx942_64x64x8_f16_f32_cta64_shared | pass | 0.042466 | 0.044487 | 0.955x | 2.14577e-06 |
| GEMM_F16_MFMA_CTA64_PIPELINED | m=512,n=1024,k=1024,compute=F32,cta=64x64x8,lds_staged=true,two_slot=true,r6_panel_copy=true | Navatala HIP MFMA kernel navatala_transformer_tiled_gemm_f16_mfma_cta64_pipelined | rocBLAS rocblas_gemm_ex F16 inputs/F32 accumulation | mfma_f16 | hip_mfma_gfx942_64x64x8_f16_f32_cta64_pipelined | pass | 0.045585 | 0.045617 | 0.999x | 2.14577e-06 |
| GEMM_F16_MFMA_CTA128 | m=512,n=1024,k=1024,compute=F32,cta=128x128x32 | Navatala HIP MFMA kernel navatala_transformer_tiled_gemm_f16_mfma_cta128 | rocBLAS rocblas_gemm_ex F16 inputs/F32 accumulation | mfma_f16 | hip_mfma_gfx942_128x128x32_f16_f32_cta128 | pass | 0.051824 | 0.045040 | 1.151x | 2.14577e-06 |
| GEMM_F16_MFMA_CTA64_SHARED_EDGE | m=512,n=1024,k=1024,compute=F32,edge=true | Navatala HIP MFMA kernel navatala_transformer_tiled_gemm_f16_mfma_cta64_shared_edge | rocBLAS rocblas_gemm_ex F16 inputs/F32 accumulation | mfma_f16 | hip_mfma_gfx942_64x64x8_f16_f32_cta64_shared_edge | pass | 0.066834 | 0.044880 | 1.489x | 2.14577e-06 |
| GEMM_F16_MFMA_CTA128_EDGE | m=512,n=1024,k=1024,compute=F32,edge=true | Navatala HIP MFMA kernel navatala_transformer_tiled_gemm_f16_mfma_cta128_edge | rocBLAS rocblas_gemm_ex F16 inputs/F32 accumulation | mfma_f16 | hip_mfma_gfx942_128x128x32_f16_f32_cta128_edge | pass | 0.132249 | 0.045048 | 2.936x | 2.14577e-06 |
| GEMM_F16_PORTABLE_F32OUT | m=2048,n=512,k=1024,output=F32,compute=F32 | Navatala HIP kernel navatala_transformer_tiled_gemm_f16_f32_out | rocBLAS rocblas_gemm_ex F16 input/F32 output/F32 accumulation | scalar | portable_f16_f32out_tiled | pass | 0.157460 | 0.045345 | 3.472x | 2.5034e-06 |
| GEMM_F16_MFMA_CTA64_SHARED | m=2048,n=512,k=1024,compute=F32,cta=64x64x8,lds_staged=true | Navatala HIP MFMA kernel navatala_transformer_tiled_gemm_f16_mfma_cta64_shared | rocBLAS rocblas_gemm_ex F16 inputs/F32 accumulation | mfma_f16 | hip_mfma_gfx942_64x64x8_f16_f32_cta64_shared | pass | 0.060395 | 0.045441 | 1.329x | 1.3113e-06 |
| GEMM_F16_MFMA_CTA64_PIPELINED | m=2048,n=512,k=1024,compute=F32,cta=64x64x8,lds_staged=true,two_slot=true,r6_panel_copy=true | Navatala HIP MFMA kernel navatala_transformer_tiled_gemm_f16_mfma_cta64_pipelined | rocBLAS rocblas_gemm_ex F16 inputs/F32 accumulation | mfma_f16 | hip_mfma_gfx942_64x64x8_f16_f32_cta64_pipelined | pass | 0.062665 | 0.045713 | 1.371x | 1.3113e-06 |
| GEMM_F16_MFMA_CTA128 | m=2048,n=512,k=1024,compute=F32,cta=128x128x32 | Navatala HIP MFMA kernel navatala_transformer_tiled_gemm_f16_mfma_cta128 | rocBLAS rocblas_gemm_ex F16 inputs/F32 accumulation | mfma_f16 | hip_mfma_gfx942_128x128x32_f16_f32_cta128 | pass | 0.058776 | 0.045802 | 1.283x | 1.3113e-06 |
| GEMM_F16_MFMA_CTA64_SHARED_EDGE | m=2048,n=512,k=1024,compute=F32,edge=true | Navatala HIP MFMA kernel navatala_transformer_tiled_gemm_f16_mfma_cta64_shared_edge | rocBLAS rocblas_gemm_ex F16 inputs/F32 accumulation | mfma_f16 | hip_mfma_gfx942_64x64x8_f16_f32_cta64_shared_edge | pass | 0.098114 | 0.045208 | 2.170x | 1.3113e-06 |
| GEMM_F16_MFMA_CTA128_EDGE | m=2048,n=512,k=1024,compute=F32,edge=true | Navatala HIP MFMA kernel navatala_transformer_tiled_gemm_f16_mfma_cta128_edge | rocBLAS rocblas_gemm_ex F16 inputs/F32 accumulation | mfma_f16 | hip_mfma_gfx942_128x128x32_f16_f32_cta128_edge | pass | 0.185957 | 0.045729 | 4.066x | 1.3113e-06 |

## Interpretation Notes

- `generatedMeanMs` and `vendorMeanMs` are back-to-back throughput means per launch, not isolated launch latency.
- Correctness is measured against the vendor baseline in the benchmark harness for the listed shape.
- This report covers only the operations listed above; it is not a full HIP backend certification.
- hipSPARSELt is reported as a capability only; no hipSPARSELt performance baseline is claimed.

## Validation Warnings

- hipSPARSELt is capability reporting only; no hipSPARSELt benchmark row is claimed
- iterations<10 with quickMode=false; report may be too noisy for publication
