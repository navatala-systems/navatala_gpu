// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Navatala Systems (OPC) Pvt Ltd
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#ifndef NAVATALA_GPU_HIP_MFMA_F16_F32_32X32X8_HELPERS
#define NAVATALA_GPU_HIP_MFMA_F16_F32_32X32X8_HELPERS
#if defined(NAVATALA_GPU_ROCM_VERSION) && NAVATALA_GPU_ROCM_VERSION < 60200
#error "matrix-intrinsic kernels require ROCm 6.2+"
#endif
typedef _Float16 navatala_mfma_f16x4 __attribute__((ext_vector_type(4)));
typedef float navatala_mfma_acc_f32_32x32x8 __attribute__((ext_vector_type(16)));
__device__ inline navatala_mfma_acc_f32_32x32x8 navatala_mfma_zero_acc_f32_32x32x8() {
  return {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
}
__device__ inline navatala_mfma_f16x4 navatala_mfma_load_a_f16_32x32x8(const __half* base, unsigned int row, unsigned int col, unsigned int stride) {
  const unsigned int lane = (unsigned int)(threadIdx.x & 63u);
  const unsigned int localRow = lane & 31u;
  const unsigned int kBase = (lane >> 5u) * 4u;
  return {(_Float16)base[(row + localRow) * stride + (col + kBase + 0u)],
          (_Float16)base[(row + localRow) * stride + (col + kBase + 1u)],
          (_Float16)base[(row + localRow) * stride + (col + kBase + 2u)],
          (_Float16)base[(row + localRow) * stride + (col + kBase + 3u)]};
}
__device__ inline navatala_mfma_f16x4 navatala_mfma_load_b_f16_32x32x8(const __half* base, unsigned int row, unsigned int col, unsigned int stride) {
  const unsigned int lane = (unsigned int)(threadIdx.x & 63u);
  const unsigned int localCol = lane & 31u;
  const unsigned int kBase = (lane >> 5u) * 4u;
  return {(_Float16)base[(row + kBase + 0u) * stride + (col + localCol)],
          (_Float16)base[(row + kBase + 1u) * stride + (col + localCol)],
          (_Float16)base[(row + kBase + 2u) * stride + (col + localCol)],
          (_Float16)base[(row + kBase + 3u) * stride + (col + localCol)]};
}
__device__ inline void navatala_mfma_store_acc_f32_32x32x8(float* dst, navatala_mfma_acc_f32_32x32x8 acc, unsigned int stride) {
  const unsigned int lane = (unsigned int)(threadIdx.x & 63u);
  const unsigned int localCol = lane & 31u;
  const unsigned int rowGroup = 4u * (lane >> 5u);
  #pragma unroll
  for (unsigned int i = 0; i < 16u; ++i) {
    const unsigned int localRow = (8u * (i / 4u)) + rowGroup + (i & 3u);
    dst[((unsigned int)(blockIdx.y * 32u) + localRow) * stride + ((unsigned int)(blockIdx.x * 32u) + localCol)] = acc[i];
  }
}
__device__ inline void navatala_mfma_store_acc_f32_32x32x8_at(float* dst, navatala_mfma_acc_f32_32x32x8 acc, unsigned int rowOffset, unsigned int colOffset, unsigned int stride, bool predicate) {
  if (!predicate) { return; }
  const unsigned int lane = (unsigned int)(threadIdx.x & 63u);
  const unsigned int localCol = lane & 31u;
  const unsigned int rowGroup = 4u * (lane >> 5u);
  #pragma unroll
  for (unsigned int i = 0; i < 16u; ++i) {
    const unsigned int localRow = (8u * (i / 4u)) + rowGroup + (i & 3u);
    dst[(rowOffset + localRow) * stride + (colOffset + localCol)] = acc[i];
  }
}
__device__ inline void navatala_mfma_store_acc_f32_32x32x8_at_bounded(float* dst, navatala_mfma_acc_f32_32x32x8 acc, unsigned int rowOffset, unsigned int colOffset, unsigned int stride, unsigned int rowLimit, unsigned int colLimit, bool predicate) {
  if (!predicate) { return; }
  const unsigned int lane = (unsigned int)(threadIdx.x & 63u);
  const unsigned int localCol = lane & 31u;
  const unsigned int rowGroup = 4u * (lane >> 5u);
  #pragma unroll
  for (unsigned int i = 0; i < 16u; ++i) {
    const unsigned int localRow = (8u * (i / 4u)) + rowGroup + (i & 3u);
    const unsigned int row = rowOffset + localRow;
    const unsigned int col = colOffset + localCol;
    if (row < rowLimit && col < colLimit) {
      dst[row * stride + col] = acc[i];
    }
  }
}
__device__ inline void navatala_mfma_store_acc_f32_32x32x8_at_scaled_bounded(float* dst, navatala_mfma_acc_f32_32x32x8 acc, unsigned int rowOffset, unsigned int colOffset, unsigned int stride, unsigned int rowLimit, unsigned int colLimit, float alpha, float beta, bool predicate) {
  if (!predicate) { return; }
  const unsigned int lane = (unsigned int)(threadIdx.x & 63u);
  const unsigned int localCol = lane & 31u;
  const unsigned int rowGroup = 4u * (lane >> 5u);
  #pragma unroll
  for (unsigned int i = 0; i < 16u; ++i) {
    const unsigned int localRow = (8u * (i / 4u)) + rowGroup + (i & 3u);
    const unsigned int row = rowOffset + localRow;
    const unsigned int col = colOffset + localCol;
    if (row < rowLimit && col < colLimit) {
      const unsigned int idx = row * stride + col;
      const float scaled = alpha * acc[i];
      dst[idx] = (beta == 0.0f) ? scaled : (scaled + beta * dst[idx]);
    }
  }
}
#endif  // NAVATALA_GPU_HIP_MFMA_F16_F32_32X32X8_HELPERS
#ifndef NAVATALA_GPU_HIP_DIRECT_LDS_COPY_HELPERS
#define NAVATALA_GPU_HIP_DIRECT_LDS_COPY_HELPERS
__device__ inline __amdgpu_buffer_rsrc_t navatala_gpu_make_buffer_resource(const void* base) {
  return __builtin_amdgcn_make_buffer_rsrc(const_cast<void*>(base), 0, 0xffffffffu, 0x00020000u);
}
template <typename T>
__device__ inline void navatala_gpu_direct_load_dword_to_lds_prepared(__amdgpu_buffer_rsrc_t srcResource, unsigned int srcElemOffset, T* dstBase, unsigned int dstElemOffset) {
#if defined(__gfx942__)
  const unsigned int ldsAddr = (unsigned int)reinterpret_cast<unsigned long>(&dstBase[dstElemOffset]);
  const unsigned int ldsBase = __builtin_amdgcn_readfirstlane(ldsAddr);
  const unsigned int srcByteOffset = srcElemOffset * (unsigned int)sizeof(T);
  asm volatile("s_mov_b32 m0, %0;\n\tbuffer_load_dword %1, %2, 0 offen lds;\n\t" :: "s"(ldsBase), "v"(srcByteOffset), "s"(srcResource) : "memory");
#else
  (void)srcResource;
  (void)srcElemOffset;
  (void)dstBase;
  (void)dstElemOffset;
#endif
}
template <typename T>
__device__ inline void navatala_gpu_direct_load_dword_to_lds(const T* srcBase, unsigned int srcElemOffset, T* dstBase, unsigned int dstElemOffset) {
#if defined(__gfx942__)
  const __amdgpu_buffer_rsrc_t srcResource = navatala_gpu_make_buffer_resource(srcBase);
  navatala_gpu_direct_load_dword_to_lds_prepared(srcResource, srcElemOffset, dstBase, dstElemOffset);
#else
  *reinterpret_cast<unsigned int*>(&dstBase[dstElemOffset]) = *reinterpret_cast<const unsigned int*>(&srcBase[srcElemOffset]);
#endif
}
template <typename T>
__device__ __attribute__((noinline)) void navatala_gpu_scalar_copy_global_to_shared(const T* srcBase, T* dstBase, unsigned int slot, unsigned int dstRow, unsigned int dstCol, unsigned int srcRow, unsigned int srcCol, unsigned int srcStride, unsigned int rows, unsigned int cols, unsigned int panelStride, unsigned int elemsPerSlot) {
  for (unsigned int navatala_copy_i = (unsigned int)threadIdx.x; navatala_copy_i < rows * cols; navatala_copy_i += (unsigned int)blockDim.x) {
    const unsigned int navatala_copy_r = navatala_copy_i / cols;
    const unsigned int navatala_copy_c = navatala_copy_i - (navatala_copy_r * cols);
    dstBase[(slot * elemsPerSlot) + ((dstRow + navatala_copy_r) * panelStride) + (dstCol + navatala_copy_c)] = srcBase[(srcRow + navatala_copy_r) * srcStride + (srcCol + navatala_copy_c)];
  }
}
__device__ inline void navatala_gpu_wait_direct_lds_copy() {
#if defined(__gfx942__)
  asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
#endif
}
#endif  // NAVATALA_GPU_HIP_DIRECT_LDS_COPY_HELPERS
extern "C" __global__ __launch_bounds__(256) void navatala_transformer_tiled_gemm_f16_mfma_cta128_edge(const __half* a, const __half* b, const unsigned int* mDim, const unsigned int* nDim, const unsigned int* kDim, const unsigned int* kBlocks, const unsigned int* aStride, const unsigned int* bStride, const unsigned int* cStride, const unsigned int* aBatchStride, const unsigned int* bBatchStride, const unsigned int* cBatchStride, const unsigned int* transA, const unsigned int* transB, const float* alpha, const float* beta, float* c) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int tid = ((unsigned int)((int)(threadIdx.x)));
  unsigned int batchIdx = ((unsigned int)((int)(blockIdx.z)));
  unsigned int waveId = (tid / 64u);
  unsigned int waveRowGroup = (waveId / 2u);
  unsigned int waveColGroup = (waveId % 2u);
  unsigned int blockRow = (((unsigned int)((int)(blockIdx.y))) * 128u);
  unsigned int blockCol = (((unsigned int)((int)(blockIdx.x))) * 128u);
  unsigned int waveRowLocalBase = (waveRowGroup * 64u);
  unsigned int waveColLocalBase = (waveColGroup * 64u);
  unsigned int waveRowBase = (blockRow + waveRowLocalBase);
  unsigned int waveColBase = (blockCol + waveColLocalBase);
  unsigned int mDimVal = mDim[0u];
  unsigned int nDimVal = nDim[0u];
  unsigned int kDimVal = kDim[0u];
  unsigned int kBlocksVal = kBlocks[0u];
  unsigned int aStrideVal = aStride[0u];
  unsigned int bStrideVal = bStride[0u];
  unsigned int cStrideVal = cStride[0u];
  unsigned int aBatchStrideVal = aBatchStride[0u];
  unsigned int bBatchStrideVal = bBatchStride[0u];
  unsigned int cBatchStrideVal = cBatchStride[0u];
  unsigned int transAVal = transA[0u];
  unsigned int transBVal = transB[0u];
  bool useTransA = (transAVal != 0u);
  bool useTransB = (transBVal != 0u);
  float alphaVal = alpha[0u];
  float betaVal = beta[0u];
  unsigned int aBatchBase = (batchIdx * aBatchStrideVal);
  unsigned int bBatchBase = (batchIdx * bBatchStrideVal);
  unsigned int cBatchBase = (batchIdx * cBatchStrideVal);
  unsigned int cWaveColBase = (cBatchBase + waveColBase);
  unsigned int cColLimit = (cBatchBase + nDimVal);
  __shared__ __half tileA[4096];
  __shared__ __half tileB[4096];
  navatala_mfma_acc_f32_32x32x8 acc00 = navatala_mfma_zero_acc_f32_32x32x8();
  navatala_mfma_acc_f32_32x32x8 acc01 = navatala_mfma_zero_acc_f32_32x32x8();
  navatala_mfma_acc_f32_32x32x8 acc10 = navatala_mfma_zero_acc_f32_32x32x8();
  navatala_mfma_acc_f32_32x32x8 acc11 = navatala_mfma_zero_acc_f32_32x32x8();
  for (int kBlock = 0; kBlock < (int)(kBlocksVal); ++kBlock) {
    unsigned int kBlockU32 = ((unsigned int)(kBlock));
    unsigned int kBase = (kBlockU32 * 32u);
    for (int loadIter = 0; loadIter < (int)(16u); ++loadIter) {
      unsigned int loadIterU32 = ((unsigned int)(loadIter));
      unsigned int _flat = ((loadIterU32 * 256u) + tid);
      unsigned int aRowLocal = (_flat / 32u);
      unsigned int aColLocal = (_flat % 32u);
      unsigned int aGlobalRow = (blockRow + aRowLocal);
      unsigned int aGlobalCol = (kBase + aColLocal);
      unsigned int aIdxNN = ((aGlobalRow * aStrideVal) + aGlobalCol);
      unsigned int aIdxT = ((aGlobalCol * aStrideVal) + aGlobalRow);
      unsigned int aIdx = ((useTransA) ? (aIdxT) : (aIdxNN));
      unsigned int aIdxBatch = (aBatchBase + aIdx);
      bool aRowValid = (aGlobalRow < mDimVal);
      bool aColValid = (aGlobalCol < kDimVal);
      bool aLoadValid = (aRowValid && aColValid);
      __half aVal = ((aLoadValid) ? (a[aIdxBatch]) : (__float2half(0.000000f)));
      unsigned int bRowLocal = (_flat / 128u);
      unsigned int bColLocal = (_flat % 128u);
      unsigned int bGlobalRow = (kBase + bRowLocal);
      unsigned int bGlobalCol = (blockCol + bColLocal);
      unsigned int bIdxNN = ((bGlobalRow * bStrideVal) + bGlobalCol);
      unsigned int bIdxT = ((bGlobalCol * bStrideVal) + bGlobalRow);
      unsigned int bIdx = ((useTransB) ? (bIdxT) : (bIdxNN));
      unsigned int bIdxBatch = (bBatchBase + bIdx);
      bool bRowValid = (bGlobalRow < kDimVal);
      bool bColValid = (bGlobalCol < nDimVal);
      bool bLoadValid = (bRowValid && bColValid);
      __half bVal = ((bLoadValid) ? (b[bIdxBatch]) : (__float2half(0.000000f)));
      tileA[_flat] = aVal;
      tileB[_flat] = bVal;
    }
    __syncthreads();
    for (int kkTile = 0; kkTile < (int)(4u); ++kkTile) {
      unsigned int kkTileU32 = ((unsigned int)(kkTile));
      unsigned int kkBase = (kkTileU32 * 8u);
      navatala_mfma_f16x4 aFrag0 = navatala_mfma_load_a_f16_32x32x8(tileA, waveRowLocalBase, kkBase, 32u);
      navatala_mfma_f16x4 aFrag1 = navatala_mfma_load_a_f16_32x32x8(tileA, (waveRowLocalBase + 32u), kkBase, 32u);
      navatala_mfma_f16x4 bFrag0 = navatala_mfma_load_b_f16_32x32x8(tileB, kkBase, waveColLocalBase, 128u);
      navatala_mfma_f16x4 bFrag1 = navatala_mfma_load_b_f16_32x32x8(tileB, kkBase, (waveColLocalBase + 32u), 128u);
      acc00 = __builtin_amdgcn_mfma_f32_32x32x8f16(aFrag0, bFrag0, acc00, 0, 0, 0);
      acc01 = __builtin_amdgcn_mfma_f32_32x32x8f16(aFrag0, bFrag1, acc01, 0, 0, 0);
      acc10 = __builtin_amdgcn_mfma_f32_32x32x8f16(aFrag1, bFrag0, acc10, 0, 0, 0);
      acc11 = __builtin_amdgcn_mfma_f32_32x32x8f16(aFrag1, bFrag1, acc11, 0, 0, 0);
    }
    __syncthreads();
  }
  navatala_mfma_store_acc_f32_32x32x8_at_scaled_bounded(c, acc00, waveRowBase, cWaveColBase, cStrideVal, mDimVal, cColLimit, alphaVal, betaVal, true);
  navatala_mfma_store_acc_f32_32x32x8_at_scaled_bounded(c, acc01, waveRowBase, (cWaveColBase + 32u), cStrideVal, mDimVal, cColLimit, alphaVal, betaVal, true);
  navatala_mfma_store_acc_f32_32x32x8_at_scaled_bounded(c, acc10, (waveRowBase + 32u), cWaveColBase, cStrideVal, mDimVal, cColLimit, alphaVal, betaVal, true);
  navatala_mfma_store_acc_f32_32x32x8_at_scaled_bounded(c, acc11, (waveRowBase + 32u), (cWaveColBase + 32u), cStrideVal, mDimVal, cColLimit, alphaVal, betaVal, true);
}
