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
  if (beta == 0.0f) {
    if (alpha == 1.0f) {
      #pragma unroll
      for (unsigned int i = 0; i < 16u; ++i) {
        const unsigned int localRow = (8u * (i / 4u)) + rowGroup + (i & 3u);
        const unsigned int row = rowOffset + localRow;
        const unsigned int col = colOffset + localCol;
        if (row < rowLimit && col < colLimit) {
          dst[row * stride + col] = acc[i];
        }
      }
      return;
    }
    #pragma unroll
    for (unsigned int i = 0; i < 16u; ++i) {
      const unsigned int localRow = (8u * (i / 4u)) + rowGroup + (i & 3u);
      const unsigned int row = rowOffset + localRow;
      const unsigned int col = colOffset + localCol;
      if (row < rowLimit && col < colLimit) {
        dst[row * stride + col] = alpha * acc[i];
      }
    }
    return;
  }
  #pragma unroll
  for (unsigned int i = 0; i < 16u; ++i) {
    const unsigned int localRow = (8u * (i / 4u)) + rowGroup + (i & 3u);
    const unsigned int row = rowOffset + localRow;
    const unsigned int col = colOffset + localCol;
    if (row < rowLimit && col < colLimit) {
      const unsigned int idx = row * stride + col;
      dst[idx] = alpha * acc[i] + beta * dst[idx];
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
extern "C" __global__ __launch_bounds__(256) void navatala_transformer_tiled_gemm_f16_mfma_cta64_pipelined(const __half* a, const __half* b, const unsigned int* kTiles, const unsigned int* kStride, const unsigned int* nStride, float* c) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int tid = ((unsigned int)((int)(threadIdx.x)));
  unsigned int waveId = (tid / 64u);
  unsigned int waveRowGroup = (waveId / 2u);
  unsigned int waveColGroup = (waveId % 2u);
  unsigned int blockRow = (((unsigned int)((int)(blockIdx.y))) * 64u);
  unsigned int blockCol = (((unsigned int)((int)(blockIdx.x))) * 64u);
  unsigned int tileRow = (blockRow + (waveRowGroup * 32u));
  unsigned int tileCol = (blockCol + (waveColGroup * 32u));
  unsigned int waveRowLocal = (waveRowGroup * 32u);
  unsigned int waveColLocal = (waveColGroup * 32u);
  unsigned int kTilesVal = kTiles[0u];
  unsigned int kStrideVal = kStride[0u];
  unsigned int nStrideVal = nStride[0u];
  __shared__ __align__(16) __half panelA[1024]; /* R6 shared panel: GPU.MatrixPanelRole.A, slots=2 */
  __shared__ __align__(16) __half panelB[1024]; /* R6 shared panel: GPU.MatrixPanelRole.B, slots=2 */
  navatala_mfma_acc_f32_32x32x8 acc = navatala_mfma_zero_acc_f32_32x32x8();
  for (int kTile = 0; kTile < (int)(kTilesVal); ++kTile) {
    unsigned int kTileU32 = ((unsigned int)(kTile));
    unsigned int slot = (kTileU32 % 2u);
    unsigned int kBase = (kTileU32 * 8u);
    bool isFirstTile = (kTileU32 == 0u);
    {
      /* R6 CopyGlobalToShared HIP direct global-to-LDS dword path for vector-b16; group=slot */
      const bool navatala_copy_direct_ok = ((unsigned int)(kStrideVal) % 2u) == 0u && ((unsigned int)(kBase) % 2u) == 0u && ((unsigned int)(0u) % 2u) == 0u;
      if (isFirstTile) {
        if (__builtin_expect(navatala_copy_direct_ok, 1)) {
          const __amdgpu_buffer_rsrc_t navatala_copy_src_resource = navatala_gpu_make_buffer_resource(&a[0]);
          for (unsigned int navatala_copy_dw = (unsigned int)threadIdx.x; navatala_copy_dw < 256u; navatala_copy_dw += (unsigned int)blockDim.x) {
            const unsigned int navatala_copy_elem = navatala_copy_dw * 2u;
            const unsigned int navatala_copy_r = navatala_copy_elem / 8u;
            const unsigned int navatala_copy_c = navatala_copy_elem - (navatala_copy_r * 8u);
            const unsigned int navatala_dst_idx = ((slot) * 512u) + (((0u) + navatala_copy_r) * 8u) + ((0u) + navatala_copy_c);
            const unsigned int navatala_src_idx = (((blockRow) + navatala_copy_r) * (kStrideVal)) + ((kBase) + navatala_copy_c);
            navatala_gpu_direct_load_dword_to_lds_prepared(navatala_copy_src_resource, navatala_src_idx, &panelA[0], navatala_dst_idx);
          }
        } else {
          navatala_gpu_scalar_copy_global_to_shared(&a[0], &panelA[0], (unsigned int)(slot), (unsigned int)(0u), (unsigned int)(0u), (unsigned int)(blockRow), (unsigned int)(kBase), (unsigned int)(kStrideVal), 64u, 8u, 8u, 512u);
        }
      }
    }
    {
      /* R6 CopyGlobalToShared HIP direct global-to-LDS dword path for vector-b16; group=slot */
      const bool navatala_copy_direct_ok = ((unsigned int)(nStrideVal) % 2u) == 0u && ((unsigned int)(blockCol) % 2u) == 0u && ((unsigned int)(0u) % 2u) == 0u;
      if (isFirstTile) {
        if (__builtin_expect(navatala_copy_direct_ok, 1)) {
          const __amdgpu_buffer_rsrc_t navatala_copy_src_resource = navatala_gpu_make_buffer_resource(&b[0]);
          for (unsigned int navatala_copy_dw = (unsigned int)threadIdx.x; navatala_copy_dw < 256u; navatala_copy_dw += (unsigned int)blockDim.x) {
            const unsigned int navatala_copy_elem = navatala_copy_dw * 2u;
            const unsigned int navatala_copy_r = navatala_copy_elem / 64u;
            const unsigned int navatala_copy_c = navatala_copy_elem - (navatala_copy_r * 64u);
            const unsigned int navatala_dst_idx = ((slot) * 512u) + (((0u) + navatala_copy_r) * 64u) + ((0u) + navatala_copy_c);
            const unsigned int navatala_src_idx = (((kBase) + navatala_copy_r) * (nStrideVal)) + ((blockCol) + navatala_copy_c);
            navatala_gpu_direct_load_dword_to_lds_prepared(navatala_copy_src_resource, navatala_src_idx, &panelB[0], navatala_dst_idx);
          }
        } else {
          navatala_gpu_scalar_copy_global_to_shared(&b[0], &panelB[0], (unsigned int)(slot), (unsigned int)(0u), (unsigned int)(0u), (unsigned int)(kBase), (unsigned int)(blockCol), (unsigned int)(nStrideVal), 8u, 64u, 64u, 512u);
        }
      }
    }
    /* R6 AwaitCopyGroup HIP direct-LDS wait: group=slot */
    navatala_gpu_wait_direct_lds_copy();
    __syncthreads();
    unsigned int nextTileU32 = (kTileU32 + 1u);
    bool hasNextTile = (nextTileU32 < kTilesVal);
    unsigned int nextSlot = (nextTileU32 % 2u);
    unsigned int nextKBase = (nextTileU32 * 8u);
    {
      /* R6 CopyGlobalToShared HIP direct global-to-LDS dword path for vector-b16; group=nextSlot */
      const bool navatala_copy_direct_ok = ((unsigned int)(kStrideVal) % 2u) == 0u && ((unsigned int)(nextKBase) % 2u) == 0u && ((unsigned int)(0u) % 2u) == 0u;
      if (hasNextTile) {
        if (__builtin_expect(navatala_copy_direct_ok, 1)) {
          const __amdgpu_buffer_rsrc_t navatala_copy_src_resource = navatala_gpu_make_buffer_resource(&a[0]);
          for (unsigned int navatala_copy_dw = (unsigned int)threadIdx.x; navatala_copy_dw < 256u; navatala_copy_dw += (unsigned int)blockDim.x) {
            const unsigned int navatala_copy_elem = navatala_copy_dw * 2u;
            const unsigned int navatala_copy_r = navatala_copy_elem / 8u;
            const unsigned int navatala_copy_c = navatala_copy_elem - (navatala_copy_r * 8u);
            const unsigned int navatala_dst_idx = ((nextSlot) * 512u) + (((0u) + navatala_copy_r) * 8u) + ((0u) + navatala_copy_c);
            const unsigned int navatala_src_idx = (((blockRow) + navatala_copy_r) * (kStrideVal)) + ((nextKBase) + navatala_copy_c);
            navatala_gpu_direct_load_dword_to_lds_prepared(navatala_copy_src_resource, navatala_src_idx, &panelA[0], navatala_dst_idx);
          }
        } else {
          navatala_gpu_scalar_copy_global_to_shared(&a[0], &panelA[0], (unsigned int)(nextSlot), (unsigned int)(0u), (unsigned int)(0u), (unsigned int)(blockRow), (unsigned int)(nextKBase), (unsigned int)(kStrideVal), 64u, 8u, 8u, 512u);
        }
      }
    }
    {
      /* R6 CopyGlobalToShared HIP direct global-to-LDS dword path for vector-b16; group=nextSlot */
      const bool navatala_copy_direct_ok = ((unsigned int)(nStrideVal) % 2u) == 0u && ((unsigned int)(blockCol) % 2u) == 0u && ((unsigned int)(0u) % 2u) == 0u;
      if (hasNextTile) {
        if (__builtin_expect(navatala_copy_direct_ok, 1)) {
          const __amdgpu_buffer_rsrc_t navatala_copy_src_resource = navatala_gpu_make_buffer_resource(&b[0]);
          for (unsigned int navatala_copy_dw = (unsigned int)threadIdx.x; navatala_copy_dw < 256u; navatala_copy_dw += (unsigned int)blockDim.x) {
            const unsigned int navatala_copy_elem = navatala_copy_dw * 2u;
            const unsigned int navatala_copy_r = navatala_copy_elem / 64u;
            const unsigned int navatala_copy_c = navatala_copy_elem - (navatala_copy_r * 64u);
            const unsigned int navatala_dst_idx = ((nextSlot) * 512u) + (((0u) + navatala_copy_r) * 64u) + ((0u) + navatala_copy_c);
            const unsigned int navatala_src_idx = (((nextKBase) + navatala_copy_r) * (nStrideVal)) + ((blockCol) + navatala_copy_c);
            navatala_gpu_direct_load_dword_to_lds_prepared(navatala_copy_src_resource, navatala_src_idx, &panelB[0], navatala_dst_idx);
          }
        } else {
          navatala_gpu_scalar_copy_global_to_shared(&b[0], &panelB[0], (unsigned int)(nextSlot), (unsigned int)(0u), (unsigned int)(0u), (unsigned int)(nextKBase), (unsigned int)(blockCol), (unsigned int)(nStrideVal), 8u, 64u, 64u, 512u);
        }
      }
    }
    navatala_mfma_f16x4 aFrag = navatala_mfma_load_a_f16_32x32x8(((panelA) + ((slot) * 512u)), waveRowLocal, 0u, 8u);
    navatala_mfma_f16x4 bFrag = navatala_mfma_load_b_f16_32x32x8(((panelB) + ((slot) * 512u)), 0u, waveColLocal, 64u);
    acc = __builtin_amdgcn_mfma_f32_32x32x8f16(aFrag, bFrag, acc, 0, 0, 0);
    __syncthreads();
  }
  navatala_mfma_store_acc_f32_32x32x8_at(c, acc, tileRow, tileCol, nStrideVal, true);
}
