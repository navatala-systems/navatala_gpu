// SPDX-License-Identifier: Apache-2.0
// Kernel registry partition for the `hip_nccl` bucket. Compiles
// against the declarations in `navatala/_registry.hpp` and links
// against the trampoline in `_registry.cpp`. Partitioning keeps each
// translation unit well under the 50K-line threshold so a typical
// build can parallelise across the buckets.
#include "navatala/_registry.hpp"
#include <cstdint>
#include <string>
#include <string_view>

const char* k_hip_navatala_nccl_direct_send_f32 = R"kernel(
#include <hip/hip_runtime.h>
extern "C" __global__ void navatala_nccl_direct_send_f32(const float* src, const unsigned int* count, float* dst) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int countVal = count[0];
  if (gid < countVal) {
    float val = src[gid];
    dst[gid] = val;
  }
}

)kernel";
const char* k_hip_navatala_nccl_direct_send_f16 = R"kernel(
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
extern "C" __global__ void navatala_nccl_direct_send_f16(const __half* src, const unsigned int* count, __half* dst) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int countVal = count[0];
  if (gid < countVal) {
    __half val = src[gid];
    dst[gid] = val;
  }
}

)kernel";
const char* k_hip_navatala_nccl_direct_send_i32 = R"kernel(
#include <hip/hip_runtime.h>
extern "C" __global__ void navatala_nccl_direct_send_i32(const int* src, const unsigned int* count, int* dst) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int countVal = count[0];
  if (gid < countVal) {
    int val = src[gid];
    dst[gid] = val;
  }
}

)kernel";
const char* k_hip_navatala_nccl_direct_recv_f32 = R"kernel(
#include <hip/hip_runtime.h>
extern "C" __global__ void navatala_nccl_direct_recv_f32(const float* src, const unsigned int* count, float* dst) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int countVal = count[0];
  if (gid < countVal) {
    float val = src[gid];
    dst[gid] = val;
  }
}

)kernel";
const char* k_hip_navatala_nccl_direct_recv_f16 = R"kernel(
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
extern "C" __global__ void navatala_nccl_direct_recv_f16(const __half* src, const unsigned int* count, __half* dst) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int countVal = count[0];
  if (gid < countVal) {
    __half val = src[gid];
    dst[gid] = val;
  }
}

)kernel";
const char* k_hip_navatala_nccl_direct_recv_i32 = R"kernel(
#include <hip/hip_runtime.h>
extern "C" __global__ void navatala_nccl_direct_recv_i32(const int* src, const unsigned int* count, int* dst) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int countVal = count[0];
  if (gid < countVal) {
    int val = src[gid];
    dst[gid] = val;
  }
}

)kernel";
const char* k_hip_navatala_nccl_reduce_local_sum_f32 = R"kernel(
#include <hip/hip_runtime.h>
extern "C" __global__ void navatala_nccl_reduce_local_sum_f32(const float* a, const float* b, const unsigned int* count, float* _out) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int countVal = count[0];
  if (gid < countVal) {
    float va = a[gid];
    float vb = b[gid];
    float result = (va + vb);
    _out[gid] = result;
  }
}

)kernel";
const char* k_hip_navatala_nccl_reduce_local_sum_f16 = R"kernel(
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
extern "C" __global__ void navatala_nccl_reduce_local_sum_f16(const __half* a, const __half* b, const unsigned int* count, __half* _out) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int countVal = count[0];
  if (gid < countVal) {
    __half va = a[gid];
    __half vb = b[gid];
    __half result = (va + vb);
    _out[gid] = result;
  }
}

)kernel";
const char* k_hip_navatala_nccl_reduce_local_sum_i32 = R"kernel(
#include <hip/hip_runtime.h>
extern "C" __global__ void navatala_nccl_reduce_local_sum_i32(const int* a, const int* b, const unsigned int* count, int* _out) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int countVal = count[0];
  if (gid < countVal) {
    int va = a[gid];
    int vb = b[gid];
    int result = (va + vb);
    _out[gid] = result;
  }
}

)kernel";
const char* k_hip_navatala_nccl_reduce_local_min_f32 = R"kernel(
#include <hip/hip_runtime.h>
extern "C" __global__ void navatala_nccl_reduce_local_min_f32(const float* a, const float* b, const unsigned int* count, float* _out) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int countVal = count[0];
  if (gid < countVal) {
    float va = a[gid];
    float vb = b[gid];
    bool aLtB = (va < vb);
    float result = ((aLtB) ? (va) : (vb));
    _out[gid] = result;
  }
}

)kernel";
const char* k_hip_navatala_nccl_reduce_local_min_f16 = R"kernel(
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
extern "C" __global__ void navatala_nccl_reduce_local_min_f16(const __half* a, const __half* b, const unsigned int* count, __half* _out) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int countVal = count[0];
  if (gid < countVal) {
    __half va = a[gid];
    __half vb = b[gid];
    bool aLtB = (va < vb);
    __half result = ((aLtB) ? (va) : (vb));
    _out[gid] = result;
  }
}

)kernel";
const char* k_hip_navatala_nccl_reduce_local_min_i32 = R"kernel(
#include <hip/hip_runtime.h>
extern "C" __global__ void navatala_nccl_reduce_local_min_i32(const int* a, const int* b, const unsigned int* count, int* _out) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int countVal = count[0];
  if (gid < countVal) {
    int va = a[gid];
    int vb = b[gid];
    bool aLtB = (va < vb);
    int result = ((aLtB) ? (va) : (vb));
    _out[gid] = result;
  }
}

)kernel";
const char* k_hip_navatala_nccl_reduce_local_max_f32 = R"kernel(
#include <hip/hip_runtime.h>
extern "C" __global__ void navatala_nccl_reduce_local_max_f32(const float* a, const float* b, const unsigned int* count, float* _out) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int countVal = count[0];
  if (gid < countVal) {
    float va = a[gid];
    float vb = b[gid];
    bool aGtB = (va > vb);
    float result = ((aGtB) ? (va) : (vb));
    _out[gid] = result;
  }
}

)kernel";
const char* k_hip_navatala_nccl_reduce_local_max_f16 = R"kernel(
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
extern "C" __global__ void navatala_nccl_reduce_local_max_f16(const __half* a, const __half* b, const unsigned int* count, __half* _out) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int countVal = count[0];
  if (gid < countVal) {
    __half va = a[gid];
    __half vb = b[gid];
    bool aGtB = (va > vb);
    __half result = ((aGtB) ? (va) : (vb));
    _out[gid] = result;
  }
}

)kernel";
const char* k_hip_navatala_nccl_reduce_local_max_i32 = R"kernel(
#include <hip/hip_runtime.h>
extern "C" __global__ void navatala_nccl_reduce_local_max_i32(const int* a, const int* b, const unsigned int* count, int* _out) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int countVal = count[0];
  if (gid < countVal) {
    int va = a[gid];
    int vb = b[gid];
    bool aGtB = (va > vb);
    int result = ((aGtB) ? (va) : (vb));
    _out[gid] = result;
  }
}

)kernel";
const char* k_hip_navatala_nccl_ring_reduce_scatter_step_sum_f32 = R"kernel(
#include <hip/hip_runtime.h>
extern "C" __global__ void navatala_nccl_ring_reduce_scatter_step_sum_f32(const float* incoming, const float* local_buf, const unsigned int* count, float* outgoing) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int countVal = count[0];
  if (gid < countVal) {
    float inVal = incoming[gid];
    float locVal = local_buf[gid];
    float reduced = (inVal + locVal);
    outgoing[gid] = reduced;
  }
}

)kernel";
const char* k_hip_navatala_nccl_ring_reduce_scatter_step_sum_f16 = R"kernel(
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
extern "C" __global__ void navatala_nccl_ring_reduce_scatter_step_sum_f16(const __half* incoming, const __half* local_buf, const unsigned int* count, __half* outgoing) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int countVal = count[0];
  if (gid < countVal) {
    __half inVal = incoming[gid];
    __half locVal = local_buf[gid];
    __half reduced = (inVal + locVal);
    outgoing[gid] = reduced;
  }
}

)kernel";
const char* k_hip_navatala_nccl_ring_reduce_scatter_step_sum_i32 = R"kernel(
#include <hip/hip_runtime.h>
extern "C" __global__ void navatala_nccl_ring_reduce_scatter_step_sum_i32(const int* incoming, const int* local_buf, const unsigned int* count, int* outgoing) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int countVal = count[0];
  if (gid < countVal) {
    int inVal = incoming[gid];
    int locVal = local_buf[gid];
    int reduced = (inVal + locVal);
    outgoing[gid] = reduced;
  }
}

)kernel";
const char* k_hip_navatala_nccl_ring_reduce_scatter_step_min_f32 = R"kernel(
#include <hip/hip_runtime.h>
extern "C" __global__ void navatala_nccl_ring_reduce_scatter_step_min_f32(const float* incoming, const float* local_buf, const unsigned int* count, float* outgoing) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int countVal = count[0];
  if (gid < countVal) {
    float inVal = incoming[gid];
    float locVal = local_buf[gid];
    bool cmp = (inVal < locVal);
    float reduced = ((cmp) ? (inVal) : (locVal));
    outgoing[gid] = reduced;
  }
}

)kernel";
const char* k_hip_navatala_nccl_ring_reduce_scatter_step_min_f16 = R"kernel(
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
extern "C" __global__ void navatala_nccl_ring_reduce_scatter_step_min_f16(const __half* incoming, const __half* local_buf, const unsigned int* count, __half* outgoing) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int countVal = count[0];
  if (gid < countVal) {
    __half inVal = incoming[gid];
    __half locVal = local_buf[gid];
    bool cmp = (inVal < locVal);
    __half reduced = ((cmp) ? (inVal) : (locVal));
    outgoing[gid] = reduced;
  }
}

)kernel";
const char* k_hip_navatala_nccl_ring_reduce_scatter_step_min_i32 = R"kernel(
#include <hip/hip_runtime.h>
extern "C" __global__ void navatala_nccl_ring_reduce_scatter_step_min_i32(const int* incoming, const int* local_buf, const unsigned int* count, int* outgoing) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int countVal = count[0];
  if (gid < countVal) {
    int inVal = incoming[gid];
    int locVal = local_buf[gid];
    bool cmp = (inVal < locVal);
    int reduced = ((cmp) ? (inVal) : (locVal));
    outgoing[gid] = reduced;
  }
}

)kernel";
const char* k_hip_navatala_nccl_ring_reduce_scatter_step_max_f32 = R"kernel(
#include <hip/hip_runtime.h>
extern "C" __global__ void navatala_nccl_ring_reduce_scatter_step_max_f32(const float* incoming, const float* local_buf, const unsigned int* count, float* outgoing) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int countVal = count[0];
  if (gid < countVal) {
    float inVal = incoming[gid];
    float locVal = local_buf[gid];
    bool cmp = (inVal > locVal);
    float reduced = ((cmp) ? (inVal) : (locVal));
    outgoing[gid] = reduced;
  }
}

)kernel";
const char* k_hip_navatala_nccl_ring_reduce_scatter_step_max_f16 = R"kernel(
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
extern "C" __global__ void navatala_nccl_ring_reduce_scatter_step_max_f16(const __half* incoming, const __half* local_buf, const unsigned int* count, __half* outgoing) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int countVal = count[0];
  if (gid < countVal) {
    __half inVal = incoming[gid];
    __half locVal = local_buf[gid];
    bool cmp = (inVal > locVal);
    __half reduced = ((cmp) ? (inVal) : (locVal));
    outgoing[gid] = reduced;
  }
}

)kernel";
const char* k_hip_navatala_nccl_ring_reduce_scatter_step_max_i32 = R"kernel(
#include <hip/hip_runtime.h>
extern "C" __global__ void navatala_nccl_ring_reduce_scatter_step_max_i32(const int* incoming, const int* local_buf, const unsigned int* count, int* outgoing) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int countVal = count[0];
  if (gid < countVal) {
    int inVal = incoming[gid];
    int locVal = local_buf[gid];
    bool cmp = (inVal > locVal);
    int reduced = ((cmp) ? (inVal) : (locVal));
    outgoing[gid] = reduced;
  }
}

)kernel";
const char* k_hip_navatala_nccl_ring_all_gather_step_f32 = R"kernel(
#include <hip/hip_runtime.h>
extern "C" __global__ void navatala_nccl_ring_all_gather_step_f32(const float* incoming, const unsigned int* count, float* local_chunk, float* outgoing) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int countVal = count[0];
  if (gid < countVal) {
    float val = incoming[gid];
    local_chunk[gid] = val;
    outgoing[gid] = val;
  }
}

)kernel";
const char* k_hip_navatala_nccl_ring_all_gather_step_f16 = R"kernel(
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
extern "C" __global__ void navatala_nccl_ring_all_gather_step_f16(const __half* incoming, const unsigned int* count, __half* local_chunk, __half* outgoing) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int countVal = count[0];
  if (gid < countVal) {
    __half val = incoming[gid];
    local_chunk[gid] = val;
    outgoing[gid] = val;
  }
}

)kernel";
const char* k_hip_navatala_nccl_ring_all_gather_step_i32 = R"kernel(
#include <hip/hip_runtime.h>
extern "C" __global__ void navatala_nccl_ring_all_gather_step_i32(const int* incoming, const unsigned int* count, int* local_chunk, int* outgoing) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int countVal = count[0];
  if (gid < countVal) {
    int val = incoming[gid];
    local_chunk[gid] = val;
    outgoing[gid] = val;
  }
}

)kernel";
const char* k_hip_navatala_nccl_ring_all_reduce_sum_f32 = R"kernel(
#include <hip/hip_runtime.h>
extern "C" __global__ void navatala_nccl_ring_all_reduce_sum_f32(const float* sendbuf, const float* peer_buf, const unsigned int* count, float* recvbuf) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int countVal = count[0];
  if (gid < countVal) {
    float myVal = sendbuf[gid];
    float peerVal = peer_buf[gid];
    float reduced = (myVal + peerVal);
    recvbuf[gid] = reduced;
  }
}

)kernel";
const char* k_hip_navatala_nccl_ring_all_reduce_sum_f16 = R"kernel(
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
extern "C" __global__ void navatala_nccl_ring_all_reduce_sum_f16(const __half* sendbuf, const __half* peer_buf, const unsigned int* count, __half* recvbuf) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int countVal = count[0];
  if (gid < countVal) {
    __half myVal = sendbuf[gid];
    __half peerVal = peer_buf[gid];
    __half reduced = (myVal + peerVal);
    recvbuf[gid] = reduced;
  }
}

)kernel";
const char* k_hip_navatala_nccl_ring_all_reduce_sum_i32 = R"kernel(
#include <hip/hip_runtime.h>
extern "C" __global__ void navatala_nccl_ring_all_reduce_sum_i32(const int* sendbuf, const int* peer_buf, const unsigned int* count, int* recvbuf) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int countVal = count[0];
  if (gid < countVal) {
    int myVal = sendbuf[gid];
    int peerVal = peer_buf[gid];
    int reduced = (myVal + peerVal);
    recvbuf[gid] = reduced;
  }
}

)kernel";
const char* k_hip_navatala_nccl_ring_all_gather_f32 = R"kernel(
#include <hip/hip_runtime.h>
extern "C" __global__ void navatala_nccl_ring_all_gather_f32(const float* incoming, const unsigned int* count, float* local_slot, float* outgoing) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int countVal = count[0];
  if (gid < countVal) {
    float val = incoming[gid];
    local_slot[gid] = val;
    outgoing[gid] = val;
  }
}

)kernel";
const char* k_hip_navatala_nccl_ring_all_gather_f16 = R"kernel(
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
extern "C" __global__ void navatala_nccl_ring_all_gather_f16(const __half* incoming, const unsigned int* count, __half* local_slot, __half* outgoing) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int countVal = count[0];
  if (gid < countVal) {
    __half val = incoming[gid];
    local_slot[gid] = val;
    outgoing[gid] = val;
  }
}

)kernel";
const char* k_hip_navatala_nccl_ring_all_gather_i32 = R"kernel(
#include <hip/hip_runtime.h>
extern "C" __global__ void navatala_nccl_ring_all_gather_i32(const int* incoming, const unsigned int* count, int* local_slot, int* outgoing) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int countVal = count[0];
  if (gid < countVal) {
    int val = incoming[gid];
    local_slot[gid] = val;
    outgoing[gid] = val;
  }
}

)kernel";
const char* k_hip_navatala_nccl_linear_broadcast_f32 = R"kernel(
#include <hip/hip_runtime.h>
extern "C" __global__ void navatala_nccl_linear_broadcast_f32(const float* src, const unsigned int* count, float* dst) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int countVal = count[0];
  if (gid < countVal) {
    float val = src[gid];
    dst[gid] = val;
  }
}

)kernel";
const char* k_hip_navatala_nccl_linear_broadcast_f16 = R"kernel(
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
extern "C" __global__ void navatala_nccl_linear_broadcast_f16(const __half* src, const unsigned int* count, __half* dst) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int countVal = count[0];
  if (gid < countVal) {
    __half val = src[gid];
    dst[gid] = val;
  }
}

)kernel";
const char* k_hip_navatala_nccl_linear_broadcast_i32 = R"kernel(
#include <hip/hip_runtime.h>
extern "C" __global__ void navatala_nccl_linear_broadcast_i32(const int* src, const unsigned int* count, int* dst) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int countVal = count[0];
  if (gid < countVal) {
    int val = src[gid];
    dst[gid] = val;
  }
}

)kernel";

namespace NavatalaRegistry {

const KernelArgumentInfo kAbiArgs_hip_navatala_nccl_direct_send_f32[] = {
  { "src", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "dst", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_nccl_direct_send_f32 = {
  1,
  "navatala_nccl_direct_send_f32",
  "hip",
  "navatala_nccl_direct_send_f32",
  "kernel:hip:navatala_nccl_direct_send_f32",
  "abi-r1:hip:navatala_nccl_direct_send_f32",
  "abi-r1:hip:navatala_nccl_direct_send_f32",
  3,
  kAbiArgs_hip_navatala_nccl_direct_send_f32
};

const KernelArgumentInfo kAbiArgs_hip_navatala_nccl_direct_send_f16[] = {
  { "src", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "dst", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_nccl_direct_send_f16 = {
  1,
  "navatala_nccl_direct_send_f16",
  "hip",
  "navatala_nccl_direct_send_f16",
  "kernel:hip:navatala_nccl_direct_send_f16",
  "abi-r1:hip:navatala_nccl_direct_send_f16",
  "abi-r1:hip:navatala_nccl_direct_send_f16",
  3,
  kAbiArgs_hip_navatala_nccl_direct_send_f16
};

const KernelArgumentInfo kAbiArgs_hip_navatala_nccl_direct_send_i32[] = {
  { "src", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "dst", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_nccl_direct_send_i32 = {
  1,
  "navatala_nccl_direct_send_i32",
  "hip",
  "navatala_nccl_direct_send_i32",
  "kernel:hip:navatala_nccl_direct_send_i32",
  "abi-r1:hip:navatala_nccl_direct_send_i32",
  "abi-r1:hip:navatala_nccl_direct_send_i32",
  3,
  kAbiArgs_hip_navatala_nccl_direct_send_i32
};

const KernelArgumentInfo kAbiArgs_hip_navatala_nccl_direct_recv_f32[] = {
  { "src", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "dst", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_nccl_direct_recv_f32 = {
  1,
  "navatala_nccl_direct_recv_f32",
  "hip",
  "navatala_nccl_direct_recv_f32",
  "kernel:hip:navatala_nccl_direct_recv_f32",
  "abi-r1:hip:navatala_nccl_direct_recv_f32",
  "abi-r1:hip:navatala_nccl_direct_recv_f32",
  3,
  kAbiArgs_hip_navatala_nccl_direct_recv_f32
};

const KernelArgumentInfo kAbiArgs_hip_navatala_nccl_direct_recv_f16[] = {
  { "src", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "dst", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_nccl_direct_recv_f16 = {
  1,
  "navatala_nccl_direct_recv_f16",
  "hip",
  "navatala_nccl_direct_recv_f16",
  "kernel:hip:navatala_nccl_direct_recv_f16",
  "abi-r1:hip:navatala_nccl_direct_recv_f16",
  "abi-r1:hip:navatala_nccl_direct_recv_f16",
  3,
  kAbiArgs_hip_navatala_nccl_direct_recv_f16
};

const KernelArgumentInfo kAbiArgs_hip_navatala_nccl_direct_recv_i32[] = {
  { "src", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "dst", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_nccl_direct_recv_i32 = {
  1,
  "navatala_nccl_direct_recv_i32",
  "hip",
  "navatala_nccl_direct_recv_i32",
  "kernel:hip:navatala_nccl_direct_recv_i32",
  "abi-r1:hip:navatala_nccl_direct_recv_i32",
  "abi-r1:hip:navatala_nccl_direct_recv_i32",
  3,
  kAbiArgs_hip_navatala_nccl_direct_recv_i32
};

const KernelArgumentInfo kAbiArgs_hip_navatala_nccl_reduce_local_sum_f32[] = {
  { "a", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "b", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "out", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_nccl_reduce_local_sum_f32 = {
  1,
  "navatala_nccl_reduce_local_sum_f32",
  "hip",
  "navatala_nccl_reduce_local_sum_f32",
  "kernel:hip:navatala_nccl_reduce_local_sum_f32",
  "abi-r1:hip:navatala_nccl_reduce_local_sum_f32",
  "abi-r1:hip:navatala_nccl_reduce_local_sum_f32",
  4,
  kAbiArgs_hip_navatala_nccl_reduce_local_sum_f32
};

const KernelArgumentInfo kAbiArgs_hip_navatala_nccl_reduce_local_sum_f16[] = {
  { "a", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "b", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "out", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_nccl_reduce_local_sum_f16 = {
  1,
  "navatala_nccl_reduce_local_sum_f16",
  "hip",
  "navatala_nccl_reduce_local_sum_f16",
  "kernel:hip:navatala_nccl_reduce_local_sum_f16",
  "abi-r1:hip:navatala_nccl_reduce_local_sum_f16",
  "abi-r1:hip:navatala_nccl_reduce_local_sum_f16",
  4,
  kAbiArgs_hip_navatala_nccl_reduce_local_sum_f16
};

const KernelArgumentInfo kAbiArgs_hip_navatala_nccl_reduce_local_sum_i32[] = {
  { "a", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "b", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "out", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_nccl_reduce_local_sum_i32 = {
  1,
  "navatala_nccl_reduce_local_sum_i32",
  "hip",
  "navatala_nccl_reduce_local_sum_i32",
  "kernel:hip:navatala_nccl_reduce_local_sum_i32",
  "abi-r1:hip:navatala_nccl_reduce_local_sum_i32",
  "abi-r1:hip:navatala_nccl_reduce_local_sum_i32",
  4,
  kAbiArgs_hip_navatala_nccl_reduce_local_sum_i32
};

const KernelArgumentInfo kAbiArgs_hip_navatala_nccl_reduce_local_min_f32[] = {
  { "a", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "b", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "out", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_nccl_reduce_local_min_f32 = {
  1,
  "navatala_nccl_reduce_local_min_f32",
  "hip",
  "navatala_nccl_reduce_local_min_f32",
  "kernel:hip:navatala_nccl_reduce_local_min_f32",
  "abi-r1:hip:navatala_nccl_reduce_local_min_f32",
  "abi-r1:hip:navatala_nccl_reduce_local_min_f32",
  4,
  kAbiArgs_hip_navatala_nccl_reduce_local_min_f32
};

const KernelArgumentInfo kAbiArgs_hip_navatala_nccl_reduce_local_min_f16[] = {
  { "a", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "b", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "out", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_nccl_reduce_local_min_f16 = {
  1,
  "navatala_nccl_reduce_local_min_f16",
  "hip",
  "navatala_nccl_reduce_local_min_f16",
  "kernel:hip:navatala_nccl_reduce_local_min_f16",
  "abi-r1:hip:navatala_nccl_reduce_local_min_f16",
  "abi-r1:hip:navatala_nccl_reduce_local_min_f16",
  4,
  kAbiArgs_hip_navatala_nccl_reduce_local_min_f16
};

const KernelArgumentInfo kAbiArgs_hip_navatala_nccl_reduce_local_min_i32[] = {
  { "a", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "b", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "out", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_nccl_reduce_local_min_i32 = {
  1,
  "navatala_nccl_reduce_local_min_i32",
  "hip",
  "navatala_nccl_reduce_local_min_i32",
  "kernel:hip:navatala_nccl_reduce_local_min_i32",
  "abi-r1:hip:navatala_nccl_reduce_local_min_i32",
  "abi-r1:hip:navatala_nccl_reduce_local_min_i32",
  4,
  kAbiArgs_hip_navatala_nccl_reduce_local_min_i32
};

const KernelArgumentInfo kAbiArgs_hip_navatala_nccl_reduce_local_max_f32[] = {
  { "a", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "b", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "out", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_nccl_reduce_local_max_f32 = {
  1,
  "navatala_nccl_reduce_local_max_f32",
  "hip",
  "navatala_nccl_reduce_local_max_f32",
  "kernel:hip:navatala_nccl_reduce_local_max_f32",
  "abi-r1:hip:navatala_nccl_reduce_local_max_f32",
  "abi-r1:hip:navatala_nccl_reduce_local_max_f32",
  4,
  kAbiArgs_hip_navatala_nccl_reduce_local_max_f32
};

const KernelArgumentInfo kAbiArgs_hip_navatala_nccl_reduce_local_max_f16[] = {
  { "a", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "b", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "out", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_nccl_reduce_local_max_f16 = {
  1,
  "navatala_nccl_reduce_local_max_f16",
  "hip",
  "navatala_nccl_reduce_local_max_f16",
  "kernel:hip:navatala_nccl_reduce_local_max_f16",
  "abi-r1:hip:navatala_nccl_reduce_local_max_f16",
  "abi-r1:hip:navatala_nccl_reduce_local_max_f16",
  4,
  kAbiArgs_hip_navatala_nccl_reduce_local_max_f16
};

const KernelArgumentInfo kAbiArgs_hip_navatala_nccl_reduce_local_max_i32[] = {
  { "a", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "b", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "out", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_nccl_reduce_local_max_i32 = {
  1,
  "navatala_nccl_reduce_local_max_i32",
  "hip",
  "navatala_nccl_reduce_local_max_i32",
  "kernel:hip:navatala_nccl_reduce_local_max_i32",
  "abi-r1:hip:navatala_nccl_reduce_local_max_i32",
  "abi-r1:hip:navatala_nccl_reduce_local_max_i32",
  4,
  kAbiArgs_hip_navatala_nccl_reduce_local_max_i32
};

const KernelArgumentInfo kAbiArgs_hip_navatala_nccl_ring_reduce_scatter_step_sum_f32[] = {
  { "incoming", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "local_buf", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "outgoing", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_nccl_ring_reduce_scatter_step_sum_f32 = {
  1,
  "navatala_nccl_ring_reduce_scatter_step_sum_f32",
  "hip",
  "navatala_nccl_ring_reduce_scatter_step_sum_f32",
  "kernel:hip:navatala_nccl_ring_reduce_scatter_step_sum_f32",
  "abi-r1:hip:navatala_nccl_ring_reduce_scatter_step_sum_f32",
  "abi-r1:hip:navatala_nccl_ring_reduce_scatter_step_sum_f32",
  4,
  kAbiArgs_hip_navatala_nccl_ring_reduce_scatter_step_sum_f32
};

const KernelArgumentInfo kAbiArgs_hip_navatala_nccl_ring_reduce_scatter_step_sum_f16[] = {
  { "incoming", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "local_buf", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "outgoing", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_nccl_ring_reduce_scatter_step_sum_f16 = {
  1,
  "navatala_nccl_ring_reduce_scatter_step_sum_f16",
  "hip",
  "navatala_nccl_ring_reduce_scatter_step_sum_f16",
  "kernel:hip:navatala_nccl_ring_reduce_scatter_step_sum_f16",
  "abi-r1:hip:navatala_nccl_ring_reduce_scatter_step_sum_f16",
  "abi-r1:hip:navatala_nccl_ring_reduce_scatter_step_sum_f16",
  4,
  kAbiArgs_hip_navatala_nccl_ring_reduce_scatter_step_sum_f16
};

const KernelArgumentInfo kAbiArgs_hip_navatala_nccl_ring_reduce_scatter_step_sum_i32[] = {
  { "incoming", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "local_buf", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "outgoing", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_nccl_ring_reduce_scatter_step_sum_i32 = {
  1,
  "navatala_nccl_ring_reduce_scatter_step_sum_i32",
  "hip",
  "navatala_nccl_ring_reduce_scatter_step_sum_i32",
  "kernel:hip:navatala_nccl_ring_reduce_scatter_step_sum_i32",
  "abi-r1:hip:navatala_nccl_ring_reduce_scatter_step_sum_i32",
  "abi-r1:hip:navatala_nccl_ring_reduce_scatter_step_sum_i32",
  4,
  kAbiArgs_hip_navatala_nccl_ring_reduce_scatter_step_sum_i32
};

const KernelArgumentInfo kAbiArgs_hip_navatala_nccl_ring_reduce_scatter_step_min_f32[] = {
  { "incoming", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "local_buf", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "outgoing", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_nccl_ring_reduce_scatter_step_min_f32 = {
  1,
  "navatala_nccl_ring_reduce_scatter_step_min_f32",
  "hip",
  "navatala_nccl_ring_reduce_scatter_step_min_f32",
  "kernel:hip:navatala_nccl_ring_reduce_scatter_step_min_f32",
  "abi-r1:hip:navatala_nccl_ring_reduce_scatter_step_min_f32",
  "abi-r1:hip:navatala_nccl_ring_reduce_scatter_step_min_f32",
  4,
  kAbiArgs_hip_navatala_nccl_ring_reduce_scatter_step_min_f32
};

const KernelArgumentInfo kAbiArgs_hip_navatala_nccl_ring_reduce_scatter_step_min_f16[] = {
  { "incoming", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "local_buf", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "outgoing", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_nccl_ring_reduce_scatter_step_min_f16 = {
  1,
  "navatala_nccl_ring_reduce_scatter_step_min_f16",
  "hip",
  "navatala_nccl_ring_reduce_scatter_step_min_f16",
  "kernel:hip:navatala_nccl_ring_reduce_scatter_step_min_f16",
  "abi-r1:hip:navatala_nccl_ring_reduce_scatter_step_min_f16",
  "abi-r1:hip:navatala_nccl_ring_reduce_scatter_step_min_f16",
  4,
  kAbiArgs_hip_navatala_nccl_ring_reduce_scatter_step_min_f16
};

const KernelArgumentInfo kAbiArgs_hip_navatala_nccl_ring_reduce_scatter_step_min_i32[] = {
  { "incoming", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "local_buf", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "outgoing", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_nccl_ring_reduce_scatter_step_min_i32 = {
  1,
  "navatala_nccl_ring_reduce_scatter_step_min_i32",
  "hip",
  "navatala_nccl_ring_reduce_scatter_step_min_i32",
  "kernel:hip:navatala_nccl_ring_reduce_scatter_step_min_i32",
  "abi-r1:hip:navatala_nccl_ring_reduce_scatter_step_min_i32",
  "abi-r1:hip:navatala_nccl_ring_reduce_scatter_step_min_i32",
  4,
  kAbiArgs_hip_navatala_nccl_ring_reduce_scatter_step_min_i32
};

const KernelArgumentInfo kAbiArgs_hip_navatala_nccl_ring_reduce_scatter_step_max_f32[] = {
  { "incoming", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "local_buf", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "outgoing", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_nccl_ring_reduce_scatter_step_max_f32 = {
  1,
  "navatala_nccl_ring_reduce_scatter_step_max_f32",
  "hip",
  "navatala_nccl_ring_reduce_scatter_step_max_f32",
  "kernel:hip:navatala_nccl_ring_reduce_scatter_step_max_f32",
  "abi-r1:hip:navatala_nccl_ring_reduce_scatter_step_max_f32",
  "abi-r1:hip:navatala_nccl_ring_reduce_scatter_step_max_f32",
  4,
  kAbiArgs_hip_navatala_nccl_ring_reduce_scatter_step_max_f32
};

const KernelArgumentInfo kAbiArgs_hip_navatala_nccl_ring_reduce_scatter_step_max_f16[] = {
  { "incoming", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "local_buf", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "outgoing", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_nccl_ring_reduce_scatter_step_max_f16 = {
  1,
  "navatala_nccl_ring_reduce_scatter_step_max_f16",
  "hip",
  "navatala_nccl_ring_reduce_scatter_step_max_f16",
  "kernel:hip:navatala_nccl_ring_reduce_scatter_step_max_f16",
  "abi-r1:hip:navatala_nccl_ring_reduce_scatter_step_max_f16",
  "abi-r1:hip:navatala_nccl_ring_reduce_scatter_step_max_f16",
  4,
  kAbiArgs_hip_navatala_nccl_ring_reduce_scatter_step_max_f16
};

const KernelArgumentInfo kAbiArgs_hip_navatala_nccl_ring_reduce_scatter_step_max_i32[] = {
  { "incoming", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "local_buf", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "outgoing", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_nccl_ring_reduce_scatter_step_max_i32 = {
  1,
  "navatala_nccl_ring_reduce_scatter_step_max_i32",
  "hip",
  "navatala_nccl_ring_reduce_scatter_step_max_i32",
  "kernel:hip:navatala_nccl_ring_reduce_scatter_step_max_i32",
  "abi-r1:hip:navatala_nccl_ring_reduce_scatter_step_max_i32",
  "abi-r1:hip:navatala_nccl_ring_reduce_scatter_step_max_i32",
  4,
  kAbiArgs_hip_navatala_nccl_ring_reduce_scatter_step_max_i32
};

const KernelArgumentInfo kAbiArgs_hip_navatala_nccl_ring_all_gather_step_f32[] = {
  { "incoming", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "local_chunk", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "outgoing", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_nccl_ring_all_gather_step_f32 = {
  1,
  "navatala_nccl_ring_all_gather_step_f32",
  "hip",
  "navatala_nccl_ring_all_gather_step_f32",
  "kernel:hip:navatala_nccl_ring_all_gather_step_f32",
  "abi-r1:hip:navatala_nccl_ring_all_gather_step_f32",
  "abi-r1:hip:navatala_nccl_ring_all_gather_step_f32",
  4,
  kAbiArgs_hip_navatala_nccl_ring_all_gather_step_f32
};

const KernelArgumentInfo kAbiArgs_hip_navatala_nccl_ring_all_gather_step_f16[] = {
  { "incoming", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "local_chunk", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "outgoing", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_nccl_ring_all_gather_step_f16 = {
  1,
  "navatala_nccl_ring_all_gather_step_f16",
  "hip",
  "navatala_nccl_ring_all_gather_step_f16",
  "kernel:hip:navatala_nccl_ring_all_gather_step_f16",
  "abi-r1:hip:navatala_nccl_ring_all_gather_step_f16",
  "abi-r1:hip:navatala_nccl_ring_all_gather_step_f16",
  4,
  kAbiArgs_hip_navatala_nccl_ring_all_gather_step_f16
};

const KernelArgumentInfo kAbiArgs_hip_navatala_nccl_ring_all_gather_step_i32[] = {
  { "incoming", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "local_chunk", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "outgoing", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_nccl_ring_all_gather_step_i32 = {
  1,
  "navatala_nccl_ring_all_gather_step_i32",
  "hip",
  "navatala_nccl_ring_all_gather_step_i32",
  "kernel:hip:navatala_nccl_ring_all_gather_step_i32",
  "abi-r1:hip:navatala_nccl_ring_all_gather_step_i32",
  "abi-r1:hip:navatala_nccl_ring_all_gather_step_i32",
  4,
  kAbiArgs_hip_navatala_nccl_ring_all_gather_step_i32
};

const KernelArgumentInfo kAbiArgs_hip_navatala_nccl_ring_all_reduce_sum_f32[] = {
  { "sendbuf", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "peer_buf", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "recvbuf", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_nccl_ring_all_reduce_sum_f32 = {
  1,
  "navatala_nccl_ring_all_reduce_sum_f32",
  "hip",
  "navatala_nccl_ring_all_reduce_sum_f32",
  "kernel:hip:navatala_nccl_ring_all_reduce_sum_f32",
  "abi-r1:hip:navatala_nccl_ring_all_reduce_sum_f32",
  "abi-r1:hip:navatala_nccl_ring_all_reduce_sum_f32",
  4,
  kAbiArgs_hip_navatala_nccl_ring_all_reduce_sum_f32
};

const KernelArgumentInfo kAbiArgs_hip_navatala_nccl_ring_all_reduce_sum_f16[] = {
  { "sendbuf", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "peer_buf", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "recvbuf", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_nccl_ring_all_reduce_sum_f16 = {
  1,
  "navatala_nccl_ring_all_reduce_sum_f16",
  "hip",
  "navatala_nccl_ring_all_reduce_sum_f16",
  "kernel:hip:navatala_nccl_ring_all_reduce_sum_f16",
  "abi-r1:hip:navatala_nccl_ring_all_reduce_sum_f16",
  "abi-r1:hip:navatala_nccl_ring_all_reduce_sum_f16",
  4,
  kAbiArgs_hip_navatala_nccl_ring_all_reduce_sum_f16
};

const KernelArgumentInfo kAbiArgs_hip_navatala_nccl_ring_all_reduce_sum_i32[] = {
  { "sendbuf", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "peer_buf", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "recvbuf", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_nccl_ring_all_reduce_sum_i32 = {
  1,
  "navatala_nccl_ring_all_reduce_sum_i32",
  "hip",
  "navatala_nccl_ring_all_reduce_sum_i32",
  "kernel:hip:navatala_nccl_ring_all_reduce_sum_i32",
  "abi-r1:hip:navatala_nccl_ring_all_reduce_sum_i32",
  "abi-r1:hip:navatala_nccl_ring_all_reduce_sum_i32",
  4,
  kAbiArgs_hip_navatala_nccl_ring_all_reduce_sum_i32
};

const KernelArgumentInfo kAbiArgs_hip_navatala_nccl_ring_all_gather_f32[] = {
  { "incoming", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "local_slot", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "outgoing", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_nccl_ring_all_gather_f32 = {
  1,
  "navatala_nccl_ring_all_gather_f32",
  "hip",
  "navatala_nccl_ring_all_gather_f32",
  "kernel:hip:navatala_nccl_ring_all_gather_f32",
  "abi-r1:hip:navatala_nccl_ring_all_gather_f32",
  "abi-r1:hip:navatala_nccl_ring_all_gather_f32",
  4,
  kAbiArgs_hip_navatala_nccl_ring_all_gather_f32
};

const KernelArgumentInfo kAbiArgs_hip_navatala_nccl_ring_all_gather_f16[] = {
  { "incoming", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "local_slot", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "outgoing", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_nccl_ring_all_gather_f16 = {
  1,
  "navatala_nccl_ring_all_gather_f16",
  "hip",
  "navatala_nccl_ring_all_gather_f16",
  "kernel:hip:navatala_nccl_ring_all_gather_f16",
  "abi-r1:hip:navatala_nccl_ring_all_gather_f16",
  "abi-r1:hip:navatala_nccl_ring_all_gather_f16",
  4,
  kAbiArgs_hip_navatala_nccl_ring_all_gather_f16
};

const KernelArgumentInfo kAbiArgs_hip_navatala_nccl_ring_all_gather_i32[] = {
  { "incoming", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "local_slot", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "outgoing", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_nccl_ring_all_gather_i32 = {
  1,
  "navatala_nccl_ring_all_gather_i32",
  "hip",
  "navatala_nccl_ring_all_gather_i32",
  "kernel:hip:navatala_nccl_ring_all_gather_i32",
  "abi-r1:hip:navatala_nccl_ring_all_gather_i32",
  "abi-r1:hip:navatala_nccl_ring_all_gather_i32",
  4,
  kAbiArgs_hip_navatala_nccl_ring_all_gather_i32
};

const KernelArgumentInfo kAbiArgs_hip_navatala_nccl_linear_broadcast_f32[] = {
  { "src", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "dst", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_nccl_linear_broadcast_f32 = {
  1,
  "navatala_nccl_linear_broadcast_f32",
  "hip",
  "navatala_nccl_linear_broadcast_f32",
  "kernel:hip:navatala_nccl_linear_broadcast_f32",
  "abi-r1:hip:navatala_nccl_linear_broadcast_f32",
  "abi-r1:hip:navatala_nccl_linear_broadcast_f32",
  3,
  kAbiArgs_hip_navatala_nccl_linear_broadcast_f32
};

const KernelArgumentInfo kAbiArgs_hip_navatala_nccl_linear_broadcast_f16[] = {
  { "src", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "dst", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_nccl_linear_broadcast_f16 = {
  1,
  "navatala_nccl_linear_broadcast_f16",
  "hip",
  "navatala_nccl_linear_broadcast_f16",
  "kernel:hip:navatala_nccl_linear_broadcast_f16",
  "abi-r1:hip:navatala_nccl_linear_broadcast_f16",
  "abi-r1:hip:navatala_nccl_linear_broadcast_f16",
  3,
  kAbiArgs_hip_navatala_nccl_linear_broadcast_f16
};

const KernelArgumentInfo kAbiArgs_hip_navatala_nccl_linear_broadcast_i32[] = {
  { "src", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "dst", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_nccl_linear_broadcast_i32 = {
  1,
  "navatala_nccl_linear_broadcast_i32",
  "hip",
  "navatala_nccl_linear_broadcast_i32",
  "kernel:hip:navatala_nccl_linear_broadcast_i32",
  "abi-r1:hip:navatala_nccl_linear_broadcast_i32",
  "abi-r1:hip:navatala_nccl_linear_broadcast_i32",
  3,
  kAbiArgs_hip_navatala_nccl_linear_broadcast_i32
};

bool tryGetKernelAbiManifest_hip_nccl(const std::string& backend, const std::string& kernelName, const KernelAbiManifestInfo*& out) {
  if (backend == "hip" && kernelName == "navatala_nccl_direct_send_f32") {
    out = &kAbiManifest_hip_navatala_nccl_direct_send_f32;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_nccl_direct_send_f16") {
    out = &kAbiManifest_hip_navatala_nccl_direct_send_f16;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_nccl_direct_send_i32") {
    out = &kAbiManifest_hip_navatala_nccl_direct_send_i32;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_nccl_direct_recv_f32") {
    out = &kAbiManifest_hip_navatala_nccl_direct_recv_f32;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_nccl_direct_recv_f16") {
    out = &kAbiManifest_hip_navatala_nccl_direct_recv_f16;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_nccl_direct_recv_i32") {
    out = &kAbiManifest_hip_navatala_nccl_direct_recv_i32;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_nccl_reduce_local_sum_f32") {
    out = &kAbiManifest_hip_navatala_nccl_reduce_local_sum_f32;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_nccl_reduce_local_sum_f16") {
    out = &kAbiManifest_hip_navatala_nccl_reduce_local_sum_f16;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_nccl_reduce_local_sum_i32") {
    out = &kAbiManifest_hip_navatala_nccl_reduce_local_sum_i32;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_nccl_reduce_local_min_f32") {
    out = &kAbiManifest_hip_navatala_nccl_reduce_local_min_f32;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_nccl_reduce_local_min_f16") {
    out = &kAbiManifest_hip_navatala_nccl_reduce_local_min_f16;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_nccl_reduce_local_min_i32") {
    out = &kAbiManifest_hip_navatala_nccl_reduce_local_min_i32;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_nccl_reduce_local_max_f32") {
    out = &kAbiManifest_hip_navatala_nccl_reduce_local_max_f32;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_nccl_reduce_local_max_f16") {
    out = &kAbiManifest_hip_navatala_nccl_reduce_local_max_f16;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_nccl_reduce_local_max_i32") {
    out = &kAbiManifest_hip_navatala_nccl_reduce_local_max_i32;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_nccl_ring_reduce_scatter_step_sum_f32") {
    out = &kAbiManifest_hip_navatala_nccl_ring_reduce_scatter_step_sum_f32;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_nccl_ring_reduce_scatter_step_sum_f16") {
    out = &kAbiManifest_hip_navatala_nccl_ring_reduce_scatter_step_sum_f16;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_nccl_ring_reduce_scatter_step_sum_i32") {
    out = &kAbiManifest_hip_navatala_nccl_ring_reduce_scatter_step_sum_i32;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_nccl_ring_reduce_scatter_step_min_f32") {
    out = &kAbiManifest_hip_navatala_nccl_ring_reduce_scatter_step_min_f32;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_nccl_ring_reduce_scatter_step_min_f16") {
    out = &kAbiManifest_hip_navatala_nccl_ring_reduce_scatter_step_min_f16;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_nccl_ring_reduce_scatter_step_min_i32") {
    out = &kAbiManifest_hip_navatala_nccl_ring_reduce_scatter_step_min_i32;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_nccl_ring_reduce_scatter_step_max_f32") {
    out = &kAbiManifest_hip_navatala_nccl_ring_reduce_scatter_step_max_f32;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_nccl_ring_reduce_scatter_step_max_f16") {
    out = &kAbiManifest_hip_navatala_nccl_ring_reduce_scatter_step_max_f16;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_nccl_ring_reduce_scatter_step_max_i32") {
    out = &kAbiManifest_hip_navatala_nccl_ring_reduce_scatter_step_max_i32;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_nccl_ring_all_gather_step_f32") {
    out = &kAbiManifest_hip_navatala_nccl_ring_all_gather_step_f32;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_nccl_ring_all_gather_step_f16") {
    out = &kAbiManifest_hip_navatala_nccl_ring_all_gather_step_f16;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_nccl_ring_all_gather_step_i32") {
    out = &kAbiManifest_hip_navatala_nccl_ring_all_gather_step_i32;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_nccl_ring_all_reduce_sum_f32") {
    out = &kAbiManifest_hip_navatala_nccl_ring_all_reduce_sum_f32;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_nccl_ring_all_reduce_sum_f16") {
    out = &kAbiManifest_hip_navatala_nccl_ring_all_reduce_sum_f16;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_nccl_ring_all_reduce_sum_i32") {
    out = &kAbiManifest_hip_navatala_nccl_ring_all_reduce_sum_i32;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_nccl_ring_all_gather_f32") {
    out = &kAbiManifest_hip_navatala_nccl_ring_all_gather_f32;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_nccl_ring_all_gather_f16") {
    out = &kAbiManifest_hip_navatala_nccl_ring_all_gather_f16;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_nccl_ring_all_gather_i32") {
    out = &kAbiManifest_hip_navatala_nccl_ring_all_gather_i32;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_nccl_linear_broadcast_f32") {
    out = &kAbiManifest_hip_navatala_nccl_linear_broadcast_f32;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_nccl_linear_broadcast_f16") {
    out = &kAbiManifest_hip_navatala_nccl_linear_broadcast_f16;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_nccl_linear_broadcast_i32") {
    out = &kAbiManifest_hip_navatala_nccl_linear_broadcast_i32;
    return true;
  }
  out = nullptr;
  return false;
}

bool tryGetKernelSource_hip_nccl(const std::string& backend, const std::string& kernelName, GpuRuntime::ProgramSource& out) {
  if (backend == "hip" && kernelName == "navatala_nccl_direct_send_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_nccl_direct_send_f32";
    std::string_view sv(k_hip_navatala_nccl_direct_send_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_nccl_direct_send_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_nccl_direct_send_f16";
    std::string_view sv(k_hip_navatala_nccl_direct_send_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_nccl_direct_send_i32") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_nccl_direct_send_i32";
    std::string_view sv(k_hip_navatala_nccl_direct_send_i32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_nccl_direct_recv_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_nccl_direct_recv_f32";
    std::string_view sv(k_hip_navatala_nccl_direct_recv_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_nccl_direct_recv_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_nccl_direct_recv_f16";
    std::string_view sv(k_hip_navatala_nccl_direct_recv_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_nccl_direct_recv_i32") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_nccl_direct_recv_i32";
    std::string_view sv(k_hip_navatala_nccl_direct_recv_i32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_nccl_reduce_local_sum_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_nccl_reduce_local_sum_f32";
    std::string_view sv(k_hip_navatala_nccl_reduce_local_sum_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_nccl_reduce_local_sum_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_nccl_reduce_local_sum_f16";
    std::string_view sv(k_hip_navatala_nccl_reduce_local_sum_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_nccl_reduce_local_sum_i32") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_nccl_reduce_local_sum_i32";
    std::string_view sv(k_hip_navatala_nccl_reduce_local_sum_i32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_nccl_reduce_local_min_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_nccl_reduce_local_min_f32";
    std::string_view sv(k_hip_navatala_nccl_reduce_local_min_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_nccl_reduce_local_min_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_nccl_reduce_local_min_f16";
    std::string_view sv(k_hip_navatala_nccl_reduce_local_min_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_nccl_reduce_local_min_i32") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_nccl_reduce_local_min_i32";
    std::string_view sv(k_hip_navatala_nccl_reduce_local_min_i32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_nccl_reduce_local_max_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_nccl_reduce_local_max_f32";
    std::string_view sv(k_hip_navatala_nccl_reduce_local_max_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_nccl_reduce_local_max_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_nccl_reduce_local_max_f16";
    std::string_view sv(k_hip_navatala_nccl_reduce_local_max_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_nccl_reduce_local_max_i32") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_nccl_reduce_local_max_i32";
    std::string_view sv(k_hip_navatala_nccl_reduce_local_max_i32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_nccl_ring_reduce_scatter_step_sum_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_nccl_ring_reduce_scatter_step_sum_f32";
    std::string_view sv(k_hip_navatala_nccl_ring_reduce_scatter_step_sum_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_nccl_ring_reduce_scatter_step_sum_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_nccl_ring_reduce_scatter_step_sum_f16";
    std::string_view sv(k_hip_navatala_nccl_ring_reduce_scatter_step_sum_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_nccl_ring_reduce_scatter_step_sum_i32") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_nccl_ring_reduce_scatter_step_sum_i32";
    std::string_view sv(k_hip_navatala_nccl_ring_reduce_scatter_step_sum_i32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_nccl_ring_reduce_scatter_step_min_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_nccl_ring_reduce_scatter_step_min_f32";
    std::string_view sv(k_hip_navatala_nccl_ring_reduce_scatter_step_min_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_nccl_ring_reduce_scatter_step_min_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_nccl_ring_reduce_scatter_step_min_f16";
    std::string_view sv(k_hip_navatala_nccl_ring_reduce_scatter_step_min_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_nccl_ring_reduce_scatter_step_min_i32") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_nccl_ring_reduce_scatter_step_min_i32";
    std::string_view sv(k_hip_navatala_nccl_ring_reduce_scatter_step_min_i32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_nccl_ring_reduce_scatter_step_max_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_nccl_ring_reduce_scatter_step_max_f32";
    std::string_view sv(k_hip_navatala_nccl_ring_reduce_scatter_step_max_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_nccl_ring_reduce_scatter_step_max_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_nccl_ring_reduce_scatter_step_max_f16";
    std::string_view sv(k_hip_navatala_nccl_ring_reduce_scatter_step_max_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_nccl_ring_reduce_scatter_step_max_i32") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_nccl_ring_reduce_scatter_step_max_i32";
    std::string_view sv(k_hip_navatala_nccl_ring_reduce_scatter_step_max_i32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_nccl_ring_all_gather_step_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_nccl_ring_all_gather_step_f32";
    std::string_view sv(k_hip_navatala_nccl_ring_all_gather_step_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_nccl_ring_all_gather_step_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_nccl_ring_all_gather_step_f16";
    std::string_view sv(k_hip_navatala_nccl_ring_all_gather_step_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_nccl_ring_all_gather_step_i32") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_nccl_ring_all_gather_step_i32";
    std::string_view sv(k_hip_navatala_nccl_ring_all_gather_step_i32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_nccl_ring_all_reduce_sum_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_nccl_ring_all_reduce_sum_f32";
    std::string_view sv(k_hip_navatala_nccl_ring_all_reduce_sum_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_nccl_ring_all_reduce_sum_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_nccl_ring_all_reduce_sum_f16";
    std::string_view sv(k_hip_navatala_nccl_ring_all_reduce_sum_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_nccl_ring_all_reduce_sum_i32") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_nccl_ring_all_reduce_sum_i32";
    std::string_view sv(k_hip_navatala_nccl_ring_all_reduce_sum_i32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_nccl_ring_all_gather_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_nccl_ring_all_gather_f32";
    std::string_view sv(k_hip_navatala_nccl_ring_all_gather_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_nccl_ring_all_gather_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_nccl_ring_all_gather_f16";
    std::string_view sv(k_hip_navatala_nccl_ring_all_gather_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_nccl_ring_all_gather_i32") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_nccl_ring_all_gather_i32";
    std::string_view sv(k_hip_navatala_nccl_ring_all_gather_i32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_nccl_linear_broadcast_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_nccl_linear_broadcast_f32";
    std::string_view sv(k_hip_navatala_nccl_linear_broadcast_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_nccl_linear_broadcast_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_nccl_linear_broadcast_f16";
    std::string_view sv(k_hip_navatala_nccl_linear_broadcast_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_nccl_linear_broadcast_i32") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_nccl_linear_broadcast_i32";
    std::string_view sv(k_hip_navatala_nccl_linear_broadcast_i32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  return false;
}

} // namespace NavatalaRegistry

