// SPDX-License-Identifier: Apache-2.0
// Kernel registry partition for the `metal_collectives` bucket. Compiles
// against the declarations in `navatala/_registry.hpp` and links
// against the trampoline in `_registry.cpp`. Partitioning keeps each
// translation unit well under the 50K-line threshold so a typical
// build can parallelise across the buckets.
#include "navatala/_registry.hpp"
#include <cstdint>
#include <string>
#include <string_view>

const char* k_metal_navatala_collectives_direct_send_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_collectives_direct_send_f32(device const float* src [[buffer(0)]], device const uint* count [[buffer(1)]], device float* dst [[buffer(2)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint countVal = count[0];
  if (gid < countVal) {
    float val = src[gid];
    dst[gid] = val;
  }
}

)kernel";
const char* k_metal_navatala_collectives_direct_send_f16 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_collectives_direct_send_f16(device const half* src [[buffer(0)]], device const uint* count [[buffer(1)]], device half* dst [[buffer(2)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint countVal = count[0];
  if (gid < countVal) {
    half val = src[gid];
    dst[gid] = val;
  }
}

)kernel";
const char* k_metal_navatala_collectives_direct_send_i32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_collectives_direct_send_i32(device const int* src [[buffer(0)]], device const uint* count [[buffer(1)]], device int* dst [[buffer(2)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint countVal = count[0];
  if (gid < countVal) {
    int val = src[gid];
    dst[gid] = val;
  }
}

)kernel";
const char* k_metal_navatala_collectives_direct_recv_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_collectives_direct_recv_f32(device const float* src [[buffer(0)]], device const uint* count [[buffer(1)]], device float* dst [[buffer(2)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint countVal = count[0];
  if (gid < countVal) {
    float val = src[gid];
    dst[gid] = val;
  }
}

)kernel";
const char* k_metal_navatala_collectives_direct_recv_f16 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_collectives_direct_recv_f16(device const half* src [[buffer(0)]], device const uint* count [[buffer(1)]], device half* dst [[buffer(2)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint countVal = count[0];
  if (gid < countVal) {
    half val = src[gid];
    dst[gid] = val;
  }
}

)kernel";
const char* k_metal_navatala_collectives_direct_recv_i32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_collectives_direct_recv_i32(device const int* src [[buffer(0)]], device const uint* count [[buffer(1)]], device int* dst [[buffer(2)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint countVal = count[0];
  if (gid < countVal) {
    int val = src[gid];
    dst[gid] = val;
  }
}

)kernel";
const char* k_metal_navatala_collectives_reduce_local_sum_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_collectives_reduce_local_sum_f32(device const float* a [[buffer(0)]], device const float* b [[buffer(1)]], device const uint* count [[buffer(2)]], device float* _out [[buffer(3)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint countVal = count[0];
  if (gid < countVal) {
    float va = a[gid];
    float vb = b[gid];
    float result = (va + vb);
    _out[gid] = result;
  }
}

)kernel";
const char* k_metal_navatala_collectives_reduce_local_sum_f16 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_collectives_reduce_local_sum_f16(device const half* a [[buffer(0)]], device const half* b [[buffer(1)]], device const uint* count [[buffer(2)]], device half* _out [[buffer(3)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint countVal = count[0];
  if (gid < countVal) {
    half va = a[gid];
    half vb = b[gid];
    half result = (va + vb);
    _out[gid] = result;
  }
}

)kernel";
const char* k_metal_navatala_collectives_reduce_local_sum_i32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_collectives_reduce_local_sum_i32(device const int* a [[buffer(0)]], device const int* b [[buffer(1)]], device const uint* count [[buffer(2)]], device int* _out [[buffer(3)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint countVal = count[0];
  if (gid < countVal) {
    int va = a[gid];
    int vb = b[gid];
    int result = (va + vb);
    _out[gid] = result;
  }
}

)kernel";
const char* k_metal_navatala_collectives_reduce_local_min_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_collectives_reduce_local_min_f32(device const float* a [[buffer(0)]], device const float* b [[buffer(1)]], device const uint* count [[buffer(2)]], device float* _out [[buffer(3)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint countVal = count[0];
  if (gid < countVal) {
    float va = a[gid];
    float vb = b[gid];
    bool aLtB = (va < vb);
    float result = ((aLtB) ? (va) : (vb));
    _out[gid] = result;
  }
}

)kernel";
const char* k_metal_navatala_collectives_reduce_local_min_f16 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_collectives_reduce_local_min_f16(device const half* a [[buffer(0)]], device const half* b [[buffer(1)]], device const uint* count [[buffer(2)]], device half* _out [[buffer(3)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint countVal = count[0];
  if (gid < countVal) {
    half va = a[gid];
    half vb = b[gid];
    bool aLtB = (va < vb);
    half result = ((aLtB) ? (va) : (vb));
    _out[gid] = result;
  }
}

)kernel";
const char* k_metal_navatala_collectives_reduce_local_min_i32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_collectives_reduce_local_min_i32(device const int* a [[buffer(0)]], device const int* b [[buffer(1)]], device const uint* count [[buffer(2)]], device int* _out [[buffer(3)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint countVal = count[0];
  if (gid < countVal) {
    int va = a[gid];
    int vb = b[gid];
    bool aLtB = (va < vb);
    int result = ((aLtB) ? (va) : (vb));
    _out[gid] = result;
  }
}

)kernel";
const char* k_metal_navatala_collectives_reduce_local_max_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_collectives_reduce_local_max_f32(device const float* a [[buffer(0)]], device const float* b [[buffer(1)]], device const uint* count [[buffer(2)]], device float* _out [[buffer(3)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint countVal = count[0];
  if (gid < countVal) {
    float va = a[gid];
    float vb = b[gid];
    bool aGtB = (va > vb);
    float result = ((aGtB) ? (va) : (vb));
    _out[gid] = result;
  }
}

)kernel";
const char* k_metal_navatala_collectives_reduce_local_max_f16 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_collectives_reduce_local_max_f16(device const half* a [[buffer(0)]], device const half* b [[buffer(1)]], device const uint* count [[buffer(2)]], device half* _out [[buffer(3)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint countVal = count[0];
  if (gid < countVal) {
    half va = a[gid];
    half vb = b[gid];
    bool aGtB = (va > vb);
    half result = ((aGtB) ? (va) : (vb));
    _out[gid] = result;
  }
}

)kernel";
const char* k_metal_navatala_collectives_reduce_local_max_i32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_collectives_reduce_local_max_i32(device const int* a [[buffer(0)]], device const int* b [[buffer(1)]], device const uint* count [[buffer(2)]], device int* _out [[buffer(3)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint countVal = count[0];
  if (gid < countVal) {
    int va = a[gid];
    int vb = b[gid];
    bool aGtB = (va > vb);
    int result = ((aGtB) ? (va) : (vb));
    _out[gid] = result;
  }
}

)kernel";
const char* k_metal_navatala_collectives_ring_reduce_scatter_step_sum_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_collectives_ring_reduce_scatter_step_sum_f32(device const float* incoming [[buffer(0)]], device const float* local_buf [[buffer(1)]], device const uint* count [[buffer(2)]], device float* outgoing [[buffer(3)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint countVal = count[0];
  if (gid < countVal) {
    float inVal = incoming[gid];
    float locVal = local_buf[gid];
    float reduced = (inVal + locVal);
    outgoing[gid] = reduced;
  }
}

)kernel";
const char* k_metal_navatala_collectives_ring_reduce_scatter_step_sum_f16 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_collectives_ring_reduce_scatter_step_sum_f16(device const half* incoming [[buffer(0)]], device const half* local_buf [[buffer(1)]], device const uint* count [[buffer(2)]], device half* outgoing [[buffer(3)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint countVal = count[0];
  if (gid < countVal) {
    half inVal = incoming[gid];
    half locVal = local_buf[gid];
    half reduced = (inVal + locVal);
    outgoing[gid] = reduced;
  }
}

)kernel";
const char* k_metal_navatala_collectives_ring_reduce_scatter_step_sum_i32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_collectives_ring_reduce_scatter_step_sum_i32(device const int* incoming [[buffer(0)]], device const int* local_buf [[buffer(1)]], device const uint* count [[buffer(2)]], device int* outgoing [[buffer(3)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint countVal = count[0];
  if (gid < countVal) {
    int inVal = incoming[gid];
    int locVal = local_buf[gid];
    int reduced = (inVal + locVal);
    outgoing[gid] = reduced;
  }
}

)kernel";
const char* k_metal_navatala_collectives_ring_reduce_scatter_step_min_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_collectives_ring_reduce_scatter_step_min_f32(device const float* incoming [[buffer(0)]], device const float* local_buf [[buffer(1)]], device const uint* count [[buffer(2)]], device float* outgoing [[buffer(3)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint countVal = count[0];
  if (gid < countVal) {
    float inVal = incoming[gid];
    float locVal = local_buf[gid];
    bool cmp = (inVal < locVal);
    float reduced = ((cmp) ? (inVal) : (locVal));
    outgoing[gid] = reduced;
  }
}

)kernel";
const char* k_metal_navatala_collectives_ring_reduce_scatter_step_min_f16 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_collectives_ring_reduce_scatter_step_min_f16(device const half* incoming [[buffer(0)]], device const half* local_buf [[buffer(1)]], device const uint* count [[buffer(2)]], device half* outgoing [[buffer(3)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint countVal = count[0];
  if (gid < countVal) {
    half inVal = incoming[gid];
    half locVal = local_buf[gid];
    bool cmp = (inVal < locVal);
    half reduced = ((cmp) ? (inVal) : (locVal));
    outgoing[gid] = reduced;
  }
}

)kernel";
const char* k_metal_navatala_collectives_ring_reduce_scatter_step_min_i32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_collectives_ring_reduce_scatter_step_min_i32(device const int* incoming [[buffer(0)]], device const int* local_buf [[buffer(1)]], device const uint* count [[buffer(2)]], device int* outgoing [[buffer(3)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint countVal = count[0];
  if (gid < countVal) {
    int inVal = incoming[gid];
    int locVal = local_buf[gid];
    bool cmp = (inVal < locVal);
    int reduced = ((cmp) ? (inVal) : (locVal));
    outgoing[gid] = reduced;
  }
}

)kernel";
const char* k_metal_navatala_collectives_ring_reduce_scatter_step_max_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_collectives_ring_reduce_scatter_step_max_f32(device const float* incoming [[buffer(0)]], device const float* local_buf [[buffer(1)]], device const uint* count [[buffer(2)]], device float* outgoing [[buffer(3)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint countVal = count[0];
  if (gid < countVal) {
    float inVal = incoming[gid];
    float locVal = local_buf[gid];
    bool cmp = (inVal > locVal);
    float reduced = ((cmp) ? (inVal) : (locVal));
    outgoing[gid] = reduced;
  }
}

)kernel";
const char* k_metal_navatala_collectives_ring_reduce_scatter_step_max_f16 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_collectives_ring_reduce_scatter_step_max_f16(device const half* incoming [[buffer(0)]], device const half* local_buf [[buffer(1)]], device const uint* count [[buffer(2)]], device half* outgoing [[buffer(3)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint countVal = count[0];
  if (gid < countVal) {
    half inVal = incoming[gid];
    half locVal = local_buf[gid];
    bool cmp = (inVal > locVal);
    half reduced = ((cmp) ? (inVal) : (locVal));
    outgoing[gid] = reduced;
  }
}

)kernel";
const char* k_metal_navatala_collectives_ring_reduce_scatter_step_max_i32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_collectives_ring_reduce_scatter_step_max_i32(device const int* incoming [[buffer(0)]], device const int* local_buf [[buffer(1)]], device const uint* count [[buffer(2)]], device int* outgoing [[buffer(3)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint countVal = count[0];
  if (gid < countVal) {
    int inVal = incoming[gid];
    int locVal = local_buf[gid];
    bool cmp = (inVal > locVal);
    int reduced = ((cmp) ? (inVal) : (locVal));
    outgoing[gid] = reduced;
  }
}

)kernel";
const char* k_metal_navatala_collectives_ring_all_gather_step_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_collectives_ring_all_gather_step_f32(device const float* incoming [[buffer(0)]], device const uint* count [[buffer(1)]], device float* local_chunk [[buffer(2)]], device float* outgoing [[buffer(3)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint countVal = count[0];
  if (gid < countVal) {
    float val = incoming[gid];
    local_chunk[gid] = val;
    outgoing[gid] = val;
  }
}

)kernel";
const char* k_metal_navatala_collectives_ring_all_gather_step_f16 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_collectives_ring_all_gather_step_f16(device const half* incoming [[buffer(0)]], device const uint* count [[buffer(1)]], device half* local_chunk [[buffer(2)]], device half* outgoing [[buffer(3)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint countVal = count[0];
  if (gid < countVal) {
    half val = incoming[gid];
    local_chunk[gid] = val;
    outgoing[gid] = val;
  }
}

)kernel";
const char* k_metal_navatala_collectives_ring_all_gather_step_i32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_collectives_ring_all_gather_step_i32(device const int* incoming [[buffer(0)]], device const uint* count [[buffer(1)]], device int* local_chunk [[buffer(2)]], device int* outgoing [[buffer(3)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint countVal = count[0];
  if (gid < countVal) {
    int val = incoming[gid];
    local_chunk[gid] = val;
    outgoing[gid] = val;
  }
}

)kernel";
const char* k_metal_navatala_collectives_ring_all_reduce_sum_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_collectives_ring_all_reduce_sum_f32(device const float* sendbuf [[buffer(0)]], device const float* peer_buf [[buffer(1)]], device const uint* count [[buffer(2)]], device float* recvbuf [[buffer(3)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint countVal = count[0];
  if (gid < countVal) {
    float myVal = sendbuf[gid];
    float peerVal = peer_buf[gid];
    float reduced = (myVal + peerVal);
    recvbuf[gid] = reduced;
  }
}

)kernel";
const char* k_metal_navatala_collectives_ring_all_reduce_sum_f16 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_collectives_ring_all_reduce_sum_f16(device const half* sendbuf [[buffer(0)]], device const half* peer_buf [[buffer(1)]], device const uint* count [[buffer(2)]], device half* recvbuf [[buffer(3)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint countVal = count[0];
  if (gid < countVal) {
    half myVal = sendbuf[gid];
    half peerVal = peer_buf[gid];
    half reduced = (myVal + peerVal);
    recvbuf[gid] = reduced;
  }
}

)kernel";
const char* k_metal_navatala_collectives_ring_all_reduce_sum_i32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_collectives_ring_all_reduce_sum_i32(device const int* sendbuf [[buffer(0)]], device const int* peer_buf [[buffer(1)]], device const uint* count [[buffer(2)]], device int* recvbuf [[buffer(3)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint countVal = count[0];
  if (gid < countVal) {
    int myVal = sendbuf[gid];
    int peerVal = peer_buf[gid];
    int reduced = (myVal + peerVal);
    recvbuf[gid] = reduced;
  }
}

)kernel";
const char* k_metal_navatala_collectives_ring_all_gather_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_collectives_ring_all_gather_f32(device const float* incoming [[buffer(0)]], device const uint* count [[buffer(1)]], device float* local_slot [[buffer(2)]], device float* outgoing [[buffer(3)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint countVal = count[0];
  if (gid < countVal) {
    float val = incoming[gid];
    local_slot[gid] = val;
    outgoing[gid] = val;
  }
}

)kernel";
const char* k_metal_navatala_collectives_ring_all_gather_f16 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_collectives_ring_all_gather_f16(device const half* incoming [[buffer(0)]], device const uint* count [[buffer(1)]], device half* local_slot [[buffer(2)]], device half* outgoing [[buffer(3)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint countVal = count[0];
  if (gid < countVal) {
    half val = incoming[gid];
    local_slot[gid] = val;
    outgoing[gid] = val;
  }
}

)kernel";
const char* k_metal_navatala_collectives_ring_all_gather_i32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_collectives_ring_all_gather_i32(device const int* incoming [[buffer(0)]], device const uint* count [[buffer(1)]], device int* local_slot [[buffer(2)]], device int* outgoing [[buffer(3)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint countVal = count[0];
  if (gid < countVal) {
    int val = incoming[gid];
    local_slot[gid] = val;
    outgoing[gid] = val;
  }
}

)kernel";
const char* k_metal_navatala_collectives_linear_broadcast_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_collectives_linear_broadcast_f32(device const float* src [[buffer(0)]], device const uint* count [[buffer(1)]], device float* dst [[buffer(2)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint countVal = count[0];
  if (gid < countVal) {
    float val = src[gid];
    dst[gid] = val;
  }
}

)kernel";
const char* k_metal_navatala_collectives_linear_broadcast_f16 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_collectives_linear_broadcast_f16(device const half* src [[buffer(0)]], device const uint* count [[buffer(1)]], device half* dst [[buffer(2)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint countVal = count[0];
  if (gid < countVal) {
    half val = src[gid];
    dst[gid] = val;
  }
}

)kernel";
const char* k_metal_navatala_collectives_linear_broadcast_i32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_collectives_linear_broadcast_i32(device const int* src [[buffer(0)]], device const uint* count [[buffer(1)]], device int* dst [[buffer(2)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint countVal = count[0];
  if (gid < countVal) {
    int val = src[gid];
    dst[gid] = val;
  }
}

)kernel";

namespace NavatalaRegistry {

const KernelArgumentInfo kAbiArgs_metal_navatala_collectives_direct_send_f32[] = {
  { "src", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "dst", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_collectives_direct_send_f32 = {
  1,
  "navatala_collectives_direct_send_f32",
  "metal",
  "navatala_collectives_direct_send_f32",
  "kernel:metal:navatala_collectives_direct_send_f32",
  "abi-r1:metal:navatala_collectives_direct_send_f32",
  "abi-r1:metal:navatala_collectives_direct_send_f32",
  3,
  kAbiArgs_metal_navatala_collectives_direct_send_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_collectives_direct_send_f16[] = {
  { "src", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "dst", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_collectives_direct_send_f16 = {
  1,
  "navatala_collectives_direct_send_f16",
  "metal",
  "navatala_collectives_direct_send_f16",
  "kernel:metal:navatala_collectives_direct_send_f16",
  "abi-r1:metal:navatala_collectives_direct_send_f16",
  "abi-r1:metal:navatala_collectives_direct_send_f16",
  3,
  kAbiArgs_metal_navatala_collectives_direct_send_f16
};

const KernelArgumentInfo kAbiArgs_metal_navatala_collectives_direct_send_i32[] = {
  { "src", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "dst", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_collectives_direct_send_i32 = {
  1,
  "navatala_collectives_direct_send_i32",
  "metal",
  "navatala_collectives_direct_send_i32",
  "kernel:metal:navatala_collectives_direct_send_i32",
  "abi-r1:metal:navatala_collectives_direct_send_i32",
  "abi-r1:metal:navatala_collectives_direct_send_i32",
  3,
  kAbiArgs_metal_navatala_collectives_direct_send_i32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_collectives_direct_recv_f32[] = {
  { "src", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "dst", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_collectives_direct_recv_f32 = {
  1,
  "navatala_collectives_direct_recv_f32",
  "metal",
  "navatala_collectives_direct_recv_f32",
  "kernel:metal:navatala_collectives_direct_recv_f32",
  "abi-r1:metal:navatala_collectives_direct_recv_f32",
  "abi-r1:metal:navatala_collectives_direct_recv_f32",
  3,
  kAbiArgs_metal_navatala_collectives_direct_recv_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_collectives_direct_recv_f16[] = {
  { "src", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "dst", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_collectives_direct_recv_f16 = {
  1,
  "navatala_collectives_direct_recv_f16",
  "metal",
  "navatala_collectives_direct_recv_f16",
  "kernel:metal:navatala_collectives_direct_recv_f16",
  "abi-r1:metal:navatala_collectives_direct_recv_f16",
  "abi-r1:metal:navatala_collectives_direct_recv_f16",
  3,
  kAbiArgs_metal_navatala_collectives_direct_recv_f16
};

const KernelArgumentInfo kAbiArgs_metal_navatala_collectives_direct_recv_i32[] = {
  { "src", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "dst", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_collectives_direct_recv_i32 = {
  1,
  "navatala_collectives_direct_recv_i32",
  "metal",
  "navatala_collectives_direct_recv_i32",
  "kernel:metal:navatala_collectives_direct_recv_i32",
  "abi-r1:metal:navatala_collectives_direct_recv_i32",
  "abi-r1:metal:navatala_collectives_direct_recv_i32",
  3,
  kAbiArgs_metal_navatala_collectives_direct_recv_i32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_collectives_reduce_local_sum_f32[] = {
  { "a", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "b", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "out", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_collectives_reduce_local_sum_f32 = {
  1,
  "navatala_collectives_reduce_local_sum_f32",
  "metal",
  "navatala_collectives_reduce_local_sum_f32",
  "kernel:metal:navatala_collectives_reduce_local_sum_f32",
  "abi-r1:metal:navatala_collectives_reduce_local_sum_f32",
  "abi-r1:metal:navatala_collectives_reduce_local_sum_f32",
  4,
  kAbiArgs_metal_navatala_collectives_reduce_local_sum_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_collectives_reduce_local_sum_f16[] = {
  { "a", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "b", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "out", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_collectives_reduce_local_sum_f16 = {
  1,
  "navatala_collectives_reduce_local_sum_f16",
  "metal",
  "navatala_collectives_reduce_local_sum_f16",
  "kernel:metal:navatala_collectives_reduce_local_sum_f16",
  "abi-r1:metal:navatala_collectives_reduce_local_sum_f16",
  "abi-r1:metal:navatala_collectives_reduce_local_sum_f16",
  4,
  kAbiArgs_metal_navatala_collectives_reduce_local_sum_f16
};

const KernelArgumentInfo kAbiArgs_metal_navatala_collectives_reduce_local_sum_i32[] = {
  { "a", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "b", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "out", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_collectives_reduce_local_sum_i32 = {
  1,
  "navatala_collectives_reduce_local_sum_i32",
  "metal",
  "navatala_collectives_reduce_local_sum_i32",
  "kernel:metal:navatala_collectives_reduce_local_sum_i32",
  "abi-r1:metal:navatala_collectives_reduce_local_sum_i32",
  "abi-r1:metal:navatala_collectives_reduce_local_sum_i32",
  4,
  kAbiArgs_metal_navatala_collectives_reduce_local_sum_i32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_collectives_reduce_local_min_f32[] = {
  { "a", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "b", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "out", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_collectives_reduce_local_min_f32 = {
  1,
  "navatala_collectives_reduce_local_min_f32",
  "metal",
  "navatala_collectives_reduce_local_min_f32",
  "kernel:metal:navatala_collectives_reduce_local_min_f32",
  "abi-r1:metal:navatala_collectives_reduce_local_min_f32",
  "abi-r1:metal:navatala_collectives_reduce_local_min_f32",
  4,
  kAbiArgs_metal_navatala_collectives_reduce_local_min_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_collectives_reduce_local_min_f16[] = {
  { "a", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "b", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "out", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_collectives_reduce_local_min_f16 = {
  1,
  "navatala_collectives_reduce_local_min_f16",
  "metal",
  "navatala_collectives_reduce_local_min_f16",
  "kernel:metal:navatala_collectives_reduce_local_min_f16",
  "abi-r1:metal:navatala_collectives_reduce_local_min_f16",
  "abi-r1:metal:navatala_collectives_reduce_local_min_f16",
  4,
  kAbiArgs_metal_navatala_collectives_reduce_local_min_f16
};

const KernelArgumentInfo kAbiArgs_metal_navatala_collectives_reduce_local_min_i32[] = {
  { "a", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "b", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "out", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_collectives_reduce_local_min_i32 = {
  1,
  "navatala_collectives_reduce_local_min_i32",
  "metal",
  "navatala_collectives_reduce_local_min_i32",
  "kernel:metal:navatala_collectives_reduce_local_min_i32",
  "abi-r1:metal:navatala_collectives_reduce_local_min_i32",
  "abi-r1:metal:navatala_collectives_reduce_local_min_i32",
  4,
  kAbiArgs_metal_navatala_collectives_reduce_local_min_i32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_collectives_reduce_local_max_f32[] = {
  { "a", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "b", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "out", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_collectives_reduce_local_max_f32 = {
  1,
  "navatala_collectives_reduce_local_max_f32",
  "metal",
  "navatala_collectives_reduce_local_max_f32",
  "kernel:metal:navatala_collectives_reduce_local_max_f32",
  "abi-r1:metal:navatala_collectives_reduce_local_max_f32",
  "abi-r1:metal:navatala_collectives_reduce_local_max_f32",
  4,
  kAbiArgs_metal_navatala_collectives_reduce_local_max_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_collectives_reduce_local_max_f16[] = {
  { "a", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "b", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "out", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_collectives_reduce_local_max_f16 = {
  1,
  "navatala_collectives_reduce_local_max_f16",
  "metal",
  "navatala_collectives_reduce_local_max_f16",
  "kernel:metal:navatala_collectives_reduce_local_max_f16",
  "abi-r1:metal:navatala_collectives_reduce_local_max_f16",
  "abi-r1:metal:navatala_collectives_reduce_local_max_f16",
  4,
  kAbiArgs_metal_navatala_collectives_reduce_local_max_f16
};

const KernelArgumentInfo kAbiArgs_metal_navatala_collectives_reduce_local_max_i32[] = {
  { "a", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "b", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "out", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_collectives_reduce_local_max_i32 = {
  1,
  "navatala_collectives_reduce_local_max_i32",
  "metal",
  "navatala_collectives_reduce_local_max_i32",
  "kernel:metal:navatala_collectives_reduce_local_max_i32",
  "abi-r1:metal:navatala_collectives_reduce_local_max_i32",
  "abi-r1:metal:navatala_collectives_reduce_local_max_i32",
  4,
  kAbiArgs_metal_navatala_collectives_reduce_local_max_i32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_collectives_ring_reduce_scatter_step_sum_f32[] = {
  { "incoming", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "local_buf", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "outgoing", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_collectives_ring_reduce_scatter_step_sum_f32 = {
  1,
  "navatala_collectives_ring_reduce_scatter_step_sum_f32",
  "metal",
  "navatala_collectives_ring_reduce_scatter_step_sum_f32",
  "kernel:metal:navatala_collectives_ring_reduce_scatter_step_sum_f32",
  "abi-r1:metal:navatala_collectives_ring_reduce_scatter_step_sum_f32",
  "abi-r1:metal:navatala_collectives_ring_reduce_scatter_step_sum_f32",
  4,
  kAbiArgs_metal_navatala_collectives_ring_reduce_scatter_step_sum_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_collectives_ring_reduce_scatter_step_sum_f16[] = {
  { "incoming", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "local_buf", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "outgoing", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_collectives_ring_reduce_scatter_step_sum_f16 = {
  1,
  "navatala_collectives_ring_reduce_scatter_step_sum_f16",
  "metal",
  "navatala_collectives_ring_reduce_scatter_step_sum_f16",
  "kernel:metal:navatala_collectives_ring_reduce_scatter_step_sum_f16",
  "abi-r1:metal:navatala_collectives_ring_reduce_scatter_step_sum_f16",
  "abi-r1:metal:navatala_collectives_ring_reduce_scatter_step_sum_f16",
  4,
  kAbiArgs_metal_navatala_collectives_ring_reduce_scatter_step_sum_f16
};

const KernelArgumentInfo kAbiArgs_metal_navatala_collectives_ring_reduce_scatter_step_sum_i32[] = {
  { "incoming", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "local_buf", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "outgoing", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_collectives_ring_reduce_scatter_step_sum_i32 = {
  1,
  "navatala_collectives_ring_reduce_scatter_step_sum_i32",
  "metal",
  "navatala_collectives_ring_reduce_scatter_step_sum_i32",
  "kernel:metal:navatala_collectives_ring_reduce_scatter_step_sum_i32",
  "abi-r1:metal:navatala_collectives_ring_reduce_scatter_step_sum_i32",
  "abi-r1:metal:navatala_collectives_ring_reduce_scatter_step_sum_i32",
  4,
  kAbiArgs_metal_navatala_collectives_ring_reduce_scatter_step_sum_i32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_collectives_ring_reduce_scatter_step_min_f32[] = {
  { "incoming", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "local_buf", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "outgoing", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_collectives_ring_reduce_scatter_step_min_f32 = {
  1,
  "navatala_collectives_ring_reduce_scatter_step_min_f32",
  "metal",
  "navatala_collectives_ring_reduce_scatter_step_min_f32",
  "kernel:metal:navatala_collectives_ring_reduce_scatter_step_min_f32",
  "abi-r1:metal:navatala_collectives_ring_reduce_scatter_step_min_f32",
  "abi-r1:metal:navatala_collectives_ring_reduce_scatter_step_min_f32",
  4,
  kAbiArgs_metal_navatala_collectives_ring_reduce_scatter_step_min_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_collectives_ring_reduce_scatter_step_min_f16[] = {
  { "incoming", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "local_buf", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "outgoing", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_collectives_ring_reduce_scatter_step_min_f16 = {
  1,
  "navatala_collectives_ring_reduce_scatter_step_min_f16",
  "metal",
  "navatala_collectives_ring_reduce_scatter_step_min_f16",
  "kernel:metal:navatala_collectives_ring_reduce_scatter_step_min_f16",
  "abi-r1:metal:navatala_collectives_ring_reduce_scatter_step_min_f16",
  "abi-r1:metal:navatala_collectives_ring_reduce_scatter_step_min_f16",
  4,
  kAbiArgs_metal_navatala_collectives_ring_reduce_scatter_step_min_f16
};

const KernelArgumentInfo kAbiArgs_metal_navatala_collectives_ring_reduce_scatter_step_min_i32[] = {
  { "incoming", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "local_buf", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "outgoing", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_collectives_ring_reduce_scatter_step_min_i32 = {
  1,
  "navatala_collectives_ring_reduce_scatter_step_min_i32",
  "metal",
  "navatala_collectives_ring_reduce_scatter_step_min_i32",
  "kernel:metal:navatala_collectives_ring_reduce_scatter_step_min_i32",
  "abi-r1:metal:navatala_collectives_ring_reduce_scatter_step_min_i32",
  "abi-r1:metal:navatala_collectives_ring_reduce_scatter_step_min_i32",
  4,
  kAbiArgs_metal_navatala_collectives_ring_reduce_scatter_step_min_i32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_collectives_ring_reduce_scatter_step_max_f32[] = {
  { "incoming", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "local_buf", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "outgoing", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_collectives_ring_reduce_scatter_step_max_f32 = {
  1,
  "navatala_collectives_ring_reduce_scatter_step_max_f32",
  "metal",
  "navatala_collectives_ring_reduce_scatter_step_max_f32",
  "kernel:metal:navatala_collectives_ring_reduce_scatter_step_max_f32",
  "abi-r1:metal:navatala_collectives_ring_reduce_scatter_step_max_f32",
  "abi-r1:metal:navatala_collectives_ring_reduce_scatter_step_max_f32",
  4,
  kAbiArgs_metal_navatala_collectives_ring_reduce_scatter_step_max_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_collectives_ring_reduce_scatter_step_max_f16[] = {
  { "incoming", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "local_buf", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "outgoing", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_collectives_ring_reduce_scatter_step_max_f16 = {
  1,
  "navatala_collectives_ring_reduce_scatter_step_max_f16",
  "metal",
  "navatala_collectives_ring_reduce_scatter_step_max_f16",
  "kernel:metal:navatala_collectives_ring_reduce_scatter_step_max_f16",
  "abi-r1:metal:navatala_collectives_ring_reduce_scatter_step_max_f16",
  "abi-r1:metal:navatala_collectives_ring_reduce_scatter_step_max_f16",
  4,
  kAbiArgs_metal_navatala_collectives_ring_reduce_scatter_step_max_f16
};

const KernelArgumentInfo kAbiArgs_metal_navatala_collectives_ring_reduce_scatter_step_max_i32[] = {
  { "incoming", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "local_buf", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "outgoing", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_collectives_ring_reduce_scatter_step_max_i32 = {
  1,
  "navatala_collectives_ring_reduce_scatter_step_max_i32",
  "metal",
  "navatala_collectives_ring_reduce_scatter_step_max_i32",
  "kernel:metal:navatala_collectives_ring_reduce_scatter_step_max_i32",
  "abi-r1:metal:navatala_collectives_ring_reduce_scatter_step_max_i32",
  "abi-r1:metal:navatala_collectives_ring_reduce_scatter_step_max_i32",
  4,
  kAbiArgs_metal_navatala_collectives_ring_reduce_scatter_step_max_i32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_collectives_ring_all_gather_step_f32[] = {
  { "incoming", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "local_chunk", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "outgoing", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_collectives_ring_all_gather_step_f32 = {
  1,
  "navatala_collectives_ring_all_gather_step_f32",
  "metal",
  "navatala_collectives_ring_all_gather_step_f32",
  "kernel:metal:navatala_collectives_ring_all_gather_step_f32",
  "abi-r1:metal:navatala_collectives_ring_all_gather_step_f32",
  "abi-r1:metal:navatala_collectives_ring_all_gather_step_f32",
  4,
  kAbiArgs_metal_navatala_collectives_ring_all_gather_step_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_collectives_ring_all_gather_step_f16[] = {
  { "incoming", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "local_chunk", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "outgoing", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_collectives_ring_all_gather_step_f16 = {
  1,
  "navatala_collectives_ring_all_gather_step_f16",
  "metal",
  "navatala_collectives_ring_all_gather_step_f16",
  "kernel:metal:navatala_collectives_ring_all_gather_step_f16",
  "abi-r1:metal:navatala_collectives_ring_all_gather_step_f16",
  "abi-r1:metal:navatala_collectives_ring_all_gather_step_f16",
  4,
  kAbiArgs_metal_navatala_collectives_ring_all_gather_step_f16
};

const KernelArgumentInfo kAbiArgs_metal_navatala_collectives_ring_all_gather_step_i32[] = {
  { "incoming", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "local_chunk", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "outgoing", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_collectives_ring_all_gather_step_i32 = {
  1,
  "navatala_collectives_ring_all_gather_step_i32",
  "metal",
  "navatala_collectives_ring_all_gather_step_i32",
  "kernel:metal:navatala_collectives_ring_all_gather_step_i32",
  "abi-r1:metal:navatala_collectives_ring_all_gather_step_i32",
  "abi-r1:metal:navatala_collectives_ring_all_gather_step_i32",
  4,
  kAbiArgs_metal_navatala_collectives_ring_all_gather_step_i32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_collectives_ring_all_reduce_sum_f32[] = {
  { "sendbuf", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "peer_buf", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "recvbuf", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_collectives_ring_all_reduce_sum_f32 = {
  1,
  "navatala_collectives_ring_all_reduce_sum_f32",
  "metal",
  "navatala_collectives_ring_all_reduce_sum_f32",
  "kernel:metal:navatala_collectives_ring_all_reduce_sum_f32",
  "abi-r1:metal:navatala_collectives_ring_all_reduce_sum_f32",
  "abi-r1:metal:navatala_collectives_ring_all_reduce_sum_f32",
  4,
  kAbiArgs_metal_navatala_collectives_ring_all_reduce_sum_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_collectives_ring_all_reduce_sum_f16[] = {
  { "sendbuf", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "peer_buf", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "recvbuf", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_collectives_ring_all_reduce_sum_f16 = {
  1,
  "navatala_collectives_ring_all_reduce_sum_f16",
  "metal",
  "navatala_collectives_ring_all_reduce_sum_f16",
  "kernel:metal:navatala_collectives_ring_all_reduce_sum_f16",
  "abi-r1:metal:navatala_collectives_ring_all_reduce_sum_f16",
  "abi-r1:metal:navatala_collectives_ring_all_reduce_sum_f16",
  4,
  kAbiArgs_metal_navatala_collectives_ring_all_reduce_sum_f16
};

const KernelArgumentInfo kAbiArgs_metal_navatala_collectives_ring_all_reduce_sum_i32[] = {
  { "sendbuf", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "peer_buf", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "recvbuf", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_collectives_ring_all_reduce_sum_i32 = {
  1,
  "navatala_collectives_ring_all_reduce_sum_i32",
  "metal",
  "navatala_collectives_ring_all_reduce_sum_i32",
  "kernel:metal:navatala_collectives_ring_all_reduce_sum_i32",
  "abi-r1:metal:navatala_collectives_ring_all_reduce_sum_i32",
  "abi-r1:metal:navatala_collectives_ring_all_reduce_sum_i32",
  4,
  kAbiArgs_metal_navatala_collectives_ring_all_reduce_sum_i32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_collectives_ring_all_gather_f32[] = {
  { "incoming", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "local_slot", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "outgoing", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_collectives_ring_all_gather_f32 = {
  1,
  "navatala_collectives_ring_all_gather_f32",
  "metal",
  "navatala_collectives_ring_all_gather_f32",
  "kernel:metal:navatala_collectives_ring_all_gather_f32",
  "abi-r1:metal:navatala_collectives_ring_all_gather_f32",
  "abi-r1:metal:navatala_collectives_ring_all_gather_f32",
  4,
  kAbiArgs_metal_navatala_collectives_ring_all_gather_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_collectives_ring_all_gather_f16[] = {
  { "incoming", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "local_slot", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "outgoing", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_collectives_ring_all_gather_f16 = {
  1,
  "navatala_collectives_ring_all_gather_f16",
  "metal",
  "navatala_collectives_ring_all_gather_f16",
  "kernel:metal:navatala_collectives_ring_all_gather_f16",
  "abi-r1:metal:navatala_collectives_ring_all_gather_f16",
  "abi-r1:metal:navatala_collectives_ring_all_gather_f16",
  4,
  kAbiArgs_metal_navatala_collectives_ring_all_gather_f16
};

const KernelArgumentInfo kAbiArgs_metal_navatala_collectives_ring_all_gather_i32[] = {
  { "incoming", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "local_slot", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "outgoing", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_collectives_ring_all_gather_i32 = {
  1,
  "navatala_collectives_ring_all_gather_i32",
  "metal",
  "navatala_collectives_ring_all_gather_i32",
  "kernel:metal:navatala_collectives_ring_all_gather_i32",
  "abi-r1:metal:navatala_collectives_ring_all_gather_i32",
  "abi-r1:metal:navatala_collectives_ring_all_gather_i32",
  4,
  kAbiArgs_metal_navatala_collectives_ring_all_gather_i32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_collectives_linear_broadcast_f32[] = {
  { "src", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "dst", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_collectives_linear_broadcast_f32 = {
  1,
  "navatala_collectives_linear_broadcast_f32",
  "metal",
  "navatala_collectives_linear_broadcast_f32",
  "kernel:metal:navatala_collectives_linear_broadcast_f32",
  "abi-r1:metal:navatala_collectives_linear_broadcast_f32",
  "abi-r1:metal:navatala_collectives_linear_broadcast_f32",
  3,
  kAbiArgs_metal_navatala_collectives_linear_broadcast_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_collectives_linear_broadcast_f16[] = {
  { "src", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "dst", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_collectives_linear_broadcast_f16 = {
  1,
  "navatala_collectives_linear_broadcast_f16",
  "metal",
  "navatala_collectives_linear_broadcast_f16",
  "kernel:metal:navatala_collectives_linear_broadcast_f16",
  "abi-r1:metal:navatala_collectives_linear_broadcast_f16",
  "abi-r1:metal:navatala_collectives_linear_broadcast_f16",
  3,
  kAbiArgs_metal_navatala_collectives_linear_broadcast_f16
};

const KernelArgumentInfo kAbiArgs_metal_navatala_collectives_linear_broadcast_i32[] = {
  { "src", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "dst", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_collectives_linear_broadcast_i32 = {
  1,
  "navatala_collectives_linear_broadcast_i32",
  "metal",
  "navatala_collectives_linear_broadcast_i32",
  "kernel:metal:navatala_collectives_linear_broadcast_i32",
  "abi-r1:metal:navatala_collectives_linear_broadcast_i32",
  "abi-r1:metal:navatala_collectives_linear_broadcast_i32",
  3,
  kAbiArgs_metal_navatala_collectives_linear_broadcast_i32
};

bool tryGetKernelAbiManifest_metal_collectives(const std::string& backend, const std::string& kernelName, const KernelAbiManifestInfo*& out) {
  if (backend == "metal" && kernelName == "navatala_collectives_direct_send_f32") {
    out = &kAbiManifest_metal_navatala_collectives_direct_send_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_collectives_direct_send_f16") {
    out = &kAbiManifest_metal_navatala_collectives_direct_send_f16;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_collectives_direct_send_i32") {
    out = &kAbiManifest_metal_navatala_collectives_direct_send_i32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_collectives_direct_recv_f32") {
    out = &kAbiManifest_metal_navatala_collectives_direct_recv_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_collectives_direct_recv_f16") {
    out = &kAbiManifest_metal_navatala_collectives_direct_recv_f16;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_collectives_direct_recv_i32") {
    out = &kAbiManifest_metal_navatala_collectives_direct_recv_i32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_collectives_reduce_local_sum_f32") {
    out = &kAbiManifest_metal_navatala_collectives_reduce_local_sum_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_collectives_reduce_local_sum_f16") {
    out = &kAbiManifest_metal_navatala_collectives_reduce_local_sum_f16;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_collectives_reduce_local_sum_i32") {
    out = &kAbiManifest_metal_navatala_collectives_reduce_local_sum_i32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_collectives_reduce_local_min_f32") {
    out = &kAbiManifest_metal_navatala_collectives_reduce_local_min_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_collectives_reduce_local_min_f16") {
    out = &kAbiManifest_metal_navatala_collectives_reduce_local_min_f16;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_collectives_reduce_local_min_i32") {
    out = &kAbiManifest_metal_navatala_collectives_reduce_local_min_i32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_collectives_reduce_local_max_f32") {
    out = &kAbiManifest_metal_navatala_collectives_reduce_local_max_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_collectives_reduce_local_max_f16") {
    out = &kAbiManifest_metal_navatala_collectives_reduce_local_max_f16;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_collectives_reduce_local_max_i32") {
    out = &kAbiManifest_metal_navatala_collectives_reduce_local_max_i32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_collectives_ring_reduce_scatter_step_sum_f32") {
    out = &kAbiManifest_metal_navatala_collectives_ring_reduce_scatter_step_sum_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_collectives_ring_reduce_scatter_step_sum_f16") {
    out = &kAbiManifest_metal_navatala_collectives_ring_reduce_scatter_step_sum_f16;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_collectives_ring_reduce_scatter_step_sum_i32") {
    out = &kAbiManifest_metal_navatala_collectives_ring_reduce_scatter_step_sum_i32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_collectives_ring_reduce_scatter_step_min_f32") {
    out = &kAbiManifest_metal_navatala_collectives_ring_reduce_scatter_step_min_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_collectives_ring_reduce_scatter_step_min_f16") {
    out = &kAbiManifest_metal_navatala_collectives_ring_reduce_scatter_step_min_f16;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_collectives_ring_reduce_scatter_step_min_i32") {
    out = &kAbiManifest_metal_navatala_collectives_ring_reduce_scatter_step_min_i32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_collectives_ring_reduce_scatter_step_max_f32") {
    out = &kAbiManifest_metal_navatala_collectives_ring_reduce_scatter_step_max_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_collectives_ring_reduce_scatter_step_max_f16") {
    out = &kAbiManifest_metal_navatala_collectives_ring_reduce_scatter_step_max_f16;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_collectives_ring_reduce_scatter_step_max_i32") {
    out = &kAbiManifest_metal_navatala_collectives_ring_reduce_scatter_step_max_i32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_collectives_ring_all_gather_step_f32") {
    out = &kAbiManifest_metal_navatala_collectives_ring_all_gather_step_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_collectives_ring_all_gather_step_f16") {
    out = &kAbiManifest_metal_navatala_collectives_ring_all_gather_step_f16;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_collectives_ring_all_gather_step_i32") {
    out = &kAbiManifest_metal_navatala_collectives_ring_all_gather_step_i32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_collectives_ring_all_reduce_sum_f32") {
    out = &kAbiManifest_metal_navatala_collectives_ring_all_reduce_sum_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_collectives_ring_all_reduce_sum_f16") {
    out = &kAbiManifest_metal_navatala_collectives_ring_all_reduce_sum_f16;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_collectives_ring_all_reduce_sum_i32") {
    out = &kAbiManifest_metal_navatala_collectives_ring_all_reduce_sum_i32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_collectives_ring_all_gather_f32") {
    out = &kAbiManifest_metal_navatala_collectives_ring_all_gather_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_collectives_ring_all_gather_f16") {
    out = &kAbiManifest_metal_navatala_collectives_ring_all_gather_f16;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_collectives_ring_all_gather_i32") {
    out = &kAbiManifest_metal_navatala_collectives_ring_all_gather_i32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_collectives_linear_broadcast_f32") {
    out = &kAbiManifest_metal_navatala_collectives_linear_broadcast_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_collectives_linear_broadcast_f16") {
    out = &kAbiManifest_metal_navatala_collectives_linear_broadcast_f16;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_collectives_linear_broadcast_i32") {
    out = &kAbiManifest_metal_navatala_collectives_linear_broadcast_i32;
    return true;
  }
  out = nullptr;
  return false;
}

bool tryGetKernelSource_metal_collectives(const std::string& backend, const std::string& kernelName, GpuRuntime::ProgramSource& out) {
  if (backend == "metal" && kernelName == "navatala_collectives_direct_send_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_collectives_direct_send_f32";
    std::string_view sv(k_metal_navatala_collectives_direct_send_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_collectives_direct_send_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_collectives_direct_send_f16";
    std::string_view sv(k_metal_navatala_collectives_direct_send_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_collectives_direct_send_i32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_collectives_direct_send_i32";
    std::string_view sv(k_metal_navatala_collectives_direct_send_i32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_collectives_direct_recv_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_collectives_direct_recv_f32";
    std::string_view sv(k_metal_navatala_collectives_direct_recv_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_collectives_direct_recv_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_collectives_direct_recv_f16";
    std::string_view sv(k_metal_navatala_collectives_direct_recv_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_collectives_direct_recv_i32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_collectives_direct_recv_i32";
    std::string_view sv(k_metal_navatala_collectives_direct_recv_i32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_collectives_reduce_local_sum_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_collectives_reduce_local_sum_f32";
    std::string_view sv(k_metal_navatala_collectives_reduce_local_sum_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_collectives_reduce_local_sum_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_collectives_reduce_local_sum_f16";
    std::string_view sv(k_metal_navatala_collectives_reduce_local_sum_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_collectives_reduce_local_sum_i32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_collectives_reduce_local_sum_i32";
    std::string_view sv(k_metal_navatala_collectives_reduce_local_sum_i32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_collectives_reduce_local_min_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_collectives_reduce_local_min_f32";
    std::string_view sv(k_metal_navatala_collectives_reduce_local_min_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_collectives_reduce_local_min_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_collectives_reduce_local_min_f16";
    std::string_view sv(k_metal_navatala_collectives_reduce_local_min_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_collectives_reduce_local_min_i32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_collectives_reduce_local_min_i32";
    std::string_view sv(k_metal_navatala_collectives_reduce_local_min_i32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_collectives_reduce_local_max_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_collectives_reduce_local_max_f32";
    std::string_view sv(k_metal_navatala_collectives_reduce_local_max_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_collectives_reduce_local_max_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_collectives_reduce_local_max_f16";
    std::string_view sv(k_metal_navatala_collectives_reduce_local_max_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_collectives_reduce_local_max_i32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_collectives_reduce_local_max_i32";
    std::string_view sv(k_metal_navatala_collectives_reduce_local_max_i32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_collectives_ring_reduce_scatter_step_sum_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_collectives_ring_reduce_scatter_step_sum_f32";
    std::string_view sv(k_metal_navatala_collectives_ring_reduce_scatter_step_sum_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_collectives_ring_reduce_scatter_step_sum_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_collectives_ring_reduce_scatter_step_sum_f16";
    std::string_view sv(k_metal_navatala_collectives_ring_reduce_scatter_step_sum_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_collectives_ring_reduce_scatter_step_sum_i32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_collectives_ring_reduce_scatter_step_sum_i32";
    std::string_view sv(k_metal_navatala_collectives_ring_reduce_scatter_step_sum_i32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_collectives_ring_reduce_scatter_step_min_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_collectives_ring_reduce_scatter_step_min_f32";
    std::string_view sv(k_metal_navatala_collectives_ring_reduce_scatter_step_min_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_collectives_ring_reduce_scatter_step_min_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_collectives_ring_reduce_scatter_step_min_f16";
    std::string_view sv(k_metal_navatala_collectives_ring_reduce_scatter_step_min_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_collectives_ring_reduce_scatter_step_min_i32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_collectives_ring_reduce_scatter_step_min_i32";
    std::string_view sv(k_metal_navatala_collectives_ring_reduce_scatter_step_min_i32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_collectives_ring_reduce_scatter_step_max_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_collectives_ring_reduce_scatter_step_max_f32";
    std::string_view sv(k_metal_navatala_collectives_ring_reduce_scatter_step_max_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_collectives_ring_reduce_scatter_step_max_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_collectives_ring_reduce_scatter_step_max_f16";
    std::string_view sv(k_metal_navatala_collectives_ring_reduce_scatter_step_max_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_collectives_ring_reduce_scatter_step_max_i32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_collectives_ring_reduce_scatter_step_max_i32";
    std::string_view sv(k_metal_navatala_collectives_ring_reduce_scatter_step_max_i32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_collectives_ring_all_gather_step_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_collectives_ring_all_gather_step_f32";
    std::string_view sv(k_metal_navatala_collectives_ring_all_gather_step_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_collectives_ring_all_gather_step_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_collectives_ring_all_gather_step_f16";
    std::string_view sv(k_metal_navatala_collectives_ring_all_gather_step_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_collectives_ring_all_gather_step_i32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_collectives_ring_all_gather_step_i32";
    std::string_view sv(k_metal_navatala_collectives_ring_all_gather_step_i32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_collectives_ring_all_reduce_sum_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_collectives_ring_all_reduce_sum_f32";
    std::string_view sv(k_metal_navatala_collectives_ring_all_reduce_sum_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_collectives_ring_all_reduce_sum_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_collectives_ring_all_reduce_sum_f16";
    std::string_view sv(k_metal_navatala_collectives_ring_all_reduce_sum_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_collectives_ring_all_reduce_sum_i32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_collectives_ring_all_reduce_sum_i32";
    std::string_view sv(k_metal_navatala_collectives_ring_all_reduce_sum_i32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_collectives_ring_all_gather_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_collectives_ring_all_gather_f32";
    std::string_view sv(k_metal_navatala_collectives_ring_all_gather_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_collectives_ring_all_gather_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_collectives_ring_all_gather_f16";
    std::string_view sv(k_metal_navatala_collectives_ring_all_gather_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_collectives_ring_all_gather_i32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_collectives_ring_all_gather_i32";
    std::string_view sv(k_metal_navatala_collectives_ring_all_gather_i32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_collectives_linear_broadcast_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_collectives_linear_broadcast_f32";
    std::string_view sv(k_metal_navatala_collectives_linear_broadcast_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_collectives_linear_broadcast_f16") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_collectives_linear_broadcast_f16";
    std::string_view sv(k_metal_navatala_collectives_linear_broadcast_f16);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_collectives_linear_broadcast_i32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_collectives_linear_broadcast_i32";
    std::string_view sv(k_metal_navatala_collectives_linear_broadcast_i32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  return false;
}

} // namespace NavatalaRegistry

