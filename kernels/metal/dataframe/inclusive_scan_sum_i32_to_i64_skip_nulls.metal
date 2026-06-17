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

#include <metal_stdlib>
using namespace metal;

static inline int gpu_atomic_cas_int(device atomic_int* ptr, int expected, int desired) {
  int exp = expected;
  while (!atomic_compare_exchange_weak_explicit(ptr, &exp, desired,
      memory_order_relaxed, memory_order_relaxed) && exp == expected) {}
  return exp;
}

static inline uint gpu_atomic_cas_uint(device atomic_uint* ptr, uint expected, uint desired) {
  uint exp = expected;
  while (!atomic_compare_exchange_weak_explicit(ptr, &exp, desired,
      memory_order_relaxed, memory_order_relaxed) && exp == expected) {}
  return exp;
}

kernel void navatala_dataframe_inclusive_scan_sum_i32_to_i64_skip_nulls(device const int* _input [[buffer(0)]], device const uint* inputValid [[buffer(1)]], device const uint* count [[buffer(2)]], device long* _output [[buffer(3)]], device uint* outputValid [[buffer(4)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint lid = ((uint)(int(__tid.x)));
  uint n = count[0u];
  threadgroup long scanData[256];
  threadgroup uint scanValid[256];
  bool inBounds = (gid < n);
  uint wordIdx = (gid / 32u);
  uint bitIdx = (gid % 32u);
  uint validWord = ((inBounds) ? (inputValid[wordIdx]) : (0u));
  uint validBit = ((validWord >> bitIdx) & 1u);
  bool isValid = (validBit == 1u);
  long inputVal = (((inBounds && isValid)) ? (((long)(_input[gid]))) : (0));
  scanData[lid] = inputVal;
  scanValid[lid] = (((inBounds && isValid)) ? (1u) : (0u));
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool hasNeighbor1 = (lid >= 1u);
  if (hasNeighbor1) {
    uint neighborIdx1 = (lid - 1u);
    long neighborVal1 = scanData[neighborIdx1];
    long myVal1 = scanData[lid];
    long newVal1 = (myVal1 + neighborVal1);
    scanData[lid] = newVal1;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool hasNeighbor2 = (lid >= 2u);
  if (hasNeighbor2) {
    uint neighborIdx2 = (lid - 2u);
    long neighborVal2 = scanData[neighborIdx2];
    long myVal2 = scanData[lid];
    long newVal2 = (myVal2 + neighborVal2);
    scanData[lid] = newVal2;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool hasNeighbor4 = (lid >= 4u);
  if (hasNeighbor4) {
    uint neighborIdx4 = (lid - 4u);
    long neighborVal4 = scanData[neighborIdx4];
    long myVal4 = scanData[lid];
    long newVal4 = (myVal4 + neighborVal4);
    scanData[lid] = newVal4;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool hasNeighbor8 = (lid >= 8u);
  if (hasNeighbor8) {
    uint neighborIdx8 = (lid - 8u);
    long neighborVal8 = scanData[neighborIdx8];
    long myVal8 = scanData[lid];
    long newVal8 = (myVal8 + neighborVal8);
    scanData[lid] = newVal8;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool hasNeighbor16 = (lid >= 16u);
  if (hasNeighbor16) {
    uint neighborIdx16 = (lid - 16u);
    long neighborVal16 = scanData[neighborIdx16];
    long myVal16 = scanData[lid];
    long newVal16 = (myVal16 + neighborVal16);
    scanData[lid] = newVal16;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool hasNeighbor32 = (lid >= 32u);
  if (hasNeighbor32) {
    uint neighborIdx32 = (lid - 32u);
    long neighborVal32 = scanData[neighborIdx32];
    long myVal32 = scanData[lid];
    long newVal32 = (myVal32 + neighborVal32);
    scanData[lid] = newVal32;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool hasNeighbor64 = (lid >= 64u);
  if (hasNeighbor64) {
    uint neighborIdx64 = (lid - 64u);
    long neighborVal64 = scanData[neighborIdx64];
    long myVal64 = scanData[lid];
    long newVal64 = (myVal64 + neighborVal64);
    scanData[lid] = newVal64;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool hasNeighbor128 = (lid >= 128u);
  if (hasNeighbor128) {
    uint neighborIdx128 = (lid - 128u);
    long neighborVal128 = scanData[neighborIdx128];
    long myVal128 = scanData[lid];
    long newVal128 = (myVal128 + neighborVal128);
    scanData[lid] = newVal128;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (inBounds) {
    long scanResult = scanData[lid];
    uint rowValid = scanValid[lid];
    _output[gid] = scanResult;
    uint outBit = (rowValid << bitIdx);
    atomic_fetch_or_explicit((device atomic_uint*)(&outputValid[wordIdx]), outBit, memory_order_relaxed);
  }
}
