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

kernel void navatala_dataframe_extract_day_of_year_from_seconds(device const long* timestamps [[buffer(0)]], device const uint* validity [[buffer(1)]], device const uint* count [[buffer(2)]], device int* _output [[buffer(3)]], device uint* outputValid [[buffer(4)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint n = count[0u];
  bool inBounds = (gid < n);
  if (inBounds) {
    uint wordIdx = (gid / 32u);
    uint bitIdx = (gid % 32u);
    uint validWord = validity[wordIdx];
    uint validBit = ((validWord >> bitIdx) & 1u);
    bool isValid = (validBit == 1u);
    long ts = timestamps[gid];
    long secsInDay = 86400;
    long rawDays = (ts / secsInDay);
    long remainder = (ts % secsInDay);
    bool needsAdjust = ((ts < 0) & (remainder != 0));
    long days = ((needsAdjust) ? ((rawDays - 1)) : (rawDays));
    long daysFromYear0 = (days + 719528);
    long rem400 = (daysFromYear0 % 146097);
    long cycles100 = (rem400 / 36524);
    long adj100 = (((cycles100 == 4)) ? (3) : (cycles100));
    long rem100 = (rem400 - (adj100 * 36524));
    long rem4 = (rem100 % 1461);
    long years1 = (rem4 / 365);
    long adjYears1 = (((years1 == 4)) ? (3) : (years1));
    long dayOfYear0 = (rem4 - (adjYears1 * 365));
    long dayOfYear = (dayOfYear0 + 1);
    int doyI32 = ((int)(dayOfYear));
    int outVal = ((isValid) ? (doyI32) : (0));
    _output[gid] = outVal;
    uint outBit = (validBit << bitIdx);
    atomic_fetch_or_explicit((device atomic_uint*)(&outputValid[wordIdx]), outBit, memory_order_relaxed);
  }
}
