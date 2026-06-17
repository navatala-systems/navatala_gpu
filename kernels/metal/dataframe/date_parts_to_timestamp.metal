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

kernel void navatala_dataframe_date_parts_to_timestamp(device const int* yearsIn [[buffer(0)]], device const int* monthsIn [[buffer(1)]], device const int* daysIn [[buffer(2)]], device const uint* validity [[buffer(3)]], device const uint* count [[buffer(4)]], device long* _output [[buffer(5)]], device uint* outputValid [[buffer(6)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint n = count[0u];
  bool inBounds = (gid < n);
  if (inBounds) {
    uint wordIdx = (gid / 32u);
    uint bitIdx = (gid % 32u);
    uint validWord = validity[wordIdx];
    uint validBit = ((validWord >> bitIdx) & 1u);
    bool isValid = (validBit == 1u);
    int yearI32 = yearsIn[gid];
    int monthI32 = monthsIn[gid];
    int dayI32 = daysIn[gid];
    long year = ((long)(yearI32));
    long month = ((long)(monthI32));
    long day = ((long)(dayI32));
    bool div4 = ((year % 4) == 0);
    bool div100 = ((year % 100) == 0);
    bool div400 = ((year % 400) == 0);
    bool isLeap = (div4 && ((!div100) || div400));
    long leapAdj = ((isLeap) ? (1) : (0));
    bool isJan = (month == 1);
    bool isFeb = (month == 2);
    bool isMar = (month == 3);
    bool isApr = (month == 4);
    bool isMay = (month == 5);
    bool isJun = (month == 6);
    bool isJul = (month == 7);
    bool isAug = (month == 8);
    bool isSep = (month == 9);
    bool isOct = (month == 10);
    bool isNov = (month == 11);
    long daysBeforeMonth = ((isJan) ? (0) : (((isFeb) ? (31) : (((isMar) ? ((59 + leapAdj)) : (((isApr) ? ((90 + leapAdj)) : (((isMay) ? ((120 + leapAdj)) : (((isJun) ? ((151 + leapAdj)) : (((isJul) ? ((181 + leapAdj)) : (((isAug) ? ((212 + leapAdj)) : (((isSep) ? ((243 + leapAdj)) : (((isOct) ? ((273 + leapAdj)) : (((isNov) ? ((304 + leapAdj)) : ((334 + leapAdj)))))))))))))))))))))));
    long dayOfYear = (daysBeforeMonth + (day - 1));
    long yearM1 = (year - 1);
    long baseDays = (yearM1 * 365);
    long leap4 = (yearM1 / 4);
    long leap100 = (yearM1 / 100);
    long leap400 = (yearM1 / 400);
    long daysFromYear0 = (((baseDays + leap4) - leap100) + leap400);
    long totalDaysFromYear0 = (daysFromYear0 + dayOfYear);
    long daysSinceEpoch = (totalDaysFromYear0 - 719527);
    long usPerDay = 86400000000;
    long tsUs = (daysSinceEpoch * usPerDay);
    long outVal = ((isValid) ? (tsUs) : (0));
    _output[gid] = outVal;
    uint outBit = (validBit << bitIdx);
    atomic_fetch_or_explicit((device atomic_uint*)(&outputValid[wordIdx]), outBit, memory_order_relaxed);
  }
}
