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

kernel void navatala_transformer_hadamard_transform_f16(device const half* _input [[buffer(0)]], device const uint* batchSize [[buffer(1)]], device const uint* vectorSize [[buffer(2)]], device half* _output [[buffer(3)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint lid = ((uint)(int(__tid.x)));
  uint batchIdx = ((uint)(int(__tgid.x)));
  uint bs = batchSize[0u];
  uint vs = vectorSize[0u];
  threadgroup float sharedBuf[256];
  bool batchValid = (batchIdx < bs);
  bool threadValid = (lid < vs);
  uint globalIdx = ((batchIdx * vs) + lid);
  if (batchValid && threadValid) {
    half valF16 = _input[globalIdx];
    float val = ((float)(valF16));
    sharedBuf[lid] = val;
  } else {
    sharedBuf[lid] = as_type<float>(0x00000000u);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  uint stride1 = 1u;
  uint partner1 = (lid ^ stride1);
  float myVal1 = sharedBuf[lid];
  float partnerVal1 = sharedBuf[partner1];
  bool lowBit1 = ((lid & stride1) == 0u);
  float newVal1 = ((lowBit1) ? ((myVal1 + partnerVal1)) : ((partnerVal1 - myVal1)));
  threadgroup_barrier(mem_flags::mem_threadgroup);
  sharedBuf[lid] = newVal1;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  uint stride2 = 2u;
  uint partner2 = (lid ^ stride2);
  float myVal2 = sharedBuf[lid];
  float partnerVal2 = sharedBuf[partner2];
  bool lowBit2 = ((lid & stride2) == 0u);
  float newVal2 = ((lowBit2) ? ((myVal2 + partnerVal2)) : ((partnerVal2 - myVal2)));
  threadgroup_barrier(mem_flags::mem_threadgroup);
  sharedBuf[lid] = newVal2;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  uint stride3 = 4u;
  uint partner3 = (lid ^ stride3);
  float myVal3 = sharedBuf[lid];
  float partnerVal3 = sharedBuf[partner3];
  bool lowBit3 = ((lid & stride3) == 0u);
  float newVal3 = ((lowBit3) ? ((myVal3 + partnerVal3)) : ((partnerVal3 - myVal3)));
  threadgroup_barrier(mem_flags::mem_threadgroup);
  sharedBuf[lid] = newVal3;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  uint stride4 = 8u;
  uint partner4 = (lid ^ stride4);
  float myVal4 = sharedBuf[lid];
  float partnerVal4 = sharedBuf[partner4];
  bool lowBit4 = ((lid & stride4) == 0u);
  float newVal4 = ((lowBit4) ? ((myVal4 + partnerVal4)) : ((partnerVal4 - myVal4)));
  float vsF32 = ((float)(vs));
  float scale = sqrt(vsF32);
  float normalized = (newVal4 / scale);
  half resultF16 = ((half)(normalized));
  if (batchValid && threadValid) {
    _output[globalIdx] = resultF16;
  }
}
