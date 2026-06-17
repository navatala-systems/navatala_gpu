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

kernel void navatala_dataframe_argsort_bitonic_step_sort_by_key_i32(device const uint* count [[buffer(0)]], device const uint* stepSize [[buffer(1)]], device const uint* halfBlockSize [[buffer(2)]], device int* keys [[buffer(3)]], device int* values [[buffer(4)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint n = count[0u];
  uint step = stepSize[0u];
  uint halfBlock = halfBlockSize[0u];
  uint pairIdx = (gid * 2u);
  uint blockIdx = (pairIdx / step);
  uint inBlock = (pairIdx % step);
  uint i = ((blockIdx * step) + inBlock);
  uint j = (i + halfBlock);
  bool inBounds = (j < n);
  if (inBounds) {
    int keyI = keys[i];
    int keyJ = keys[j];
    int valI = values[i];
    int valJ = values[j];
    uint blockPairIdx = (i / (step * 2u));
    bool ascending = ((blockPairIdx % 2u) == 0u);
    bool needSwapAsc = (keyI > keyJ);
    bool needSwapDesc = (keyI < keyJ);
    bool needSwap = ((ascending) ? (needSwapAsc) : (needSwapDesc));
    int newKeyI = ((needSwap) ? (keyJ) : (keyI));
    int newKeyJ = ((needSwap) ? (keyI) : (keyJ));
    int newValI = ((needSwap) ? (valJ) : (valI));
    int newValJ = ((needSwap) ? (valI) : (valJ));
    keys[i] = newKeyI;
    keys[j] = newKeyJ;
    values[i] = newValI;
    values[j] = newValJ;
  }
}
