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

kernel void navatala_transformer_unpermute_tokens_f32(device const float* permutedStates [[buffer(0)]], device const int* tokenIndices [[buffer(1)]], device const uint* numTokens [[buffer(2)]], device const uint* hiddenSize [[buffer(3)]], device float* hiddenStates [[buffer(4)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint lid = ((uint)(int(__tid.x)));
  uint permutedIdx = ((uint)(int(__tgid.x)));
  uint nt = numTokens[0u];
  uint hs = hiddenSize[0u];
  bool valid = (permutedIdx < nt);
  if (valid) {
    int origIdxI32 = tokenIndices[permutedIdx];
    uint origIdx = ((uint)(origIdxI32));
    bool origValid = (origIdx < nt);
    if (origValid) {
      for (int hIter = 0; hIter < (int)(hs); ++hIter) {
        uint hIdx = (lid + (256u * hIter));
        if ((hIdx < hs)) {
          uint srcIdx = ((permutedIdx * hs) + hIdx);
          uint dstIdx = ((origIdx * hs) + hIdx);
          float val = permutedStates[srcIdx];
          hiddenStates[dstIdx] = val;
        }
      }
    }
  }
}
