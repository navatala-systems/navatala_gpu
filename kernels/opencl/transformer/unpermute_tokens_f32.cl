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

__kernel void navatala_transformer_unpermute_tokens_f32(__global const float* permutedStates, __global const int* tokenIndices, __global const uint* numTokens, __global const uint* hiddenSize, __global float* hiddenStates) {
  int gid0 = (int)get_global_id(0);
  uint lid = ((uint)((int)(get_local_id(0))));
  uint permutedIdx = ((uint)((int)(get_group_id(0))));
  uint nt = numTokens[(uint)(0u)];
  uint hs = hiddenSize[(uint)(0u)];
  bool valid = (permutedIdx < nt);
  if (valid) {
    int origIdxI32 = tokenIndices[permutedIdx];
    uint origIdx = ((uint)(origIdxI32));
    bool origValid = (origIdx < nt);
    if (origValid) {
      for (int hIter = 0; hIter < (int)(hs); ++hIter) {
        uint hIdx = (lid + ((uint)(256u) * hIter));
        if (hIdx < hs) {
          uint srcIdx = ((permutedIdx * hs) + hIdx);
          uint dstIdx = ((origIdx * hs) + hIdx);
          float val = permutedStates[srcIdx];
          hiddenStates[dstIdx] = val;
        }
      }
    }
  }
}
