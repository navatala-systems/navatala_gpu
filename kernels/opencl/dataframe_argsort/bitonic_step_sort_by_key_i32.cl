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

__kernel void navatala_dataframe_argsort_bitonic_step_sort_by_key_i32(__global const uint* count, __global const uint* stepSize, __global const uint* halfBlockSize, __global int* keys, __global int* values) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint n = count[(uint)(0u)];
  uint step = stepSize[(uint)(0u)];
  uint halfBlock = halfBlockSize[(uint)(0u)];
  uint pairIdx = (gid * (uint)(2u));
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
    uint blockPairIdx = (i / (step * (uint)(2u)));
    bool ascending = ((blockPairIdx % (uint)(2u)) == (uint)(0u));
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
