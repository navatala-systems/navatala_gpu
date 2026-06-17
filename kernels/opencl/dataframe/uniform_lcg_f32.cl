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

__kernel void navatala_dataframe_uniform_lcg_f32(__global ulong* state, __global const uint* count, __global float* _output) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint countVal = count[(uint)(0u)];
  bool inBounds = (gid < countVal);
  if (inBounds) {
    ulong s = state[gid];
    ulong aU64 = ((ulong)((uint)(1103515245u)));
    ulong cU64 = ((ulong)((uint)(12345u)));
    ulong prod = (aU64 * s);
    ulong sum = (prod + cU64);
    ulong newState = (sum & ((ulong)((uint)(2147483647u))));
    state[gid] = newState;
    float stateF32 = ((float)(newState));
    float randVal = (stateF32 / as_float(0x4f000000u));
    _output[gid] = randVal;
  }
}
