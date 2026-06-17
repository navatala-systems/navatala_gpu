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

#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__kernel void navatala_transformer_generate_causal_mask_f16(__global const uint* seqLen, __global half* _output) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint sl = seqLen[(uint)(0u)];
  uint total = (sl * sl);
  bool inBounds = (gid < total);
  if (inBounds) {
    uint row = (gid / sl);
    uint col = (gid % sl);
    bool canAttend = (col <= row);
    if (canAttend) {
      _output[gid] = (half)(0.000000f);
    } else {
      _output[gid] = (half)(-65000.000000f);
    }
  }
}
