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

__kernel void navatala_dataframe_scan_exclusive_write_total_u32(__global const uint* _input, __global const uint* count, __global uint* _output) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  if ((gid == (uint)(0u))) {
    uint n = count[(uint)(0u)];
    if ((n == (uint)(0u))) {
      _output[(uint)(0u)] = (uint)(0u);
    } else {
      uint nMinus1 = (n - (uint)(1u));
      uint prevExc = _output[nMinus1];
      uint lastIn = _input[nMinus1];
      _output[n] = (prevExc + lastIn);
    }
  }
}
