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

#pragma OPENCL EXTENSION cl_khr_fp64 : enable
__kernel void navatala_dataframe_shuffle_rows_f64(__global const double* _input, __global const uint* random, __global const uint* m, __global const uint* n, __global double* _output) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint mVal = m[(uint)(0u)];
  uint nVal = n[(uint)(0u)];
  bool inBounds = (gid < mVal);
  if (inBounds) {
    uint randVal = random[gid];
    uint srcRow = (randVal % mVal);
    for (int col = 0; col < (int)(nVal); ++col) {
      uint colU32 = ((uint)(col));
      uint srcIdx = ((srcRow * nVal) + colU32);
      uint dstIdx = ((gid * nVal) + colU32);
      double val = _input[srcIdx];
      _output[dstIdx] = val;
    }
  }
}
