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
__kernel void navatala_dataframe_column_product_i64(__global const long* data, __global const uint* validity, __global const uint* count, __global long* sumLogResult, __global uint* negCount, __global uint* zeroFlag) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint n = count[(uint)(0u)];
  bool inBounds = (gid < n);
  if (inBounds) {
    uint wordIdx = (gid / (uint)(32u));
    uint bitIdx = (gid % (uint)(32u));
    uint validWord = validity[wordIdx];
    uint validBit = ((validWord >> bitIdx) & (uint)(1u));
    bool isValid = (validBit == (uint)(1u));
    if (isValid) {
      long val = data[gid];
      bool isZero = (val == 0);
      if (isZero) {
        atomic_max(&zeroFlag[(uint)(0u)], (uint)(1u));
      } else {
        bool isNeg = (val < 0);
        if (isNeg) {
          atomic_add(&negCount[(uint)(0u)], (uint)(1u));
        }
        long absVal = ((isNeg) ? ((0 - val)) : (val));
        double absF = ((double)(absVal));
        double logVal = log(absF);
        double scaledLog = (logVal * as_double(0x412e848000000000ul));
        long logFixed = ((long)(scaledLog));
        atomic_add(&sumLogResult[(uint)(0u)], logFixed);
      }
    }
  }
}
