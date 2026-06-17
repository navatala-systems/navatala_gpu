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
__kernel void navatala_dataframe_gather_f64(__global const double* _input, __global const uint* inputValid, __global const int* indices, __global const uint* inputSize, __global const uint* outputSize, __global double* _output, __global uint* outputValid) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint nIn = inputSize[(uint)(0u)];
  uint nOut = outputSize[(uint)(0u)];
  bool inBounds = (gid < nOut);
  if (inBounds) {
    int idx = indices[gid];
    uint idxU32 = ((uint)(idx));
    bool idxValid = ((idx >= 0) && (idxU32 < nIn));
    uint srcWordIdx = (idxU32 / (uint)(32u));
    uint srcBitIdx = (idxU32 % (uint)(32u));
    uint srcValidWord = ((idxValid) ? (inputValid[srcWordIdx]) : ((uint)(0u)));
    uint srcValidBit = ((srcValidWord >> srcBitIdx) & (uint)(1u));
    bool srcIsValid = (srcValidBit == (uint)(1u));
    bool outIsValid = (idxValid && srcIsValid);
    double val = ((outIsValid) ? (_input[idxU32]) : (as_double(0x0000000000000000ul)));
    _output[gid] = val;
    uint outWordIdx = (gid / (uint)(32u));
    uint outBitIdx = (gid % (uint)(32u));
    uint outBit = ((outIsValid) ? (((uint)(1u) << outBitIdx)) : ((uint)(0u)));
    atomic_or(&outputValid[outWordIdx], outBit);
  }
}
