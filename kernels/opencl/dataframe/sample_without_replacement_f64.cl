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
__kernel void navatala_dataframe_sample_without_replacement_f64(__global const double* data, __global const uint* random, __global const uint* n, __global const uint* k, __global double* _output, __global uint* indices) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint nVal = n[(uint)(0u)];
  uint kVal = k[(uint)(0u)];
  bool inBounds = (gid < kVal);
  if (inBounds) {
    uint randVal = random[gid];
    uint idx = (randVal % nVal);
    double val = data[idx];
    _output[gid] = val;
    indices[gid] = idx;
  }
}
