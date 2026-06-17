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
__kernel void navatala_ml_initialize_embedding_f64(__global const uint* seed, __global const uint* n_samples, __global const uint* n_components, __global const double* spread, __global double* embedding) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint nSamples = n_samples[0];
  uint nComp = n_components[0];
  uint totalSize = (nSamples * nComp);
  uint seedVal = seed[0];
  double spreadVal = spread[0];
  bool inBounds = (gid < totalSize);
  if (inBounds) {
    uint state = (seedVal + gid);
    uint a = (uint)(1664525u);
    uint c = (uint)(1013904223u);
    uint newState = ((a * state) + c);
    double randBitsF = ((double)(newState));
    double maxU32 = as_double(0x41efffffffe00000ul);
    double normalized = (randBitsF / maxU32);
    double centered = ((normalized * as_double(0x4000000000000000ul)) - as_double(0x3ff0000000000000ul));
    double scaled = (centered * spreadVal);
    embedding[gid] = scaled;
  }
}
