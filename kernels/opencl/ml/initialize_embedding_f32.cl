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

__kernel void navatala_ml_initialize_embedding_f32(__global const uint* seed, __global const uint* n_samples, __global const uint* n_components, __global const float* spread, __global float* embedding) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint nSamples = n_samples[0];
  uint nComp = n_components[0];
  uint totalSize = (nSamples * nComp);
  uint seedVal = seed[0];
  float spreadVal = spread[0];
  bool inBounds = (gid < totalSize);
  if (inBounds) {
    uint state = (seedVal + gid);
    uint a = (uint)(1664525u);
    uint c = (uint)(1013904223u);
    uint newState = ((a * state) + c);
    float randBitsF = ((float)(newState));
    float maxU32 = as_float(0x4f800000u);
    float normalized = (randBitsF / maxU32);
    float centered = ((normalized * as_float(0x40000000u)) - as_float(0x3f800000u));
    float scaled = (centered * spreadVal);
    embedding[gid] = scaled;
  }
}
