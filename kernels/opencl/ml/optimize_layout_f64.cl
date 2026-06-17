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
__kernel void navatala_ml_optimize_layout_f64(__global const double* embedding, __global const double* gradients, __global const uint* n_samples, __global const uint* n_components, __global const double* alpha, __global double* embedding_out) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint nSamples = n_samples[0];
  uint nComp = n_components[0];
  uint totalSize = (nSamples * nComp);
  double alphaVal = alpha[0];
  bool inBounds = (gid < totalSize);
  if (inBounds) {
    double embVal = embedding[gid];
    double gradVal = gradients[gid];
    double gradClipped = (((gradVal > as_double(0x4010000000000000ul))) ? (as_double(0x4010000000000000ul)) : ((((gradVal < as_double(0xc010000000000000ul))) ? (as_double(0xc010000000000000ul)) : (gradVal))));
    double update = (alphaVal * gradClipped);
    double newVal = (embVal + update);
    embedding_out[gid] = newVal;
  }
}
