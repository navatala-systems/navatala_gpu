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

__kernel void navatala_ml_kde_bandwidth_silverman_f32(__global const float* stddev, __global const uint* n, __global float* bandwidth) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  if ((gid == (uint)(0u))) {
    float sigma = stddev[0];
    uint count = n[0];
    float countF = ((float)(count));
    float logN = log(countF);
    float negLogN = (as_float(0xbe4ccccdu) * logN);
    float nPowNeg02 = exp(negLogN);
    float h = (as_float(0x3f87ae14u) * (sigma * nPowNeg02));
    bandwidth[0] = h;
  }
}
