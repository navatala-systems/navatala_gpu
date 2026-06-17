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

__kernel void navatala_sparse_sparse_cg_iter_f32(__global const uint* rowPtr, __global const uint* colInd, __global const float* values, __global const float* x, __global const float* p, __global const float* r, __global float* q, __global const float* rTr, __global const float* alpha, __global const uint* n, __global float* x_new, __global float* r_new) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint i = gid;
  uint nVal = n[(uint)(0u)];
  if ((i < nVal)) {
    float alphaVal = alpha[(uint)(0u)];
    float xi = x[i];
    float pi = p[i];
    float ri = r[i];
    float qi = q[i];
    float alphaPi = (alphaVal * pi);
    float xNew = (xi + alphaPi);
    x_new[i] = xNew;
    float alphaQi = (alphaVal * qi);
    float rNew = (ri - alphaQi);
    r_new[i] = rNew;
  }
}
