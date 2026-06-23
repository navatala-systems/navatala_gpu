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

__kernel void navatala_sparse_block_gram_schmidt_f32(__global const float* X, __global const float* coeffs, __global const uint* n, __global const uint* j, __global const uint* k, __global float* Xj) {
  int gid0 = (int)get_global_id(0);
  int gid = (int)(get_global_id(0));
  int N = ((int)(n[0]));
  if (gid < N) {
    int jVal = ((int)(j[0]));
    float xij = X[((jVal * N) + gid)];
    float v = xij;
    int kVal = ((int)(k[0]));
    for (int ii = 0; ii < (int)(jVal); ++ii) {
      float dot = coeffs[ii];
      float q = X[((ii * N) + gid)];
      v = (v - (dot * q));
    }
    Xj[gid] = v;
  }
}
