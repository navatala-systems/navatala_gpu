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

__kernel void navatala_sparse_dense_sym_eig_f32(__global const float* M, __global const uint* k, __global float* eigenvalues, __global float* eigenvectors) {
  int gid0 = (int)get_global_id(0);
  int kVal = ((int)(k[0]));
  for (int i = 0; i < (int)(kVal); ++i) {
    for (int j = 0; j < (int)(kVal); ++j) {
      float val = M[((i * kVal) + j)];
      float isOne = (((i == j)) ? (as_float(0x3f800000u)) : (as_float(0x00000000u)));
      eigenvalues[i] = M[((i * kVal) + i)];
      eigenvectors[((i * kVal) + j)] = isOne;
    }
  }
}
