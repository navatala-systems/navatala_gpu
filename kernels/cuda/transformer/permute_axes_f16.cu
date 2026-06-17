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

#include <cuda_runtime.h>
#include <cuda_fp16.h>
extern "C" __global__ void navatala_transformer_permute_axes_f16(const __half* _input, const unsigned int* dim0, const unsigned int* dim1, const unsigned int* dim2, const unsigned int* dim3, const unsigned int* perm0, const unsigned int* perm1, const unsigned int* perm2, const unsigned int* perm3, __half* _output) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int d0 = dim0[0u];
  unsigned int d1 = dim1[0u];
  unsigned int d2 = dim2[0u];
  unsigned int d3 = dim3[0u];
  unsigned int p0 = perm0[0u];
  unsigned int p1 = perm1[0u];
  unsigned int p2 = perm2[0u];
  unsigned int p3 = perm3[0u];
  unsigned int total = (d0 * (d1 * (d2 * d3)));
  bool inBounds = (gid < total);
  if (inBounds) {
    unsigned int stride0 = (d1 * (d2 * d3));
    unsigned int stride1 = (d2 * d3);
    unsigned int stride2 = d3;
    unsigned int i0 = (gid / stride0);
    unsigned int rem0 = (gid % stride0);
    unsigned int i1 = (rem0 / stride1);
    unsigned int rem1 = (rem0 % stride1);
    unsigned int i2 = (rem1 / stride2);
    unsigned int i3 = (rem1 % stride2);
    __half val = _input[gid];
    unsigned int od0 = (((p0 == 0u)) ? (d0) : ((((p0 == 1u)) ? (d1) : ((((p0 == 2u)) ? (d2) : (d3))))));
    unsigned int od1 = (((p1 == 0u)) ? (d0) : ((((p1 == 1u)) ? (d1) : ((((p1 == 2u)) ? (d2) : (d3))))));
    unsigned int od2 = (((p2 == 0u)) ? (d0) : ((((p2 == 1u)) ? (d1) : ((((p2 == 2u)) ? (d2) : (d3))))));
    unsigned int od3 = (((p3 == 0u)) ? (d0) : ((((p3 == 1u)) ? (d1) : ((((p3 == 2u)) ? (d2) : (d3))))));
    unsigned int oi0 = (((p0 == 0u)) ? (i0) : ((((p0 == 1u)) ? (i1) : ((((p0 == 2u)) ? (i2) : (i3))))));
    unsigned int oi1 = (((p1 == 0u)) ? (i0) : ((((p1 == 1u)) ? (i1) : ((((p1 == 2u)) ? (i2) : (i3))))));
    unsigned int oi2 = (((p2 == 0u)) ? (i0) : ((((p2 == 1u)) ? (i1) : ((((p2 == 2u)) ? (i2) : (i3))))));
    unsigned int oi3 = (((p3 == 0u)) ? (i0) : ((((p3 == 1u)) ? (i1) : ((((p3 == 2u)) ? (i2) : (i3))))));
    unsigned int ostride0 = (od1 * (od2 * od3));
    unsigned int ostride1 = (od2 * od3);
    unsigned int ostride2 = od3;
    unsigned int outIdx = ((oi0 * ostride0) + ((oi1 * ostride1) + ((oi2 * ostride2) + oi3)));
    _output[outIdx] = val;
  }
}
