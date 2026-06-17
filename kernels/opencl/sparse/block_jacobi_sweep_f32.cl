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

__kernel void navatala_sparse_block_jacobi_sweep_f32(__global const uint* rowPtr, __global const uint* colIdx, __global const float* values, __global const float* diagBlocks, __global const float* b, __global const float* x, __global const float* omega, __global const uint* nBlockRows, __global const uint* blockDim, __global float* xNew) {
  int gid0 = (int)get_global_id(0);
  int blockRow = (int)(get_global_id(0));
  int N = ((int)(nBlockRows[0]));
  if ((blockRow < N)) {
    int bdm = ((int)(blockDim[0]));
    int bdm2 = (bdm * bdm);
    int rs = ((int)(rowPtr[blockRow]));
    int re = ((int)(rowPtr[(blockRow + 1)]));
    float om = omega[0];
    if ((bdm == 4)) {
      int rowBase4 = (blockRow * 4);
      float offDiag0 = as_float(0x00000000u);
      float offDiag1 = as_float(0x00000000u);
      float offDiag2 = as_float(0x00000000u);
      float offDiag3 = as_float(0x00000000u);
      for (int jj = 0; jj < (int)((re - rs)); ++jj) {
        int k = (rs + jj);
        int col = ((int)(colIdx[k]));
        if ((col != blockRow)) {
          int kBase = (k * 16);
          int colBase4 = (col * 4);
          float xv0 = x[colBase4];
          float xv1 = x[(colBase4 + 1)];
          float xv2 = x[(colBase4 + 2)];
          float xv3 = x[(colBase4 + 3)];
          offDiag0 = (offDiag0 + (((values[kBase] * xv0) + (values[(kBase + 1)] * xv1)) + ((values[(kBase + 2)] * xv2) + (values[(kBase + 3)] * xv3))));
          offDiag1 = (offDiag1 + (((values[(kBase + 4)] * xv0) + (values[(kBase + 5)] * xv1)) + ((values[(kBase + 6)] * xv2) + (values[(kBase + 7)] * xv3))));
          offDiag2 = (offDiag2 + (((values[(kBase + 8)] * xv0) + (values[(kBase + 9)] * xv1)) + ((values[(kBase + 10)] * xv2) + (values[(kBase + 11)] * xv3))));
          offDiag3 = (offDiag3 + (((values[(kBase + 12)] * xv0) + (values[(kBase + 13)] * xv1)) + ((values[(kBase + 14)] * xv2) + (values[(kBase + 15)] * xv3))));
        }
      }
      float bMinus0 = (b[rowBase4] - offDiag0);
      float bMinus1 = (b[(rowBase4 + 1)] - offDiag1);
      float bMinus2 = (b[(rowBase4 + 2)] - offDiag2);
      float bMinus3 = (b[(rowBase4 + 3)] - offDiag3);
      int dBase = (blockRow * 16);
      float xr0 = x[rowBase4];
      float xr1 = x[(rowBase4 + 1)];
      float xr2 = x[(rowBase4 + 2)];
      float xr3 = x[(rowBase4 + 3)];
      float oneMinusOm = (as_float(0x3f800000u) - om);
      xNew[rowBase4] = ((oneMinusOm * xr0) + (om * (((diagBlocks[dBase] * bMinus0) + (diagBlocks[(dBase + 1)] * bMinus1)) + ((diagBlocks[(dBase + 2)] * bMinus2) + (diagBlocks[(dBase + 3)] * bMinus3)))));
      xNew[(rowBase4 + 1)] = ((oneMinusOm * xr1) + (om * (((diagBlocks[(dBase + 4)] * bMinus0) + (diagBlocks[(dBase + 5)] * bMinus1)) + ((diagBlocks[(dBase + 6)] * bMinus2) + (diagBlocks[(dBase + 7)] * bMinus3)))));
      xNew[(rowBase4 + 2)] = ((oneMinusOm * xr2) + (om * (((diagBlocks[(dBase + 8)] * bMinus0) + (diagBlocks[(dBase + 9)] * bMinus1)) + ((diagBlocks[(dBase + 10)] * bMinus2) + (diagBlocks[(dBase + 11)] * bMinus3)))));
      xNew[(rowBase4 + 3)] = ((oneMinusOm * xr3) + (om * (((diagBlocks[(dBase + 12)] * bMinus0) + (diagBlocks[(dBase + 13)] * bMinus1)) + ((diagBlocks[(dBase + 14)] * bMinus2) + (diagBlocks[(dBase + 15)] * bMinus3)))));
    } else {
      for (int eq = 0; eq < (int)(bdm); ++eq) {
        int grow = ((blockRow * bdm) + eq);
        float offDiag = as_float(0x00000000u);
        for (int jj = 0; jj < (int)((re - rs)); ++jj) {
          int k = (rs + jj);
          int col = ((int)(colIdx[k]));
          if ((col != blockRow)) {
            int kBase = (k * bdm2);
            int rowBase = (kBase + (eq * bdm));
            for (int eq2 = 0; eq2 < (int)(bdm); ++eq2) {
              int grow2 = ((col * bdm) + eq2);
              float aval = values[(rowBase + eq2)];
              float xval = x[grow2];
              offDiag = (offDiag + (aval * xval));
            }
          }
        }
        float rhs = (b[grow] - offDiag);
        int invIdx = (((blockRow * bdm2) + (eq * bdm)) + eq);
        float invVal = diagBlocks[invIdx];
        float xr = x[grow];
        xNew[grow] = (((as_float(0x3f800000u) - om) * xr) + (om * (invVal * rhs)));
      }
    }
  }
}
