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

__kernel void navatala_cfd_attention_row_softmax(__global const float* scores, __global const int* numQ, __global const int* numK, __global float* weights) {
  int gid0 = (int)get_global_id(0);
  int qi = (int)(get_global_id(0));
  int nq = numQ[0];
  int nk = numK[0];
  if (qi < nq) {
    int rowBase = (qi * nk);
    float maxScore = as_float(0xf149f2cau);
    for (int j = 0; j < (int)(nk); ++j) {
      int idx = (rowBase + j);
      float s = scores[idx];
      maxScore = (((s > maxScore)) ? (s) : (maxScore));
    }
    float sumExp = as_float(0x00000000u);
    for (int j = 0; j < (int)(nk); ++j) {
      int idx = (rowBase + j);
      float s = scores[idx];
      float e = exp((s - maxScore));
      sumExp = (sumExp + e);
    }
    float denom = (sumExp + as_float(0x3089705fu));
    for (int j = 0; j < (int)(nk); ++j) {
      int idx = (rowBase + j);
      float s = scores[idx];
      float e = exp((s - maxScore));
      float w = (e / denom);
      weights[idx] = w;
    }
  }
}
