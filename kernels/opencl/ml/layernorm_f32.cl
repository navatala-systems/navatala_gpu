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

__kernel void navatala_ml_layernorm_f32(__global const float* x, __global const float* gamma, __global const float* beta, __global const uint* count, __global const float* eps, __global float* _output) {
  int gid0 = (int)get_global_id(0);
  uint lid = ((uint)((int)(get_local_id(0))));
  __local float sdata[256];
  uint countVal = count[0];
  float n = ((float)(countVal));
  uint numIters = ((countVal + (uint)(255u)) / (uint)(256u));
  float epsVal = eps[0];
  float gsM = as_float(0x00000000u);
  for (int itM = 0; itM < (int)(numIters); ++itM) {
    if (((lid + (((uint)(itM)) * (uint)(256u))) < countVal)) {
      gsM = (gsM + x[(lid + (((uint)(itM)) * (uint)(256u)))]);
    }
  }
  sdata[lid] = gsM;
  barrier(CLK_LOCAL_MEM_FENCE);
  uint meanStr = (uint)(128u);
  for (int redStep = 0; redStep < (int)(8); ++redStep) {
    uint stride = meanStr;
    if ((lid < stride)) {
      float other = sdata[(lid + stride)];
      float mine = sdata[lid];
      float acc = (mine + other);
      sdata[lid] = acc;
    }
    uint strideToHalve = meanStr;
    uint nextStride = (strideToHalve >> (uint)(1u));
    meanStr = nextStride;
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  float mean = (sdata[0] / n);
  barrier(CLK_LOCAL_MEM_FENCE);
  float gsV = as_float(0x00000000u);
  for (int itV = 0; itV < (int)(numIters); ++itV) {
    if (((lid + (((uint)(itV)) * (uint)(256u))) < countVal)) {
      gsV = (gsV + ((x[(lid + (((uint)(itV)) * (uint)(256u)))] - mean) * (x[(lid + (((uint)(itV)) * (uint)(256u)))] - mean)));
    }
  }
  sdata[lid] = gsV;
  barrier(CLK_LOCAL_MEM_FENCE);
  uint varStr = (uint)(128u);
  for (int redStep = 0; redStep < (int)(8); ++redStep) {
    uint stride = varStr;
    if ((lid < stride)) {
      float other = sdata[(lid + stride)];
      float mine = sdata[lid];
      float acc = (mine + other);
      sdata[lid] = acc;
    }
    uint strideToHalve = varStr;
    uint nextStride = (strideToHalve >> (uint)(1u));
    varStr = nextStride;
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  float var = (sdata[0] / n);
  float denom = sqrt((var + epsVal));
  for (int itW = 0; itW < (int)(numIters); ++itW) {
    uint idxW = (lid + (((uint)(itW)) * (uint)(256u)));
    if ((idxW < countVal)) {
      float xnW = ((x[idxW] - mean) / denom);
      float outF = ((gamma[idxW] * xnW) + beta[idxW]);
      _output[idxW] = outF;
    }
  }
}
