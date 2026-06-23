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

__kernel void navatala_ml_instancenorm_f32(__global const float* x, __global const float* gamma, __global const float* beta, __global const uint* C, __global const uint* HW, __global const float* eps, __global float* _output) {
  int gid0 = (int)get_global_id(0);
  uint grp = ((uint)((int)(get_group_id(0))));
  uint lid = ((uint)((int)(get_local_id(0))));
  __local float sdata[256];
  uint Cv = C[0];
  uint HWv = HW[0];
  uint chan = (grp % Cv);
  uint base = (grp * HWv);
  uint countVal = HWv;
  float n = ((float)(countVal));
  uint numIters = ((countVal + (uint)(255u)) / (uint)(256u));
  float epsVal = eps[0];
  float gsM = as_float(0x00000000u);
  for (int itM = 0; itM < (int)(numIters); ++itM) {
    if ((lid + (((uint)(itM)) * (uint)(256u))) < countVal) {
      gsM = (gsM + x[(base + (lid + (((uint)(itM)) * (uint)(256u))))]);
    }
  }
  sdata[lid] = gsM;
  barrier(CLK_LOCAL_MEM_FENCE);
  uint meanStr = (uint)(128u);
  for (int redStep = 0; redStep < (int)(8); ++redStep) {
    uint stride = meanStr;
    if (lid < stride) {
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
    if ((lid + (((uint)(itV)) * (uint)(256u))) < countVal) {
      gsV = (gsV + ((x[(base + (lid + (((uint)(itV)) * (uint)(256u))))] - mean) * (x[(base + (lid + (((uint)(itV)) * (uint)(256u))))] - mean)));
    }
  }
  sdata[lid] = gsV;
  barrier(CLK_LOCAL_MEM_FENCE);
  uint varStr = (uint)(128u);
  for (int redStep = 0; redStep < (int)(8); ++redStep) {
    uint stride = varStr;
    if (lid < stride) {
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
    if ((lid + (((uint)(itW)) * (uint)(256u))) < countVal) {
      float gv = gamma[chan];
      float bv = beta[chan];
      float xv = x[(base + (lid + (((uint)(itW)) * (uint)(256u))))];
      float xn = ((xv - mean) / denom);
      _output[(base + (lid + (((uint)(itW)) * (uint)(256u))))] = ((gv * xn) + bv);
    }
  }
}
