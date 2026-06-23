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

__kernel void navatala_ml_layernorm_backward_f32(__global const float* x, __global const float* dy, __global const float* gamma, __global const uint* count, __global const float* eps, __global float* dx, __global float* dgamma, __global float* dbeta) {
  int gid0 = (int)get_global_id(0);
  uint lid = ((uint)((int)(get_local_id(0))));
  __local float sdata[256];
  uint countVal = count[0];
  float n = ((float)(countVal));
  uint numIters = ((countVal + (uint)(255u)) / (uint)(256u));
  float epsVal = eps[0];
  float gM = as_float(0x00000000u);
  for (int itm = 0; itm < (int)(numIters); ++itm) {
    if ((lid + (((uint)(itm)) * (uint)(256u))) < countVal) {
      gM = (gM + x[(lid + (((uint)(itm)) * (uint)(256u)))]);
    }
  }
  sdata[lid] = gM;
  barrier(CLK_LOCAL_MEM_FENCE);
  uint mStr = (uint)(128u);
  for (int redStep = 0; redStep < (int)(8); ++redStep) {
    uint stride = mStr;
    if (lid < stride) {
      float other = sdata[(lid + stride)];
      float mine = sdata[lid];
      float acc = (mine + other);
      sdata[lid] = acc;
    }
    uint strideToHalve = mStr;
    uint nextStride = (strideToHalve >> (uint)(1u));
    mStr = nextStride;
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  float mean = (sdata[0] / n);
  barrier(CLK_LOCAL_MEM_FENCE);
  float gV = as_float(0x00000000u);
  for (int itv = 0; itv < (int)(numIters); ++itv) {
    if ((lid + (((uint)(itv)) * (uint)(256u))) < countVal) {
      gV = (gV + ((x[(lid + (((uint)(itv)) * (uint)(256u)))] - mean) * (x[(lid + (((uint)(itv)) * (uint)(256u)))] - mean)));
    }
  }
  sdata[lid] = gV;
  barrier(CLK_LOCAL_MEM_FENCE);
  uint vStr = (uint)(128u);
  for (int redStep = 0; redStep < (int)(8); ++redStep) {
    uint stride = vStr;
    if (lid < stride) {
      float other = sdata[(lid + stride)];
      float mine = sdata[lid];
      float acc = (mine + other);
      sdata[lid] = acc;
    }
    uint strideToHalve = vStr;
    uint nextStride = (strideToHalve >> (uint)(1u));
    vStr = nextStride;
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  float var = (sdata[0] / n);
  float std = sqrt((var + epsVal));
  barrier(CLK_LOCAL_MEM_FENCE);
  float g1 = as_float(0x00000000u);
  for (int it1 = 0; it1 < (int)(numIters); ++it1) {
    if ((lid + (((uint)(it1)) * (uint)(256u))) < countVal) {
      g1 = (g1 + (dy[(lid + (((uint)(it1)) * (uint)(256u)))] * gamma[(lid + (((uint)(it1)) * (uint)(256u)))]));
    }
  }
  sdata[lid] = g1;
  barrier(CLK_LOCAL_MEM_FENCE);
  uint s1Str = (uint)(128u);
  for (int redStep = 0; redStep < (int)(8); ++redStep) {
    uint stride = s1Str;
    if (lid < stride) {
      float other = sdata[(lid + stride)];
      float mine = sdata[lid];
      float acc = (mine + other);
      sdata[lid] = acc;
    }
    uint strideToHalve = s1Str;
    uint nextStride = (strideToHalve >> (uint)(1u));
    s1Str = nextStride;
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  float m1 = (sdata[0] / n);
  barrier(CLK_LOCAL_MEM_FENCE);
  float g2 = as_float(0x00000000u);
  for (int it2 = 0; it2 < (int)(numIters); ++it2) {
    if ((lid + (((uint)(it2)) * (uint)(256u))) < countVal) {
      g2 = (g2 + ((dy[(lid + (((uint)(it2)) * (uint)(256u)))] * gamma[(lid + (((uint)(it2)) * (uint)(256u)))]) * ((x[(lid + (((uint)(it2)) * (uint)(256u)))] - mean) / std)));
    }
  }
  sdata[lid] = g2;
  barrier(CLK_LOCAL_MEM_FENCE);
  uint s2Str = (uint)(128u);
  for (int redStep = 0; redStep < (int)(8); ++redStep) {
    uint stride = s2Str;
    if (lid < stride) {
      float other = sdata[(lid + stride)];
      float mine = sdata[lid];
      float acc = (mine + other);
      sdata[lid] = acc;
    }
    uint strideToHalve = s2Str;
    uint nextStride = (strideToHalve >> (uint)(1u));
    s2Str = nextStride;
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  float m2 = (sdata[0] / n);
  for (int itw = 0; itw < (int)(numIters); ++itw) {
    uint idw = (lid + (((uint)(itw)) * (uint)(256u)));
    if (idw < countVal) {
      float xhatw = ((x[idw] - mean) / std);
      float dyvw = dy[idw];
      float dxhw = (dyvw * gamma[idw]);
      float dxFw = (((dxhw - m1) - (xhatw * m2)) / std);
      dx[idw] = dxFw;
      dgamma[idw] = (dyvw * xhatw);
      dbeta[idw] = dyvw;
    }
  }
}
