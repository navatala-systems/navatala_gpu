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

#include <hip/hip_runtime.h>
extern "C" __global__ void navatala_ml_layernorm_backward_f32(const float* x, const float* dy, const float* gamma, const unsigned int* count, const float* eps, float* dx, float* dgamma, float* dbeta) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int lid = ((unsigned int)((int)(threadIdx.x)));
  __shared__ float sdata[256];
  unsigned int countVal = count[0];
  float n = ((float)(countVal));
  unsigned int numIters = ((countVal + 255u) / 256u);
  float epsVal = eps[0];
  float gM = __uint_as_float(0x00000000u);
  for (int itm = 0; itm < (int)(numIters); ++itm) {
    if ((lid + (((unsigned int)(itm)) * 256u)) < countVal) {
      gM = (gM + x[(lid + (((unsigned int)(itm)) * 256u))]);
    }
  }
  sdata[lid] = gM;
  __syncthreads();
  unsigned int mStr = 128u;
  for (int redStep = 0; redStep < (int)(8); ++redStep) {
    unsigned int stride = mStr;
    if (lid < stride) {
      float other = sdata[(lid + stride)];
      float mine = sdata[lid];
      float acc = (mine + other);
      sdata[lid] = acc;
    }
    unsigned int strideToHalve = mStr;
    unsigned int nextStride = (strideToHalve >> 1u);
    mStr = nextStride;
    __syncthreads();
  }
  float mean = (sdata[0] / n);
  __syncthreads();
  float gV = __uint_as_float(0x00000000u);
  for (int itv = 0; itv < (int)(numIters); ++itv) {
    if ((lid + (((unsigned int)(itv)) * 256u)) < countVal) {
      gV = (gV + ((x[(lid + (((unsigned int)(itv)) * 256u))] - mean) * (x[(lid + (((unsigned int)(itv)) * 256u))] - mean)));
    }
  }
  sdata[lid] = gV;
  __syncthreads();
  unsigned int vStr = 128u;
  for (int redStep = 0; redStep < (int)(8); ++redStep) {
    unsigned int stride = vStr;
    if (lid < stride) {
      float other = sdata[(lid + stride)];
      float mine = sdata[lid];
      float acc = (mine + other);
      sdata[lid] = acc;
    }
    unsigned int strideToHalve = vStr;
    unsigned int nextStride = (strideToHalve >> 1u);
    vStr = nextStride;
    __syncthreads();
  }
  float var = (sdata[0] / n);
  float std = sqrt((var + epsVal));
  __syncthreads();
  float g1 = __uint_as_float(0x00000000u);
  for (int it1 = 0; it1 < (int)(numIters); ++it1) {
    if ((lid + (((unsigned int)(it1)) * 256u)) < countVal) {
      g1 = (g1 + (dy[(lid + (((unsigned int)(it1)) * 256u))] * gamma[(lid + (((unsigned int)(it1)) * 256u))]));
    }
  }
  sdata[lid] = g1;
  __syncthreads();
  unsigned int s1Str = 128u;
  for (int redStep = 0; redStep < (int)(8); ++redStep) {
    unsigned int stride = s1Str;
    if (lid < stride) {
      float other = sdata[(lid + stride)];
      float mine = sdata[lid];
      float acc = (mine + other);
      sdata[lid] = acc;
    }
    unsigned int strideToHalve = s1Str;
    unsigned int nextStride = (strideToHalve >> 1u);
    s1Str = nextStride;
    __syncthreads();
  }
  float m1 = (sdata[0] / n);
  __syncthreads();
  float g2 = __uint_as_float(0x00000000u);
  for (int it2 = 0; it2 < (int)(numIters); ++it2) {
    if ((lid + (((unsigned int)(it2)) * 256u)) < countVal) {
      g2 = (g2 + ((dy[(lid + (((unsigned int)(it2)) * 256u))] * gamma[(lid + (((unsigned int)(it2)) * 256u))]) * ((x[(lid + (((unsigned int)(it2)) * 256u))] - mean) / std)));
    }
  }
  sdata[lid] = g2;
  __syncthreads();
  unsigned int s2Str = 128u;
  for (int redStep = 0; redStep < (int)(8); ++redStep) {
    unsigned int stride = s2Str;
    if (lid < stride) {
      float other = sdata[(lid + stride)];
      float mine = sdata[lid];
      float acc = (mine + other);
      sdata[lid] = acc;
    }
    unsigned int strideToHalve = s2Str;
    unsigned int nextStride = (strideToHalve >> 1u);
    s2Str = nextStride;
    __syncthreads();
  }
  float m2 = (sdata[0] / n);
  for (int itw = 0; itw < (int)(numIters); ++itw) {
    unsigned int idw = (lid + (((unsigned int)(itw)) * 256u));
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
