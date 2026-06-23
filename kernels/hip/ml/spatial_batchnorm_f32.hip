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
extern "C" __global__ void navatala_ml_spatial_batchnorm_f32(const float* x, const float* gamma, const float* beta, const unsigned int* N, const unsigned int* C, const unsigned int* HW, const float* eps, float* _output) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int grp = ((unsigned int)((int)(blockIdx.x)));
  unsigned int lid = ((unsigned int)((int)(threadIdx.x)));
  __shared__ float sdata[256];
  unsigned int Nv = N[0];
  unsigned int Cv = C[0];
  unsigned int HWv = HW[0];
  unsigned int CHW = (Cv * HWv);
  unsigned int chBase = (grp * HWv);
  unsigned int countVal = (Nv * HWv);
  float n = ((float)(countVal));
  unsigned int numIters = ((countVal + 255u) / 256u);
  float epsVal = eps[0];
  float gsM = __uint_as_float(0x00000000u);
  for (int itM = 0; itM < (int)(numIters); ++itM) {
    if ((lid + (((unsigned int)(itM)) * 256u)) < countVal) {
      gsM = (gsM + x[(((((lid + (((unsigned int)(itM)) * 256u)) / HWv) * CHW) + chBase) + ((lid + (((unsigned int)(itM)) * 256u)) % HWv))]);
    }
  }
  sdata[lid] = gsM;
  __syncthreads();
  unsigned int meanStr = 128u;
  for (int redStep = 0; redStep < (int)(8); ++redStep) {
    unsigned int stride = meanStr;
    if (lid < stride) {
      float other = sdata[(lid + stride)];
      float mine = sdata[lid];
      float acc = (mine + other);
      sdata[lid] = acc;
    }
    unsigned int strideToHalve = meanStr;
    unsigned int nextStride = (strideToHalve >> 1u);
    meanStr = nextStride;
    __syncthreads();
  }
  float mean = (sdata[0] / n);
  __syncthreads();
  float gsV = __uint_as_float(0x00000000u);
  for (int itV = 0; itV < (int)(numIters); ++itV) {
    if ((lid + (((unsigned int)(itV)) * 256u)) < countVal) {
      gsV = (gsV + ((x[(((((lid + (((unsigned int)(itV)) * 256u)) / HWv) * CHW) + chBase) + ((lid + (((unsigned int)(itV)) * 256u)) % HWv))] - mean) * (x[(((((lid + (((unsigned int)(itV)) * 256u)) / HWv) * CHW) + chBase) + ((lid + (((unsigned int)(itV)) * 256u)) % HWv))] - mean)));
    }
  }
  sdata[lid] = gsV;
  __syncthreads();
  unsigned int varStr = 128u;
  for (int redStep = 0; redStep < (int)(8); ++redStep) {
    unsigned int stride = varStr;
    if (lid < stride) {
      float other = sdata[(lid + stride)];
      float mine = sdata[lid];
      float acc = (mine + other);
      sdata[lid] = acc;
    }
    unsigned int strideToHalve = varStr;
    unsigned int nextStride = (strideToHalve >> 1u);
    varStr = nextStride;
    __syncthreads();
  }
  float var = (sdata[0] / n);
  float denom = sqrt((var + epsVal));
  for (int itW = 0; itW < (int)(numIters); ++itW) {
    if ((lid + (((unsigned int)(itW)) * 256u)) < countVal) {
      float gv = gamma[grp];
      float bv = beta[grp];
      float xv = x[(((((lid + (((unsigned int)(itW)) * 256u)) / HWv) * CHW) + chBase) + ((lid + (((unsigned int)(itW)) * 256u)) % HWv))];
      float xn = ((xv - mean) / denom);
      _output[(((((lid + (((unsigned int)(itW)) * 256u)) / HWv) * CHW) + chBase) + ((lid + (((unsigned int)(itW)) * 256u)) % HWv))] = ((gv * xn) + bv);
    }
  }
}
