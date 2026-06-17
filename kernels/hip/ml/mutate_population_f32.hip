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
extern "C" __global__ void navatala_ml_mutate_population_f32(const float* population, const unsigned long long* rngState, const float* mutationRate, const float* mutationStrength, const unsigned int* totalGenes, float* mutated, unsigned long long* rngStateOut) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int totalGenesVal = totalGenes[0];
  float mutRateVal = mutationRate[0];
  float mutStrengthVal = mutationStrength[0];
  bool inBounds = (gid < totalGenesVal);
  if (inBounds) {
    float geneVal = population[gid];
    unsigned long long s = rngState[gid];
    unsigned long long aU64 = ((unsigned long long)(1103515245u));
    unsigned long long cU64 = ((unsigned long long)(12345u));
    unsigned long long s1 = (((aU64 * s) + cU64) & ((unsigned long long)(2147483647u)));
    float u1 = (((float)(s1)) / __uint_as_float(0x4f000000u));
    unsigned long long s2 = (((aU64 * s1) + cU64) & ((unsigned long long)(2147483647u)));
    float u2 = (((float)(s2)) / __uint_as_float(0x4f000000u));
    unsigned long long s3 = (((aU64 * s2) + cU64) & ((unsigned long long)(2147483647u)));
    float u3 = (((float)(s3)) / __uint_as_float(0x4f000000u));
    rngStateOut[gid] = s3;
    float u2Clamped = (((u2 < __uint_as_float(0x33d6bf95u))) ? (__uint_as_float(0x33d6bf95u)) : (u2));
    float logU2 = log(u2Clamped);
    float negTwoLogU2 = (__uint_as_float(0xc0000000u) * logU2);
    float sqrtTerm = sqrt(negTwoLogU2);
    float angle = (__uint_as_float(0x40c90fdbu) * u3);
    float cosTerm = cos(angle);
    float gaussian = (sqrtTerm * cosTerm);
    float perturbation = (gaussian * mutStrengthVal);
    bool shouldMutate = (u1 < mutRateVal);
    float mutatedVal = (geneVal + perturbation);
    float finalVal = ((shouldMutate) ? (mutatedVal) : (geneVal));
    mutated[gid] = finalVal;
  }
}
