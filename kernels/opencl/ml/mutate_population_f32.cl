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

__kernel void navatala_ml_mutate_population_f32(__global const float* population, __global const ulong* rngState, __global const float* mutationRate, __global const float* mutationStrength, __global const uint* totalGenes, __global float* mutated, __global ulong* rngStateOut) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint totalGenesVal = totalGenes[0];
  float mutRateVal = mutationRate[0];
  float mutStrengthVal = mutationStrength[0];
  bool inBounds = (gid < totalGenesVal);
  if (inBounds) {
    float geneVal = population[gid];
    ulong s = rngState[gid];
    ulong aU64 = ((ulong)((uint)(1103515245u)));
    ulong cU64 = ((ulong)((uint)(12345u)));
    ulong s1 = (((aU64 * s) + cU64) & ((ulong)((uint)(2147483647u))));
    float u1 = (((float)(s1)) / as_float(0x4f000000u));
    ulong s2 = (((aU64 * s1) + cU64) & ((ulong)((uint)(2147483647u))));
    float u2 = (((float)(s2)) / as_float(0x4f000000u));
    ulong s3 = (((aU64 * s2) + cU64) & ((ulong)((uint)(2147483647u))));
    float u3 = (((float)(s3)) / as_float(0x4f000000u));
    rngStateOut[gid] = s3;
    float u2Clamped = (((u2 < as_float(0x33d6bf95u))) ? (as_float(0x33d6bf95u)) : (u2));
    float logU2 = log(u2Clamped);
    float negTwoLogU2 = (as_float(0xc0000000u) * logU2);
    float sqrtTerm = sqrt(negTwoLogU2);
    float angle = (as_float(0x40c90fdbu) * u3);
    float cosTerm = cos(angle);
    float gaussian = (sqrtTerm * cosTerm);
    float perturbation = (gaussian * mutStrengthVal);
    bool shouldMutate = (u1 < mutRateVal);
    float mutatedVal = (geneVal + perturbation);
    float finalVal = ((shouldMutate) ? (mutatedVal) : (geneVal));
    mutated[gid] = finalVal;
  }
}
