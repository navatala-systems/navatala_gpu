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

#pragma OPENCL EXTENSION cl_khr_fp64 : enable
__kernel void navatala_ml_mutate_population_f64(__global const double* population, __global const ulong* rngState, __global const double* mutationRate, __global const double* mutationStrength, __global const uint* totalGenes, __global double* mutated, __global ulong* rngStateOut) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint totalGenesVal = totalGenes[0];
  double mutRateVal = mutationRate[0];
  double mutStrengthVal = mutationStrength[0];
  bool inBounds = (gid < totalGenesVal);
  if (inBounds) {
    double geneVal = population[gid];
    ulong s = rngState[gid];
    ulong aU64 = ((ulong)((uint)(1103515245u)));
    ulong cU64 = ((ulong)((uint)(12345u)));
    ulong s1 = (((aU64 * s) + cU64) & ((ulong)((uint)(2147483647u))));
    double u1 = (((double)(s1)) / as_double(0x41e0000000000000ul));
    ulong s2 = (((aU64 * s1) + cU64) & ((ulong)((uint)(2147483647u))));
    double u2 = (((double)(s2)) / as_double(0x41e0000000000000ul));
    ulong s3 = (((aU64 * s2) + cU64) & ((ulong)((uint)(2147483647u))));
    double u3 = (((double)(s3)) / as_double(0x41e0000000000000ul));
    rngStateOut[gid] = s3;
    double u2Clamped = (((u2 < as_double(0x3e7ad7f29abcaf48ul))) ? (as_double(0x3e7ad7f29abcaf48ul)) : (u2));
    double logU2 = log(u2Clamped);
    double negTwoLogU2 = (as_double(0xc000000000000000ul) * logU2);
    double sqrtTerm = sqrt(negTwoLogU2);
    double angle = (as_double(0x401921fb54442d18ul) * u3);
    double cosTerm = cos(angle);
    double gaussian = (sqrtTerm * cosTerm);
    double perturbation = (gaussian * mutStrengthVal);
    bool shouldMutate = (u1 < mutRateVal);
    double mutatedVal = (geneVal + perturbation);
    double finalVal = ((shouldMutate) ? (mutatedVal) : (geneVal));
    mutated[gid] = finalVal;
  }
}
