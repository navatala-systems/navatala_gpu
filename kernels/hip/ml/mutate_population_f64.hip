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
extern "C" __global__ void navatala_ml_mutate_population_f64(const double* population, const unsigned long long* rngState, const double* mutationRate, const double* mutationStrength, const unsigned int* totalGenes, double* mutated, unsigned long long* rngStateOut) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int totalGenesVal = totalGenes[0];
  double mutRateVal = mutationRate[0];
  double mutStrengthVal = mutationStrength[0];
  bool inBounds = (gid < totalGenesVal);
  if (inBounds) {
    double geneVal = population[gid];
    unsigned long long s = rngState[gid];
    unsigned long long aU64 = ((unsigned long long)(1103515245u));
    unsigned long long cU64 = ((unsigned long long)(12345u));
    unsigned long long s1 = (((aU64 * s) + cU64) & ((unsigned long long)(2147483647u)));
    double u1 = (((double)(s1)) / __longlong_as_double(0x41e0000000000000ull));
    unsigned long long s2 = (((aU64 * s1) + cU64) & ((unsigned long long)(2147483647u)));
    double u2 = (((double)(s2)) / __longlong_as_double(0x41e0000000000000ull));
    unsigned long long s3 = (((aU64 * s2) + cU64) & ((unsigned long long)(2147483647u)));
    double u3 = (((double)(s3)) / __longlong_as_double(0x41e0000000000000ull));
    rngStateOut[gid] = s3;
    double u2Clamped = (((u2 < __longlong_as_double(0x3e7ad7f29abcaf48ull))) ? (__longlong_as_double(0x3e7ad7f29abcaf48ull)) : (u2));
    double logU2 = log(u2Clamped);
    double negTwoLogU2 = (__longlong_as_double(0xc000000000000000ull) * logU2);
    double sqrtTerm = sqrt(negTwoLogU2);
    double angle = (__longlong_as_double(0x401921fb54442d18ull) * u3);
    double cosTerm = cos(angle);
    double gaussian = (sqrtTerm * cosTerm);
    double perturbation = (gaussian * mutStrengthVal);
    bool shouldMutate = (u1 < mutRateVal);
    double mutatedVal = (geneVal + perturbation);
    double finalVal = ((shouldMutate) ? (mutatedVal) : (geneVal));
    mutated[gid] = finalVal;
  }
}
