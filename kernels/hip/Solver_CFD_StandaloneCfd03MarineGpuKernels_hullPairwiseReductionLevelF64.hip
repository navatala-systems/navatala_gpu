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
extern "C" __global__ void Solver_CFD_StandaloneCfd03MarineGpuKernels_hullPairwiseReductionLevelF64(const double* pressureForceXIn, const double* pressureForceYIn, const double* pressureForceZIn, const double* viscousForceXIn, const double* viscousForceYIn, const double* viscousForceZIn, const double* pressureMomentXIn, const double* pressureMomentYIn, const double* pressureMomentZIn, const double* viscousMomentXIn, const double* viscousMomentYIn, const double* viscousMomentZIn, const unsigned int* activeCount, double* pressureForceXOut, double* pressureForceYOut, double* pressureForceZOut, double* viscousForceXOut, double* viscousForceYOut, double* viscousForceZOut, double* pressureMomentXOut, double* pressureMomentYOut, double* pressureMomentZOut, double* viscousMomentXOut, double* viscousMomentYOut, double* viscousMomentZOut) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int _active = activeCount[0];
  unsigned int activeHalf = (_active / 2u);
  unsigned int outputCount = (activeHalf + (_active % 2u));
  if (gid < outputCount) {
    unsigned int leftIndex = (gid * 2u);
    unsigned int rightIndex = (leftIndex + 1u);
    bool hasRight = (rightIndex < _active);
    double pressureForceXLeft = pressureForceXIn[leftIndex];
    double pressureForceYLeft = pressureForceYIn[leftIndex];
    double pressureForceZLeft = pressureForceZIn[leftIndex];
    double viscousForceXLeft = viscousForceXIn[leftIndex];
    double viscousForceYLeft = viscousForceYIn[leftIndex];
    double viscousForceZLeft = viscousForceZIn[leftIndex];
    double pressureMomentXLeft = pressureMomentXIn[leftIndex];
    double pressureMomentYLeft = pressureMomentYIn[leftIndex];
    double pressureMomentZLeft = pressureMomentZIn[leftIndex];
    double viscousMomentXLeft = viscousMomentXIn[leftIndex];
    double viscousMomentYLeft = viscousMomentYIn[leftIndex];
    double viscousMomentZLeft = viscousMomentZIn[leftIndex];
    pressureForceXOut[gid] = ((hasRight) ? ((pressureForceXLeft + pressureForceXIn[rightIndex])) : (pressureForceXLeft));
    pressureForceYOut[gid] = ((hasRight) ? ((pressureForceYLeft + pressureForceYIn[rightIndex])) : (pressureForceYLeft));
    pressureForceZOut[gid] = ((hasRight) ? ((pressureForceZLeft + pressureForceZIn[rightIndex])) : (pressureForceZLeft));
    viscousForceXOut[gid] = ((hasRight) ? ((viscousForceXLeft + viscousForceXIn[rightIndex])) : (viscousForceXLeft));
    viscousForceYOut[gid] = ((hasRight) ? ((viscousForceYLeft + viscousForceYIn[rightIndex])) : (viscousForceYLeft));
    viscousForceZOut[gid] = ((hasRight) ? ((viscousForceZLeft + viscousForceZIn[rightIndex])) : (viscousForceZLeft));
    pressureMomentXOut[gid] = ((hasRight) ? ((pressureMomentXLeft + pressureMomentXIn[rightIndex])) : (pressureMomentXLeft));
    pressureMomentYOut[gid] = ((hasRight) ? ((pressureMomentYLeft + pressureMomentYIn[rightIndex])) : (pressureMomentYLeft));
    pressureMomentZOut[gid] = ((hasRight) ? ((pressureMomentZLeft + pressureMomentZIn[rightIndex])) : (pressureMomentZLeft));
    viscousMomentXOut[gid] = ((hasRight) ? ((viscousMomentXLeft + viscousMomentXIn[rightIndex])) : (viscousMomentXLeft));
    viscousMomentYOut[gid] = ((hasRight) ? ((viscousMomentYLeft + viscousMomentYIn[rightIndex])) : (viscousMomentYLeft));
    viscousMomentZOut[gid] = ((hasRight) ? ((viscousMomentZLeft + viscousMomentZIn[rightIndex])) : (viscousMomentZLeft));
  }
}
