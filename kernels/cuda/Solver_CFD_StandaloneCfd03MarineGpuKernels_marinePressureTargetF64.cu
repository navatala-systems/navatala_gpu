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

#include <cuda_runtime.h>
extern "C" __global__ void Solver_CFD_StandaloneCfd03MarineGpuKernels_marinePressureTargetF64(const double* pointX, const double* pointY, const double* pointZ, const double* liquidFraction, const double* datum, const unsigned int* nCells, double* densityOut, double* pRghOut, double* absolutePressureOut) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  int gid = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  int cellCount = ((int)(nCells[0]));
  if (gid < cellCount) {
    double alpha = liquidFraction[gid];
    double density = ((alpha * datum[10]) + ((__longlong_as_double(0x3ff0000000000000ull) - alpha) * datum[11]));
    double datumGravityHead = (((datum[7] * (datum[1] - datum[4])) + (datum[8] * (datum[2] - datum[5]))) + (datum[9] * (datum[3] - datum[6])));
    double pRgh = (datum[0] + (density * datumGravityHead));
    double pointGravityHead = (((datum[7] * (pointX[gid] - datum[1])) + (datum[8] * (pointY[gid] - datum[2]))) + (datum[9] * (pointZ[gid] - datum[3])));
    double absolutePressure = (pRgh + (density * pointGravityHead));
    densityOut[gid] = density;
    pRghOut[gid] = pRgh;
    absolutePressureOut[gid] = absolutePressure;
  }
}
