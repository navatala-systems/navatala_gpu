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
extern "C" __global__ void Solver_CFD_StandaloneCfd03MarineGpuKernels_marineMomentumDampingF64(const double* density, const double* volume, const double* streamwiseCoordinate, const double* acceptedVelocityX, const double* acceptedVelocityY, const double* acceptedVelocityZ, const double* dampingDatum, const unsigned int* nCells, double* lambdaOut, double* implicitDiagonalOut, double* implicitTargetRhsXOut, double* implicitTargetRhsYOut, double* implicitTargetRhsZOut, double* momentumSourceXOut, double* momentumSourceYOut, double* momentumSourceZOut, double* signedPowerOut, double* signedWorkOut, unsigned int* activeOut) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  int gid = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  int cellCount = ((int)(nCells[0]));
  if (gid < cellCount) {
    double rawXi = ((streamwiseCoordinate[gid] - dampingDatum[1]) / (dampingDatum[2] - dampingDatum[1]));
    double xi = (((rawXi < __longlong_as_double(0x0000000000000000ull))) ? (__longlong_as_double(0x0000000000000000ull)) : ((((__longlong_as_double(0x3ff0000000000000ull) < rawXi)) ? (__longlong_as_double(0x3ff0000000000000ull)) : (rawXi))));
    double ramp = ((xi * xi) * (__longlong_as_double(0x4008000000000000ull) - (__longlong_as_double(0x4000000000000000ull) * xi)));
    double lambda = (dampingDatum[0] * ramp);
    double scale = ((density[gid] * lambda) * volume[gid]);
    double targetRhsX = (scale * dampingDatum[3]);
    double targetRhsY = (scale * dampingDatum[4]);
    double targetRhsZ = (scale * dampingDatum[5]);
    double sourceX = (targetRhsX - (scale * acceptedVelocityX[gid]));
    double sourceY = (targetRhsY - (scale * acceptedVelocityY[gid]));
    double sourceZ = (targetRhsZ - (scale * acceptedVelocityZ[gid]));
    double signedPower = (((acceptedVelocityX[gid] * sourceX) + (acceptedVelocityY[gid] * sourceY)) + (acceptedVelocityZ[gid] * sourceZ));
    double signedWork = (dampingDatum[6] * signedPower);
    lambdaOut[gid] = lambda;
    implicitDiagonalOut[gid] = scale;
    implicitTargetRhsXOut[gid] = targetRhsX;
    implicitTargetRhsYOut[gid] = targetRhsY;
    implicitTargetRhsZOut[gid] = targetRhsZ;
    momentumSourceXOut[gid] = sourceX;
    momentumSourceYOut[gid] = sourceY;
    momentumSourceZOut[gid] = sourceZ;
    signedPowerOut[gid] = signedPower;
    signedWorkOut[gid] = signedWork;
    activeOut[gid] = (((__longlong_as_double(0x0000000000000000ull) < lambda)) ? (1u) : (0u));
  }
}
