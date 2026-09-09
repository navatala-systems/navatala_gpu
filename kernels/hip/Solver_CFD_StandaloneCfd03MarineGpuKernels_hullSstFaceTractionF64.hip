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
extern "C" __global__ void Solver_CFD_StandaloneCfd03MarineGpuKernels_hullSstFaceTractionF64(const double* wallShearOnFluidX, const double* wallShearOnFluidY, const double* wallShearOnFluidZ, const double* fluidAreaVectorX, const double* fluidAreaVectorY, const double* fluidAreaVectorZ, const double* faceCentroidX, const double* faceCentroidY, const double* faceCentroidZ, const double* hullSstDatum, const unsigned int* nFaces, double* areaOut, double* forceXOut, double* forceYOut, double* forceZOut, double* momentXOut, double* momentYOut, double* momentZOut) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  int gid = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  int faceCount = ((int)(nFaces[0]));
  if (gid < faceCount) {
    double areaX = fluidAreaVectorX[gid];
    double areaY = fluidAreaVectorY[gid];
    double areaZ = fluidAreaVectorZ[gid];
    double area = sqrt((((areaX * areaX) + (areaY * areaY)) + (areaZ * areaZ)));
    double forceX = ((-wallShearOnFluidX[gid]) * area);
    double forceY = ((-wallShearOnFluidY[gid]) * area);
    double forceZ = ((-wallShearOnFluidZ[gid]) * area);
    double momentArmX = (faceCentroidX[gid] - hullSstDatum[0]);
    double momentArmY = (faceCentroidY[gid] - hullSstDatum[1]);
    double momentArmZ = (faceCentroidZ[gid] - hullSstDatum[2]);
    double momentX = ((momentArmY * forceZ) - (momentArmZ * forceY));
    double momentY = ((momentArmZ * forceX) - (momentArmX * forceZ));
    double momentZ = ((momentArmX * forceY) - (momentArmY * forceX));
    areaOut[gid] = area;
    forceXOut[gid] = forceX;
    forceYOut[gid] = forceY;
    forceZOut[gid] = forceZ;
    momentXOut[gid] = momentX;
    momentYOut[gid] = momentY;
    momentZOut[gid] = momentZ;
  }
}
