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
extern "C" __global__ void Solver_CFD_StandaloneCfd03MarineGpuKernels_hullLaminarFaceTractionF64(const double* velocityGradient00, const double* velocityGradient01, const double* velocityGradient02, const double* velocityGradient10, const double* velocityGradient11, const double* velocityGradient12, const double* velocityGradient20, const double* velocityGradient21, const double* velocityGradient22, const double* dynamicViscosity, const double* fluidAreaVectorX, const double* fluidAreaVectorY, const double* fluidAreaVectorZ, const double* faceCentroidX, const double* faceCentroidY, const double* faceCentroidZ, const double* hullLaminarDatum, const unsigned int* nFaces, double* areaOut, double* forceXOut, double* forceYOut, double* forceZOut, double* momentXOut, double* momentYOut, double* momentZOut) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  int gid = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  int faceCount = ((int)(nFaces[0]));
  if (gid < faceCount) {
    double gradient00 = velocityGradient00[gid];
    double gradient01 = velocityGradient01[gid];
    double gradient02 = velocityGradient02[gid];
    double gradient10 = velocityGradient10[gid];
    double gradient11 = velocityGradient11[gid];
    double gradient12 = velocityGradient12[gid];
    double gradient20 = velocityGradient20[gid];
    double gradient21 = velocityGradient21[gid];
    double gradient22 = velocityGradient22[gid];
    double trace = ((gradient00 + gradient11) + gradient22);
    double traceThird = (trace / __longlong_as_double(0x4008000000000000ull));
    double twoMu = (__longlong_as_double(0x4000000000000000ull) * dynamicViscosity[gid]);
    double strain00 = ((__longlong_as_double(0x3fe0000000000000ull) * (gradient00 + gradient00)) - traceThird);
    double strain01 = (__longlong_as_double(0x3fe0000000000000ull) * (gradient01 + gradient10));
    double strain02 = (__longlong_as_double(0x3fe0000000000000ull) * (gradient02 + gradient20));
    double strain10 = (__longlong_as_double(0x3fe0000000000000ull) * (gradient10 + gradient01));
    double strain11 = ((__longlong_as_double(0x3fe0000000000000ull) * (gradient11 + gradient11)) - traceThird);
    double strain12 = (__longlong_as_double(0x3fe0000000000000ull) * (gradient12 + gradient21));
    double strain20 = (__longlong_as_double(0x3fe0000000000000ull) * (gradient20 + gradient02));
    double strain21 = (__longlong_as_double(0x3fe0000000000000ull) * (gradient21 + gradient12));
    double strain22 = ((__longlong_as_double(0x3fe0000000000000ull) * (gradient22 + gradient22)) - traceThird);
    double tau00 = (twoMu * strain00);
    double tau01 = (twoMu * strain01);
    double tau02 = (twoMu * strain02);
    double tau10 = (twoMu * strain10);
    double tau11 = (twoMu * strain11);
    double tau12 = (twoMu * strain12);
    double tau20 = (twoMu * strain20);
    double tau21 = (twoMu * strain21);
    double tau22 = (twoMu * strain22);
    double areaX = fluidAreaVectorX[gid];
    double areaY = fluidAreaVectorY[gid];
    double areaZ = fluidAreaVectorZ[gid];
    double area = sqrt((((areaX * areaX) + (areaY * areaY)) + (areaZ * areaZ)));
    double minusAreaX = (areaX * __longlong_as_double(0xbff0000000000000ull));
    double minusAreaY = (areaY * __longlong_as_double(0xbff0000000000000ull));
    double minusAreaZ = (areaZ * __longlong_as_double(0xbff0000000000000ull));
    double forceX = (((tau00 * minusAreaX) + (tau01 * minusAreaY)) + (tau02 * minusAreaZ));
    double forceY = (((tau10 * minusAreaX) + (tau11 * minusAreaY)) + (tau12 * minusAreaZ));
    double forceZ = (((tau20 * minusAreaX) + (tau21 * minusAreaY)) + (tau22 * minusAreaZ));
    double momentArmX = (faceCentroidX[gid] - hullLaminarDatum[0]);
    double momentArmY = (faceCentroidY[gid] - hullLaminarDatum[1]);
    double momentArmZ = (faceCentroidZ[gid] - hullLaminarDatum[2]);
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
