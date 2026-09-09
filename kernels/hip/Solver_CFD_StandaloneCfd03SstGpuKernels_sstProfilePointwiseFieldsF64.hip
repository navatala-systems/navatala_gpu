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
extern "C" __global__ void Solver_CFD_StandaloneCfd03SstGpuKernels_sstProfilePointwiseFieldsF64(const double* alpha1, const double* k, const double* omega, const double* wallDistance, const double* gradientDotProduct, const double* strainMagnitudeSquared2, const double* datum, const unsigned int* nCells, double* densityOut, double* dynamicViscosityOut, double* kinematicViscosityOut, double* crossDiffusionOut, double* positiveCrossDiffusionOut, double* argument1Out, double* argument2Out, double* f1Out, double* f2Out, double* turbulentKinematicViscosityOut, double* turbulentDynamicViscosityOut, double* effectiveDynamicViscosityOut, double* nutToNuOut, double* gammaOut, double* betaOut, double* sigmaKOut, double* sigmaOmegaOut, double* productionRawOut, double* productionLimitedOut, double* kExplicitSourceOut, double* kImplicitSinkOut, double* omegaBaseSourceOut, double* crossLagOut, double* omegaCrossExplicitSourceOut, double* omegaCrossImplicitSinkOut, double* omegaImplicitSinkOut, double* kEffectiveDiffusivityOut, double* omegaEffectiveDiffusivityOut) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int cellCount = nCells[0];
  if (gid < cellCount) {
    double alpha = alpha1[gid];
    double kValue = k[gid];
    double omegaValue = omega[gid];
    double distance = wallDistance[gid];
    double gradientDot = gradientDotProduct[gid];
    double strain2 = strainMagnitudeSquared2[gid];
    double oneMinusAlpha = (__longlong_as_double(0x3ff0000000000000ull) - alpha);
    double density = ((alpha * datum[0]) + (oneMinusAlpha * datum[1]));
    double dynamicViscosity = ((alpha * datum[2]) + (oneMinusAlpha * datum[3]));
    double kinematicViscosity = (dynamicViscosity / density);
    double crossDiffusion = (((__longlong_as_double(0x4000000000000000ull) * __longlong_as_double(0x3feb645a1cac0831ull)) / omegaValue) * gradientDot);
    double positiveCrossDiffusion = (((crossDiffusion < __longlong_as_double(0x3ddb7cdfd9d7bdbbull))) ? (__longlong_as_double(0x3ddb7cdfd9d7bdbbull)) : (crossDiffusion));
    double rootK = sqrt(kValue);
    double distanceSquared = (distance * distance);
    double firstA = (rootK / ((__longlong_as_double(0x3fb70a3d70a3d70aull) * omegaValue) * distance));
    double firstB = ((__longlong_as_double(0x407f400000000000ull) * kinematicViscosity) / (distanceSquared * omegaValue));
    double firstC = (((__longlong_as_double(0x4010000000000000ull) * __longlong_as_double(0x3feb645a1cac0831ull)) * kValue) / (positiveCrossDiffusion * distanceSquared));
    double argument1 = (((firstC < (((firstA < firstB)) ? (firstB) : (firstA)))) ? (firstC) : ((((firstA < firstB)) ? (firstB) : (firstA))));
    double argument2 = ((((__longlong_as_double(0x4000000000000000ull) * firstA) < firstB)) ? (firstB) : ((__longlong_as_double(0x4000000000000000ull) * firstA)));
    double f1Value = (((((argument1 * argument1) * (argument1 * argument1)) < __longlong_as_double(0x4034000000000000ull))) ? (tanh(((argument1 * argument1) * (argument1 * argument1)))) : (__longlong_as_double(0x3ff0000000000000ull)));
    double f2Value = ((((argument2 * argument2) < __longlong_as_double(0x4034000000000000ull))) ? (tanh((argument2 * argument2))) : (__longlong_as_double(0x3ff0000000000000ull)));
    double strainMagnitude = sqrt(strain2);
    double nutDenominator = ((((__longlong_as_double(0x3fd3d70a3d70a3d7ull) * omegaValue) < ((__longlong_as_double(0x3ff0000000000000ull) * f2Value) * strainMagnitude))) ? (((__longlong_as_double(0x3ff0000000000000ull) * f2Value) * strainMagnitude)) : ((__longlong_as_double(0x3fd3d70a3d70a3d7ull) * omegaValue)));
    double nut = ((__longlong_as_double(0x3fd3d70a3d70a3d7ull) * kValue) / nutDenominator);
    double turbulentDynamicViscosity = (density * nut);
    double effectiveDynamicViscosity = (dynamicViscosity + turbulentDynamicViscosity);
    double nutToNu = (nut / (dynamicViscosity / density));
    double oneMinusF1 = (__longlong_as_double(0x3ff0000000000000ull) - f1Value);
    double gamma = ((f1Value * __longlong_as_double(0x3fe1c71c71c71c72ull)) + (oneMinusF1 * __longlong_as_double(0x3fdc28f5c28f5c29ull)));
    double beta = ((f1Value * __longlong_as_double(0x3fb3333333333333ull)) + (oneMinusF1 * __longlong_as_double(0x3fb532617c1bda51ull)));
    double sigmaK = ((f1Value * __longlong_as_double(0x3feb333333333333ull)) + (oneMinusF1 * __longlong_as_double(0x3ff0000000000000ull)));
    double sigmaOmega = ((f1Value * __longlong_as_double(0x3fe0000000000000ull)) + (oneMinusF1 * __longlong_as_double(0x3feb645a1cac0831ull)));
    double productionRaw = (turbulentDynamicViscosity * strain2);
    double productionLimit = ((((__longlong_as_double(0x4024000000000000ull) * __longlong_as_double(0x3fb70a3d70a3d70aull)) * density) * kValue) * omegaValue);
    double productionLimited = (((productionLimit < productionRaw)) ? (productionLimit) : (productionRaw));
    double kExplicitSource = (((productionLimited < __longlong_as_double(0x0000000000000000ull))) ? (__longlong_as_double(0x0000000000000000ull)) : (productionLimited));
    double kImplicitSink = ((__longlong_as_double(0x3fb70a3d70a3d70aull) * density) * omegaValue);
    double omegaBaseSource = ((gamma * density) * strain2);
    double admittedOmega = (((omegaValue < __longlong_as_double(0x3d719799812dea11ull))) ? (__longlong_as_double(0x3d719799812dea11ull)) : (omegaValue));
    double crossLag = (((((__longlong_as_double(0x4000000000000000ull) * oneMinusF1) * density) * __longlong_as_double(0x3feb645a1cac0831ull)) / admittedOmega) * gradientDot);
    double omegaCrossExplicitSource = (((crossLag < __longlong_as_double(0x0000000000000000ull))) ? (__longlong_as_double(0x0000000000000000ull)) : (crossLag));
    double omegaCrossImplicitSink = (((((-crossLag) < __longlong_as_double(0x0000000000000000ull))) ? (__longlong_as_double(0x0000000000000000ull)) : ((-crossLag))) / admittedOmega);
    double omegaImplicitSink = (((beta * density) * omegaValue) + omegaCrossImplicitSink);
    double kEffectiveDiffusivity = (dynamicViscosity + (sigmaK * turbulentDynamicViscosity));
    double omegaEffectiveDiffusivity = (dynamicViscosity + (sigmaOmega * turbulentDynamicViscosity));
    densityOut[gid] = density;
    dynamicViscosityOut[gid] = dynamicViscosity;
    kinematicViscosityOut[gid] = kinematicViscosity;
    crossDiffusionOut[gid] = crossDiffusion;
    positiveCrossDiffusionOut[gid] = positiveCrossDiffusion;
    argument1Out[gid] = argument1;
    argument2Out[gid] = argument2;
    f1Out[gid] = f1Value;
    f2Out[gid] = f2Value;
    turbulentKinematicViscosityOut[gid] = nut;
    turbulentDynamicViscosityOut[gid] = turbulentDynamicViscosity;
    effectiveDynamicViscosityOut[gid] = effectiveDynamicViscosity;
    nutToNuOut[gid] = nutToNu;
    gammaOut[gid] = gamma;
    betaOut[gid] = beta;
    sigmaKOut[gid] = sigmaK;
    sigmaOmegaOut[gid] = sigmaOmega;
    productionRawOut[gid] = productionRaw;
    productionLimitedOut[gid] = productionLimited;
    kExplicitSourceOut[gid] = kExplicitSource;
    kImplicitSinkOut[gid] = kImplicitSink;
    omegaBaseSourceOut[gid] = omegaBaseSource;
    crossLagOut[gid] = crossLag;
    omegaCrossExplicitSourceOut[gid] = omegaCrossExplicitSource;
    omegaCrossImplicitSinkOut[gid] = omegaCrossImplicitSink;
    omegaImplicitSinkOut[gid] = omegaImplicitSink;
    kEffectiveDiffusivityOut[gid] = kEffectiveDiffusivity;
    omegaEffectiveDiffusivityOut[gid] = omegaEffectiveDiffusivity;
  }
}
