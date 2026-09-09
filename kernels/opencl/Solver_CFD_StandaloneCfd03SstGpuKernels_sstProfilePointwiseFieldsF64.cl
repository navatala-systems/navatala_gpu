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
__kernel void Solver_CFD_StandaloneCfd03SstGpuKernels_sstProfilePointwiseFieldsF64(__global const double* alpha1, __global const double* k, __global const double* omega, __global const double* wallDistance, __global const double* gradientDotProduct, __global const double* strainMagnitudeSquared2, __global const double* datum, __global const uint* nCells, __global double* densityOut, __global double* dynamicViscosityOut, __global double* kinematicViscosityOut, __global double* crossDiffusionOut, __global double* positiveCrossDiffusionOut, __global double* argument1Out, __global double* argument2Out, __global double* f1Out, __global double* f2Out, __global double* turbulentKinematicViscosityOut, __global double* turbulentDynamicViscosityOut, __global double* effectiveDynamicViscosityOut, __global double* nutToNuOut, __global double* gammaOut, __global double* betaOut, __global double* sigmaKOut, __global double* sigmaOmegaOut, __global double* productionRawOut, __global double* productionLimitedOut, __global double* kExplicitSourceOut, __global double* kImplicitSinkOut, __global double* omegaBaseSourceOut, __global double* crossLagOut, __global double* omegaCrossExplicitSourceOut, __global double* omegaCrossImplicitSinkOut, __global double* omegaImplicitSinkOut, __global double* kEffectiveDiffusivityOut, __global double* omegaEffectiveDiffusivityOut) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint cellCount = nCells[0];
  if (gid < cellCount) {
    double alpha = alpha1[gid];
    double kValue = k[gid];
    double omegaValue = omega[gid];
    double distance = wallDistance[gid];
    double gradientDot = gradientDotProduct[gid];
    double strain2 = strainMagnitudeSquared2[gid];
    double oneMinusAlpha = (as_double(0x3ff0000000000000ul) - alpha);
    double density = ((alpha * datum[0]) + (oneMinusAlpha * datum[1]));
    double dynamicViscosity = ((alpha * datum[2]) + (oneMinusAlpha * datum[3]));
    double kinematicViscosity = (dynamicViscosity / density);
    double crossDiffusion = (((as_double(0x4000000000000000ul) * as_double(0x3feb645a1cac0831ul)) / omegaValue) * gradientDot);
    double positiveCrossDiffusion = (((crossDiffusion < as_double(0x3ddb7cdfd9d7bdbbul))) ? (as_double(0x3ddb7cdfd9d7bdbbul)) : (crossDiffusion));
    double rootK = sqrt(kValue);
    double distanceSquared = (distance * distance);
    double firstA = (rootK / ((as_double(0x3fb70a3d70a3d70aul) * omegaValue) * distance));
    double firstB = ((as_double(0x407f400000000000ul) * kinematicViscosity) / (distanceSquared * omegaValue));
    double firstC = (((as_double(0x4010000000000000ul) * as_double(0x3feb645a1cac0831ul)) * kValue) / (positiveCrossDiffusion * distanceSquared));
    double argument1 = (((firstC < (((firstA < firstB)) ? (firstB) : (firstA)))) ? (firstC) : ((((firstA < firstB)) ? (firstB) : (firstA))));
    double argument2 = ((((as_double(0x4000000000000000ul) * firstA) < firstB)) ? (firstB) : ((as_double(0x4000000000000000ul) * firstA)));
    double f1Value = (((((argument1 * argument1) * (argument1 * argument1)) < as_double(0x4034000000000000ul))) ? (tanh(((argument1 * argument1) * (argument1 * argument1)))) : (as_double(0x3ff0000000000000ul)));
    double f2Value = ((((argument2 * argument2) < as_double(0x4034000000000000ul))) ? (tanh((argument2 * argument2))) : (as_double(0x3ff0000000000000ul)));
    double strainMagnitude = sqrt(strain2);
    double nutDenominator = ((((as_double(0x3fd3d70a3d70a3d7ul) * omegaValue) < ((as_double(0x3ff0000000000000ul) * f2Value) * strainMagnitude))) ? (((as_double(0x3ff0000000000000ul) * f2Value) * strainMagnitude)) : ((as_double(0x3fd3d70a3d70a3d7ul) * omegaValue)));
    double nut = ((as_double(0x3fd3d70a3d70a3d7ul) * kValue) / nutDenominator);
    double turbulentDynamicViscosity = (density * nut);
    double effectiveDynamicViscosity = (dynamicViscosity + turbulentDynamicViscosity);
    double nutToNu = (nut / (dynamicViscosity / density));
    double oneMinusF1 = (as_double(0x3ff0000000000000ul) - f1Value);
    double gamma = ((f1Value * as_double(0x3fe1c71c71c71c72ul)) + (oneMinusF1 * as_double(0x3fdc28f5c28f5c29ul)));
    double beta = ((f1Value * as_double(0x3fb3333333333333ul)) + (oneMinusF1 * as_double(0x3fb532617c1bda51ul)));
    double sigmaK = ((f1Value * as_double(0x3feb333333333333ul)) + (oneMinusF1 * as_double(0x3ff0000000000000ul)));
    double sigmaOmega = ((f1Value * as_double(0x3fe0000000000000ul)) + (oneMinusF1 * as_double(0x3feb645a1cac0831ul)));
    double productionRaw = (turbulentDynamicViscosity * strain2);
    double productionLimit = ((((as_double(0x4024000000000000ul) * as_double(0x3fb70a3d70a3d70aul)) * density) * kValue) * omegaValue);
    double productionLimited = (((productionLimit < productionRaw)) ? (productionLimit) : (productionRaw));
    double kExplicitSource = (((productionLimited < as_double(0x0000000000000000ul))) ? (as_double(0x0000000000000000ul)) : (productionLimited));
    double kImplicitSink = ((as_double(0x3fb70a3d70a3d70aul) * density) * omegaValue);
    double omegaBaseSource = ((gamma * density) * strain2);
    double admittedOmega = (((omegaValue < as_double(0x3d719799812dea11ul))) ? (as_double(0x3d719799812dea11ul)) : (omegaValue));
    double crossLag = (((((as_double(0x4000000000000000ul) * oneMinusF1) * density) * as_double(0x3feb645a1cac0831ul)) / admittedOmega) * gradientDot);
    double omegaCrossExplicitSource = (((crossLag < as_double(0x0000000000000000ul))) ? (as_double(0x0000000000000000ul)) : (crossLag));
    double omegaCrossImplicitSink = (((((-crossLag) < as_double(0x0000000000000000ul))) ? (as_double(0x0000000000000000ul)) : ((-crossLag))) / admittedOmega);
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
