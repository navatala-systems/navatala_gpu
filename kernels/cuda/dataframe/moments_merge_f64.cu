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
extern "C" __global__ void navatala_dataframe_moments_merge_f64(const unsigned int* nA, const double* M1A, const double* M2A, const double* M3A, const double* M4A, const unsigned int* nB, const double* M1B, const double* M2B, const double* M3B, const double* M4B, unsigned int* nOut, double* M1Out, double* M2Out, double* M3Out, double* M4Out) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  if ((int)(blockIdx.x * blockDim.x + threadIdx.x) == 0) {
    unsigned int nAVal = nA[0];
    double m1AVal = M1A[0];
    double m2AVal = M2A[0];
    double m3AVal = M3A[0];
    double m4AVal = M4A[0];
    unsigned int nBVal = nB[0];
    double m1BVal = M1B[0];
    double m2BVal = M2B[0];
    double m3BVal = M3B[0];
    double m4BVal = M4B[0];
    unsigned int nTotal = (nAVal + nBVal);
    double nAf = ((double)(nAVal));
    double nBf = ((double)(nBVal));
    double nTotalf = ((double)(nTotal));
    double nTotalSafe = (((nTotal == 0u)) ? (__longlong_as_double(0x3ff0000000000000ull)) : (nTotalf));
    double delta = (m1BVal - m1AVal);
    double delta2 = (delta * delta);
    double delta3 = (delta2 * delta);
    double delta4 = (delta3 * delta);
    double nAnB = (nAf * nBf);
    double m1New = (m1AVal + (delta * (nBf / nTotalSafe)));
    double m2Term = ((delta2 * nAnB) / nTotalSafe);
    double m2New = ((m2AVal + m2BVal) + m2Term);
    double n2 = (nTotalSafe * nTotalSafe);
    double nAMinusNB = (nAf - nBf);
    double m3Term1 = (((delta3 * nAnB) * nAMinusNB) / n2);
    double nAM2B = (nAf * m2BVal);
    double nBM2A = (nBf * m2AVal);
    double m3Term2 = (((__longlong_as_double(0x4008000000000000ull) * delta) * (nAM2B - nBM2A)) / nTotalSafe);
    double m3New = (((m3AVal + m3BVal) + m3Term1) + m3Term2);
    double n3 = (n2 * nTotalSafe);
    double nA2 = (nAf * nAf);
    double nB2 = (nBf * nBf);
    double nA2MinusNAnB = (nA2 - nAnB);
    double m4Coeff = (nA2MinusNAnB + nB2);
    double m4Term1 = (((delta4 * nAnB) * m4Coeff) / n3);
    double nA2M2B = (nA2 * m2BVal);
    double nB2M2A = (nB2 * m2AVal);
    double m4Term2 = (((__longlong_as_double(0x4018000000000000ull) * delta2) * (nA2M2B + nB2M2A)) / n2);
    double nAM3B = (nAf * m3BVal);
    double nBM3A = (nBf * m3AVal);
    double m4Term3 = (((__longlong_as_double(0x4010000000000000ull) * delta) * (nAM3B - nBM3A)) / nTotalSafe);
    double m4New = ((((m4AVal + m4BVal) + m4Term1) + m4Term2) + m4Term3);
    nOut[0] = nTotal;
    M1Out[0] = m1New;
    M2Out[0] = m2New;
    M3Out[0] = m3New;
    M4Out[0] = m4New;
  }
}
