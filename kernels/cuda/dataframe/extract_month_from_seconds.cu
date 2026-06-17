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
extern "C" __global__ void navatala_dataframe_extract_month_from_seconds(const long long* timestamps, const unsigned int* validity, const unsigned int* count, int* _output, unsigned int* outputValid) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int n = count[0u];
  bool inBounds = (gid < n);
  if (inBounds) {
    unsigned int wordIdx = (gid / 32u);
    unsigned int bitIdx = (gid % 32u);
    unsigned int validWord = validity[wordIdx];
    unsigned int validBit = ((validWord >> bitIdx) & 1u);
    bool isValid = (validBit == 1u);
    long long ts = timestamps[gid];
    long long secsInDay = 86400;
    long long rawDays = (ts / secsInDay);
    long long remainder = (ts % secsInDay);
    bool needsAdjust = ((ts < 0) & (remainder != 0));
    long long days = ((needsAdjust) ? ((rawDays - 1)) : (rawDays));
    long long daysFromYear0 = (days + 719528);
    long long cycles400 = (daysFromYear0 / 146097);
    long long rem400 = (daysFromYear0 % 146097);
    long long cycles100 = (rem400 / 36524);
    long long adj100 = (((cycles100 == 4)) ? (3) : (cycles100));
    long long rem100 = (rem400 - (adj100 * 36524));
    long long cycles4 = (rem100 / 1461);
    long long rem4 = (rem100 % 1461);
    long long years1 = (rem4 / 365);
    long long adjYears1 = (((years1 == 4)) ? (3) : (years1));
    long long year = ((((cycles400 * 400) + (adj100 * 100)) + (cycles4 * 4)) + adjYears1);
    long long dayOfYear = (rem4 - (adjYears1 * 365));
    bool div4 = ((year % 4) == 0);
    bool div100 = ((year % 100) == 0);
    bool div400 = ((year % 400) == 0);
    bool isLeap = (div4 && ((!div100) || div400));
    long long doy1 = (dayOfYear + 1);
    long long leapAdj = ((isLeap) ? (1) : (0));
    bool m12 = (doy1 >= (335 + leapAdj));
    bool m11 = (doy1 >= (305 + leapAdj));
    bool m10 = (doy1 >= (274 + leapAdj));
    bool m9 = (doy1 >= (244 + leapAdj));
    bool m8 = (doy1 >= (213 + leapAdj));
    bool m7 = (doy1 >= (182 + leapAdj));
    bool m6 = (doy1 >= (152 + leapAdj));
    bool m5 = (doy1 >= (121 + leapAdj));
    bool m4 = (doy1 >= (91 + leapAdj));
    bool m3 = (doy1 >= 60);
    bool m2 = (doy1 >= 32);
    long long month = ((m12) ? (12) : (((m11) ? (11) : (((m10) ? (10) : (((m9) ? (9) : (((m8) ? (8) : (((m7) ? (7) : (((m6) ? (6) : (((m5) ? (5) : (((m4) ? (4) : (((m3) ? (3) : (((m2) ? (2) : (1))))))))))))))))))))));
    int monthI32 = ((int)(month));
    int outVal = ((isValid) ? (monthI32) : (0));
    _output[gid] = outVal;
    unsigned int outBit = (validBit << bitIdx);
    atomicOr(&outputValid[wordIdx], outBit);
  }
}
