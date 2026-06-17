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
extern "C" __global__ void navatala_dataframe_date_parts_to_timestamp(const int* yearsIn, const int* monthsIn, const int* daysIn, const unsigned int* validity, const unsigned int* count, long long* _output, unsigned int* outputValid) {
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
    int yearI32 = yearsIn[gid];
    int monthI32 = monthsIn[gid];
    int dayI32 = daysIn[gid];
    long long year = ((long long)(yearI32));
    long long month = ((long long)(monthI32));
    long long day = ((long long)(dayI32));
    bool div4 = ((year % 4) == 0);
    bool div100 = ((year % 100) == 0);
    bool div400 = ((year % 400) == 0);
    bool isLeap = (div4 && ((!div100) || div400));
    long long leapAdj = ((isLeap) ? (1) : (0));
    bool isJan = (month == 1);
    bool isFeb = (month == 2);
    bool isMar = (month == 3);
    bool isApr = (month == 4);
    bool isMay = (month == 5);
    bool isJun = (month == 6);
    bool isJul = (month == 7);
    bool isAug = (month == 8);
    bool isSep = (month == 9);
    bool isOct = (month == 10);
    bool isNov = (month == 11);
    long long daysBeforeMonth = ((isJan) ? (0) : (((isFeb) ? (31) : (((isMar) ? ((59 + leapAdj)) : (((isApr) ? ((90 + leapAdj)) : (((isMay) ? ((120 + leapAdj)) : (((isJun) ? ((151 + leapAdj)) : (((isJul) ? ((181 + leapAdj)) : (((isAug) ? ((212 + leapAdj)) : (((isSep) ? ((243 + leapAdj)) : (((isOct) ? ((273 + leapAdj)) : (((isNov) ? ((304 + leapAdj)) : ((334 + leapAdj)))))))))))))))))))))));
    long long dayOfYear = (daysBeforeMonth + (day - 1));
    long long yearM1 = (year - 1);
    long long baseDays = (yearM1 * 365);
    long long leap4 = (yearM1 / 4);
    long long leap100 = (yearM1 / 100);
    long long leap400 = (yearM1 / 400);
    long long daysFromYear0 = (((baseDays + leap4) - leap100) + leap400);
    long long totalDaysFromYear0 = (daysFromYear0 + dayOfYear);
    long long daysSinceEpoch = (totalDaysFromYear0 - 719527);
    long long usPerDay = 86400000000;
    long long tsUs = (daysSinceEpoch * usPerDay);
    long long outVal = ((isValid) ? (tsUs) : (0));
    _output[gid] = outVal;
    unsigned int outBit = (validBit << bitIdx);
    atomicOr(&outputValid[wordIdx], outBit);
  }
}
