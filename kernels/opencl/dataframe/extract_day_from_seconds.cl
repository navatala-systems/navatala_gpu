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

__kernel void navatala_dataframe_extract_day_from_seconds(__global const long* timestamps, __global const uint* validity, __global const uint* count, __global int* _output, __global uint* outputValid) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint n = count[(uint)(0u)];
  bool inBounds = (gid < n);
  if (inBounds) {
    uint wordIdx = (gid / (uint)(32u));
    uint bitIdx = (gid % (uint)(32u));
    uint validWord = validity[wordIdx];
    uint validBit = ((validWord >> bitIdx) & (uint)(1u));
    bool isValid = (validBit == (uint)(1u));
    long ts = timestamps[gid];
    long secsInDay = 86400;
    long rawDays = (ts / secsInDay);
    long remainder = (ts % secsInDay);
    bool needsAdjust = ((ts < 0) & (remainder != 0));
    long days = ((needsAdjust) ? ((rawDays - 1)) : (rawDays));
    long daysFromYear0 = (days + 719528);
    long cycles400 = (daysFromYear0 / 146097);
    long rem400 = (daysFromYear0 % 146097);
    long cycles100 = (rem400 / 36524);
    long adj100 = (((cycles100 == 4)) ? (3) : (cycles100));
    long rem100 = (rem400 - (adj100 * 36524));
    long cycles4 = (rem100 / 1461);
    long rem4 = (rem100 % 1461);
    long years1 = (rem4 / 365);
    long adjYears1 = (((years1 == 4)) ? (3) : (years1));
    long year = ((((cycles400 * 400) + (adj100 * 100)) + (cycles4 * 4)) + adjYears1);
    long dayOfYear = (rem4 - (adjYears1 * 365));
    bool div4 = ((year % 4) == 0);
    bool div100 = ((year % 100) == 0);
    bool div400 = ((year % 400) == 0);
    bool isLeap = (div4 && ((!div100) || div400));
    long doy1 = (dayOfYear + 1);
    long leapAdj = ((isLeap) ? (1) : (0));
    bool m12 = (doy1 >= (335 + leapAdj));
    bool m11 = (doy1 >= (305 + leapAdj));
    bool m10 = (doy1 >= (274 + leapAdj));
    bool m9 = (doy1 >= (244 + leapAdj));
    bool m8 = (doy1 >= (213 + leapAdj));
    bool m7 = (doy1 >= (182 + leapAdj));
    bool m6 = (doy1 >= (152 + leapAdj));
    bool m5 = (doy1 >= (121 + leapAdj));
    bool m4 = (doy1 >= (91 + leapAdj));
    bool m3 = (doy1 >= (60 + leapAdj));
    bool m2 = (doy1 >= 32);
    long monthStart = ((m12) ? ((335 + leapAdj)) : (((m11) ? ((305 + leapAdj)) : (((m10) ? ((274 + leapAdj)) : (((m9) ? ((244 + leapAdj)) : (((m8) ? ((213 + leapAdj)) : (((m7) ? ((182 + leapAdj)) : (((m6) ? ((152 + leapAdj)) : (((m5) ? ((121 + leapAdj)) : (((m4) ? ((91 + leapAdj)) : (((m3) ? ((60 + leapAdj)) : (((m2) ? (32) : (1))))))))))))))))))))));
    long dayOfMonth = ((doy1 - monthStart) + 1);
    int dayI32 = ((int)(dayOfMonth));
    int outVal = ((isValid) ? (dayI32) : (0));
    _output[gid] = outVal;
    uint outBit = (validBit << bitIdx);
    atomic_or(&outputValid[wordIdx], outBit);
  }
}
