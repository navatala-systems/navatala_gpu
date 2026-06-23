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

__kernel void navatala_transformer_hadamard_transform_f32(__global const float* _input, __global const uint* batchSize, __global const uint* vectorSize, __global float* _output) {
  int gid0 = (int)get_global_id(0);
  uint lid = ((uint)((int)(get_local_id(0))));
  uint batchIdx = ((uint)((int)(get_group_id(0))));
  uint bs = batchSize[(uint)(0u)];
  uint vs = vectorSize[(uint)(0u)];
  __local float sharedBuf[256];
  bool batchValid = (batchIdx < bs);
  bool threadValid = (lid < vs);
  uint globalIdx = ((batchIdx * vs) + lid);
  if (batchValid && threadValid) {
    float val = _input[globalIdx];
    sharedBuf[lid] = val;
  } else {
    sharedBuf[lid] = as_float(0x00000000u);
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  uint stride1 = (uint)(1u);
  uint partner1 = (lid ^ stride1);
  float myVal1 = sharedBuf[lid];
  float partnerVal1 = sharedBuf[partner1];
  bool lowBit1 = ((lid & stride1) == (uint)(0u));
  float newVal1 = ((lowBit1) ? ((myVal1 + partnerVal1)) : ((partnerVal1 - myVal1)));
  barrier(CLK_LOCAL_MEM_FENCE);
  sharedBuf[lid] = newVal1;
  barrier(CLK_LOCAL_MEM_FENCE);
  uint stride2 = (uint)(2u);
  uint partner2 = (lid ^ stride2);
  float myVal2 = sharedBuf[lid];
  float partnerVal2 = sharedBuf[partner2];
  bool lowBit2 = ((lid & stride2) == (uint)(0u));
  float newVal2 = ((lowBit2) ? ((myVal2 + partnerVal2)) : ((partnerVal2 - myVal2)));
  barrier(CLK_LOCAL_MEM_FENCE);
  sharedBuf[lid] = newVal2;
  barrier(CLK_LOCAL_MEM_FENCE);
  uint stride3 = (uint)(4u);
  uint partner3 = (lid ^ stride3);
  float myVal3 = sharedBuf[lid];
  float partnerVal3 = sharedBuf[partner3];
  bool lowBit3 = ((lid & stride3) == (uint)(0u));
  float newVal3 = ((lowBit3) ? ((myVal3 + partnerVal3)) : ((partnerVal3 - myVal3)));
  barrier(CLK_LOCAL_MEM_FENCE);
  sharedBuf[lid] = newVal3;
  barrier(CLK_LOCAL_MEM_FENCE);
  uint stride4 = (uint)(8u);
  uint partner4 = (lid ^ stride4);
  float myVal4 = sharedBuf[lid];
  float partnerVal4 = sharedBuf[partner4];
  bool lowBit4 = ((lid & stride4) == (uint)(0u));
  float newVal4 = ((lowBit4) ? ((myVal4 + partnerVal4)) : ((partnerVal4 - myVal4)));
  barrier(CLK_LOCAL_MEM_FENCE);
  sharedBuf[lid] = newVal4;
  barrier(CLK_LOCAL_MEM_FENCE);
  uint stride5 = (uint)(16u);
  uint partner5 = (lid ^ stride5);
  float myVal5 = sharedBuf[lid];
  float partnerVal5 = sharedBuf[partner5];
  bool lowBit5 = ((lid & stride5) == (uint)(0u));
  float newVal5 = ((lowBit5) ? ((myVal5 + partnerVal5)) : ((partnerVal5 - myVal5)));
  barrier(CLK_LOCAL_MEM_FENCE);
  sharedBuf[lid] = newVal5;
  barrier(CLK_LOCAL_MEM_FENCE);
  uint stride6 = (uint)(32u);
  uint partner6 = (lid ^ stride6);
  float myVal6 = sharedBuf[lid];
  float partnerVal6 = sharedBuf[partner6];
  bool lowBit6 = ((lid & stride6) == (uint)(0u));
  float newVal6 = ((lowBit6) ? ((myVal6 + partnerVal6)) : ((partnerVal6 - myVal6)));
  barrier(CLK_LOCAL_MEM_FENCE);
  sharedBuf[lid] = newVal6;
  barrier(CLK_LOCAL_MEM_FENCE);
  uint stride7 = (uint)(64u);
  uint partner7 = (lid ^ stride7);
  float myVal7 = sharedBuf[lid];
  float partnerVal7 = sharedBuf[partner7];
  bool lowBit7 = ((lid & stride7) == (uint)(0u));
  float newVal7 = ((lowBit7) ? ((myVal7 + partnerVal7)) : ((partnerVal7 - myVal7)));
  barrier(CLK_LOCAL_MEM_FENCE);
  sharedBuf[lid] = newVal7;
  barrier(CLK_LOCAL_MEM_FENCE);
  uint stride8 = (uint)(128u);
  uint partner8 = (lid ^ stride8);
  float myVal8 = sharedBuf[lid];
  float partnerVal8 = sharedBuf[partner8];
  bool lowBit8 = ((lid & stride8) == (uint)(0u));
  float newVal8 = ((lowBit8) ? ((myVal8 + partnerVal8)) : ((partnerVal8 - myVal8)));
  float vsF32 = ((float)(vs));
  float scale = sqrt(vsF32);
  float normalized = (newVal8 / scale);
  if (batchValid && threadValid) {
    _output[globalIdx] = normalized;
  }
}
