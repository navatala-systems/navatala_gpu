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

#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__kernel void navatala_transformer_pad_sequence_f16(__global const half* _input, __global const uint* seqLengths, __global const uint* seqOffsets, __global const uint* batchSize, __global const uint* maxSeqLen, __global const uint* hiddenSize, __global half* _output) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint bs = batchSize[(uint)(0u)];
  uint msl = maxSeqLen[(uint)(0u)];
  uint hs = hiddenSize[(uint)(0u)];
  uint totalOut = (bs * (msl * hs));
  bool inBounds = (gid < totalOut);
  if (inBounds) {
    uint seqHidden = (msl * hs);
    uint batchIdx = (gid / seqHidden);
    uint remainder = (gid % seqHidden);
    uint seqIdx = (remainder / hs);
    uint hiddenIdx = (remainder % hs);
    uint seqLen = seqLengths[batchIdx];
    bool isValid = (seqIdx < seqLen);
    if (isValid) {
      uint seqOffset = seqOffsets[batchIdx];
      uint inputIdx = ((seqOffset * hs) + ((seqIdx * hs) + hiddenIdx));
      half val = _input[inputIdx];
      _output[gid] = val;
    } else {
      _output[gid] = (half)(0.000000f);
    }
  }
}
