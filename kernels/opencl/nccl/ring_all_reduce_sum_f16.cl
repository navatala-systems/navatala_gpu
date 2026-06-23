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
__kernel void navatala_nccl_ring_all_reduce_sum_f16(__global const half* sendbuf, __global const half* peer_buf, __global const uint* count, __global half* recvbuf) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint countVal = count[0];
  if (gid < countVal) {
    half myVal = sendbuf[gid];
    half peerVal = peer_buf[gid];
    half reduced = (myVal + peerVal);
    recvbuf[gid] = reduced;
  }
}
