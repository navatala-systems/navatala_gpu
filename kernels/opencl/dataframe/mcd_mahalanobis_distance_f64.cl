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
__kernel void navatala_dataframe_mcd_mahalanobis_distance_f64(__global const double* dataX, __global const double* dataY, __global const double* robustMeanX, __global const double* robustMeanY, __global const double* robustCov, __global const double* varX, __global const double* varY, __global const uint* n, __global double* distances) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint len = n[0];
  bool inBounds = (gid < len);
  if (inBounds) {
    double mx = robustMeanX[0];
    double my = robustMeanY[0];
    double cov = robustCov[0];
    double vx = varX[0];
    double vy = varY[0];
    double x = dataX[gid];
    double y = dataY[gid];
    double dx = (x - mx);
    double dy = (y - my);
    double covSq = (cov * cov);
    double det = ((vx * vy) - covSq);
    double safeDet = (((det > as_double(0x3e112e0be826d695ul))) ? (det) : (as_double(0x3ff0000000000000ul)));
    double invDet = (as_double(0x3ff0000000000000ul) / safeDet);
    double dx2 = (dx * dx);
    double dy2 = (dy * dy);
    double dxdy = (dx * dy);
    double term1 = (dx2 * vy);
    double term2 = (dy2 * vx);
    double term3 = (as_double(0x4000000000000000ul) * (dxdy * cov));
    double num = ((term1 + term2) - term3);
    double dist2 = (num * invDet);
    distances[gid] = dist2;
  }
}
