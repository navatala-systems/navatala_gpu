// SPDX-License-Identifier: Apache-2.0
// Kernel registry partition for the `opencl_cfd` bucket. Compiles
// against the declarations in `navatala/_registry.hpp` and links
// against the trampoline in `_registry.cpp`. Partitioning keeps each
// translation unit well under the 50K-line threshold so a typical
// build can parallelise across the buckets.
#include "navatala/_registry.hpp"
#include <cstdint>
#include <string>
#include <string_view>

const char* k_opencl_navatala_cfd_boundary_force_partials = R"kernel(
__kernel void navatala_cfd_boundary_force_partials(__global const float* pAllFaces, __global const float* sfComponent, __global const int* counts, __global float* outPartials) {
  int gid0 = (int)get_global_id(0);
  const int nSafeMax = (((int)(counts[0])) > 0 ? ((int)(counts[0])) - 1 : 0);
  const int safeIdx = (gid0 < nSafeMax ? gid0 : nSafeMax);
  if (gid0 >= ((int)(counts[0]))) return;
  __local float tmp[256];
  float v = as_float(0x00000000u);
  if (((int)((int)(get_global_id(0)))) < counts[0]) {
    int faceIdx = (counts[1] + ((int)((int)(get_global_id(0)))));
    float pVal = pAllFaces[faceIdx];
    float sfVal = sfComponent[faceIdx];
    v = (pVal * sfVal);
  }
  tmp[((int)((int)(get_local_id(0))))] = v;
  barrier(CLK_LOCAL_MEM_FENCE);
  if (((int)((int)(get_local_id(0)))) < 128) {
    tmp[((int)((int)(get_local_id(0))))] = (tmp[((int)((int)(get_local_id(0))))] + tmp[(((int)((int)(get_local_id(0)))) + 128)]);
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  if (((int)((int)(get_local_id(0)))) < 64) {
    tmp[((int)((int)(get_local_id(0))))] = (tmp[((int)((int)(get_local_id(0))))] + tmp[(((int)((int)(get_local_id(0)))) + 64)]);
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  if (((int)((int)(get_local_id(0)))) < 32) {
    tmp[((int)((int)(get_local_id(0))))] = (tmp[((int)((int)(get_local_id(0))))] + tmp[(((int)((int)(get_local_id(0)))) + 32)]);
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  if (((int)((int)(get_local_id(0)))) < 16) {
    tmp[((int)((int)(get_local_id(0))))] = (tmp[((int)((int)(get_local_id(0))))] + tmp[(((int)((int)(get_local_id(0)))) + 16)]);
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  if (((int)((int)(get_local_id(0)))) < 8) {
    tmp[((int)((int)(get_local_id(0))))] = (tmp[((int)((int)(get_local_id(0))))] + tmp[(((int)((int)(get_local_id(0)))) + 8)]);
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  if (((int)((int)(get_local_id(0)))) < 4) {
    tmp[((int)((int)(get_local_id(0))))] = (tmp[((int)((int)(get_local_id(0))))] + tmp[(((int)((int)(get_local_id(0)))) + 4)]);
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  if (((int)((int)(get_local_id(0)))) < 2) {
    tmp[((int)((int)(get_local_id(0))))] = (tmp[((int)((int)(get_local_id(0))))] + tmp[(((int)((int)(get_local_id(0)))) + 2)]);
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  if (((int)((int)(get_local_id(0)))) < 1) {
    tmp[((int)((int)(get_local_id(0))))] = (tmp[((int)((int)(get_local_id(0))))] + tmp[(((int)((int)(get_local_id(0)))) + 1)]);
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  if (((int)((int)(get_local_id(0)))) == 0) {
    outPartials[((int)((int)(get_group_id(0))))] = tmp[0];
  }
}

)kernel";
const char* k_opencl_navatala_cfd_pack_owner_values = R"kernel(
__kernel void navatala_cfd_pack_owner_values(__global const float* x, __global const int* owner, __global const int* procFaces, __global const int* counts, __global float* outSend) {
  int gid0 = (int)get_global_id(0);
  const int nSafeMax = (((int)(counts[0])) > 0 ? ((int)(counts[0])) - 1 : 0);
  const int safeIdx = (gid0 < nSafeMax ? gid0 : nSafeMax);
  if (gid0 >= ((int)(counts[0]))) return;
  if ((int)(get_global_id(0)) >= counts[0]) {
    return;
  } else {
    int face = procFaces[(int)(get_global_id(0))];
    int o = owner[face];
    float v = x[o];
    outSend[(int)(get_global_id(0))] = v;
  }
}

)kernel";
const char* k_opencl_navatala_cfd_scatter_values_and_mask = R"kernel(
__kernel void navatala_cfd_scatter_values_and_mask(__global const int* procFaces, __global const float* recv, __global const int* counts, __global float* bcVal, __global int* bcMask) {
  int gid0 = (int)get_global_id(0);
  const int nSafeMax = (((int)(counts[0])) > 0 ? ((int)(counts[0])) - 1 : 0);
  const int safeIdx = (gid0 < nSafeMax ? gid0 : nSafeMax);
  if (gid0 >= ((int)(counts[0]))) return;
  if ((int)(get_global_id(0)) >= counts[0]) {
    return;
  } else {
    int face = procFaces[(int)(get_global_id(0))];
    bcVal[face] = recv[(int)(get_global_id(0))];
    bcMask[face] = 1;
  }
}

)kernel";
const char* k_opencl_navatala_cfd_pack_mu_grad_owner = R"kernel(
__kernel void navatala_cfd_pack_mu_grad_owner(__global const float* muCell, __global const float* gXX, __global const float* gXY, __global const float* gXZ, __global const float* gYX, __global const float* gYY, __global const float* gYZ, __global const float* gZX, __global const float* gZY, __global const float* gZZ, __global const int* owner, __global const int* procFaces, __global const int* counts, __global float* outSend) {
  int gid0 = (int)get_global_id(0);
  const int nSafeMax = (((int)(counts[0])) > 0 ? ((int)(counts[0])) - 1 : 0);
  const int safeIdx = (gid0 < nSafeMax ? gid0 : nSafeMax);
  if (gid0 >= ((int)(counts[0]))) return;
  if ((int)(get_global_id(0)) >= ((int)(counts[0]))) {
    return;
  } else {
    int f = procFaces[(int)(get_global_id(0))];
    int o = owner[f];
    int base = ((int)(get_global_id(0)) * 10);
    outSend[base] = muCell[o];
    int b1 = (base + 1);
    outSend[b1] = gXX[o];
    int b2 = (base + 2);
    outSend[b2] = gXY[o];
    int b3 = (base + 3);
    outSend[b3] = gXZ[o];
    int b4 = (base + 4);
    outSend[b4] = gYX[o];
    int b5 = (base + 5);
    outSend[b5] = gYY[o];
    int b6 = (base + 6);
    outSend[b6] = gYZ[o];
    int b7 = (base + 7);
    outSend[b7] = gZX[o];
    int b8 = (base + 8);
    outSend[b8] = gZY[o];
    int b9 = (base + 9);
    outSend[b9] = gZZ[o];
  }
}

)kernel";
const char* k_opencl_navatala_cfd_pack_owner_rho_vec3 = R"kernel(
__kernel void navatala_cfd_pack_owner_rho_vec3(__global const float* rhoCell, __global const float* ux, __global const float* uy, __global const float* uz, __global const int* owner, __global const int* procFaces, __global const int* counts, __global float* outSend) {
  int gid0 = (int)get_global_id(0);
  const int nSafeMax = (((int)(counts[0])) > 0 ? ((int)(counts[0])) - 1 : 0);
  const int safeIdx = (gid0 < nSafeMax ? gid0 : nSafeMax);
  if (gid0 >= ((int)(counts[0]))) return;
  if ((int)(get_global_id(0)) >= ((int)(counts[0]))) {
    return;
  } else {
    int f = procFaces[(int)(get_global_id(0))];
    int o = owner[f];
    int base = ((int)(get_global_id(0)) * 4);
    outSend[base] = rhoCell[o];
    int b1 = (base + 1);
    outSend[b1] = ux[o];
    int b2 = (base + 2);
    outSend[b2] = uy[o];
    int b3 = (base + 3);
    outSend[b3] = uz[o];
  }
}

)kernel";
const char* k_opencl_navatala_cfd_pack_owner_vec3 = R"kernel(
__kernel void navatala_cfd_pack_owner_vec3(__global const float* x, __global const float* y, __global const float* z, __global const int* owner, __global const int* procFaces, __global const int* counts, __global float* outSend) {
  int gid0 = (int)get_global_id(0);
  const int nSafeMax = (((int)(counts[0])) > 0 ? ((int)(counts[0])) - 1 : 0);
  const int safeIdx = (gid0 < nSafeMax ? gid0 : nSafeMax);
  if (gid0 >= ((int)(counts[0]))) return;
  if ((int)(get_global_id(0)) >= ((int)(counts[0]))) {
    return;
  } else {
    int f = procFaces[(int)(get_global_id(0))];
    int o = owner[f];
    int base = ((int)(get_global_id(0)) * 3);
    outSend[base] = x[o];
    int b1 = (base + 1);
    outSend[b1] = y[o];
    int b2 = (base + 2);
    outSend[b2] = z[o];
  }
}

)kernel";
const char* k_opencl_navatala_cfd_pack_owner_vec3_mu = R"kernel(
__kernel void navatala_cfd_pack_owner_vec3_mu(__global const float* ux, __global const float* uy, __global const float* uz, __global const float* muCell, __global const int* owner, __global const int* procFaces, __global const int* counts, __global float* outSend) {
  int gid0 = (int)get_global_id(0);
  const int nSafeMax = (((int)(counts[0])) > 0 ? ((int)(counts[0])) - 1 : 0);
  const int safeIdx = (gid0 < nSafeMax ? gid0 : nSafeMax);
  if (gid0 >= ((int)(counts[0]))) return;
  if ((int)(get_global_id(0)) >= ((int)(counts[0]))) {
    return;
  } else {
    int f = procFaces[(int)(get_global_id(0))];
    int o = owner[f];
    int base = ((int)(get_global_id(0)) * 4);
    outSend[base] = ux[o];
    int b1 = (base + 1);
    outSend[b1] = uy[o];
    int b2 = (base + 2);
    outSend[b2] = uz[o];
    int b3 = (base + 3);
    outSend[b3] = muCell[o];
  }
}

)kernel";
const char* k_opencl_navatala_cfd_scatter_mu_grad_ghost = R"kernel(
__kernel void navatala_cfd_scatter_mu_grad_ghost(__global const float* recvVals, __global const int* counts, __global float* muCell, __global float* gXX, __global float* gXY, __global float* gXZ, __global float* gYX, __global float* gYY, __global float* gYZ, __global float* gZX, __global float* gZY, __global float* gZZ) {
  int gid0 = (int)get_global_id(0);
  const int nSafeMax = (((int)(counts[0])) > 0 ? ((int)(counts[0])) - 1 : 0);
  const int safeIdx = (gid0 < nSafeMax ? gid0 : nSafeMax);
  if (gid0 >= ((int)(counts[0]))) return;
  if ((int)(get_global_id(0)) >= ((int)(counts[0]))) {
    return;
  } else {
    int idx = (((int)(counts[1])) + (int)(get_global_id(0)));
    int base = ((int)(get_global_id(0)) * 10);
    muCell[idx] = recvVals[base];
    int b1 = (base + 1);
    gXX[idx] = recvVals[b1];
    int b2 = (base + 2);
    gXY[idx] = recvVals[b2];
    int b3 = (base + 3);
    gXZ[idx] = recvVals[b3];
    int b4 = (base + 4);
    gYX[idx] = recvVals[b4];
    int b5 = (base + 5);
    gYY[idx] = recvVals[b5];
    int b6 = (base + 6);
    gYZ[idx] = recvVals[b6];
    int b7 = (base + 7);
    gZX[idx] = recvVals[b7];
    int b8 = (base + 8);
    gZY[idx] = recvVals[b8];
    int b9 = (base + 9);
    gZZ[idx] = recvVals[b9];
  }
}

)kernel";
const char* k_opencl_navatala_cfd_scatter_rho_vec3_and_mask = R"kernel(
__kernel void navatala_cfd_scatter_rho_vec3_and_mask(__global const int* procFaces, __global const float* recvVals, __global const int* counts, __global float* rhoBcVal, __global float* uBcX, __global float* uBcY, __global float* uBcZ, __global uint* rhoBcMask, __global uint* uBcMask) {
  int gid0 = (int)get_global_id(0);
  const int nSafeMax = (((int)(counts[0])) > 0 ? ((int)(counts[0])) - 1 : 0);
  const int safeIdx = (gid0 < nSafeMax ? gid0 : nSafeMax);
  if (gid0 >= ((int)(counts[0]))) return;
  if ((int)(get_global_id(0)) >= ((int)(counts[0]))) {
    return;
  } else {
    int f = procFaces[(int)(get_global_id(0))];
    int base = ((int)(get_global_id(0)) * 4);
    rhoBcVal[f] = recvVals[base];
    int b1 = (base + 1);
    uBcX[f] = recvVals[b1];
    int b2 = (base + 2);
    uBcY[f] = recvVals[b2];
    int b3 = (base + 3);
    uBcZ[f] = recvVals[b3];
    rhoBcMask[f] = ((uint)(1));
    uBcMask[f] = ((uint)(1));
  }
}

)kernel";
const char* k_opencl_navatala_cfd_scatter_vec3_and_mask = R"kernel(
__kernel void navatala_cfd_scatter_vec3_and_mask(__global const int* procFaces, __global const float* recvVals, __global const int* counts, __global float* bcX, __global float* bcY, __global float* bcZ, __global uint* bcMask) {
  int gid0 = (int)get_global_id(0);
  const int nSafeMax = (((int)(counts[0])) > 0 ? ((int)(counts[0])) - 1 : 0);
  const int safeIdx = (gid0 < nSafeMax ? gid0 : nSafeMax);
  if (gid0 >= ((int)(counts[0]))) return;
  if ((int)(get_global_id(0)) >= ((int)(counts[0]))) {
    return;
  } else {
    int f = procFaces[(int)(get_global_id(0))];
    int base = ((int)(get_global_id(0)) * 3);
    int b1 = (base + 1);
    int b2 = (base + 2);
    bcX[f] = recvVals[base];
    bcY[f] = recvVals[b1];
    bcZ[f] = recvVals[b2];
    bcMask[f] = ((uint)(1));
  }
}

)kernel";
const char* k_opencl_navatala_cfd_scatter_vec3_mu_and_mask = R"kernel(
__kernel void navatala_cfd_scatter_vec3_mu_and_mask(__global const int* procFaces, __global const float* recvVals, __global const int* counts, __global float* bcX, __global float* bcY, __global float* bcZ, __global float* muBc, __global uint* bcMask) {
  int gid0 = (int)get_global_id(0);
  const int nSafeMax = (((int)(counts[0])) > 0 ? ((int)(counts[0])) - 1 : 0);
  const int safeIdx = (gid0 < nSafeMax ? gid0 : nSafeMax);
  if (gid0 >= ((int)(counts[0]))) return;
  if ((int)(get_global_id(0)) >= ((int)(counts[0]))) {
    return;
  } else {
    int f = procFaces[(int)(get_global_id(0))];
    int base = ((int)(get_global_id(0)) * 4);
    bcX[f] = recvVals[base];
    int b1 = (base + 1);
    bcY[f] = recvVals[b1];
    int b2 = (base + 2);
    bcZ[f] = recvVals[b2];
    int b3 = (base + 3);
    muBc[f] = recvVals[b3];
    bcMask[f] = ((uint)(1));
  }
}

)kernel";
const char* k_opencl_navatala_cfd_cf_mesh_volume_optimizer_gradient_f32 = R"kernel(
__kernel void navatala_cfd_cf_mesh_volume_optimizer_gradient_f32(__global const float* pointX, __global const float* pointY, __global const float* pointZ, __global const int* tetConn, __global const int* tetOffsets, __global const int* tetCounts, __global const int* fixedMask, __global const float* kPerStencil, __global const int* counts, __global float* outGradX, __global float* outGradY, __global float* outGradZ) {
  int gid0 = (int)get_global_id(0);
  const int nSafeMax = (((int)(counts[0])) > 0 ? ((int)(counts[0])) - 1 : 0);
  const int safeIdx = (gid0 < nSafeMax ? gid0 : nSafeMax);
  if (gid0 >= ((int)(counts[0]))) return;
  if (((int)((int)(get_global_id(0)))) >= counts[0]) {
    return;
  } else {
    if (fixedMask[((int)((int)(get_global_id(0))))] != 0) {
      outGradX[((int)((int)(get_global_id(0))))] = as_float(0x00000000u);
      outGradY[((int)((int)(get_global_id(0))))] = as_float(0x00000000u);
      outGradZ[((int)((int)(get_global_id(0))))] = as_float(0x00000000u);
    } else {
      float k = kPerStencil[((int)((int)(get_global_id(0))))];
      float gxSum = as_float(0x00000000u);
      float gySum = as_float(0x00000000u);
      float gzSum = as_float(0x00000000u);
      int tetBeg = tetOffsets[((int)((int)(get_global_id(0))))];
      int tetN = tetCounts[((int)((int)(get_global_id(0))))];
      for (int t = 0; t < (int)(tetN); ++t) {
        int aId = tetConn[(((tetBeg + t) * 4) + 0)];
        int bId = tetConn[(((tetBeg + t) * 4) + 1)];
        int cId = tetConn[(((tetBeg + t) * 4) + 2)];
        int dId = tetConn[(((tetBeg + t) * 4) + 3)];
        float ax = pointX[aId];
        float ay = pointY[aId];
        float az = pointZ[aId];
        float bx = pointX[bId];
        float by = pointY[bId];
        float bz = pointZ[bId];
        float cx = pointX[cId];
        float cy = pointY[cId];
        float cz = pointZ[cId];
        float dx = pointX[dId];
        float dy = pointY[dId];
        float dz = pointZ[dId];
        float bax = (bx - ax);
        float bay = (by - ay);
        float baz = (bz - az);
        float cax = (cx - ax);
        float cay = (cy - ay);
        float caz = (cz - az);
        float dax = (dx - ax);
        float day = (dy - ay);
        float daz = (dz - az);
        float dbx = (dx - bx);
        float dby = (dy - by);
        float dbz = (dz - bz);
        float dcx = (dx - cx);
        float dcy = (dy - cy);
        float dcz = (dz - cz);
        float crossX = ((bay * caz) - (baz * cay));
        float crossY = ((baz * cax) - (bax * caz));
        float crossZ = ((bax * cay) - (bay * cax));
        float dotVol = (((dax * crossX) + (day * crossY)) + (daz * crossZ));
        float vtri = (dotVol / as_float(0x40c00000u));
        float lsqr = (((((dax * dax) + (day * day)) + (daz * daz)) + (((dbx * dbx) + (dby * dby)) + (dbz * dbz))) + (((dcx * dcx) + (dcy * dcy)) + (dcz * dcz)));
        float gradVx = (crossX / as_float(0x40c00000u));
        float gradVy = (crossY / as_float(0x40c00000u));
        float gradVz = (crossZ / as_float(0x40c00000u));
        float stab = sqrt(((vtri * vtri) + k));
        float vs = (((vtri < as_float(0x00000000u))) ? (((as_float(0x3f000000u) * k) / (stab - vtri))) : ((as_float(0x3f000000u) * (vtri + stab))));
        float gradStabX = ((vtri * gradVx) / stab);
        float gradStabY = ((vtri * gradVy) / stab);
        float gradStabZ = ((vtri * gradVz) / stab);
        float gradVsX = (((vtri < as_float(0x00000000u))) ? (((as_float(0xbf000000u) * (k * (gradStabX - gradVx))) / ((stab - vtri) * (stab - vtri)))) : ((as_float(0x3f000000u) * (gradVx + gradStabX))));
        float gradVsY = (((vtri < as_float(0x00000000u))) ? (((as_float(0xbf000000u) * (k * (gradStabY - gradVy))) / ((stab - vtri) * (stab - vtri)))) : ((as_float(0x3f000000u) * (gradVy + gradStabY))));
        float gradVsZ = (((vtri < as_float(0x00000000u))) ? (((as_float(0xbf000000u) * (k * (gradStabZ - gradVz))) / ((stab - vtri) * (stab - vtri)))) : ((as_float(0x3f000000u) * (gradVz + gradStabZ))));
        float gradLsqX = (as_float(0x40000000u) * ((as_float(0x40400000u) * dx) - ((ax + bx) + cx)));
        float gradLsqY = (as_float(0x40000000u) * ((as_float(0x40400000u) * dy) - ((ay + by) + cy)));
        float gradLsqZ = (as_float(0x40000000u) * ((as_float(0x40400000u) * dz) - ((az + bz) + cz)));
        float c0 = (as_float(0x3f2aaaabu) * pow(as_float(0x3f000000u), as_float(0x3f2aaaabu)));
        float vs13 = pow((as_float(0x40000000u) * vs), as_float(0x3eaaaaabu));
        float vstab = pow(vs, as_float(0x3f2aaaabu));
        float sqrVstab = (vstab * vstab);
        float gradVstabX = ((c0 * (as_float(0x40000000u) * gradVsX)) / vs13);
        float gradVstabY = ((c0 * (as_float(0x40000000u) * gradVsY)) / vs13);
        float gradVstabZ = ((c0 * (as_float(0x40000000u) * gradVsZ)) / vs13);
        float termLX = (gradLsqX / vstab);
        float termLY = (gradLsqY / vstab);
        float termLZ = (gradLsqZ / vstab);
        float termVX = ((lsqr * gradVstabX) / sqrVstab);
        float termVY = ((lsqr * gradVstabY) / sqrVstab);
        float termVZ = ((lsqr * gradVstabZ) / sqrVstab);
        float gx = (termLX - termVX);
        float gy = (termLY - termVY);
        float gz = (termLZ - termVZ);
        gxSum = (gxSum + gx);
        gySum = (gySum + gy);
        gzSum = (gzSum + gz);
      }
      outGradX[((int)((int)(get_global_id(0))))] = gxSum;
      outGradY[((int)((int)(get_global_id(0))))] = gySum;
      outGradZ[((int)((int)(get_global_id(0))))] = gzSum;
    }
  }
}

)kernel";
const char* k_opencl_navatala_cfd_cf_mesh_volume_optimizer_gradient_f64 = R"kernel(
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
__kernel void navatala_cfd_cf_mesh_volume_optimizer_gradient_f64(__global const double* pointX, __global const double* pointY, __global const double* pointZ, __global const int* tetConn, __global const int* tetOffsets, __global const int* tetCounts, __global const int* fixedMask, __global const double* kPerStencil, __global const int* counts, __global double* outGradX, __global double* outGradY, __global double* outGradZ) {
  int gid0 = (int)get_global_id(0);
  const int nSafeMax = (((int)(counts[0])) > 0 ? ((int)(counts[0])) - 1 : 0);
  const int safeIdx = (gid0 < nSafeMax ? gid0 : nSafeMax);
  if (gid0 >= ((int)(counts[0]))) return;
  if (((int)((int)(get_global_id(0)))) >= counts[0]) {
    return;
  } else {
    if (fixedMask[((int)((int)(get_global_id(0))))] != 0) {
      outGradX[((int)((int)(get_global_id(0))))] = as_double(0x0000000000000000ul);
      outGradY[((int)((int)(get_global_id(0))))] = as_double(0x0000000000000000ul);
      outGradZ[((int)((int)(get_global_id(0))))] = as_double(0x0000000000000000ul);
    } else {
      double k = kPerStencil[((int)((int)(get_global_id(0))))];
      double gxSum = as_double(0x0000000000000000ul);
      double gySum = as_double(0x0000000000000000ul);
      double gzSum = as_double(0x0000000000000000ul);
      int tetBeg = tetOffsets[((int)((int)(get_global_id(0))))];
      int tetN = tetCounts[((int)((int)(get_global_id(0))))];
      for (int t = 0; t < (int)(tetN); ++t) {
        int aId = tetConn[(((tetBeg + t) * 4) + 0)];
        int bId = tetConn[(((tetBeg + t) * 4) + 1)];
        int cId = tetConn[(((tetBeg + t) * 4) + 2)];
        int dId = tetConn[(((tetBeg + t) * 4) + 3)];
        double ax = pointX[aId];
        double ay = pointY[aId];
        double az = pointZ[aId];
        double bx = pointX[bId];
        double by = pointY[bId];
        double bz = pointZ[bId];
        double cx = pointX[cId];
        double cy = pointY[cId];
        double cz = pointZ[cId];
        double dx = pointX[dId];
        double dy = pointY[dId];
        double dz = pointZ[dId];
        double bax = (bx - ax);
        double bay = (by - ay);
        double baz = (bz - az);
        double cax = (cx - ax);
        double cay = (cy - ay);
        double caz = (cz - az);
        double dax = (dx - ax);
        double day = (dy - ay);
        double daz = (dz - az);
        double dbx = (dx - bx);
        double dby = (dy - by);
        double dbz = (dz - bz);
        double dcx = (dx - cx);
        double dcy = (dy - cy);
        double dcz = (dz - cz);
        double crossX = ((bay * caz) - (baz * cay));
        double crossY = ((baz * cax) - (bax * caz));
        double crossZ = ((bax * cay) - (bay * cax));
        double dotVol = (((dax * crossX) + (day * crossY)) + (daz * crossZ));
        double vtri = (dotVol / as_double(0x4018000000000000ul));
        double lsqr = (((((dax * dax) + (day * day)) + (daz * daz)) + (((dbx * dbx) + (dby * dby)) + (dbz * dbz))) + (((dcx * dcx) + (dcy * dcy)) + (dcz * dcz)));
        double gradVx = (crossX / as_double(0x4018000000000000ul));
        double gradVy = (crossY / as_double(0x4018000000000000ul));
        double gradVz = (crossZ / as_double(0x4018000000000000ul));
        double stab = sqrt(((vtri * vtri) + k));
        double vs = (((vtri < as_double(0x0000000000000000ul))) ? (((as_double(0x3fe0000000000000ul) * k) / (stab - vtri))) : ((as_double(0x3fe0000000000000ul) * (vtri + stab))));
        double gradStabX = ((vtri * gradVx) / stab);
        double gradStabY = ((vtri * gradVy) / stab);
        double gradStabZ = ((vtri * gradVz) / stab);
        double gradVsX = (((vtri < as_double(0x0000000000000000ul))) ? (((as_double(0xbfe0000000000000ul) * (k * (gradStabX - gradVx))) / ((stab - vtri) * (stab - vtri)))) : ((as_double(0x3fe0000000000000ul) * (gradVx + gradStabX))));
        double gradVsY = (((vtri < as_double(0x0000000000000000ul))) ? (((as_double(0xbfe0000000000000ul) * (k * (gradStabY - gradVy))) / ((stab - vtri) * (stab - vtri)))) : ((as_double(0x3fe0000000000000ul) * (gradVy + gradStabY))));
        double gradVsZ = (((vtri < as_double(0x0000000000000000ul))) ? (((as_double(0xbfe0000000000000ul) * (k * (gradStabZ - gradVz))) / ((stab - vtri) * (stab - vtri)))) : ((as_double(0x3fe0000000000000ul) * (gradVz + gradStabZ))));
        double gradLsqX = (as_double(0x4000000000000000ul) * ((as_double(0x4008000000000000ul) * dx) - ((ax + bx) + cx)));
        double gradLsqY = (as_double(0x4000000000000000ul) * ((as_double(0x4008000000000000ul) * dy) - ((ay + by) + cy)));
        double gradLsqZ = (as_double(0x4000000000000000ul) * ((as_double(0x4008000000000000ul) * dz) - ((az + bz) + cz)));
        double c0 = (as_double(0x3fe5555555555555ul) * pow(as_double(0x3fe0000000000000ul), as_double(0x3fe5555555555555ul)));
        double vs13 = pow((as_double(0x4000000000000000ul) * vs), as_double(0x3fd5555555555555ul));
        double vstab = pow(vs, as_double(0x3fe5555555555555ul));
        double sqrVstab = (vstab * vstab);
        double gradVstabX = ((c0 * (as_double(0x4000000000000000ul) * gradVsX)) / vs13);
        double gradVstabY = ((c0 * (as_double(0x4000000000000000ul) * gradVsY)) / vs13);
        double gradVstabZ = ((c0 * (as_double(0x4000000000000000ul) * gradVsZ)) / vs13);
        double termLX = (gradLsqX / vstab);
        double termLY = (gradLsqY / vstab);
        double termLZ = (gradLsqZ / vstab);
        double termVX = ((lsqr * gradVstabX) / sqrVstab);
        double termVY = ((lsqr * gradVstabY) / sqrVstab);
        double termVZ = ((lsqr * gradVstabZ) / sqrVstab);
        double gx = (termLX - termVX);
        double gy = (termLY - termVY);
        double gz = (termLZ - termVZ);
        gxSum = (gxSum + gx);
        gySum = (gySum + gy);
        gzSum = (gzSum + gz);
      }
      outGradX[((int)((int)(get_global_id(0))))] = gxSum;
      outGradY[((int)((int)(get_global_id(0))))] = gySum;
      outGradZ[((int)((int)(get_global_id(0))))] = gzSum;
    }
  }
}

)kernel";
const char* k_opencl_navatala_cfd_cf_mesh_volume_optimizer_k_prepass_f32 = R"kernel(
__kernel void navatala_cfd_cf_mesh_volume_optimizer_k_prepass_f32(__global const float* pointX, __global const float* pointY, __global const float* pointZ, __global const int* tetConn, __global const int* tetOffsets, __global const int* tetCounts, __global const int* fixedMask, __global const float* params, __global const int* counts, __global float* outK) {
  int gid0 = (int)get_global_id(0);
  const int nSafeMax = (((int)(counts[0])) > 0 ? ((int)(counts[0])) - 1 : 0);
  const int safeIdx = (gid0 < nSafeMax ? gid0 : nSafeMax);
  if (gid0 >= ((int)(counts[0]))) return;
  if (((int)((int)(get_global_id(0)))) >= counts[0]) {
    return;
  } else {
    if (fixedMask[((int)((int)(get_global_id(0))))] != 0) {
      outK[((int)((int)(get_global_id(0))))] = as_float(0x00000000u);
    } else {
      float small = params[0];
      float vMin = as_float(0x7149f2cau);
      float lSqrMax = as_float(0x00000000u);
      int tetBeg = tetOffsets[((int)((int)(get_global_id(0))))];
      int tetN = tetCounts[((int)((int)(get_global_id(0))))];
      for (int t = 0; t < (int)(tetN); ++t) {
        int aId = tetConn[(((tetBeg + t) * 4) + 0)];
        int bId = tetConn[(((tetBeg + t) * 4) + 1)];
        int cId = tetConn[(((tetBeg + t) * 4) + 2)];
        int dId = tetConn[(((tetBeg + t) * 4) + 3)];
        float ax = pointX[aId];
        float ay = pointY[aId];
        float az = pointZ[aId];
        float bx = pointX[bId];
        float by = pointY[bId];
        float bz = pointZ[bId];
        float cx = pointX[cId];
        float cy = pointY[cId];
        float cz = pointZ[cId];
        float dx = pointX[dId];
        float dy = pointY[dId];
        float dz = pointZ[dId];
        float bax = (bx - ax);
        float bay = (by - ay);
        float baz = (bz - az);
        float cax = (cx - ax);
        float cay = (cy - ay);
        float caz = (cz - az);
        float dax = (dx - ax);
        float day = (dy - ay);
        float daz = (dz - az);
        float dbx = (dx - bx);
        float dby = (dy - by);
        float dbz = (dz - bz);
        float dcx = (dx - cx);
        float dcy = (dy - cy);
        float dcz = (dz - cz);
        float crossX = ((bay * caz) - (baz * cay));
        float crossY = ((baz * cax) - (bax * caz));
        float crossZ = ((bax * cay) - (bay * cax));
        float dotVol = (((dax * crossX) + (day * crossY)) + (daz * crossZ));
        float vtri = (dotVol / as_float(0x40c00000u));
        float lsqr = (((((dax * dax) + (day * day)) + (daz * daz)) + (((dbx * dbx) + (dby * dby)) + (dbz * dbz))) + (((dcx * dcx) + (dcy * dcy)) + (dcz * dcz)));
        vMin = (((vtri < vMin)) ? (vtri) : (vMin));
        lSqrMax = (((lsqr > lSqrMax)) ? (lsqr) : (lSqrMax));
      }
      float threshold = (small * lSqrMax);
      float kVal = (((vMin < threshold)) ? (threshold) : (as_float(0x00000000u)));
      outK[((int)((int)(get_global_id(0))))] = kVal;
    }
  }
}

)kernel";
const char* k_opencl_navatala_cfd_cf_mesh_volume_optimizer_k_prepass_f64 = R"kernel(
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
__kernel void navatala_cfd_cf_mesh_volume_optimizer_k_prepass_f64(__global const double* pointX, __global const double* pointY, __global const double* pointZ, __global const int* tetConn, __global const int* tetOffsets, __global const int* tetCounts, __global const int* fixedMask, __global const double* params, __global const int* counts, __global double* outK) {
  int gid0 = (int)get_global_id(0);
  const int nSafeMax = (((int)(counts[0])) > 0 ? ((int)(counts[0])) - 1 : 0);
  const int safeIdx = (gid0 < nSafeMax ? gid0 : nSafeMax);
  if (gid0 >= ((int)(counts[0]))) return;
  if (((int)((int)(get_global_id(0)))) >= counts[0]) {
    return;
  } else {
    if (fixedMask[((int)((int)(get_global_id(0))))] != 0) {
      outK[((int)((int)(get_global_id(0))))] = as_double(0x0000000000000000ul);
    } else {
      double small = params[0];
      double vMin = as_double(0x7e37e43c8800759cul);
      double lSqrMax = as_double(0x0000000000000000ul);
      int tetBeg = tetOffsets[((int)((int)(get_global_id(0))))];
      int tetN = tetCounts[((int)((int)(get_global_id(0))))];
      for (int t = 0; t < (int)(tetN); ++t) {
        int aId = tetConn[(((tetBeg + t) * 4) + 0)];
        int bId = tetConn[(((tetBeg + t) * 4) + 1)];
        int cId = tetConn[(((tetBeg + t) * 4) + 2)];
        int dId = tetConn[(((tetBeg + t) * 4) + 3)];
        double ax = pointX[aId];
        double ay = pointY[aId];
        double az = pointZ[aId];
        double bx = pointX[bId];
        double by = pointY[bId];
        double bz = pointZ[bId];
        double cx = pointX[cId];
        double cy = pointY[cId];
        double cz = pointZ[cId];
        double dx = pointX[dId];
        double dy = pointY[dId];
        double dz = pointZ[dId];
        double bax = (bx - ax);
        double bay = (by - ay);
        double baz = (bz - az);
        double cax = (cx - ax);
        double cay = (cy - ay);
        double caz = (cz - az);
        double dax = (dx - ax);
        double day = (dy - ay);
        double daz = (dz - az);
        double dbx = (dx - bx);
        double dby = (dy - by);
        double dbz = (dz - bz);
        double dcx = (dx - cx);
        double dcy = (dy - cy);
        double dcz = (dz - cz);
        double crossX = ((bay * caz) - (baz * cay));
        double crossY = ((baz * cax) - (bax * caz));
        double crossZ = ((bax * cay) - (bay * cax));
        double dotVol = (((dax * crossX) + (day * crossY)) + (daz * crossZ));
        double vtri = (dotVol / as_double(0x4018000000000000ul));
        double lsqr = (((((dax * dax) + (day * day)) + (daz * daz)) + (((dbx * dbx) + (dby * dby)) + (dbz * dbz))) + (((dcx * dcx) + (dcy * dcy)) + (dcz * dcz)));
        vMin = (((vtri < vMin)) ? (vtri) : (vMin));
        lSqrMax = (((lsqr > lSqrMax)) ? (lsqr) : (lSqrMax));
      }
      double threshold = (small * lSqrMax);
      double kVal = (((vMin < threshold)) ? (threshold) : (as_double(0x0000000000000000ul)));
      outK[((int)((int)(get_global_id(0))))] = kVal;
    }
  }
}

)kernel";
const char* k_opencl_navatala_cfd_cf_mesh_volume_optimizer_objective_f32 = R"kernel(
__kernel void navatala_cfd_cf_mesh_volume_optimizer_objective_f32(__global const float* pointX, __global const float* pointY, __global const float* pointZ, __global const int* tetConn, __global const int* tetOffsets, __global const int* tetCounts, __global const int* fixedMask, __global const float* kPerStencil, __global const int* counts, __global float* outObjective) {
  int gid0 = (int)get_global_id(0);
  const int nSafeMax = (((int)(counts[0])) > 0 ? ((int)(counts[0])) - 1 : 0);
  const int safeIdx = (gid0 < nSafeMax ? gid0 : nSafeMax);
  if (gid0 >= ((int)(counts[0]))) return;
  if (((int)((int)(get_global_id(0)))) >= counts[0]) {
    return;
  } else {
    if (fixedMask[((int)((int)(get_global_id(0))))] != 0) {
      outObjective[((int)((int)(get_global_id(0))))] = as_float(0x00000000u);
    } else {
      float k = kPerStencil[((int)((int)(get_global_id(0))))];
      float sum = as_float(0x00000000u);
      int tetBeg = tetOffsets[((int)((int)(get_global_id(0))))];
      int tetN = tetCounts[((int)((int)(get_global_id(0))))];
      for (int t = 0; t < (int)(tetN); ++t) {
        int aId = tetConn[(((tetBeg + t) * 4) + 0)];
        int bId = tetConn[(((tetBeg + t) * 4) + 1)];
        int cId = tetConn[(((tetBeg + t) * 4) + 2)];
        int dId = tetConn[(((tetBeg + t) * 4) + 3)];
        float ax = pointX[aId];
        float ay = pointY[aId];
        float az = pointZ[aId];
        float bx = pointX[bId];
        float by = pointY[bId];
        float bz = pointZ[bId];
        float cx = pointX[cId];
        float cy = pointY[cId];
        float cz = pointZ[cId];
        float dx = pointX[dId];
        float dy = pointY[dId];
        float dz = pointZ[dId];
        float bax = (bx - ax);
        float bay = (by - ay);
        float baz = (bz - az);
        float cax = (cx - ax);
        float cay = (cy - ay);
        float caz = (cz - az);
        float dax = (dx - ax);
        float day = (dy - ay);
        float daz = (dz - az);
        float dbx = (dx - bx);
        float dby = (dy - by);
        float dbz = (dz - bz);
        float dcx = (dx - cx);
        float dcy = (dy - cy);
        float dcz = (dz - cz);
        float crossX = ((bay * caz) - (baz * cay));
        float crossY = ((baz * cax) - (bax * caz));
        float crossZ = ((bax * cay) - (bay * cax));
        float dotVol = (((dax * crossX) + (day * crossY)) + (daz * crossZ));
        float vtri = (dotVol / as_float(0x40c00000u));
        float lsqr = (((((dax * dax) + (day * day)) + (daz * daz)) + (((dbx * dbx) + (dby * dby)) + (dbz * dbz))) + (((dcx * dcx) + (dcy * dcy)) + (dcz * dcz)));
        float sqrtArg = ((vtri * vtri) + k);
        float stab = sqrt(sqrtArg);
        float vstab = (((vtri < as_float(0x00000000u))) ? (((as_float(0x3f000000u) * k) / (stab - vtri))) : ((as_float(0x3f000000u) * (vtri + stab))));
        float denom = pow(vstab, as_float(0x3f2aaaabu));
        float contrib = (lsqr / denom);
        sum = (sum + contrib);
      }
      outObjective[((int)((int)(get_global_id(0))))] = sum;
    }
  }
}

)kernel";
const char* k_opencl_navatala_cfd_cf_mesh_volume_optimizer_objective_f64 = R"kernel(
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
__kernel void navatala_cfd_cf_mesh_volume_optimizer_objective_f64(__global const double* pointX, __global const double* pointY, __global const double* pointZ, __global const int* tetConn, __global const int* tetOffsets, __global const int* tetCounts, __global const int* fixedMask, __global const double* kPerStencil, __global const int* counts, __global double* outObjective) {
  int gid0 = (int)get_global_id(0);
  const int nSafeMax = (((int)(counts[0])) > 0 ? ((int)(counts[0])) - 1 : 0);
  const int safeIdx = (gid0 < nSafeMax ? gid0 : nSafeMax);
  if (gid0 >= ((int)(counts[0]))) return;
  if (((int)((int)(get_global_id(0)))) >= counts[0]) {
    return;
  } else {
    if (fixedMask[((int)((int)(get_global_id(0))))] != 0) {
      outObjective[((int)((int)(get_global_id(0))))] = as_double(0x0000000000000000ul);
    } else {
      double k = kPerStencil[((int)((int)(get_global_id(0))))];
      double sum = as_double(0x0000000000000000ul);
      int tetBeg = tetOffsets[((int)((int)(get_global_id(0))))];
      int tetN = tetCounts[((int)((int)(get_global_id(0))))];
      for (int t = 0; t < (int)(tetN); ++t) {
        int aId = tetConn[(((tetBeg + t) * 4) + 0)];
        int bId = tetConn[(((tetBeg + t) * 4) + 1)];
        int cId = tetConn[(((tetBeg + t) * 4) + 2)];
        int dId = tetConn[(((tetBeg + t) * 4) + 3)];
        double ax = pointX[aId];
        double ay = pointY[aId];
        double az = pointZ[aId];
        double bx = pointX[bId];
        double by = pointY[bId];
        double bz = pointZ[bId];
        double cx = pointX[cId];
        double cy = pointY[cId];
        double cz = pointZ[cId];
        double dx = pointX[dId];
        double dy = pointY[dId];
        double dz = pointZ[dId];
        double bax = (bx - ax);
        double bay = (by - ay);
        double baz = (bz - az);
        double cax = (cx - ax);
        double cay = (cy - ay);
        double caz = (cz - az);
        double dax = (dx - ax);
        double day = (dy - ay);
        double daz = (dz - az);
        double dbx = (dx - bx);
        double dby = (dy - by);
        double dbz = (dz - bz);
        double dcx = (dx - cx);
        double dcy = (dy - cy);
        double dcz = (dz - cz);
        double crossX = ((bay * caz) - (baz * cay));
        double crossY = ((baz * cax) - (bax * caz));
        double crossZ = ((bax * cay) - (bay * cax));
        double dotVol = (((dax * crossX) + (day * crossY)) + (daz * crossZ));
        double vtri = (dotVol / as_double(0x4018000000000000ul));
        double lsqr = (((((dax * dax) + (day * day)) + (daz * daz)) + (((dbx * dbx) + (dby * dby)) + (dbz * dbz))) + (((dcx * dcx) + (dcy * dcy)) + (dcz * dcz)));
        double sqrtArg = ((vtri * vtri) + k);
        double stab = sqrt(sqrtArg);
        double vstab = (((vtri < as_double(0x0000000000000000ul))) ? (((as_double(0x3fe0000000000000ul) * k) / (stab - vtri))) : ((as_double(0x3fe0000000000000ul) * (vtri + stab))));
        double denom = pow(vstab, as_double(0x3fe5555555555555ul));
        double contrib = (lsqr / denom);
        sum = (sum + contrib);
      }
      outObjective[((int)((int)(get_global_id(0))))] = sum;
    }
  }
}

)kernel";
const char* k_opencl_navatala_cfd_add_vol_vector = R"kernel(
__kernel void navatala_cfd_add_vol_vector(__global const float* ax, __global const float* ay, __global const float* az, __global const float* bx, __global const float* by_, __global const float* bz, __global const int* params, __global float* outX, __global float* outY, __global float* outZ) {
  int gid0 = (int)get_global_id(0);
  if (((int)((int)(get_global_id(0)))) >= params[0]) {
    return;
  } else {
    outX[((int)((int)(get_global_id(0))))] = (ax[((int)((int)(get_global_id(0))))] + bx[((int)((int)(get_global_id(0))))]);
    outY[((int)((int)(get_global_id(0))))] = (ay[((int)((int)(get_global_id(0))))] + by_[((int)((int)(get_global_id(0))))]);
    outZ[((int)((int)(get_global_id(0))))] = (az[((int)((int)(get_global_id(0))))] + bz[((int)((int)(get_global_id(0))))]);
  }
}

)kernel";
const char* k_opencl_navatala_cfd_div_rho_phi_u_explicit = R"kernel(
__kernel void navatala_cfd_div_rho_phi_u_explicit(__global const float* phi, __global const float* rhoCell, __global const float* ux, __global const float* uy, __global const float* uz, __global const int* owner, __global const int* neighbour, __global const float* weights, __global const int* offsets, __global const int* faceIdx, __global const float* sign, __global const float* vol, __global const float* rhoBcVal, __global const uint* rhoBcMask, __global const float* uBcX, __global const float* uBcY, __global const float* uBcZ, __global const uint* uBcMask, __global const int* params, __global float* outX, __global float* outY, __global float* outZ) {
  int gid0 = (int)get_global_id(0);
  if (((int)((int)(get_global_id(0)))) >= params[0]) {
    return;
  } else {
    float sumX = as_float(0x00000000u);
    float sumY = as_float(0x00000000u);
    float sumZ = as_float(0x00000000u);
    int beg = offsets[((int)((int)(get_global_id(0))))];
    int c1 = (((int)((int)(get_global_id(0)))) + 1);
    int end = offsets[c1];
    int len = (end - beg);
    for (int t = 0; t < (int)(len); ++t) {
      int k = (beg + t);
      int f = faceIdx[k];
      if (f < params[1]) {
        float s = sign[k];
        float flux = phi[f];
        int o = owner[f];
        float rhoF = rhoCell[o];
        float ufx = ux[o];
        float ufy = uy[o];
        float ufz = uz[o];
        if (f < params[2]) {
          int n = neighbour[f];
          float w0 = weights[f];
          float iw = (as_float(0x3f800000u) - w0);
          rhoF = ((w0 * rhoCell[o]) + (iw * rhoCell[n]));
          ufx = ((w0 * ux[o]) + (iw * ux[n]));
          ufy = ((w0 * uy[o]) + (iw * uy[n]));
          ufz = ((w0 * uz[o]) + (iw * uz[n]));
        } else {
          uint rm = rhoBcMask[f];
          if (rm == (uint)(1u)) {
            float w0 = weights[f];
            float iw = (as_float(0x3f800000u) - w0);
            rhoF = ((w0 * rhoCell[o]) + (iw * rhoBcVal[f]));
          }
          if (rm == (uint)(2u)) {
            rhoF = rhoBcVal[f];
          }
          uint um = uBcMask[f];
          if (um == (uint)(1u)) {
            float w0 = weights[f];
            float iw = (as_float(0x3f800000u) - w0);
            ufx = ((w0 * ux[o]) + (iw * uBcX[f]));
            ufy = ((w0 * uy[o]) + (iw * uBcY[f]));
            ufz = ((w0 * uz[o]) + (iw * uBcZ[f]));
          }
          if (um == (uint)(2u)) {
            ufx = uBcX[f];
            ufy = uBcY[f];
            ufz = uBcZ[f];
          }
        }
        float rhoPhi = (rhoF * flux);
        sumX = (sumX + (s * (rhoPhi * ufx)));
        sumY = (sumY + (s * (rhoPhi * ufy)));
        sumZ = (sumZ + (s * (rhoPhi * ufz)));
      }
    }
    float v = vol[((int)((int)(get_global_id(0))))];
    float invV = as_float(0x00000000u);
    if (v != as_float(0x00000000u)) {
      invV = (as_float(0x3f800000u) / v);
    }
    outX[((int)((int)(get_global_id(0))))] = (sumX * invV);
    outY[((int)((int)(get_global_id(0))))] = (sumY * invV);
    outZ[((int)((int)(get_global_id(0))))] = (sumZ * invV);
  }
}

)kernel";
const char* k_opencl_navatala_cfd_hby_a_from_u_and_terms = R"kernel(
__kernel void navatala_cfd_hby_a_from_u_and_terms(__global const float* ux, __global const float* uy, __global const float* uz, __global const float* lapx, __global const float* lapy, __global const float* lapz, __global const float* divx, __global const float* divy, __global const float* divz, __global const float* extrax, __global const float* extray, __global const float* extraz, __global const float* rau, __global const int* counts, __global const int* modeBuf, __global float* outx, __global float* outy, __global float* outz) {
  int gid0 = (int)get_global_id(0);
  const int nSafeMax = (((int)(counts[0])) > 0 ? ((int)(counts[0])) - 1 : 0);
  const int safeIdx = (gid0 < nSafeMax ? gid0 : nSafeMax);
  if (gid0 >= ((int)(counts[0]))) return;
  if (((int)((int)(get_global_id(0)))) >= counts[0]) {
    return;
  } else {
    float tx = lapx[((int)((int)(get_global_id(0))))];
    float ty = lapy[((int)((int)(get_global_id(0))))];
    float tz = lapz[((int)((int)(get_global_id(0))))];
    if ((modeBuf[0] == 1) || (modeBuf[0] == 3)) {
      tx = (tx - divx[((int)((int)(get_global_id(0))))]);
      ty = (ty - divy[((int)((int)(get_global_id(0))))]);
      tz = (tz - divz[((int)((int)(get_global_id(0))))]);
    }
    if ((modeBuf[0] == 2) || (modeBuf[0] == 3)) {
      tx = (tx + extrax[((int)((int)(get_global_id(0))))]);
      ty = (ty + extray[((int)((int)(get_global_id(0))))]);
      tz = (tz + extraz[((int)((int)(get_global_id(0))))]);
    }
    float a = rau[((int)((int)(get_global_id(0))))];
    outx[((int)((int)(get_global_id(0))))] = (ux[((int)((int)(get_global_id(0))))] + (a * tx));
    outy[((int)((int)(get_global_id(0))))] = (uy[((int)((int)(get_global_id(0))))] + (a * ty));
    outz[((int)((int)(get_global_id(0))))] = (uz[((int)((int)(get_global_id(0))))] + (a * tz));
  }
}

)kernel";
const char* k_opencl_navatala_cfd_apply_sp_to_hby_a = R"kernel(
__kernel void navatala_cfd_apply_sp_to_hby_a(__global float* hx, __global float* hy, __global float* hz, __global const float* ux, __global const float* uy, __global const float* uz, __global const float* rAU, __global const float* sp, __global const int* counts) {
  int gid0 = (int)get_global_id(0);
  const int nSafeMax = (((int)(counts[0])) > 0 ? ((int)(counts[0])) - 1 : 0);
  const int safeIdx = (gid0 < nSafeMax ? gid0 : nSafeMax);
  if (gid0 >= ((int)(counts[0]))) return;
  if (((int)((int)(get_global_id(0)))) >= counts[0]) {
    return;
  } else {
    float fac = (sp[((int)((int)(get_global_id(0))))] * rAU[((int)((int)(get_global_id(0))))]);
    hx[((int)((int)(get_global_id(0))))] = (hx[((int)((int)(get_global_id(0))))] - (fac * ux[((int)((int)(get_global_id(0))))]));
    hy[((int)((int)(get_global_id(0))))] = (hy[((int)((int)(get_global_id(0))))] - (fac * uy[((int)((int)(get_global_id(0))))]));
    hz[((int)((int)(get_global_id(0))))] = (hz[((int)((int)(get_global_id(0))))] - (fac * uz[((int)((int)(get_global_id(0))))]));
  }
}

)kernel";
const char* k_opencl_navatala_cfd_laplacian_scalar_explicit = R"kernel(
__kernel void navatala_cfd_laplacian_scalar_explicit(__global const float* muCell, __global const float* phiCell, __global const int* owner, __global const int* neighbour, __global const float* weights, __global const float* magSf, __global const float* deltaCoeffs, __global const int* offsets, __global const int* faceIdx, __global const int* sign, __global const float* vol, __global const float* bcMu, __global const uint* bcMuMask, __global const float* bcPhi, __global const uint* bcPhiMask, __global const int* params, __global float* outL) {
  int gid0 = (int)get_global_id(0);
  if (((int)((int)(get_global_id(0)))) >= params[0]) {
    return;
  } else {
    float phiC = phiCell[((int)((int)(get_global_id(0))))];
    float muC = muCell[((int)((int)(get_global_id(0))))];
    float sum = as_float(0x00000000u);
    int beg = offsets[((int)((int)(get_global_id(0))))];
    int c1 = (((int)((int)(get_global_id(0)))) + 1);
    int end = offsets[c1];
    int len = (end - beg);
    for (int t = 0; t < (int)(len); ++t) {
      int k = (beg + t);
      int f = faceIdx[k];
      if (f < params[1]) {
        int sgn = sign[k];
        int o = owner[f];
        float phiOther = phiCell[o];
        float muF = muC;
        if (sgn > 0) {
          if (f < params[2]) {
            int n = neighbour[f];
            phiOther = phiCell[n];
            float w0 = weights[f];
            float iw = (as_float(0x3f800000u) - w0);
            muF = ((w0 * muC) + (iw * muCell[n]));
          } else {
            if (bcPhiMask[f] != (uint)(0u)) {
              phiOther = bcPhi[f];
            }
            if (bcMuMask[f] != (uint)(0u)) {
              muF = bcMu[f];
            }
          }
        } else {
          float w0 = weights[f];
          float iw = (as_float(0x3f800000u) - w0);
          muF = ((w0 * muCell[o]) + (iw * muC));
        }
        float coeff = (muF * (magSf[f] * deltaCoeffs[f]));
        sum = (sum + (coeff * (phiOther - phiC)));
      }
    }
    float v = vol[((int)((int)(get_global_id(0))))];
    float invV = as_float(0x00000000u);
    if (v != as_float(0x00000000u)) {
      invV = (as_float(0x3f800000u) / v);
    }
    outL[((int)((int)(get_global_id(0))))] = (sum * invV);
  }
}

)kernel";
const char* k_opencl_navatala_cfd_laplacian_vector_explicit = R"kernel(
__kernel void navatala_cfd_laplacian_vector_explicit(__global const float* muCell, __global const float* muBc, __global const float* ux, __global const float* uy, __global const float* uz, __global const int* owner, __global const int* neighbour, __global const float* weights, __global const float* magSf, __global const float* deltaCoeffs, __global const float* corrX, __global const float* corrY, __global const float* corrZ, __global const float* gradRow0, __global const float* gradRow1, __global const float* gradRow2, __global const int* offsets, __global const int* faceIdx, __global const int* sign, __global const float* vol, __global const float* bcX, __global const float* bcY, __global const float* bcZ, __global const int* bcMask, __global const int* params, __global float* outX, __global float* outY, __global float* outZ) {
  int gid0 = (int)get_global_id(0);
  if (((int)((int)(get_global_id(0)))) >= params[0]) {
    return;
  } else {
    float ucx = ux[((int)((int)(get_global_id(0))))];
    float ucy = uy[((int)((int)(get_global_id(0))))];
    float ucz = uz[((int)((int)(get_global_id(0))))];
    float muC = muCell[((int)((int)(get_global_id(0))))];
    float sumX = as_float(0x00000000u);
    float sumY = as_float(0x00000000u);
    float sumZ = as_float(0x00000000u);
    int beg = offsets[((int)((int)(get_global_id(0))))];
    int c1 = (((int)((int)(get_global_id(0)))) + 1);
    int end = offsets[c1];
    int len = (end - beg);
    for (int t = 0; t < (int)(len); ++t) {
      int k = (beg + t);
      int f = faceIdx[k];
      if (f < params[1]) {
        int sgn = sign[k];
        int o = owner[f];
        float otherX = ux[o];
        float otherY = uy[o];
        float otherZ = uz[o];
        float muF = muC;
        if (sgn > 0) {
          if (f < params[2]) {
            int n = neighbour[f];
            otherX = ux[n];
            otherY = uy[n];
            otherZ = uz[n];
            float w0 = weights[f];
            float iw = (as_float(0x3f800000u) - w0);
            muF = ((w0 * muC) + (iw * muCell[n]));
          } else {
            muF = muBc[f];
            if (bcMask[f] == 1) {
              float w0 = weights[f];
              float iw = (as_float(0x3f800000u) - w0);
              muF = ((w0 * muC) + (iw * muBc[f]));
              otherX = bcX[f];
              otherY = bcY[f];
              otherZ = bcZ[f];
            } else {
              if (bcMask[f] != 0) {
                otherX = bcX[f];
                otherY = bcY[f];
                otherZ = bcZ[f];
              }
            }
          }
        } else {
          float w0 = weights[f];
          float iw = (as_float(0x3f800000u) - w0);
          muF = ((w0 * muCell[o]) + (iw * muC));
        }
        float coeff = (muF * (magSf[f] * deltaCoeffs[f]));
        sumX = (sumX + (coeff * (otherX - ucx)));
        sumY = (sumY + (coeff * (otherY - ucy)));
        sumZ = (sumZ + (coeff * (otherZ - ucz)));
        float kx = corrX[f];
        float ky = corrY[f];
        float kz = corrZ[f];
        float gxxF = gradRow0[((o)*3 + 0)];
        float gxyF = gradRow0[((o)*3 + 1)];
        float gxzF = gradRow0[((o)*3 + 2)];
        float gyxF = gradRow1[((o)*3 + 0)];
        float gyyF = gradRow1[((o)*3 + 1)];
        float gyzF = gradRow1[((o)*3 + 2)];
        float gzxF = gradRow2[((o)*3 + 0)];
        float gzyF = gradRow2[((o)*3 + 1)];
        float gzzF = gradRow2[((o)*3 + 2)];
        if (f < params[2]) {
          int n2 = neighbour[f];
          float w1 = weights[f];
          float iw1 = (as_float(0x3f800000u) - w1);
          gxxF = ((w1 * gradRow0[((o)*3 + 0)]) + (iw1 * gradRow0[((n2)*3 + 0)]));
          gxyF = ((w1 * gradRow0[((o)*3 + 1)]) + (iw1 * gradRow0[((n2)*3 + 1)]));
          gxzF = ((w1 * gradRow0[((o)*3 + 2)]) + (iw1 * gradRow0[((n2)*3 + 2)]));
          gyxF = ((w1 * gradRow1[((o)*3 + 0)]) + (iw1 * gradRow1[((n2)*3 + 0)]));
          gyyF = ((w1 * gradRow1[((o)*3 + 1)]) + (iw1 * gradRow1[((n2)*3 + 1)]));
          gyzF = ((w1 * gradRow1[((o)*3 + 2)]) + (iw1 * gradRow1[((n2)*3 + 2)]));
          gzxF = ((w1 * gradRow2[((o)*3 + 0)]) + (iw1 * gradRow2[((n2)*3 + 0)]));
          gzyF = ((w1 * gradRow2[((o)*3 + 1)]) + (iw1 * gradRow2[((n2)*3 + 1)]));
          gzzF = ((w1 * gradRow2[((o)*3 + 2)]) + (iw1 * gradRow2[((n2)*3 + 2)]));
        }
        float corrUx = (((kx * gxxF) + (ky * gxyF)) + (kz * gxzF));
        float corrUy = (((kx * gyxF) + (ky * gyyF)) + (kz * gyzF));
        float corrUz = (((kx * gzxF) + (ky * gzyF)) + (kz * gzzF));
        float fluxScale = (muF * magSf[f]);
        float sgnF = ((float)(sgn));
        sumX = (sumX + ((sgnF * fluxScale) * corrUx));
        sumY = (sumY + ((sgnF * fluxScale) * corrUy));
        sumZ = (sumZ + ((sgnF * fluxScale) * corrUz));
      }
    }
    float v = vol[((int)((int)(get_global_id(0))))];
    float invV = as_float(0x00000000u);
    if (v != as_float(0x00000000u)) {
      invV = (as_float(0x3f800000u) / v);
    }
    outX[((int)((int)(get_global_id(0))))] = (sumX * invV);
    outY[((int)((int)(get_global_id(0))))] = (sumY * invV);
    outZ[((int)((int)(get_global_id(0))))] = (sumZ * invV);
  }
}

)kernel";
const char* k_opencl_navatala_cfd_predictor_hby_a_from_source = R"kernel(
__kernel void navatala_cfd_predictor_hby_a_from_source(__global const float* rAU, __global const float* sourceX, __global const float* sourceY, __global const float* sourceZ, __global const int* counts1, __global float* outHbyAX, __global float* outHbyAY, __global float* outHbyAZ) {
  int gid0 = (int)get_global_id(0);
  if (((int)((int)(get_global_id(0)))) >= counts1[0]) {
    return;
  } else {
    float r = rAU[((int)((int)(get_global_id(0))))];
    outHbyAX[((int)((int)(get_global_id(0))))] = (r * sourceX[((int)((int)(get_global_id(0))))]);
    outHbyAY[((int)((int)(get_global_id(0))))] = (r * sourceY[((int)((int)(get_global_id(0))))]);
    outHbyAZ[((int)((int)(get_global_id(0))))] = (r * sourceZ[((int)((int)(get_global_id(0))))]);
  }
}

)kernel";
const char* k_opencl_navatala_cfd_predictor_inverse_diag = R"kernel(
__kernel void navatala_cfd_predictor_inverse_diag(__global const float* inputDiag, __global const int* counts1, __global float* outRAU) {
  int gid0 = (int)get_global_id(0);
  if (((int)((int)(get_global_id(0)))) >= counts1[0]) {
    return;
  } else {
    float a = inputDiag[((int)((int)(get_global_id(0))))];
    if (a != as_float(0x00000000u)) {
      outRAU[((int)((int)(get_global_id(0))))] = (as_float(0x3f800000u) / a);
    } else {
      outRAU[((int)((int)(get_global_id(0))))] = as_float(0x00000000u);
    }
  }
}

)kernel";
const char* k_opencl_navatala_cfd_predictor_phi_hby_a_fuse = R"kernel(
__kernel void navatala_cfd_predictor_phi_hby_a_fuse(__global const float* fluxHbyA, __global const float* rhoRauFace, __global const float* ddtCorr, __global const int* counts1, __global float* outPhiHbyA) {
  int gid0 = (int)get_global_id(0);
  if (((int)((int)(get_global_id(0)))) >= counts1[0]) {
    return;
  } else {
    float f = fluxHbyA[((int)((int)(get_global_id(0))))];
    float r = rhoRauFace[((int)((int)(get_global_id(0))))];
    float d = ddtCorr[((int)((int)(get_global_id(0))))];
    outPhiHbyA[((int)((int)(get_global_id(0))))] = (f + (r * d));
  }
}

)kernel";
const char* k_opencl_navatala_cfd_axpy_cells = R"kernel(
__kernel void navatala_cfd_axpy_cells(__global const float* x, __global const int* counts, __global const float* paramsF, __global float* y) {
  int gid0 = (int)get_global_id(0);
  const int nSafeMax = (((int)(counts[0])) > 0 ? ((int)(counts[0])) - 1 : 0);
  const int safeIdx = (gid0 < nSafeMax ? gid0 : nSafeMax);
  if (gid0 >= ((int)(counts[0]))) return;
  if (((int)((int)(get_global_id(0)))) >= counts[0]) {
    return;
  } else {
    float xVal = x[((int)((int)(get_global_id(0))))];
    float yVal = y[((int)((int)(get_global_id(0))))];
    float result = (yVal + (paramsF[0] * xVal));
    y[((int)((int)(get_global_id(0))))] = result;
  }
}

)kernel";
const char* k_opencl_navatala_cfd_ddt_diagonal = R"kernel(
__kernel void navatala_cfd_ddt_diagonal(__global const float* rho, __global const float* vol, __global const int* counts, __global const float* paramsF, __global float* outDiag) {
  int gid0 = (int)get_global_id(0);
  const int nSafeMax = (((int)(counts[0])) > 0 ? ((int)(counts[0])) - 1 : 0);
  const int safeIdx = (gid0 < nSafeMax ? gid0 : nSafeMax);
  if (gid0 >= ((int)(counts[0]))) return;
  if (((int)((int)(get_global_id(0)))) >= counts[0]) {
    return;
  } else {
    float rhoVal = rho[((int)((int)(get_global_id(0))))];
    float volVal = vol[((int)((int)(get_global_id(0))))];
    float diag = ((rhoVal * volVal) * paramsF[0]);
    outDiag[((int)((int)(get_global_id(0))))] = diag;
  }
}

)kernel";
const char* k_opencl_navatala_cfd_div_upwind_coeffs = R"kernel(
__kernel void navatala_cfd_div_upwind_coeffs(__global const float* weights, __global const float* flux, __global const int* counts, __global float* outLower, __global float* outUpper) {
  int gid0 = (int)get_global_id(0);
  const int nSafeMax = (((int)(counts[0])) > 0 ? ((int)(counts[0])) - 1 : 0);
  const int safeIdx = (gid0 < nSafeMax ? gid0 : nSafeMax);
  if (gid0 >= ((int)(counts[0]))) return;
  if (((int)((int)(get_global_id(0)))) >= counts[0]) {
    return;
  } else {
    float w = weights[((int)((int)(get_global_id(0))))];
    float phi = flux[((int)((int)(get_global_id(0))))];
    float low = ((as_float(0x00000000u) - w) * phi);
    float up = (low + phi);
    outLower[((int)((int)(get_global_id(0))))] = low;
    outUpper[((int)((int)(get_global_id(0))))] = up;
  }
}

)kernel";
const char* k_opencl_navatala_cfd_fvm_sp = R"kernel(
__kernel void navatala_cfd_fvm_sp(__global const float* sp, __global const float* vol, __global const int* counts, __global float* outDiag) {
  int gid0 = (int)get_global_id(0);
  const int nSafeMax = (((int)(counts[0])) > 0 ? ((int)(counts[0])) - 1 : 0);
  const int safeIdx = (gid0 < nSafeMax ? gid0 : nSafeMax);
  if (gid0 >= ((int)(counts[0]))) return;
  if (((int)((int)(get_global_id(0)))) >= counts[0]) {
    return;
  } else {
    float spVal = sp[((int)((int)(get_global_id(0))))];
    float volVal = vol[((int)((int)(get_global_id(0))))];
    float contrib = (spVal * volVal);
    float prev = outDiag[((int)((int)(get_global_id(0))))];
    float newDiag = (prev + contrib);
    outDiag[((int)((int)(get_global_id(0))))] = newDiag;
  }
}

)kernel";
const char* k_opencl_navatala_cfd_fvm_su_sp = R"kernel(
__kernel void navatala_cfd_fvm_su_sp(__global const float* sp, __global const float* psi, __global const float* vol, __global const int* counts, __global float* outDiag, __global float* outSource) {
  int gid0 = (int)get_global_id(0);
  const int nSafeMax = (((int)(counts[0])) > 0 ? ((int)(counts[0])) - 1 : 0);
  const int safeIdx = (gid0 < nSafeMax ? gid0 : nSafeMax);
  if (gid0 >= ((int)(counts[0]))) return;
  if (((int)((int)(get_global_id(0)))) >= counts[0]) {
    return;
  } else {
    float spVal = sp[((int)((int)(get_global_id(0))))];
    float psiVal = psi[((int)((int)(get_global_id(0))))];
    float volVal = vol[((int)((int)(get_global_id(0))))];
    float spVol = (spVal * volVal);
    if (spVal >= as_float(0x00000000u)) {
      float prevDiag = outDiag[((int)((int)(get_global_id(0))))];
      float newDiag = (prevDiag + spVol);
      outDiag[((int)((int)(get_global_id(0))))] = newDiag;
    } else {
      float prevSrc = outSource[((int)((int)(get_global_id(0))))];
      float contrib = (spVol * psiVal);
      float newSrc = (prevSrc - contrib);
      outSource[((int)((int)(get_global_id(0))))] = newSrc;
    }
  }
}

)kernel";
const char* k_opencl_navatala_cfd_laplacian_coeffs = R"kernel(
__kernel void navatala_cfd_laplacian_coeffs(__global const float* gamma, __global const int* owner, __global const int* neighbour, __global const float* weights, __global const float* deltaCoeffs, __global const float* magSf, __global const int* counts, __global float* outUpper, __global float* outLower) {
  int gid0 = (int)get_global_id(0);
  const int nSafeMax = (((int)(counts[0])) > 0 ? ((int)(counts[0])) - 1 : 0);
  const int safeIdx = (gid0 < nSafeMax ? gid0 : nSafeMax);
  if (gid0 >= ((int)(counts[0]))) return;
  if (((int)((int)(get_global_id(0)))) >= counts[0]) {
    return;
  } else {
    float gammaF = gamma[((int)((int)(get_global_id(0))))];
    float coeff = (deltaCoeffs[((int)((int)(get_global_id(0))))] * (gammaF * magSf[((int)((int)(get_global_id(0))))]));
    outUpper[((int)((int)(get_global_id(0))))] = coeff;
    outLower[((int)((int)(get_global_id(0))))] = coeff;
  }
}

)kernel";
const char* k_opencl_navatala_cfd_laplacian_coeffs_accum = R"kernel(
__kernel void navatala_cfd_laplacian_coeffs_accum(__global const float* gamma, __global const float* deltaCoeffs, __global const float* magSf, __global const int* counts, __global float* outUpper, __global float* outLower) {
  int gid0 = (int)get_global_id(0);
  const int nSafeMax = (((int)(counts[0])) > 0 ? ((int)(counts[0])) - 1 : 0);
  const int safeIdx = (gid0 < nSafeMax ? gid0 : nSafeMax);
  if (gid0 >= ((int)(counts[0]))) return;
  if (((int)((int)(get_global_id(0)))) >= counts[0]) {
    return;
  } else {
    float coeff = (deltaCoeffs[((int)((int)(get_global_id(0))))] * (gamma[((int)((int)(get_global_id(0))))] * magSf[((int)((int)(get_global_id(0))))]));
    float prevUpper = outUpper[((int)((int)(get_global_id(0))))];
    float prevLower = outLower[((int)((int)(get_global_id(0))))];
    outUpper[((int)((int)(get_global_id(0))))] = (prevUpper + coeff);
    outLower[((int)((int)(get_global_id(0))))] = (prevLower + coeff);
  }
}

)kernel";
const char* k_opencl_navatala_cfd_linear_upwind_correction_source = R"kernel(
__kernel void navatala_cfd_linear_upwind_correction_source(__global const float* luFaceVals, __global const float* flux, __global const int* owner, __global const int* neighbour, __global const float* psi, __global const int* offsets, __global const int* faceIdx, __global const float* signF, __global const int* counts, __global float* outSource) {
  int gid0 = (int)get_global_id(0);
  const int nSafeMax = (((int)(counts[0])) > 0 ? ((int)(counts[0])) - 1 : 0);
  const int safeIdx = (gid0 < nSafeMax ? gid0 : nSafeMax);
  if (gid0 >= ((int)(counts[0]))) return;
  if (((int)((int)(get_global_id(0)))) >= counts[0]) {
    return;
  } else {
    float sum = as_float(0x00000000u);
    int beg = offsets[((int)((int)(get_global_id(0))))];
    int c1 = (((int)((int)(get_global_id(0)))) + 1);
    int end = offsets[c1];
    int len = (end - beg);
    for (int t = 0; t < (int)(len); ++t) {
      int k = (beg + t);
      int f = faceIdx[k];
      if (f < counts[1]) {
        float fluxVal = flux[f];
        float luVal = luFaceVals[f];
        float sgnF = signF[k];
        float correction = (fluxVal * luVal);
        sum = (sum + (sgnF * correction));
      }
    }
    float prev = outSource[((int)((int)(get_global_id(0))))];
    outSource[((int)((int)(get_global_id(0))))] = (prev - sum);
  }
}

)kernel";
const char* k_opencl_navatala_cfd_matrix_relax = R"kernel(
__kernel void navatala_cfd_matrix_relax(__global const float* diagOld, __global const float* sumOff, __global const float* psi, __global const int* counts, __global const float* paramsF, __global float* diag, __global float* source) {
  int gid0 = (int)get_global_id(0);
  const int nSafeMax = (((int)(counts[0])) > 0 ? ((int)(counts[0])) - 1 : 0);
  const int safeIdx = (gid0 < nSafeMax ? gid0 : nSafeMax);
  if (gid0 >= ((int)(counts[0]))) return;
  if (((int)((int)(get_global_id(0)))) >= counts[0]) {
    return;
  } else {
    float d0 = diagOld[((int)((int)(get_global_id(0))))];
    float sOff = sumOff[((int)((int)(get_global_id(0))))];
    float x = psi[((int)((int)(get_global_id(0))))];
    float negD0 = (as_float(0x00000000u) - d0);
    float absD0 = (((d0 >= as_float(0x00000000u))) ? (d0) : (negD0));
    float maxVal = (((absD0 >= sOff)) ? (absD0) : (sOff));
    float newDiag = (maxVal / paramsF[0]);
    float srcPrev = source[((int)((int)(get_global_id(0))))];
    float delta = (newDiag - d0);
    float srcNew = (srcPrev + (delta * x));
    diag[((int)((int)(get_global_id(0))))] = newDiag;
    source[((int)((int)(get_global_id(0))))] = srcNew;
  }
}

)kernel";
const char* k_opencl_navatala_cfd_neg_sum_diag = R"kernel(
__kernel void navatala_cfd_neg_sum_diag(__global const float* upper, __global const float* lower, __global const int* offsets, __global const int* faceIdx, __global const int* sign, __global const int* counts, __global float* outDiag) {
  int gid0 = (int)get_global_id(0);
  const int nSafeMax = (((int)(counts[0])) > 0 ? ((int)(counts[0])) - 1 : 0);
  const int safeIdx = (gid0 < nSafeMax ? gid0 : nSafeMax);
  if (gid0 >= ((int)(counts[0]))) return;
  if (((int)((int)(get_global_id(0)))) >= counts[0]) {
    return;
  } else {
    float sum = as_float(0x00000000u);
    int beg = offsets[((int)((int)(get_global_id(0))))];
    int c1 = (((int)((int)(get_global_id(0)))) + 1);
    int end = offsets[c1];
    int len = (end - beg);
    for (int t = 0; t < (int)(len); ++t) {
      int k = (beg + t);
      int f = faceIdx[k];
      if (f < counts[1]) {
        int sgn = sign[k];
        if (sgn > 0) {
          sum = (sum - upper[f]);
        } else {
          sum = (sum - lower[f]);
        }
      }
    }
    float prev = outDiag[((int)((int)(get_global_id(0))))];
    outDiag[((int)((int)(get_global_id(0))))] = (prev + sum);
  }
}

)kernel";
const char* k_opencl_navatala_cfd_pack_boundary_adjacent_cells = R"kernel(
__kernel void navatala_cfd_pack_boundary_adjacent_cells(__global const float* cellField, __global const int* indexMap, __global const int* counts, __global float* outPacked) {
  int gid0 = (int)get_global_id(0);
  const int nSafeMax = (((int)(counts[0])) > 0 ? ((int)(counts[0])) - 1 : 0);
  const int safeIdx = (gid0 < nSafeMax ? gid0 : nSafeMax);
  if (gid0 >= ((int)(counts[0]))) return;
  if (((int)((int)(get_global_id(0)))) >= counts[0]) {
    return;
  } else {
    int cellIdx = indexMap[((int)((int)(get_global_id(0))))];
    float val = cellField[cellIdx];
    outPacked[((int)((int)(get_global_id(0))))] = val;
  }
}

)kernel";
const char* k_opencl_navatala_cfd_scalar_jacobi_clamp_min = R"kernel(
__kernel void navatala_cfd_scalar_jacobi_clamp_min(__global const float* lowerBound, __global const uint* counts, __global float* x) {
  int gid0 = (int)get_global_id(0);
  const int nSafeMax = (((int)(counts[0])) > 0 ? ((int)(counts[0])) - 1 : 0);
  const int safeIdx = (gid0 < nSafeMax ? gid0 : nSafeMax);
  if (gid0 >= ((int)(counts[0]))) return;
  if ((int)(get_global_id(0)) >= ((int)(counts[0]))) {
    return;
  } else {
    float lb = lowerBound[0];
    float xi = x[(int)(get_global_id(0))];
    if (xi > lb) {
      x[(int)(get_global_id(0))] = xi;
    } else {
      x[(int)(get_global_id(0))] = lb;
    }
  }
}

)kernel";
const char* k_opencl_navatala_cfd_scalar_jacobi_update = R"kernel(
__kernel void navatala_cfd_scalar_jacobi_update(__global const float* ax, __global const float* diag, __global const float* rhs, __global const float* params, __global const uint* counts, __global float* x) {
  int gid0 = (int)get_global_id(0);
  const int nSafeMax = (((int)(counts[0])) > 0 ? ((int)(counts[0])) - 1 : 0);
  const int safeIdx = (gid0 < nSafeMax ? gid0 : nSafeMax);
  if (gid0 >= ((int)(counts[0]))) return;
  if ((int)(get_global_id(0)) >= ((int)(counts[0]))) {
    return;
  } else {
    float smallDiag = params[0];
    float omega = params[1];
    float d = diag[(int)(get_global_id(0))];
    float absD = fabs(d);
    if (absD < smallDiag) {
      if (d >= as_float(0x00000000u)) {
        d = smallDiag;
      } else {
        d = (-smallDiag);
      }
    }
    float xi = x[(int)(get_global_id(0))];
    float correction = (omega * ((rhs[(int)(get_global_id(0))] - ax[(int)(get_global_id(0))]) / d));
    x[(int)(get_global_id(0))] = (xi + correction);
  }
}

)kernel";
const char* k_opencl_navatala_cfd_scalar_ldu_mat_vec = R"kernel(
__kernel void navatala_cfd_scalar_ldu_mat_vec(__global const float* diag, __global const float* upper, __global const float* lower, __global const float* x, __global const int* owner, __global const int* neighbour, __global const uint* offsets, __global const uint* faceIdx, __global const int* sign, __global const uint* counts, __global float* ax) {
  int gid0 = (int)get_global_id(0);
  const int nSafeMax = (((int)(counts[0])) > 0 ? ((int)(counts[0])) - 1 : 0);
  const int safeIdx = (gid0 < nSafeMax ? gid0 : nSafeMax);
  if (gid0 >= ((int)(counts[0]))) return;
  if ((int)(get_global_id(0)) >= ((int)(counts[0]))) {
    return;
  } else {
    float s = (diag[(int)(get_global_id(0))] * x[(int)(get_global_id(0))]);
    int beg = ((int)(offsets[(int)(get_global_id(0))]));
    int c1 = ((int)(get_global_id(0)) + 1);
    int end = ((int)(offsets[c1]));
    int len = (end - beg);
    for (int t = 0; t < (int)(len); ++t) {
      int k = (beg + t);
      uint f = faceIdx[k];
      int sg = sign[k];
      if (((int)(f)) < ((int)(counts[1]))) {
        if (sg >= 0) {
          int nb = neighbour[((int)(f))];
          s = (s + (upper[((int)(f))] * x[nb]));
        } else {
          int ow = owner[((int)(f))];
          s = (s + (lower[((int)(f))] * x[ow]));
        }
      }
    }
    ax[(int)(get_global_id(0))] = s;
  }
}

)kernel";
const char* k_opencl_navatala_cfd_scalar_ldu_multi_color_gs_sweep = R"kernel(
__kernel void navatala_cfd_scalar_ldu_multi_color_gs_sweep(__global const float* diag, __global const float* upper, __global const float* lower, __global const float* rhs, __global const int* owner, __global const int* neighbour, __global const uint* offsets, __global const uint* faceIdx, __global const int* sign, __global const uint* colorOrder, __global const uint* colorRange, __global const float* params, __global const uint* counts, __global float* x) {
  int gid0 = (int)get_global_id(0);
  const int nSafeMax = (((int)(counts[0])) > 0 ? ((int)(counts[0])) - 1 : 0);
  const int safeIdx = (gid0 < nSafeMax ? gid0 : nSafeMax);
  if (gid0 >= ((int)(counts[0]))) return;
  int colorSize = (((int)(colorRange[1])) - ((int)(colorRange[0])));
  if ((int)(get_global_id(0)) >= colorSize) {
    return;
  } else {
    int cellIdx = (((int)(colorRange[0])) + (int)(get_global_id(0)));
    uint celli = colorOrder[cellIdx];
    int ci = ((int)(celli));
    float smallDiag = params[0];
    float omega = params[1];
    float d = diag[ci];
    float absD = fabs(d);
    if (absD < smallDiag) {
      if (d >= as_float(0x00000000u)) {
        d = smallDiag;
      } else {
        d = (-smallDiag);
      }
    }
    float s = (d * x[ci]);
    int beg = ((int)(offsets[ci]));
    int c1 = (ci + 1);
    int end = ((int)(offsets[c1]));
    int len = (end - beg);
    for (int t = 0; t < (int)(len); ++t) {
      int k = (beg + t);
      uint f = faceIdx[k];
      int sg = sign[k];
      if (((int)(f)) < ((int)(counts[1]))) {
        if (sg >= 0) {
          int nb = neighbour[((int)(f))];
          s = (s + (upper[((int)(f))] * x[nb]));
        } else {
          int ow = owner[((int)(f))];
          s = (s + (lower[((int)(f))] * x[ow]));
        }
      }
    }
    x[ci] = (x[ci] + (omega * ((rhs[ci] - s) / d)));
  }
}

)kernel";
const char* k_opencl_navatala_cfd_sum_mag_off_diag = R"kernel(
__kernel void navatala_cfd_sum_mag_off_diag(__global const float* upper, __global const float* lower, __global const int* offsets, __global const int* faceIdx, __global const int* sign, __global const int* counts, __global float* outSumOff) {
  int gid0 = (int)get_global_id(0);
  const int nSafeMax = (((int)(counts[0])) > 0 ? ((int)(counts[0])) - 1 : 0);
  const int safeIdx = (gid0 < nSafeMax ? gid0 : nSafeMax);
  if (gid0 >= ((int)(counts[0]))) return;
  if (((int)((int)(get_global_id(0)))) >= counts[0]) {
    return;
  } else {
    float sum = as_float(0x00000000u);
    int beg = offsets[((int)((int)(get_global_id(0))))];
    int c1 = (((int)((int)(get_global_id(0)))) + 1);
    int end = offsets[c1];
    int len = (end - beg);
    for (int t = 0; t < (int)(len); ++t) {
      int k = (beg + t);
      int f = faceIdx[k];
      if (f < counts[1]) {
        int sgn = sign[k];
        if (sgn > 0) {
          float val = upper[f];
          float negVal = (as_float(0x00000000u) - val);
          float absVal = (((val >= as_float(0x00000000u))) ? (val) : (negVal));
          sum = (sum + absVal);
        } else {
          float val = lower[f];
          float negVal = (as_float(0x00000000u) - val);
          float absVal = (((val >= as_float(0x00000000u))) ? (val) : (negVal));
          sum = (sum + absVal);
        }
      }
    }
    outSumOff[((int)((int)(get_global_id(0))))] = sum;
  }
}

)kernel";
const char* k_opencl_navatala_cfd_gamg_axpy_in_place = R"kernel(
__kernel void navatala_cfd_gamg_axpy_in_place(__global const float* x, __global const float* alpha, __global const uint* nRows, __global float* y) {
  int gid0 = (int)get_global_id(0);
  if ((int)(get_global_id(0)) >= ((int)(nRows[0]))) {
    return;
  } else {
    float a = alpha[0];
    float xi = x[(int)(get_global_id(0))];
    float yi = y[(int)(get_global_id(0))];
    y[(int)(get_global_id(0))] = (yi + (a * xi));
  }
}

)kernel";
const char* k_opencl_navatala_cfd_gamg_compute_scale_factor = R"kernel(
__kernel void navatala_cfd_gamg_compute_scale_factor(__global const float* num, __global const float* den, __global float* sf) {
  int gid0 = (int)get_global_id(0);
  if ((int)(get_global_id(0)) >= 1) {
    return;
  } else {
    float n = num[0];
    float d = den[0];
    float dAbs = fabs(d);
    if (dAbs > as_float(0x0da24260u)) {
      sf[0] = (n / d);
    } else {
      sf[0] = as_float(0x3f800000u);
    }
  }
}

)kernel";
const char* k_opencl_navatala_cfd_gamg_dot_finalize = R"kernel(
__kernel void navatala_cfd_gamg_dot_finalize(__global const float* partials, __global const uint* count, __global float* result) {
  int gid0 = (int)get_global_id(0);
  if ((int)(get_global_id(0)) >= 1) {
    return;
  } else {
    float s = as_float(0x00000000u);
    for (int i = 0; i < (int)(((int)(count[0]))); ++i) {
      s = (s + partials[i]);
    }
    result[0] = s;
  }
}

)kernel";
const char* k_opencl_navatala_cfd_gamg_negate_scalar = R"kernel(
__kernel void navatala_cfd_gamg_negate_scalar(__global const float* _input, __global float* _output) {
  int gid0 = (int)get_global_id(0);
  if ((int)(get_global_id(0)) >= 1) {
    return;
  } else {
    float v = _input[0];
    _output[0] = (-v);
  }
}

)kernel";
const char* k_opencl_navatala_cfd_gamg_scale_correction = R"kernel(
__kernel void navatala_cfd_gamg_scale_correction(__global const float* z, __global const float* r, __global const float* Az, __global const float* diagInv, __global const float* sfPtr, __global const uint* nRows, __global float* corr) {
  int gid0 = (int)get_global_id(0);
  if ((int)(get_global_id(0)) >= ((int)(nRows[0]))) {
    return;
  } else {
    float sf = sfPtr[0];
    float zi = z[(int)(get_global_id(0))];
    float ri = r[(int)(get_global_id(0))];
    float azi = Az[(int)(get_global_id(0))];
    float di = diagInv[(int)(get_global_id(0))];
    float sfz = (sf * zi);
    float sfaz = (sf * azi);
    float residual = (ri - sfaz);
    float corrVal = (sfz + (residual * di));
    corr[(int)(get_global_id(0))] = corrVal;
  }
}

)kernel";
const char* k_opencl_navatala_cfd_mg_dic_build = R"kernel(
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
__kernel void navatala_cfd_mg_dic_build(__global const float* diag, __global const int* edgeU, __global const int* edgeV, __global const float* edgeCf, __global const int* counts, __global double* rD) {
  int gid0 = (int)get_global_id(0);
  const int nSafeMax = (((int)(counts[0])) > 0 ? ((int)(counts[0])) - 1 : 0);
  const int safeIdx = (gid0 < nSafeMax ? gid0 : nSafeMax);
  if (gid0 >= ((int)(counts[0]))) return;
  if ((int)(get_global_id(0)) >= 1) {
    return;
  } else {
    int nCells = counts[1];
    int nEdges = counts[2];
    for (int c = 0; c < (int)(nCells); ++c) {
      rD[c] = ((double)(diag[c]));
    }
    for (int e = 0; e < (int)(nEdges); ++e) {
      int u = edgeU[e];
      int v = edgeV[e];
      double dl = rD[u];
      if (dl > as_double(0x01a56e1fc2f8f359ul)) {
        double cf = ((double)(edgeCf[e]));
        rD[v] = (rD[v] - ((cf * cf) / dl));
      }
    }
    for (int c2 = 0; c2 < (int)(nCells); ++c2) {
      double d = rD[c2];
      if (d > as_double(0x01a56e1fc2f8f359ul)) {
        rD[c2] = (as_double(0x3ff0000000000000ul) / d);
      } else {
        rD[c2] = as_double(0x0000000000000000ul);
      }
    }
  }
}

)kernel";
const char* k_opencl_navatala_cfd_mg_dic_smooth = R"kernel(
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
__kernel void navatala_cfd_mg_dic_smooth(__global const float* b, __global const float* ax, __global const double* rD, __global const int* edgeU, __global const int* edgeV, __global const float* edgeCf, __global const int* counts, __global float* x, __global float* z) {
  int gid0 = (int)get_global_id(0);
  const int nSafeMax = (((int)(counts[0])) > 0 ? ((int)(counts[0])) - 1 : 0);
  const int safeIdx = (gid0 < nSafeMax ? gid0 : nSafeMax);
  if (gid0 >= ((int)(counts[0]))) return;
  if ((int)(get_global_id(0)) >= 1) {
    return;
  } else {
    int nCells = counts[1];
    int nEdges = counts[2];
    for (int c = 0; c < (int)(nCells); ++c) {
      z[c] = ((float)((rD[c] * (((double)(b[c])) - ((double)(ax[c]))))));
    }
    for (int e = 0; e < (int)(nEdges); ++e) {
      int v = edgeV[e];
      int u = edgeU[e];
      z[v] = ((float)((((double)(z[v])) - (rD[v] * (((double)(edgeCf[e])) * ((double)(z[u])))))));
    }
    for (int t = 0; t < (int)(nEdges); ++t) {
      int e = ((nEdges - 1) - t);
      int u = edgeU[e];
      int v = edgeV[e];
      z[u] = ((float)((((double)(z[u])) - (rD[u] * (((double)(edgeCf[e])) * ((double)(z[v])))))));
    }
    for (int c2 = 0; c2 < (int)(nCells); ++c2) {
      x[c2] = (x[c2] + z[c2]);
    }
  }
}

)kernel";
const char* k_opencl_navatala_cfd_mg_coarse_jacobi = R"kernel(
__kernel void navatala_cfd_mg_coarse_jacobi(__global float* x, __global const float* r, __global const float* diag, __global const int* mgCounts) {
  int gid0 = (int)get_global_id(0);
  if ((int)(get_global_id(0)) >= ((int)(mgCounts[1]))) {
    return;
  } else {
    float d = diag[(int)(get_global_id(0))];
    if (d == as_float(0x00000000u)) {
      return;
    } else {
      float omega = as_float(0x3f4ccccdu);
      x[(int)(get_global_id(0))] = (x[(int)(get_global_id(0))] + (omega * (r[(int)(get_global_id(0))] / d)));
    }
  }
}

)kernel";
const char* k_opencl_navatala_cfd_mg_coarse_matvec = R"kernel(
__kernel void navatala_cfd_mg_coarse_matvec(__global const int* edgeU, __global const int* edgeV, __global const float* edgeCf, __global const float* diag, __global const float* x, __global float* outAx, __global const int* mgCounts) {
  int gid0 = (int)get_global_id(0);
  int total = (((int)(mgCounts[1])) + ((int)(mgCounts[2])));
  if ((int)(get_global_id(0)) >= total) {
    return;
  } else {
    if ((int)(get_global_id(0)) < ((int)(mgCounts[1]))) {
      atomic_add(&outAx[(int)(get_global_id(0))], (diag[(int)(get_global_id(0))] * x[(int)(get_global_id(0))]));
    } else {
      int e = ((int)(get_global_id(0)) - ((int)(mgCounts[1])));
      int u = edgeU[e];
      int v = edgeV[e];
      float cf = edgeCf[e];
      atomic_add(&outAx[u], (cf * x[v]));
      atomic_add(&outAx[v], (cf * x[u]));
    }
  }
}

)kernel";
const char* k_opencl_navatala_cfd_mg_coarse_rhs = R"kernel(
__kernel void navatala_cfd_mg_coarse_rhs(__global const float* b, __global const float* ax, __global float* r, __global const int* mgCounts) {
  int gid0 = (int)get_global_id(0);
  if ((int)(get_global_id(0)) >= ((int)(mgCounts[1]))) {
    return;
  } else {
    r[(int)(get_global_id(0))] = (b[(int)(get_global_id(0))] - ax[(int)(get_global_id(0))]);
  }
}

)kernel";
const char* k_opencl_navatala_cfd_mg_find_best_match = R"kernel(
__kernel void navatala_cfd_mg_find_best_match(__global const uint* csrOffsets, __global const uint* csrFaceIdx, __global const int* csrSign, __global const float* strength, __global const int* owner, __global const int* nei, __global int* bestNeighbor, __global const int* counts) {
  int gid0 = (int)get_global_id(0);
  const int nSafeMax = (((int)(counts[0])) > 0 ? ((int)(counts[0])) - 1 : 0);
  const int safeIdx = (gid0 < nSafeMax ? gid0 : nSafeMax);
  if (gid0 >= ((int)(counts[0]))) return;
  if ((int)(get_global_id(0)) >= ((int)(counts[0]))) {
    return;
  } else {
    float bestS = as_float(0xbf800000u);
    int bestN = -1;
    uint beg = csrOffsets[(int)(get_global_id(0))];
    int c1 = ((int)(get_global_id(0)) + 1);
    uint end = csrOffsets[c1];
    int len = ((int)((end - beg)));
    for (int t = 0; t < (int)(len); ++t) {
      int k = (((int)(beg)) + t);
      uint f = csrFaceIdx[k];
      float s = strength[((int)(f))];
      if (s > bestS) {
        bestS = s;
        int sg = csrSign[k];
        if (sg > 0) {
          bestN = nei[((int)(f))];
        } else {
          bestN = owner[((int)(f))];
        }
      }
    }
    bestNeighbor[(int)(get_global_id(0))] = bestN;
  }
}

)kernel";
const char* k_opencl_navatala_cfd_mg_pair_match = R"kernel(
__kernel void navatala_cfd_mg_pair_match(__global const int* bestNeighbor, __global int* aggMap, __global const int* counts) {
  int gid0 = (int)get_global_id(0);
  const int nSafeMax = (((int)(counts[0])) > 0 ? ((int)(counts[0])) - 1 : 0);
  const int safeIdx = (gid0 < nSafeMax ? gid0 : nSafeMax);
  if (gid0 >= ((int)(counts[0]))) return;
  if ((int)(get_global_id(0)) >= ((int)(counts[0]))) {
    return;
  } else {
    int n = bestNeighbor[(int)(get_global_id(0))];
    if (n >= 0) {
      if ((int)(get_global_id(0)) < n) {
        aggMap[(int)(get_global_id(0))] = (int)(get_global_id(0));
      } else {
        aggMap[(int)(get_global_id(0))] = n;
      }
    } else {
      aggMap[(int)(get_global_id(0))] = (int)(get_global_id(0));
    }
  }
}

)kernel";
const char* k_opencl_navatala_cfd_mg_prolong_add = R"kernel(
__kernel void navatala_cfd_mg_prolong_add(__global const float* xCoarse, __global const int* prolongRowOffsets, __global const int* prolongCols, __global const float* prolongVals, __global float* zFine, __global const int* mgCounts) {
  int gid0 = (int)get_global_id(0);
  if ((int)(get_global_id(0)) >= ((int)(mgCounts[0]))) {
    return;
  } else {
    float accum = as_float(0x00000000u);
    int beg = prolongRowOffsets[(int)(get_global_id(0))];
    int c1 = ((int)(get_global_id(0)) + 1);
    int end = prolongRowOffsets[c1];
    int len = (end - beg);
    for (int t = 0; t < (int)(len); ++t) {
      int k = (beg + t);
      int col = prolongCols[k];
      accum = (accum + (prolongVals[k] * xCoarse[col]));
    }
    zFine[(int)(get_global_id(0))] = (zFine[(int)(get_global_id(0))] + accum);
  }
}

)kernel";
const char* k_opencl_navatala_cfd_mg_renumber_aggregates = R"kernel(
__kernel void navatala_cfd_mg_renumber_aggregates(__global int* aggMap, __global int* counter, __global const int* counts) {
  int gid0 = (int)get_global_id(0);
  const int nSafeMax = (((int)(counts[0])) > 0 ? ((int)(counts[0])) - 1 : 0);
  const int safeIdx = (gid0 < nSafeMax ? gid0 : nSafeMax);
  if (gid0 >= ((int)(counts[0]))) return;
  if ((int)(get_global_id(0)) >= ((int)(counts[0]))) {
    return;
  } else {
    if (aggMap[(int)(get_global_id(0))] == (int)(get_global_id(0))) {
      int oldCount = atomic_fetch_add_explicit((volatile __global atomic_int*)(&(counter[0])), (int)(1), memory_order_relaxed, memory_scope_device);
      aggMap[(int)(get_global_id(0))] = (-(oldCount + 1));
    }
  }
}

)kernel";
const char* k_opencl_navatala_cfd_mg_renumber_propagate = R"kernel(
__kernel void navatala_cfd_mg_renumber_propagate(__global int* aggMap, __global const int* counts) {
  int gid0 = (int)get_global_id(0);
  const int nSafeMax = (((int)(counts[0])) > 0 ? ((int)(counts[0])) - 1 : 0);
  const int safeIdx = (gid0 < nSafeMax ? gid0 : nSafeMax);
  if (gid0 >= ((int)(counts[0]))) return;
  if ((int)(get_global_id(0)) >= ((int)(counts[0]))) {
    return;
  } else {
    int root = aggMap[(int)(get_global_id(0))];
    if (root < 0) {
      aggMap[(int)(get_global_id(0))] = (-(root + 1));
    } else {
      int rootVal = aggMap[root];
      if (rootVal < 0) {
        aggMap[(int)(get_global_id(0))] = (-(rootVal + 1));
      } else {
        aggMap[(int)(get_global_id(0))] = rootVal;
      }
    }
  }
}

)kernel";
const char* k_opencl_navatala_cfd_mg_resolve_chains = R"kernel(
__kernel void navatala_cfd_mg_resolve_chains(__global int* aggMap, __global const int* counts) {
  int gid0 = (int)get_global_id(0);
  const int nSafeMax = (((int)(counts[0])) > 0 ? ((int)(counts[0])) - 1 : 0);
  const int safeIdx = (gid0 < nSafeMax ? gid0 : nSafeMax);
  if (gid0 >= ((int)(counts[0]))) return;
  if ((int)(get_global_id(0)) >= ((int)(counts[0]))) {
    return;
  } else {
    int root = aggMap[(int)(get_global_id(0))];
    if (root >= 0) {
      if (root < ((int)(counts[0]))) {
        if (root != (int)(get_global_id(0))) {
          int next = aggMap[root];
          if (next >= 0) {
            if (next < ((int)(counts[0]))) {
              aggMap[(int)(get_global_id(0))] = next;
            }
          }
        }
      }
    }
  }
}

)kernel";
const char* k_opencl_navatala_cfd_mg_restrict_sum = R"kernel(
__kernel void navatala_cfd_mg_restrict_sum(__global const float* rFine, __global const int* prolongRowOffsets, __global const int* prolongCols, __global const float* prolongVals, __global float* rCoarse, __global const int* mgCounts) {
  int gid0 = (int)get_global_id(0);
  if ((int)(get_global_id(0)) >= ((int)(mgCounts[0]))) {
    return;
  } else {
    int beg = prolongRowOffsets[(int)(get_global_id(0))];
    int c1 = ((int)(get_global_id(0)) + 1);
    int end = prolongRowOffsets[c1];
    float v = rFine[(int)(get_global_id(0))];
    int len = (end - beg);
    for (int t = 0; t < (int)(len); ++t) {
      int k = (beg + t);
      int col = prolongCols[k];
      atomic_add(&rCoarse[col], (prolongVals[k] * v));
    }
  }
}

)kernel";
const char* k_opencl_navatala_cfd_mg_strength_of_connection = R"kernel(
__kernel void navatala_cfd_mg_strength_of_connection(__global const float* diag, __global const float* upper, __global const int* owner, __global const int* nei, __global float* strength, __global const int* counts) {
  int gid0 = (int)get_global_id(0);
  const int nSafeMax = (((int)(counts[0])) > 0 ? ((int)(counts[0])) - 1 : 0);
  const int safeIdx = (gid0 < nSafeMax ? gid0 : nSafeMax);
  if (gid0 >= ((int)(counts[0]))) return;
  if ((int)(get_global_id(0)) >= ((int)(counts[1]))) {
    return;
  } else {
    int o = owner[(int)(get_global_id(0))];
    int n = nei[(int)(get_global_id(0))];
    float dO = fabs(diag[o]);
    float dN = fabs(diag[n]);
    float denomSq = (dO * dN);
    if (denomSq > as_float(0x0da24260u)) {
      strength[(int)(get_global_id(0))] = (fabs(upper[(int)(get_global_id(0))]) / sqrt(denomSq));
    } else {
      strength[(int)(get_global_id(0))] = as_float(0x00000000u);
    }
  }
}

)kernel";
const char* k_opencl_navatala_cfd_mg_zero_coarse = R"kernel(
__kernel void navatala_cfd_mg_zero_coarse(__global float* x, __global const int* mgCounts) {
  int gid0 = (int)get_global_id(0);
  if ((int)(get_global_id(0)) >= ((int)(mgCounts[1]))) {
    return;
  } else {
    x[(int)(get_global_id(0))] = as_float(0x00000000u);
  }
}

)kernel";
const char* k_opencl_navatala_cfd_phi_correct = R"kernel(
__kernel void navatala_cfd_phi_correct(__global const float* phiIn, __global const float* pCell, __global const int* owner, __global const int* neighbour, __global const float* magSf, __global const float* deltaCoeffs, __global const float* coeff, __global const float* bcVal, __global const int* bcMask, __global const float* bcSnGrad, __global const int* bcSnGradMask, __global const float* faceFluxCorrection, __global const int* counts, __global const float* paramsF, __global float* phiOut) {
  int gid0 = (int)get_global_id(0);
  const int nSafeMax = (((int)(counts[0])) > 0 ? ((int)(counts[0])) - 1 : 0);
  const int safeIdx = (gid0 < nSafeMax ? gid0 : nSafeMax);
  if (gid0 >= ((int)(counts[0]))) return;
  if (((int)((int)(get_global_id(0)))) >= counts[0]) {
    return;
  } else {
    int o = owner[((int)((int)(get_global_id(0))))];
    float po = pCell[o];
    float other = po;
    float snGrad = as_float(0x00000000u);
    if (((int)((int)(get_global_id(0)))) < counts[1]) {
      int n = neighbour[((int)((int)(get_global_id(0))))];
      other = pCell[n];
    } else {
      if (bcMask[((int)((int)(get_global_id(0))))] != 0) {
        other = bcVal[((int)((int)(get_global_id(0))))];
      }
    }
    snGrad = (deltaCoeffs[((int)((int)(get_global_id(0))))] * (other - po));
    if (bcSnGradMask[((int)((int)(get_global_id(0))))] != 0) {
      snGrad = bcSnGrad[((int)((int)(get_global_id(0))))];
    }
    float gradFace = (magSf[((int)((int)(get_global_id(0))))] * snGrad);
    phiOut[((int)((int)(get_global_id(0))))] = (phiIn[((int)((int)(get_global_id(0))))] - ((paramsF[0] * (coeff[((int)((int)(get_global_id(0))))] * gradFace)) + faceFluxCorrection[((int)((int)(get_global_id(0))))]));
  }
}

)kernel";
const char* k_opencl_navatala_cfd_phi_correct_boundary = R"kernel(
__kernel void navatala_cfd_phi_correct_boundary(__global const float* phiIn, __global const float* pCell, __global const int* owner, __global const float* magSf, __global const float* deltaCoeffs, __global const float* coeff, __global const float* bcVal, __global const int* bcMask, __global const float* bcSnGrad, __global const int* bcSnGradMask, __global const float* faceFluxCorrection, __global const int* counts, __global const float* paramsF, __global float* phiOut) {
  int gid0 = (int)get_global_id(0);
  const int nSafeMax = (((int)(counts[0])) > 0 ? ((int)(counts[0])) - 1 : 0);
  const int safeIdx = (gid0 < nSafeMax ? gid0 : nSafeMax);
  if (gid0 >= ((int)(counts[0]))) return;
  int f = (counts[1] + ((int)((int)(get_global_id(0)))));
  if (f >= counts[0]) {
    return;
  } else {
    int o = owner[f];
    float po = pCell[o];
    float other = po;
    if (bcMask[f] != 0) {
      other = bcVal[f];
    }
    float snGrad = (deltaCoeffs[f] * (other - po));
    if (bcSnGradMask[f] != 0) {
      snGrad = bcSnGrad[f];
    }
    float gradFace = (magSf[f] * snGrad);
    phiOut[f] = (phiIn[f] - ((paramsF[0] * (coeff[f] * gradFace)) + faceFluxCorrection[f]));
  }
}

)kernel";
const char* k_opencl_navatala_cfd_phi_correct_internal = R"kernel(
__kernel void navatala_cfd_phi_correct_internal(__global const float* phiIn, __global const float* pCell, __global const int* owner, __global const int* neighbour, __global const float* magSf, __global const float* deltaCoeffs, __global const float* coeff, __global const float* faceFluxCorrection, __global const int* counts, __global const float* paramsF, __global float* phiOut) {
  int gid0 = (int)get_global_id(0);
  const int nSafeMax = (((int)(counts[0])) > 0 ? ((int)(counts[0])) - 1 : 0);
  const int safeIdx = (gid0 < nSafeMax ? gid0 : nSafeMax);
  if (gid0 >= ((int)(counts[0]))) return;
  if (((int)((int)(get_global_id(0)))) >= counts[1]) {
    return;
  } else {
    int o = owner[((int)((int)(get_global_id(0))))];
    int n = neighbour[((int)((int)(get_global_id(0))))];
    float po = pCell[o];
    float pn = pCell[n];
    float snGrad = (deltaCoeffs[((int)((int)(get_global_id(0))))] * (pn - po));
    float gradFace = (magSf[((int)((int)(get_global_id(0))))] * snGrad);
    phiOut[((int)((int)(get_global_id(0))))] = (phiIn[((int)((int)(get_global_id(0))))] - ((paramsF[0] * (coeff[((int)((int)(get_global_id(0))))] * gradFace)) + faceFluxCorrection[((int)((int)(get_global_id(0))))]));
  }
}

)kernel";
const char* k_opencl_navatala_cfd_axpy_in_place = R"kernel(
__kernel void navatala_cfd_axpy_in_place(__global const float* x, __global float* y, __global const int* counts, __global const float* paramsF) {
  int gid0 = (int)get_global_id(0);
  const int nSafeMax = (((int)(counts[0])) > 0 ? ((int)(counts[0])) - 1 : 0);
  const int safeIdx = (gid0 < nSafeMax ? gid0 : nSafeMax);
  if (gid0 >= ((int)(counts[0]))) return;
  if ((int)(get_global_id(0)) >= counts[0]) {
    return;
  } else {
    float yi = y[(int)(get_global_id(0))];
    y[(int)(get_global_id(0))] = (yi + (paramsF[0] * x[(int)(get_global_id(0))]));
  }
}

)kernel";
const char* k_opencl_navatala_cfd_bc_dirichlet_face_flux = R"kernel(
__kernel void navatala_cfd_bc_dirichlet_face_flux(__global const float* cf, __global const float* bcVal, __global const int* bcMask, __global const int* counts, __global float* outFlux) {
  int gid0 = (int)get_global_id(0);
  const int nSafeMax = (((int)(counts[1])) > 0 ? ((int)(counts[1])) - 1 : 0);
  const int safeIdx = (gid0 < nSafeMax ? gid0 : nSafeMax);
  if (gid0 >= ((int)(counts[1]))) return;
  if (((int)((int)(get_global_id(0)))) >= counts[1]) {
    return;
  } else {
    float flux = as_float(0x00000000u);
    if (bcMask[((int)((int)(get_global_id(0))))] == 2) {
      flux = (cf[((int)((int)(get_global_id(0))))] * bcVal[((int)((int)(get_global_id(0))))]);
    }
    outFlux[((int)((int)(get_global_id(0))))] = flux;
  }
}

)kernel";
const char* k_opencl_navatala_cfd_bc_sn_grad_face_flux = R"kernel(
__kernel void navatala_cfd_bc_sn_grad_face_flux(__global const float* cf, __global const float* delta, __global const float* bcSnGrad, __global const int* bcSnGradMask, __global const int* counts, __global float* outFlux) {
  int gid0 = (int)get_global_id(0);
  const int nSafeMax = (((int)(counts[1])) > 0 ? ((int)(counts[1])) - 1 : 0);
  const int safeIdx = (gid0 < nSafeMax ? gid0 : nSafeMax);
  if (gid0 >= ((int)(counts[1]))) return;
  if (((int)((int)(get_global_id(0)))) >= counts[1]) {
    return;
  } else {
    float d = delta[((int)((int)(get_global_id(0))))];
    float sg = bcSnGrad[((int)((int)(get_global_id(0))))];
    float flux = as_float(0x00000000u);
    if (bcSnGradMask[((int)((int)(get_global_id(0))))] != 0) {
      if (d != as_float(0x00000000u)) {
        flux = (cf[((int)((int)(get_global_id(0))))] * (sg / d));
      }
    }
    outFlux[((int)((int)(get_global_id(0))))] = flux;
  }
}

)kernel";
const char* k_opencl_navatala_cfd_cast_f32_to_f64 = R"kernel(
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
__kernel void navatala_cfd_cast_f32_to_f64(__global const float* src, __global double* dst, __global const int* counts) {
  int gid0 = (int)get_global_id(0);
  const int nSafeMax = (((int)(counts[0])) > 0 ? ((int)(counts[0])) - 1 : 0);
  const int safeIdx = (gid0 < nSafeMax ? gid0 : nSafeMax);
  if (gid0 >= ((int)(counts[0]))) return;
  if ((int)(get_global_id(0)) >= counts[0]) {
    return;
  } else {
    dst[(int)(get_global_id(0))] = ((double)(src[(int)(get_global_id(0))]));
  }
}

)kernel";
const char* k_opencl_navatala_cfd_cast_f64_to_f32 = R"kernel(
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
__kernel void navatala_cfd_cast_f64_to_f32(__global const double* src, __global float* dst, __global const int* counts) {
  int gid0 = (int)get_global_id(0);
  const int nSafeMax = (((int)(counts[0])) > 0 ? ((int)(counts[0])) - 1 : 0);
  const int safeIdx = (gid0 < nSafeMax ? gid0 : nSafeMax);
  if (gid0 >= ((int)(counts[0]))) return;
  if ((int)(get_global_id(0)) >= counts[0]) {
    return;
  } else {
    dst[(int)(get_global_id(0))] = ((float)(src[(int)(get_global_id(0))]));
  }
}

)kernel";
const char* k_opencl_navatala_cfd_coeff_to_cf_in_place = R"kernel(
__kernel void navatala_cfd_coeff_to_cf_in_place(__global float* cf, __global const float* magSf, __global const float* delta, __global const int* counts) {
  int gid0 = (int)get_global_id(0);
  const int nSafeMax = (((int)(counts[1])) > 0 ? ((int)(counts[1])) - 1 : 0);
  const int safeIdx = (gid0 < nSafeMax ? gid0 : nSafeMax);
  if (gid0 >= ((int)(counts[1]))) return;
  if (((int)((int)(get_global_id(0)))) >= counts[1]) {
    return;
  } else {
    float v = cf[((int)((int)(get_global_id(0))))];
    cf[((int)((int)(get_global_id(0))))] = ((v * magSf[((int)((int)(get_global_id(0))))]) * delta[((int)((int)(get_global_id(0))))]);
  }
}

)kernel";
const char* k_opencl_navatala_cfd_diag_from_cf = R"kernel(
__kernel void navatala_cfd_diag_from_cf(__global const float* cf, __global const int* offsets, __global const int* faceIdx, __global const float* vol, __global const int* counts, __global float* diag) {
  int gid0 = (int)get_global_id(0);
  const int nSafeMax = (((int)(counts[0])) > 0 ? ((int)(counts[0])) - 1 : 0);
  const int safeIdx = (gid0 < nSafeMax ? gid0 : nSafeMax);
  if (gid0 >= ((int)(counts[0]))) return;
  if ((int)(get_global_id(0)) >= counts[0]) {
    return;
  } else {
    int beg = offsets[(int)(get_global_id(0))];
    int c1 = ((int)(get_global_id(0)) + 1);
    int end = offsets[c1];
    int len = (end - beg);
    float s = as_float(0x00000000u);
    for (int t = 0; t < (int)(len); ++t) {
      int k = (beg + t);
      int f = faceIdx[k];
      s = (s + cf[f]);
    }
    diag[(int)(get_global_id(0))] = (as_float(0x00000000u) - s);
  }
}

)kernel";
const char* k_opencl_navatala_cfd_dic_apply = R"kernel(
__kernel void navatala_cfd_dic_apply(__global const float* r, __global const float* rD, __global const float* upper, __global const int* owner, __global const int* neighbour, __global const int* counts, __global float* z) {
  int gid0 = (int)get_global_id(0);
  const int nSafeMax = (((int)(counts[0])) > 0 ? ((int)(counts[0])) - 1 : 0);
  const int safeIdx = (gid0 < nSafeMax ? gid0 : nSafeMax);
  if (gid0 >= ((int)(counts[0]))) return;
  for (int c = 0; c < (int)(counts[0]); ++c) {
    z[c] = (rD[c] * r[c]);
  }
  for (int f = 0; f < (int)(counts[2]); ++f) {
    int o = owner[f];
    int n = neighbour[f];
    float zn = z[n];
    z[n] = (zn - ((rD[n] * upper[f]) * z[o]));
  }
  for (int t = 0; t < (int)(counts[2]); ++t) {
    int f2 = ((counts[2] - 1) - t);
    int o2 = owner[f2];
    int n2 = neighbour[f2];
    float zo = z[o2];
    z[o2] = (zo - ((rD[o2] * upper[f2]) * z[n2]));
  }
}

)kernel";
const char* k_opencl_navatala_cfd_dic_build_reciprocal_d = R"kernel(
__kernel void navatala_cfd_dic_build_reciprocal_d(__global const float* diag, __global const float* upper, __global const int* owner, __global const int* neighbour, __global const int* counts, __global float* rD) {
  int gid0 = (int)get_global_id(0);
  const int nSafeMax = (((int)(counts[0])) > 0 ? ((int)(counts[0])) - 1 : 0);
  const int safeIdx = (gid0 < nSafeMax ? gid0 : nSafeMax);
  if (gid0 >= ((int)(counts[0]))) return;
  for (int c = 0; c < (int)(counts[0]); ++c) {
    rD[c] = diag[c];
  }
  for (int f = 0; f < (int)(counts[2]); ++f) {
    int o = owner[f];
    int n = neighbour[f];
    float u = upper[f];
    float rdo = rD[o];
    float rdn = rD[n];
    rD[n] = (rdn - ((u * u) / rdo));
  }
  for (int c2 = 0; c2 < (int)(counts[0]); ++c2) {
    float v = rD[c2];
    if (v != as_float(0x00000000u)) {
      rD[c2] = (as_float(0x3f800000u) / v);
    } else {
      rD[c2] = as_float(0x00000000u);
    }
  }
}

)kernel";
const char* k_opencl_navatala_cfd_dot_partials = R"kernel(
__kernel void navatala_cfd_dot_partials(__global const float* a, __global const float* b, __global const int* counts, __global float* outPartials) {
  int gid0 = (int)get_global_id(0);
  __local float tmp[256];
  float v = as_float(0x00000000u);
  if (((int)((int)(get_global_id(0)))) < counts[0]) {
    v = (a[((int)((int)(get_global_id(0))))] * b[((int)((int)(get_global_id(0))))]);
  }
  tmp[((int)((int)(get_local_id(0))))] = v;
  barrier(CLK_LOCAL_MEM_FENCE);
  if (((int)((int)(get_local_id(0)))) < 128) {
    tmp[((int)((int)(get_local_id(0))))] = (tmp[((int)((int)(get_local_id(0))))] + tmp[(((int)((int)(get_local_id(0)))) + 128)]);
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  if (((int)((int)(get_local_id(0)))) < 64) {
    tmp[((int)((int)(get_local_id(0))))] = (tmp[((int)((int)(get_local_id(0))))] + tmp[(((int)((int)(get_local_id(0)))) + 64)]);
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  if (((int)((int)(get_local_id(0)))) < 32) {
    tmp[((int)((int)(get_local_id(0))))] = (tmp[((int)((int)(get_local_id(0))))] + tmp[(((int)((int)(get_local_id(0)))) + 32)]);
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  if (((int)((int)(get_local_id(0)))) < 16) {
    tmp[((int)((int)(get_local_id(0))))] = (tmp[((int)((int)(get_local_id(0))))] + tmp[(((int)((int)(get_local_id(0)))) + 16)]);
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  if (((int)((int)(get_local_id(0)))) < 8) {
    tmp[((int)((int)(get_local_id(0))))] = (tmp[((int)((int)(get_local_id(0))))] + tmp[(((int)((int)(get_local_id(0)))) + 8)]);
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  if (((int)((int)(get_local_id(0)))) < 4) {
    tmp[((int)((int)(get_local_id(0))))] = (tmp[((int)((int)(get_local_id(0))))] + tmp[(((int)((int)(get_local_id(0)))) + 4)]);
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  if (((int)((int)(get_local_id(0)))) < 2) {
    tmp[((int)((int)(get_local_id(0))))] = (tmp[((int)((int)(get_local_id(0))))] + tmp[(((int)((int)(get_local_id(0)))) + 2)]);
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  if (((int)((int)(get_local_id(0)))) < 1) {
    tmp[((int)((int)(get_local_id(0))))] = (tmp[((int)((int)(get_local_id(0))))] + tmp[(((int)((int)(get_local_id(0)))) + 1)]);
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  if (((int)((int)(get_local_id(0)))) == 0) {
    outPartials[((int)((int)(get_group_id(0))))] = tmp[0];
  }
}

)kernel";
const char* k_opencl_navatala_cfd_face_flux = R"kernel(
__kernel void navatala_cfd_face_flux(__global const float* x, __global const int* owner, __global const int* neighbour, __global const float* cf, __global const float* bcVal, __global const int* bcMask, __global const int* counts, __global float* outFlux) {
  int gid0 = (int)get_global_id(0);
  const int nSafeMax = (((int)(counts[1])) > 0 ? ((int)(counts[1])) - 1 : 0);
  const int safeIdx = (gid0 < nSafeMax ? gid0 : nSafeMax);
  if (gid0 >= ((int)(counts[1]))) return;
  if (((int)((int)(get_global_id(0)))) >= counts[1]) {
    return;
  } else {
    int o = owner[((int)((int)(get_global_id(0))))];
    float xo = x[o];
    float other = xo;
    if (((int)((int)(get_global_id(0)))) < counts[2]) {
      int n = neighbour[((int)((int)(get_global_id(0))))];
      other = x[n];
    } else {
      int m = bcMask[((int)((int)(get_global_id(0))))];
      if ((m == 1) || (m == 3)) {
        other = bcVal[((int)((int)(get_global_id(0))))];
      } else {
        if (m == 2) {
          other = as_float(0x00000000u);
        } else {
          if (m == 4) {
            outFlux[((int)((int)(get_global_id(0))))] = bcVal[((int)((int)(get_global_id(0))))];
            return;
          }
        }
      }
    }
    outFlux[((int)((int)(get_global_id(0))))] = (cf[((int)((int)(get_global_id(0))))] * (other - xo));
  }
}

)kernel";
const char* k_opencl_navatala_cfd_inv_diag = R"kernel(
__kernel void navatala_cfd_inv_diag(__global const float* r, __global const float* diag, __global const int* counts, __global float* z) {
  int gid0 = (int)get_global_id(0);
  const int nSafeMax = (((int)(counts[0])) > 0 ? ((int)(counts[0])) - 1 : 0);
  const int safeIdx = (gid0 < nSafeMax ? gid0 : nSafeMax);
  if (gid0 >= ((int)(counts[0]))) return;
  if ((int)(get_global_id(0)) >= counts[0]) {
    return;
  } else {
    float d = diag[(int)(get_global_id(0))];
    float v = r[(int)(get_global_id(0))];
    if (d != as_float(0x00000000u)) {
      z[(int)(get_global_id(0))] = (v / d);
    } else {
      z[(int)(get_global_id(0))] = v;
    }
  }
}

)kernel";
const char* k_opencl_navatala_cfd_laplacian_from_face_flux = R"kernel(
__kernel void navatala_cfd_laplacian_from_face_flux(__global const float* faceFlux, __global const int* offsets, __global const int* faceIdx, __global const float* signF, __global const float* vol, __global const int* counts, __global float* outLap) {
  int gid0 = (int)get_global_id(0);
  if ((int)(get_global_id(0)) >= counts[0]) {
    return;
  } else {
    int beg = offsets[(int)(get_global_id(0))];
    int c1 = ((int)(get_global_id(0)) + 1);
    int end = offsets[c1];
    int len = (end - beg);
    float s = as_float(0x00000000u);
    for (int t = 0; t < (int)(len); ++t) {
      int k = (beg + t);
      int f = faceIdx[k];
      s = (s + (signF[k] * faceFlux[f]));
    }
    outLap[(int)(get_global_id(0))] = s;
  }
}

)kernel";
const char* k_opencl_navatala_cfd_mul_by_vol_in_place = R"kernel(
__kernel void navatala_cfd_mul_by_vol_in_place(__global float* b, __global const float* vol, __global const int* counts) {
  int gid0 = (int)get_global_id(0);
  const int nSafeMax = (((int)(counts[0])) > 0 ? ((int)(counts[0])) - 1 : 0);
  const int safeIdx = (gid0 < nSafeMax ? gid0 : nSafeMax);
  if (gid0 >= ((int)(counts[0]))) return;
  if (((int)((int)(get_global_id(0)))) >= counts[0]) {
    return;
  } else {
    float v = b[((int)((int)(get_global_id(0))))];
    b[((int)((int)(get_global_id(0))))] = (v * vol[((int)((int)(get_global_id(0))))]);
  }
}

)kernel";
const char* k_opencl_navatala_cfd_ref_add_ax = R"kernel(
__kernel void navatala_cfd_ref_add_ax(__global const float* diag, __global const float* x, __global float* outAx, __global const int* refCell) {
  int gid0 = (int)get_global_id(0);
  int c = refCell[0];
  if (c < 0) {
    return;
  } else {
    float cur = outAx[c];
    outAx[c] = (cur + (diag[c] * x[c]));
  }
}

)kernel";
const char* k_opencl_navatala_cfd_ref_add_b = R"kernel(
__kernel void navatala_cfd_ref_add_b(__global const float* diag, __global float* b, __global const int* refCell, __global const float* refVal) {
  int gid0 = (int)get_global_id(0);
  int c = refCell[0];
  if (c < 0) {
    return;
  } else {
    float cur = b[c];
    b[c] = (cur + (diag[c] * refVal[0]));
  }
}

)kernel";
const char* k_opencl_navatala_cfd_rhs_sub = R"kernel(
__kernel void navatala_cfd_rhs_sub(__global const float* b, __global const float* ax, __global const int* counts, __global float* r) {
  int gid0 = (int)get_global_id(0);
  const int nSafeMax = (((int)(counts[0])) > 0 ? ((int)(counts[0])) - 1 : 0);
  const int safeIdx = (gid0 < nSafeMax ? gid0 : nSafeMax);
  if (gid0 >= ((int)(counts[0]))) return;
  if ((int)(get_global_id(0)) >= counts[0]) {
    return;
  } else {
    r[(int)(get_global_id(0))] = (b[(int)(get_global_id(0))] - ax[(int)(get_global_id(0))]);
  }
}

)kernel";
const char* k_opencl_navatala_cfd_shift_in_place = R"kernel(
__kernel void navatala_cfd_shift_in_place(__global float* x, __global const int* counts, __global const float* paramsF) {
  int gid0 = (int)get_global_id(0);
  const int nSafeMax = (((int)(counts[0])) > 0 ? ((int)(counts[0])) - 1 : 0);
  const int safeIdx = (gid0 < nSafeMax ? gid0 : nSafeMax);
  if (gid0 >= ((int)(counts[0]))) return;
  if (((int)((int)(get_global_id(0)))) >= counts[0]) {
    return;
  } else {
    float xi = x[((int)((int)(get_global_id(0))))];
    x[((int)((int)(get_global_id(0))))] = (xi + paramsF[0]);
  }
}

)kernel";
const char* k_opencl_navatala_cfd_sum_abs_partials = R"kernel(
__kernel void navatala_cfd_sum_abs_partials(__global const float* a, __global const int* counts, __global float* outPartials) {
  int gid0 = (int)get_global_id(0);
  __local float tmp[256];
  float v = as_float(0x00000000u);
  if (((int)((int)(get_global_id(0)))) < counts[0]) {
    float ai = a[((int)((int)(get_global_id(0))))];
    if (ai >= as_float(0x00000000u)) {
      v = ai;
    } else {
      v = (as_float(0x00000000u) - ai);
    }
  }
  tmp[((int)((int)(get_local_id(0))))] = v;
  barrier(CLK_LOCAL_MEM_FENCE);
  if (((int)((int)(get_local_id(0)))) < 128) {
    tmp[((int)((int)(get_local_id(0))))] = (tmp[((int)((int)(get_local_id(0))))] + tmp[(((int)((int)(get_local_id(0)))) + 128)]);
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  if (((int)((int)(get_local_id(0)))) < 64) {
    tmp[((int)((int)(get_local_id(0))))] = (tmp[((int)((int)(get_local_id(0))))] + tmp[(((int)((int)(get_local_id(0)))) + 64)]);
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  if (((int)((int)(get_local_id(0)))) < 32) {
    tmp[((int)((int)(get_local_id(0))))] = (tmp[((int)((int)(get_local_id(0))))] + tmp[(((int)((int)(get_local_id(0)))) + 32)]);
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  if (((int)((int)(get_local_id(0)))) < 16) {
    tmp[((int)((int)(get_local_id(0))))] = (tmp[((int)((int)(get_local_id(0))))] + tmp[(((int)((int)(get_local_id(0)))) + 16)]);
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  if (((int)((int)(get_local_id(0)))) < 8) {
    tmp[((int)((int)(get_local_id(0))))] = (tmp[((int)((int)(get_local_id(0))))] + tmp[(((int)((int)(get_local_id(0)))) + 8)]);
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  if (((int)((int)(get_local_id(0)))) < 4) {
    tmp[((int)((int)(get_local_id(0))))] = (tmp[((int)((int)(get_local_id(0))))] + tmp[(((int)((int)(get_local_id(0)))) + 4)]);
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  if (((int)((int)(get_local_id(0)))) < 2) {
    tmp[((int)((int)(get_local_id(0))))] = (tmp[((int)((int)(get_local_id(0))))] + tmp[(((int)((int)(get_local_id(0)))) + 2)]);
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  if (((int)((int)(get_local_id(0)))) < 1) {
    tmp[((int)((int)(get_local_id(0))))] = (tmp[((int)((int)(get_local_id(0))))] + tmp[(((int)((int)(get_local_id(0)))) + 1)]);
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  if (((int)((int)(get_local_id(0)))) == 0) {
    outPartials[((int)((int)(get_group_id(0))))] = tmp[0];
  }
}

)kernel";
const char* k_opencl_navatala_cfd_upper_from_cf = R"kernel(
__kernel void navatala_cfd_upper_from_cf(__global const float* cf, __global const int* counts, __global float* upper) {
  int gid0 = (int)get_global_id(0);
  const int nSafeMax = (((int)(counts[2])) > 0 ? ((int)(counts[2])) - 1 : 0);
  const int safeIdx = (gid0 < nSafeMax ? gid0 : nSafeMax);
  if (gid0 >= ((int)(counts[2]))) return;
  if (((int)((int)(get_global_id(0)))) >= counts[2]) {
    return;
  } else {
    upper[((int)((int)(get_global_id(0))))] = cf[((int)((int)(get_global_id(0))))];
  }
}

)kernel";
const char* k_opencl_navatala_cfd_xpay_in_place = R"kernel(
__kernel void navatala_cfd_xpay_in_place(__global const float* z, __global const float* pIn, __global float* pOut, __global const int* counts, __global const float* paramsF) {
  int gid0 = (int)get_global_id(0);
  const int nSafeMax = (((int)(counts[0])) > 0 ? ((int)(counts[0])) - 1 : 0);
  const int safeIdx = (gid0 < nSafeMax ? gid0 : nSafeMax);
  if (gid0 >= ((int)(counts[0]))) return;
  if ((int)(get_global_id(0)) >= counts[0]) {
    return;
  } else {
    pOut[(int)(get_global_id(0))] = (z[(int)(get_global_id(0))] + (paramsF[0] * pIn[(int)(get_global_id(0))]));
  }
}

)kernel";
const char* k_opencl_navatala_cfd_dot_finalize_partials = R"kernel(
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
__kernel void navatala_cfd_dot_finalize_partials(__global const double* partials, __global const uint* count, __global double* result) {
  int gid0 = (int)get_global_id(0);
  if ((int)(get_global_id(0)) >= 1) {
    return;
  } else {
    double s = as_double(0x0000000000000000ul);
    for (int i = 0; i < (int)(((int)(count[0]))); ++i) {
      s = (s + partials[i]);
    }
    result[0] = s;
  }
}

)kernel";
const char* k_opencl_navatala_cfd_negate_scalar = R"kernel(
__kernel void navatala_cfd_negate_scalar(__global const float* _input, __global float* _output) {
  int gid0 = (int)get_global_id(0);
  if ((int)(get_global_id(0)) >= 1) {
    return;
  } else {
    float v = _input[0];
    _output[0] = (-v);
  }
}

)kernel";
const char* k_opencl_navatala_cfd_pcg_compute_alpha = R"kernel(
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
__kernel void navatala_cfd_pcg_compute_alpha(__global const double* rz, __global const double* pAp, __global float* alpha) {
  int gid0 = (int)get_global_id(0);
  if ((int)(get_global_id(0)) >= 1) {
    return;
  } else {
    double rz_val = rz[0];
    double pAp_val = pAp[0];
    if (pAp_val != as_double(0x0000000000000000ul)) {
      alpha[0] = ((float)((rz_val / pAp_val)));
    } else {
      alpha[0] = as_float(0x00000000u);
    }
  }
}

)kernel";
const char* k_opencl_navatala_cfd_pcg_compute_beta = R"kernel(
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
__kernel void navatala_cfd_pcg_compute_beta(__global const double* rzNew, __global const double* rzOld, __global float* beta) {
  int gid0 = (int)get_global_id(0);
  if ((int)(get_global_id(0)) >= 1) {
    return;
  } else {
    double new_val = rzNew[0];
    double old_val = rzOld[0];
    if (old_val != as_double(0x0000000000000000ul)) {
      beta[0] = ((float)((new_val / old_val)));
    } else {
      beta[0] = as_float(0x00000000u);
    }
  }
}

)kernel";
const char* k_opencl_navatala_cfd_exact_mat_vec = R"kernel(
__kernel void navatala_cfd_exact_mat_vec(__global const float* x, __global const int* owner, __global const int* neighbour, __global const uint* offsets, __global const uint* faceIdx, __global const int* sign, __global const float* upper, __global const float* lower, __global const float* diag, __global const int* counts, __global float* outAx) {
  int gid0 = (int)get_global_id(0);
  const int nSafeMax = (((int)(counts[0])) > 0 ? ((int)(counts[0])) - 1 : 0);
  const int safeIdx = (gid0 < nSafeMax ? gid0 : nSafeMax);
  if (gid0 >= ((int)(counts[0]))) return;
  if ((int)(get_global_id(0)) >= ((int)(counts[0]))) {
    return;
  } else {
    float s = (diag[(int)(get_global_id(0))] * x[(int)(get_global_id(0))]);
    uint beg = offsets[(int)(get_global_id(0))];
    int c1 = ((int)(get_global_id(0)) + 1);
    uint end = offsets[c1];
    int len = ((int)((end - beg)));
    for (int t = 0; t < (int)(len); ++t) {
      int k = (((int)(beg)) + t);
      uint f = faceIdx[k];
      if (((int)(f)) < ((int)(counts[2]))) {
        int sg = sign[k];
        if (sg >= 0) {
          int nb = neighbour[((int)(f))];
          s = (s + (upper[((int)(f))] * x[nb]));
        } else {
          int ow = owner[((int)(f))];
          s = (s + (lower[((int)(f))] * x[ow]));
        }
      }
    }
    outAx[(int)(get_global_id(0))] = s;
  }
}

)kernel";
const char* k_opencl_navatala_cfd_exact_mat_vec_f64 = R"kernel(
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
__kernel void navatala_cfd_exact_mat_vec_f64(__global const float* x, __global const int* owner, __global const int* neighbour, __global const uint* offsets, __global const uint* faceIdx, __global const int* sign, __global const double* upper, __global const double* lower, __global const double* diag, __global const int* counts, __global float* outAx) {
  int gid0 = (int)get_global_id(0);
  const int nSafeMax = (((int)(counts[0])) > 0 ? ((int)(counts[0])) - 1 : 0);
  const int safeIdx = (gid0 < nSafeMax ? gid0 : nSafeMax);
  if (gid0 >= ((int)(counts[0]))) return;
  if ((int)(get_global_id(0)) >= ((int)(counts[0]))) {
    return;
  } else {
    double s = (diag[(int)(get_global_id(0))] * ((double)(x[(int)(get_global_id(0))])));
    uint beg = offsets[(int)(get_global_id(0))];
    int c1 = ((int)(get_global_id(0)) + 1);
    uint end = offsets[c1];
    int len = ((int)((end - beg)));
    for (int t = 0; t < (int)(len); ++t) {
      int k = (((int)(beg)) + t);
      uint f = faceIdx[k];
      if (((int)(f)) < ((int)(counts[2]))) {
        int sg = sign[k];
        if (sg >= 0) {
          int nb = neighbour[((int)(f))];
          s = (s + (upper[((int)(f))] * ((double)(x[nb]))));
        } else {
          int ow = owner[((int)(f))];
          s = (s + (lower[((int)(f))] * ((double)(x[ow]))));
        }
      }
    }
    outAx[(int)(get_global_id(0))] = ((float)(s));
  }
}

)kernel";
const char* k_opencl_navatala_cfd_face_flux_boundary = R"kernel(
__kernel void navatala_cfd_face_flux_boundary(__global const float* x, __global const int* owner, __global const float* cf, __global const float* bcVal, __global const uint* bcMask, __global const uint* params, __global float* faceFlux) {
  int gid0 = (int)get_global_id(0);
  if ((int)(get_global_id(0)) >= ((int)(params[0]))) {
    return;
  } else {
    int gf = (((int)(params[1])) + (int)(get_global_id(0)));
    int o = owner[gf];
    float xo = x[o];
    uint m = bcMask[gf];
    int mInt = ((int)(m));
    if (mInt == 4) {
      faceFlux[gf] = bcVal[gf];
    } else {
      float other = xo;
      if (mInt == 1) {
        other = bcVal[gf];
      } else {
        if (mInt == 3) {
          other = bcVal[gf];
        } else {
          if (mInt == 2) {
            other = as_float(0x00000000u);
          }
        }
      }
      faceFlux[gf] = (cf[gf] * (other - xo));
    }
  }
}

)kernel";
const char* k_opencl_navatala_cfd_face_flux_internal = R"kernel(
__kernel void navatala_cfd_face_flux_internal(__global const float* x, __global const int* owner, __global const int* neighbour, __global const float* cf, __global const uint* params, __global float* faceFlux) {
  int gid0 = (int)get_global_id(0);
  if ((int)(get_global_id(0)) >= ((int)(params[0]))) {
    return;
  } else {
    int o = owner[(int)(get_global_id(0))];
    int n = neighbour[(int)(get_global_id(0))];
    faceFlux[(int)(get_global_id(0))] = (cf[(int)(get_global_id(0))] * (x[n] - x[o]));
  }
}

)kernel";
const char* k_opencl_navatala_cfd_pressure_face_flux_correction_corrected = R"kernel(
__kernel void navatala_cfd_pressure_face_flux_correction_corrected(__global const int* owner, __global const int* neighbour, __global const float* weights, __global const float* gradX, __global const float* gradY, __global const float* gradZ, __global const float* magSf, __global const float* dCorrX, __global const float* dCorrY, __global const float* dCorrZ, __global const float* rAUf, __global const uint* params, __global float* outFaceFlux) {
  int gid0 = (int)get_global_id(0);
  if ((int)(get_global_id(0)) >= ((int)(params[0]))) {
    return;
  } else {
    if ((int)(get_global_id(0)) >= ((int)(params[1]))) {
      outFaceFlux[(int)(get_global_id(0))] = as_float(0x00000000u);
    } else {
      int own = owner[(int)(get_global_id(0))];
      int nei = neighbour[(int)(get_global_id(0))];
      float w = weights[(int)(get_global_id(0))];
      float omw = (as_float(0x3f800000u) - w);
      float gx = ((w * gradX[own]) + (omw * gradX[nei]));
      float gy = ((w * gradY[own]) + (omw * gradY[nei]));
      float gz = ((w * gradZ[own]) + (omw * gradZ[nei]));
      float corr = (((dCorrX[(int)(get_global_id(0))] * gx) + (dCorrY[(int)(get_global_id(0))] * gy)) + (dCorrZ[(int)(get_global_id(0))] * gz));
      outFaceFlux[(int)(get_global_id(0))] = (rAUf[(int)(get_global_id(0))] * (magSf[(int)(get_global_id(0))] * corr));
    }
  }
}

)kernel";
const char* k_opencl_navatala_cfd_pressure_sn_grad = R"kernel(
__kernel void navatala_cfd_pressure_sn_grad(__global const float* phiHbyA, __global const int* ownerAll, __global const float* sfX, __global const float* sfY, __global const float* sfZ, __global const float* magSf, __global const float* ux, __global const float* uy, __global const float* uz, __global const float* uBcX, __global const float* uBcY, __global const float* uBcZ, __global const uint* uBcMask, __global const float* rAUf, __global const uint* snGradMask, __global const uint* params, __global float* outSnGrad) {
  int gid0 = (int)get_global_id(0);
  if ((int)(get_global_id(0)) >= ((int)(params[1]))) {
    return;
  } else {
    uint mask = snGradMask[(int)(get_global_id(0))];
    if (((int)(mask)) == 0) {
      outSnGrad[(int)(get_global_id(0))] = as_float(0x00000000u);
    } else {
      int o = ownerAll[(int)(get_global_id(0))];
      if (o < 0) {
        outSnGrad[(int)(get_global_id(0))] = as_float(0x00000000u);
      } else {
        if (o >= ((int)(params[0]))) {
          outSnGrad[(int)(get_global_id(0))] = as_float(0x00000000u);
        } else {
          float uxF = ux[o];
          float uyF = uy[o];
          float uzF = uz[o];
          uint bcm = uBcMask[(int)(get_global_id(0))];
          if (((int)(bcm)) == 2) {
            uxF = uBcX[(int)(get_global_id(0))];
            uyF = uBcY[(int)(get_global_id(0))];
            uzF = uBcZ[(int)(get_global_id(0))];
          }
          float phiU = (((sfX[(int)(get_global_id(0))] * uxF) + (sfY[(int)(get_global_id(0))] * uyF)) + (sfZ[(int)(get_global_id(0))] * uzF));
          float rf = rAUf[(int)(get_global_id(0))];
          float ms = magSf[(int)(get_global_id(0))];
          float denom = (rf * ms);
          float ph = phiHbyA[(int)(get_global_id(0))];
          if (fabs(denom) < as_float(0x1e3ce508u)) {
            outSnGrad[(int)(get_global_id(0))] = as_float(0x00000000u);
          } else {
            outSnGrad[(int)(get_global_id(0))] = ((ph - (rf * phiU)) / denom);
          }
        }
      }
    }
  }
}

)kernel";
const char* k_opencl_navatala_cfd_u_correct = R"kernel(
__kernel void navatala_cfd_u_correct(__global const float* pCell, __global const int* owner, __global const int* neighbour, __global const float* sfX, __global const float* sfY, __global const float* sfZ, __global const float* bcVal, __global const int* bcMask, __global const int* offsets, __global const int* faceIdx, __global const float* sign, __global const float* vol, __global const float* rAU, __global const float* ux, __global const float* uy, __global const float* uz, __global const int* counts, __global const float* paramsF, __global float* outX, __global float* outY, __global float* outZ) {
  int gid0 = (int)get_global_id(0);
  const int nSafeMax = (((int)(counts[0])) > 0 ? ((int)(counts[0])) - 1 : 0);
  const int safeIdx = (gid0 < nSafeMax ? gid0 : nSafeMax);
  if (gid0 >= ((int)(counts[0]))) return;
  if (((int)((int)(get_global_id(0)))) >= counts[0]) {
    return;
  } else {
    int beg = offsets[((int)((int)(get_global_id(0))))];
    int c1 = (((int)((int)(get_global_id(0)))) + 1);
    int end = offsets[c1];
    int len = (end - beg);
    float gx = as_float(0x00000000u);
    float gy = as_float(0x00000000u);
    float gz = as_float(0x00000000u);
    for (int t = 0; t < (int)(len); ++t) {
      int k = (beg + t);
      int f = faceIdx[k];
      int o = owner[f];
      float pf = pCell[o];
      if (f < counts[2]) {
        int n = neighbour[f];
        pf = (as_float(0x3f000000u) * (pCell[o] + pCell[n]));
      } else {
        if (bcMask[f] != 0) {
          pf = bcVal[f];
        }
      }
      float s = sign[k];
      gx = (gx + (s * (sfX[f] * pf)));
      gy = (gy + (s * (sfY[f] * pf)));
      gz = (gz + (s * (sfZ[f] * pf)));
    }
    float invV = (as_float(0x3f800000u) / vol[((int)((int)(get_global_id(0))))]);
    float sC = (paramsF[0] * rAU[((int)((int)(get_global_id(0))))]);
    float dUx = ((gx * invV) * sC);
    float dUy = ((gy * invV) * sC);
    float dUz = ((gz * invV) * sC);
    outX[((int)((int)(get_global_id(0))))] = (ux[((int)((int)(get_global_id(0))))] - dUx);
    outY[((int)((int)(get_global_id(0))))] = (uy[((int)((int)(get_global_id(0))))] - dUy);
    outZ[((int)((int)(get_global_id(0))))] = (uz[((int)((int)(get_global_id(0))))] - dUz);
  }
}

)kernel";
const char* k_opencl_navatala_cfd_u_correct_reconstruct = R"kernel(
__kernel void navatala_cfd_u_correct_reconstruct(__global const float* pCell, __global const float* rAU, __global const float* hbx, __global const float* hby, __global const float* hbz, __global const float* rAUf, __global const float* phig, __global const int* owner, __global const int* neighbour, __global const float* sfX, __global const float* sfY, __global const float* sfZ, __global const float* magSf, __global const float* deltaCoeffs, __global const int* offsets, __global const int* faceIdx, __global const float* sign, __global const float* vol, __global const float* bcVal, __global const int* bcMask, __global const float* bcSnGrad, __global const int* bcSnGradMask, __global const float* faceFluxCorrection, __global const int* counts, __global const float* paramsF, __global float* outX, __global float* outY, __global float* outZ) {
  int gid0 = (int)get_global_id(0);
  const int nSafeMax = (((int)(counts[0])) > 0 ? ((int)(counts[0])) - 1 : 0);
  const int safeIdx = (gid0 < nSafeMax ? gid0 : nSafeMax);
  if (gid0 >= ((int)(counts[0]))) return;
  if (((int)((int)(get_global_id(0)))) >= counts[0]) {
    return;
  } else {
    int beg = offsets[((int)((int)(get_global_id(0))))];
    int c1 = (((int)((int)(get_global_id(0)))) + 1);
    int end = offsets[c1];
    int len = (end - beg);
    float a00 = as_float(0x00000000u);
    float a01 = as_float(0x00000000u);
    float a02 = as_float(0x00000000u);
    float a10 = as_float(0x00000000u);
    float a11 = as_float(0x00000000u);
    float a12 = as_float(0x00000000u);
    float a20 = as_float(0x00000000u);
    float a21 = as_float(0x00000000u);
    float a22 = as_float(0x00000000u);
    float bx = as_float(0x00000000u);
    float by = as_float(0x00000000u);
    float bz = as_float(0x00000000u);
    for (int t = 0; t < (int)(len); ++t) {
      int k = (beg + t);
      int f = faceIdx[k];
      if (f < counts[1]) {
        int o = owner[f];
        float po = pCell[o];
        float other = po;
        if (f < counts[2]) {
          int n = neighbour[f];
          other = pCell[n];
        } else {
          if (bcMask[f] != 0) {
            other = bcVal[f];
          }
        }
        float snGrad = (deltaCoeffs[f] * (other - po));
        float phiTerm = as_float(0x00000000u);
        if (bcSnGradMask[f] == 2) {
          phiTerm = bcSnGrad[f];
        } else {
          if (bcSnGradMask[f] == 1) {
            snGrad = bcSnGrad[f];
          }
          float gradFace = (magSf[f] * snGrad);
          float denom = rAUf[f];
          float pFlux = ((paramsF[0] * (denom * gradFace)) + faceFluxCorrection[f]);
          if (denom != as_float(0x00000000u)) {
            phiTerm = ((phig[f] - pFlux) / denom);
          }
        }
        float ms = magSf[f];
        if (ms > as_float(0x00000000u)) {
          float nx = (sfX[f] / ms);
          float ny = (sfY[f] / ms);
          float nz = (sfZ[f] / ms);
          a00 = (a00 + (nx * sfX[f]));
          a01 = (a01 + (nx * sfY[f]));
          a02 = (a02 + (nx * sfZ[f]));
          a10 = (a10 + (ny * sfX[f]));
          a11 = (a11 + (ny * sfY[f]));
          a12 = (a12 + (ny * sfZ[f]));
          a20 = (a20 + (nz * sfX[f]));
          a21 = (a21 + (nz * sfY[f]));
          a22 = (a22 + (nz * sfZ[f]));
          bx = (bx + (nx * phiTerm));
          by = (by + (ny * phiTerm));
          bz = (bz + (nz * phiTerm));
        }
      }
    }
    float smallConst = as_float(0x33d6bf95u);
    float rootVSmall = as_float(0x1e3ce508u);
    float magSqr_xx = (a00 * a00);
    float magSqr_yy = (a11 * a11);
    float magSqr_zz = (a22 * a22);
    float threshold = (smallConst * ((magSqr_xx + magSqr_yy) + magSqr_zz));
    bool small_xx = (magSqr_xx < threshold);
    bool small_yy = (magSqr_yy < threshold);
    bool small_zz = (magSqr_zz < threshold);
    float w00 = a00;
    float w01 = a01;
    float w02 = a02;
    float w10 = a10;
    float w11 = a11;
    float w12 = a12;
    float w20 = a20;
    float w21 = a21;
    float w22 = a22;
    if (small_xx) {
      w00 = (w00 + as_float(0x3f800000u));
    }
    if (small_yy) {
      w11 = (w11 + as_float(0x3f800000u));
    }
    if (small_zz) {
      w22 = (w22 + as_float(0x3f800000u));
    }
    float det = (((((w00 * w11) * w22) + ((w01 * w12) * w20)) + ((w02 * w10) * w21)) - ((((w00 * w12) * w21) + ((w01 * w10) * w22)) + ((w02 * w11) * w20)));
    float rx = as_float(0x00000000u);
    float ry = as_float(0x00000000u);
    float rz = as_float(0x00000000u);
    if (fabs(det) >= rootVSmall) {
      float invDet = (as_float(0x3f800000u) / det);
      float i00 = (((w11 * w22) - (w21 * w12)) * invDet);
      float i01 = (((w02 * w21) - (w01 * w22)) * invDet);
      float i02 = (((w01 * w12) - (w02 * w11)) * invDet);
      float i10 = (((w12 * w20) - (w10 * w22)) * invDet);
      float i11 = (((w00 * w22) - (w02 * w20)) * invDet);
      float i12 = (((w02 * w10) - (w00 * w12)) * invDet);
      float i20 = (((w10 * w21) - (w11 * w20)) * invDet);
      float i21 = (((w01 * w20) - (w00 * w21)) * invDet);
      float i22 = (((w00 * w11) - (w10 * w01)) * invDet);
      if (small_xx) {
        i00 = (i00 - as_float(0x3f800000u));
      }
      if (small_yy) {
        i11 = (i11 - as_float(0x3f800000u));
      }
      if (small_zz) {
        i22 = (i22 - as_float(0x3f800000u));
      }
      rx = (((i00 * bx) + (i01 * by)) + (i02 * bz));
      ry = (((i10 * bx) + (i11 * by)) + (i12 * bz));
      rz = (((i20 * bx) + (i21 * by)) + (i22 * bz));
    }
    float fac = rAU[((int)((int)(get_global_id(0))))];
    outX[((int)((int)(get_global_id(0))))] = (hbx[((int)((int)(get_global_id(0))))] + (fac * rx));
    outY[((int)((int)(get_global_id(0))))] = (hby[((int)((int)(get_global_id(0))))] + (fac * ry));
    outZ[((int)((int)(get_global_id(0))))] = (hbz[((int)((int)(get_global_id(0))))] + (fac * rz));
  }
}

)kernel";
const char* k_opencl_navatala_cfd_grad_alpha_cell = R"kernel(
__kernel void navatala_cfd_grad_alpha_cell(__global const float* alphaF, __global const float* sfX, __global const float* sfY, __global const float* sfZ, __global const int* offsets, __global const int* faceIdx, __global const float* sign, __global const float* vol, __global const int* counts, __global float* gx, __global float* gy, __global float* gz) {
  int gid0 = (int)get_global_id(0);
  const int nSafeMax = (((int)(counts[0])) > 0 ? ((int)(counts[0])) - 1 : 0);
  const int safeIdx = (gid0 < nSafeMax ? gid0 : nSafeMax);
  if (gid0 >= ((int)(counts[0]))) return;
  if (((int)((int)(get_global_id(0)))) >= counts[0]) {
    return;
  } else {
    int beg = offsets[((int)((int)(get_global_id(0))))];
    int c1 = (((int)((int)(get_global_id(0)))) + 1);
    int end = offsets[c1];
    int len = (end - beg);
    float sx = as_float(0x00000000u);
    float sy = as_float(0x00000000u);
    float sz = as_float(0x00000000u);
    for (int t = 0; t < (int)(len); ++t) {
      int k = (beg + t);
      int f = faceIdx[k];
      float s = (sign[k] * alphaF[f]);
      sx = (sx + (s * sfX[f]));
      sy = (sy + (s * sfY[f]));
      sz = (sz + (s * sfZ[f]));
    }
    float invV = (as_float(0x3f800000u) / vol[((int)((int)(get_global_id(0))))]);
    gx[((int)((int)(get_global_id(0))))] = (sx * invV);
    gy[((int)((int)(get_global_id(0))))] = (sy * invV);
    gz[((int)((int)(get_global_id(0))))] = (sz * invV);
  }
}

)kernel";
const char* k_opencl_navatala_cfd_grad_alpha_cell_f64 = R"kernel(
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
__kernel void navatala_cfd_grad_alpha_cell_f64(__global const double* alphaF, __global const double* sfX, __global const double* sfY, __global const double* sfZ, __global const int* offsets, __global const int* faceIdx, __global const float* sign, __global const double* vol, __global const int* counts, __global double* gx, __global double* gy, __global double* gz) {
  int gid0 = (int)get_global_id(0);
  const int nSafeMax = (((int)(counts[0])) > 0 ? ((int)(counts[0])) - 1 : 0);
  const int safeIdx = (gid0 < nSafeMax ? gid0 : nSafeMax);
  if (gid0 >= ((int)(counts[0]))) return;
  if (((int)((int)(get_global_id(0)))) >= counts[0]) {
    return;
  } else {
    int beg = offsets[((int)((int)(get_global_id(0))))];
    int c1 = (((int)((int)(get_global_id(0)))) + 1);
    int end = offsets[c1];
    int len = (end - beg);
    double sx = as_double(0x0000000000000000ul);
    double sy = as_double(0x0000000000000000ul);
    double sz = as_double(0x0000000000000000ul);
    for (int t = 0; t < (int)(len); ++t) {
      int k = (beg + t);
      int f = faceIdx[k];
      double signV = ((double)(sign[k]));
      double aV = ((double)(alphaF[f]));
      double s = (signV * aV);
      double sfXv = ((double)(sfX[f]));
      double sfYv = ((double)(sfY[f]));
      double sfZv = ((double)(sfZ[f]));
      sx = (sx + (s * sfXv));
      sy = (sy + (s * sfYv));
      sz = (sz + (s * sfZv));
    }
    double invV = (as_double(0x3ff0000000000000ul) / ((double)(vol[((int)((int)(get_global_id(0))))])));
    gx[((int)((int)(get_global_id(0))))] = (sx * invV);
    gy[((int)((int)(get_global_id(0))))] = (sy * invV);
    gz[((int)((int)(get_global_id(0))))] = (sz * invV);
  }
}

)kernel";
const char* k_opencl_navatala_cfd_n_hatf_all = R"kernel(
__kernel void navatala_cfd_n_hatf_all(__global const float* gx, __global const float* gy, __global const float* gz, __global const float* sfX, __global const float* sfY, __global const float* sfZ, __global const int* owner, __global const int* nei, __global const float* weights, __global const float* theta, __global const uint* thetaMask, __global const float* paramsF, __global const int* counts, __global float* contactGrad, __global float* nHatf) {
  int gid0 = (int)get_global_id(0);
  const int nSafeMax = (((int)(counts[0])) > 0 ? ((int)(counts[0])) - 1 : 0);
  const int safeIdx = (gid0 < nSafeMax ? gid0 : nSafeMax);
  if (gid0 >= ((int)(counts[0]))) return;
  if (((int)((int)(get_global_id(0)))) >= counts[1]) {
    return;
  } else {
    int o = owner[((int)((int)(get_global_id(0))))];
    int n = nei[((int)((int)(get_global_id(0))))];
    bool isBoundary = (((int)((int)(get_global_id(0)))) >= counts[2]);
    float w = weights[((int)((int)(get_global_id(0))))];
    float wm = (as_float(0x3f800000u) - w);
    float gfx = ((isBoundary) ? (gx[o]) : (((w * gx[o]) + (wm * gx[n]))));
    float gfy = ((isBoundary) ? (gy[o]) : (((w * gy[o]) + (wm * gy[n]))));
    float gfz = ((isBoundary) ? (gz[o]) : (((w * gz[o]) + (wm * gz[n]))));
    float magg = sqrt((((gfx * gfx) + (gfy * gfy)) + (gfz * gfz)));
    float inv = (as_float(0x3f800000u) / (magg + paramsF[0]));
    float nx = (gfx * inv);
    float ny = (gfy * inv);
    float nz = (gfz * inv);
    contactGrad[((int)((int)(get_global_id(0))))] = as_float(0x00000000u);
    if (thetaMask[((int)((int)(get_global_id(0))))] != (uint)(0u)) {
      float sfx = sfX[((int)((int)(get_global_id(0))))];
      float sfy = sfY[((int)((int)(get_global_id(0))))];
      float sfz = sfZ[((int)((int)(get_global_id(0))))];
      float magSf = sqrt((((sfx * sfx) + (sfy * sfy)) + (sfz * sfz)));
      if (magSf > as_float(0x1e3ce508u)) {
        float nfx = (sfx / magSf);
        float nfy = (sfy / magSf);
        float nfz = (sfz / magSf);
        float a12raw = (((nx * nfx) + (ny * nfy)) + (nz * nfz));
        float a12 = a12raw;
        if (a12 < as_float(0xbf800000u)) {
          a12 = as_float(0xbf800000u);
        } else {
          if (a12 > as_float(0x3f800000u)) {
            a12 = as_float(0x3f800000u);
          }
        }
        float th = theta[((int)((int)(get_global_id(0))))];
        float b1 = cos(th);
        float acosA12 = acos(a12);
        float b2 = cos((acosA12 - th));
        float det = (as_float(0x3f800000u) - (a12 * a12));
        if (det < as_float(0x358637bdu)) {
          det = as_float(0x358637bdu);
        }
        float a = ((b1 - (a12 * b2)) / det);
        float b = ((b2 - (a12 * b1)) / det);
        float nx2 = ((a * nfx) + (b * nx));
        float ny2 = ((a * nfy) + (b * ny));
        float nz2 = ((a * nfz) + (b * nz));
        float magN = sqrt((((nx2 * nx2) + (ny2 * ny2)) + (nz2 * nz2)));
        float invN = (as_float(0x3f800000u) / (magN + paramsF[0]));
        nx = (nx2 * invN);
        ny = (ny2 * invN);
        nz = (nz2 * invN);
        float dot = (((nx * nfx) + (ny * nfy)) + (nz * nfz));
        contactGrad[((int)((int)(get_global_id(0))))] = (dot * magg);
      }
    }
    nHatf[((int)((int)(get_global_id(0))))] = (((nx * sfX[((int)((int)(get_global_id(0))))]) + (ny * sfY[((int)((int)(get_global_id(0))))])) + (nz * sfZ[((int)((int)(get_global_id(0))))]));
  }
}

)kernel";
const char* k_opencl_navatala_cfd_n_hatf_all_f64 = R"kernel(
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
__kernel void navatala_cfd_n_hatf_all_f64(__global const double* gx, __global const double* gy, __global const double* gz, __global const double* sfX, __global const double* sfY, __global const double* sfZ, __global const int* owner, __global const int* nei, __global const double* weights, __global const float* theta, __global const uint* thetaMask, __global const double* paramsF, __global const int* counts, __global double* contactGrad, __global double* nHatf) {
  int gid0 = (int)get_global_id(0);
  const int nSafeMax = (((int)(counts[0])) > 0 ? ((int)(counts[0])) - 1 : 0);
  const int safeIdx = (gid0 < nSafeMax ? gid0 : nSafeMax);
  if (gid0 >= ((int)(counts[0]))) return;
  if (((int)((int)(get_global_id(0)))) >= counts[1]) {
    return;
  } else {
    int o = owner[((int)((int)(get_global_id(0))))];
    int n = nei[((int)((int)(get_global_id(0))))];
    bool isBoundary = (((int)((int)(get_global_id(0)))) >= counts[2]);
    double w = ((double)(weights[((int)((int)(get_global_id(0))))]));
    double wm = (as_double(0x3ff0000000000000ul) - w);
    double gfx = ((isBoundary) ? (gx[o]) : (((w * gx[o]) + (wm * gx[n]))));
    double gfy = ((isBoundary) ? (gy[o]) : (((w * gy[o]) + (wm * gy[n]))));
    double gfz = ((isBoundary) ? (gz[o]) : (((w * gz[o]) + (wm * gz[n]))));
    double magg = sqrt((((gfx * gfx) + (gfy * gfy)) + (gfz * gfz)));
    double inv = (as_double(0x3ff0000000000000ul) / (magg + paramsF[0]));
    double nx = (gfx * inv);
    double ny = (gfy * inv);
    double nz = (gfz * inv);
    contactGrad[((int)((int)(get_global_id(0))))] = as_double(0x0000000000000000ul);
    if (thetaMask[((int)((int)(get_global_id(0))))] != (uint)(0u)) {
      double sfx = ((double)(sfX[((int)((int)(get_global_id(0))))]));
      double sfy = ((double)(sfY[((int)((int)(get_global_id(0))))]));
      double sfz = ((double)(sfZ[((int)((int)(get_global_id(0))))]));
      double magSf = sqrt((((sfx * sfx) + (sfy * sfy)) + (sfz * sfz)));
      if (magSf > as_double(0x3bc79ca10c924223ul)) {
        double nfx = (sfx / magSf);
        double nfy = (sfy / magSf);
        double nfz = (sfz / magSf);
        double a12raw = (((nx * nfx) + (ny * nfy)) + (nz * nfz));
        double a12 = a12raw;
        if (a12 < as_double(0xbff0000000000000ul)) {
          a12 = as_double(0xbff0000000000000ul);
        } else {
          if (a12 > as_double(0x3ff0000000000000ul)) {
            a12 = as_double(0x3ff0000000000000ul);
          }
        }
        double th = ((double)(theta[((int)((int)(get_global_id(0))))]));
        double b1 = cos(th);
        double acosA12 = acos(a12);
        double b2 = cos((acosA12 - th));
        double det = (as_double(0x3ff0000000000000ul) - (a12 * a12));
        if (det < as_double(0x3eb0c6f7a0b5ed8dul)) {
          det = as_double(0x3eb0c6f7a0b5ed8dul);
        }
        double a = ((b1 - (a12 * b2)) / det);
        double b = ((b2 - (a12 * b1)) / det);
        double nx2 = ((a * nfx) + (b * nx));
        double ny2 = ((a * nfy) + (b * ny));
        double nz2 = ((a * nfz) + (b * nz));
        double magN = sqrt((((nx2 * nx2) + (ny2 * ny2)) + (nz2 * nz2)));
        double invN = (as_double(0x3ff0000000000000ul) / (magN + paramsF[0]));
        nx = (nx2 * invN);
        ny = (ny2 * invN);
        nz = (nz2 * invN);
        double dot = (((nx * nfx) + (ny * nfy)) + (nz * nfz));
        contactGrad[((int)((int)(get_global_id(0))))] = (dot * magg);
      }
    }
    nHatf[((int)((int)(get_global_id(0))))] = (((nx * ((double)(sfX[((int)((int)(get_global_id(0))))]))) + (ny * ((double)(sfY[((int)((int)(get_global_id(0))))])))) + (nz * ((double)(sfZ[((int)((int)(get_global_id(0))))]))));
  }
}

)kernel";
const char* k_opencl_navatala_cfd_stf_all_faces = R"kernel(
__kernel void navatala_cfd_stf_all_faces(__global const float* divN, __global const float* alpha, __global const float* alphaF, __global const float* delta, __global const int* owner, __global const int* nei, __global const float* weights, __global const float* contactGrad, __global const uint* thetaMask, __global const float* paramsF, __global const int* counts, __global float* stf) {
  int gid0 = (int)get_global_id(0);
  const int nSafeMax = (((int)(counts[0])) > 0 ? ((int)(counts[0])) - 1 : 0);
  const int safeIdx = (gid0 < nSafeMax ? gid0 : nSafeMax);
  if (gid0 >= ((int)(counts[0]))) return;
  if (((int)((int)(get_global_id(0)))) >= counts[1]) {
    return;
  } else {
    int o = owner[((int)((int)(get_global_id(0))))];
    int n = nei[((int)((int)(get_global_id(0))))];
    float w = weights[((int)((int)(get_global_id(0))))];
    float wm = (as_float(0x3f800000u) - w);
    float divInterp = ((w * divN[o]) + (wm * divN[n]));
    float Kf = (as_float(0xbf800000u) * divInterp);
    float snGrad = as_float(0x00000000u);
    if (((int)((int)(get_global_id(0)))) < counts[2]) {
      snGrad = (delta[((int)((int)(get_global_id(0))))] * (alpha[n] - alpha[o]));
    } else {
      if (thetaMask[((int)((int)(get_global_id(0))))] != (uint)(0u)) {
        snGrad = contactGrad[((int)((int)(get_global_id(0))))];
      } else {
        snGrad = (delta[((int)((int)(get_global_id(0))))] * (alphaF[((int)((int)(get_global_id(0))))] - alpha[o]));
      }
    }
    stf[((int)((int)(get_global_id(0))))] = ((paramsF[0] * Kf) * snGrad);
  }
}

)kernel";
const char* k_opencl_navatala_cfd_stf_all_faces_f64 = R"kernel(
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
__kernel void navatala_cfd_stf_all_faces_f64(__global const double* divN, __global const float* alpha, __global const float* alphaF, __global const float* delta, __global const int* owner, __global const int* nei, __global const float* weights, __global const double* contactGrad, __global const uint* thetaMask, __global const float* paramsF, __global const int* counts, __global double* stf) {
  int gid0 = (int)get_global_id(0);
  const int nSafeMax = (((int)(counts[0])) > 0 ? ((int)(counts[0])) - 1 : 0);
  const int safeIdx = (gid0 < nSafeMax ? gid0 : nSafeMax);
  if (gid0 >= ((int)(counts[0]))) return;
  if (((int)((int)(get_global_id(0)))) >= counts[1]) {
    return;
  } else {
    int o = owner[((int)((int)(get_global_id(0))))];
    int n = nei[((int)((int)(get_global_id(0))))];
    double w = ((double)(weights[((int)((int)(get_global_id(0))))]));
    double wm = (as_double(0x3ff0000000000000ul) - w);
    double divInterp = ((w * divN[o]) + (wm * divN[n]));
    double Kf = (-divInterp);
    double snGrad = as_double(0x0000000000000000ul);
    if (((int)((int)(get_global_id(0)))) < counts[2]) {
      snGrad = (((double)(delta[((int)((int)(get_global_id(0))))])) * (((double)(alpha[n])) - ((double)(alpha[o]))));
    } else {
      if (thetaMask[((int)((int)(get_global_id(0))))] != (uint)(0u)) {
        snGrad = contactGrad[((int)((int)(get_global_id(0))))];
      } else {
        snGrad = (((double)(delta[((int)((int)(get_global_id(0))))])) * (((double)(alphaF[((int)((int)(get_global_id(0))))])) - ((double)(alpha[o]))));
      }
    }
    stf[((int)((int)(get_global_id(0))))] = ((((double)(paramsF[0])) * Kf) * snGrad);
  }
}

)kernel";
const char* k_opencl_navatala_cfd_k_omega_s_s_t_blending = R"kernel(
__kernel void navatala_cfd_k_omega_s_s_t_blending(__global const float* k, __global const float* omega, __global const float* nut, __global const float* nu, __global const float* y, __global const float* gradKx, __global const float* gradKy, __global const float* gradKz, __global const float* gradOmegax, __global const float* gradOmegay, __global const float* gradOmegaz, __global const int* counts, __global float* outF1, __global float* outF23, __global float* outCDkOmega) {
  int gid0 = (int)get_global_id(0);
  const int nSafeMax = (((int)(counts[0])) > 0 ? ((int)(counts[0])) - 1 : 0);
  const int safeIdx = (gid0 < nSafeMax ? gid0 : nSafeMax);
  if (gid0 >= ((int)(counts[0]))) return;
  if (((int)((int)(get_global_id(0)))) >= counts[0]) {
    return;
  } else {
    float kC = k[((int)((int)(get_global_id(0))))];
    float omC = omega[((int)((int)(get_global_id(0))))];
    float nutC = nut[((int)((int)(get_global_id(0))))];
    float nuC = nu[((int)((int)(get_global_id(0))))];
    float yC = y[((int)((int)(get_global_id(0))))];
    float gKx = gradKx[((int)((int)(get_global_id(0))))];
    float gKy = gradKy[((int)((int)(get_global_id(0))))];
    float gKz = gradKz[((int)((int)(get_global_id(0))))];
    float gOx = gradOmegax[((int)((int)(get_global_id(0))))];
    float gOy = gradOmegay[((int)((int)(get_global_id(0))))];
    float gOz = gradOmegaz[((int)((int)(get_global_id(0))))];
    float dotGrad = (((gKx * gOx) + (gKy * gOy)) + (gKz * gOz));
    float omSafe = (((omC > as_float(0x2edbe6ffu))) ? (omC) : (as_float(0x2edbe6ffu)));
    float cdkw = ((as_float(0x3fdb22d1u) * dotGrad) / omSafe);
    float cdkwPlus = (((cdkw > as_float(0x2edbe6ffu))) ? (cdkw) : (as_float(0x2edbe6ffu)));
    float ySq = (yC * yC);
    float ySqSafe = (((ySq > as_float(0x1e3ce508u))) ? (ySq) : (as_float(0x1e3ce508u)));
    float kSafe = (((kC > as_float(0x00000000u))) ? (kC) : (as_float(0x00000000u)));
    float sqrtK = sqrt(kSafe);
    float denom1 = (as_float(0x3db851ecu) * (omSafe * yC));
    float denom1Safe = (((denom1 > as_float(0x2edbe6ffu))) ? (denom1) : (as_float(0x2edbe6ffu)));
    float term1 = (sqrtK / denom1Safe);
    float denom2 = (ySqSafe * omSafe);
    float term2 = ((as_float(0x43fa0000u) * nuC) / denom2);
    float maxT12 = (((term1 >= term2)) ? (term1) : (term2));
    float denom3 = (cdkwPlus * ySqSafe);
    float denom3Safe = (((denom3 > as_float(0x2edbe6ffu))) ? (denom3) : (as_float(0x2edbe6ffu)));
    float term3 = ((as_float(0x405b22d1u) * kC) / denom3Safe);
    float minT3 = (((maxT12 <= term3)) ? (maxT12) : (term3));
    float arg1 = (((minT3 <= as_float(0x41200000u))) ? (minT3) : (as_float(0x41200000u)));
    float arg1sq = (arg1 * arg1);
    float arg1p4 = (arg1sq * arg1sq);
    float arg1p4Safe = (((arg1p4 <= as_float(0x42200000u))) ? (arg1p4) : (as_float(0x42200000u)));
    float f1Val = tanh(arg1p4Safe);
    float term2a = ((as_float(0x40000000u) * sqrtK) / denom1Safe);
    float maxF2 = (((term2a >= term2)) ? (term2a) : (term2));
    float arg2 = (((maxF2 <= as_float(0x42c80000u))) ? (maxF2) : (as_float(0x42c80000u)));
    float arg2sq = (arg2 * arg2);
    float arg2sqSafe = (((arg2sq <= as_float(0x42200000u))) ? (arg2sq) : (as_float(0x42200000u)));
    float f23Val = tanh(arg2sqSafe);
    outF1[((int)((int)(get_global_id(0))))] = f1Val;
    outF23[((int)((int)(get_global_id(0))))] = f23Val;
    outCDkOmega[((int)((int)(get_global_id(0))))] = cdkw;
  }
}

)kernel";
const char* k_opencl_navatala_cfd_k_omega_s_s_t_nut_compute = R"kernel(
__kernel void navatala_cfd_k_omega_s_s_t_nut_compute(__global const float* kVals, __global const float* omegaVals, __global const float* f23Vals, __global const float* s2Vals, __global const uint* counts, __global const float* params, __global float* outNut) {
  int gid0 = (int)get_global_id(0);
  const int nSafeMax = (((int)(counts[0])) > 0 ? ((int)(counts[0])) - 1 : 0);
  const int safeIdx = (gid0 < nSafeMax ? gid0 : nSafeMax);
  if (gid0 >= ((int)(counts[0]))) return;
  if ((int)(get_global_id(0)) >= ((int)(counts[0]))) {
    return;
  } else {
    float a1 = params[0];
    float b1 = params[1];
    float k = kVals[(int)(get_global_id(0))];
    float om = omegaVals[(int)(get_global_id(0))];
    float f23 = f23Vals[(int)(get_global_id(0))];
    float s2raw = s2Vals[(int)(get_global_id(0))];
    float denomA = (a1 * om);
    float denomB = ((b1 * f23) * sqrt((s2raw * ((float)((s2raw > as_float(0x00000000u)))))));
    float num = (a1 * k);
    if (denomA > denomB) {
      if (denomA > as_float(0x00000000u)) {
        outNut[(int)(get_global_id(0))] = (num / denomA);
      } else {
        outNut[(int)(get_global_id(0))] = as_float(0x00000000u);
      }
    } else {
      if (denomB > as_float(0x00000000u)) {
        outNut[(int)(get_global_id(0))] = (num / denomB);
      } else {
        outNut[(int)(get_global_id(0))] = as_float(0x00000000u);
      }
    }
  }
}

)kernel";
const char* k_opencl_navatala_cfd_k_omega_s_s_t_sources = R"kernel(
__kernel void navatala_cfd_k_omega_s_s_t_sources(__global const float* k, __global const float* omega, __global const float* nut, __global const float* nu, __global const float* alpha, __global const float* rho, __global const float* F1, __global const float* F23, __global const float* CDkOmega, __global const float* S2, __global const float* divU, __global const int* counts, __global float* outOmegaSp, __global float* outOmegaSuSp1, __global float* outOmegaSuSp2, __global float* outOmegaExpl, __global float* outKSp, __global float* outKSuSp, __global float* outKExpl, __global float* outDomegaEff, __global float* outDkEff) {
  int gid0 = (int)get_global_id(0);
  const int nSafeMax = (((int)(counts[0])) > 0 ? ((int)(counts[0])) - 1 : 0);
  const int safeIdx = (gid0 < nSafeMax ? gid0 : nSafeMax);
  if (gid0 >= ((int)(counts[0]))) return;
  if (((int)((int)(get_global_id(0)))) >= counts[0]) {
    return;
  } else {
    float kC = k[((int)((int)(get_global_id(0))))];
    float omC = omega[((int)((int)(get_global_id(0))))];
    float nutC = nut[((int)((int)(get_global_id(0))))];
    float nuC = nu[((int)((int)(get_global_id(0))))];
    float alphaC = alpha[((int)((int)(get_global_id(0))))];
    float rhoC = rho[((int)((int)(get_global_id(0))))];
    float f1C = F1[((int)((int)(get_global_id(0))))];
    float f23C = F23[((int)((int)(get_global_id(0))))];
    float cdkwC = CDkOmega[((int)((int)(get_global_id(0))))];
    float s2C = S2[((int)((int)(get_global_id(0))))];
    float divUC = divU[((int)((int)(get_global_id(0))))];
    float arC = (alphaC * rhoC);
    float gammaC = ((f1C * as_float(0x3decbfb1u)) + as_float(0x3ee147aeu));
    float betaC = (as_float(0x3da9930cu) - (f1C * as_float(0x3bff9724u)));
    float alphaKC = (as_float(0x3f800000u) - (f1C * as_float(0x3e19999au)));
    float alphaOmC = (as_float(0x3f5b22d1u) - (f1C * as_float(0x3eb645a2u)));
    float s2Safe = (((s2C > as_float(0x00000000u))) ? (s2C) : (as_float(0x00000000u)));
    float sqrtS2 = sqrt(s2Safe);
    float a1om = (as_float(0x3e9eb852u) * omC);
    float f23s = (f23C * sqrtS2);
    float maxTerm = (((a1om >= f23s)) ? (a1om) : (f23s));
    float limRhs = (as_float(0x4039ce07u) * (omC * maxTerm));
    float GbyNu = (((s2C <= limRhs)) ? (s2C) : (limRhs));
    float nutS2 = (nutC * s2C);
    float c1bsKOm = (as_float(0x3f666666u) * (kC * omC));
    float pk = (((nutS2 <= c1bsKOm)) ? (nutS2) : (c1bsKOm));
    float omSp = (as_float(0x00000000u) - (arC * (betaC * omC)));
    float omSuSp1 = (as_float(0x00000000u) - (as_float(0x3f2aacdau) * (arC * (gammaC * divUC))));
    float omSafe = (((omC > as_float(0x26901d7du))) ? (omC) : (as_float(0x26901d7du)));
    float f1m1 = (f1C - as_float(0x3f800000u));
    float omSuSp2 = (as_float(0x00000000u) - ((arC * (f1m1 * cdkwC)) / omSafe));
    float omExpl = (arC * (gammaC * GbyNu));
    float kSp = (as_float(0x00000000u) - (arC * (as_float(0x3db851ecu) * omC)));
    float kSuSp = (as_float(0x00000000u) - (as_float(0x3f2aacdau) * (arC * divUC)));
    float kExpl = (arC * pk);
    float domEff = (arC * ((alphaOmC * nutC) + nuC));
    float dkEff = (arC * ((alphaKC * nutC) + nuC));
    outOmegaSp[((int)((int)(get_global_id(0))))] = omSp;
    outOmegaSuSp1[((int)((int)(get_global_id(0))))] = omSuSp1;
    outOmegaSuSp2[((int)((int)(get_global_id(0))))] = omSuSp2;
    outOmegaExpl[((int)((int)(get_global_id(0))))] = omExpl;
    outKSp[((int)((int)(get_global_id(0))))] = kSp;
    outKSuSp[((int)((int)(get_global_id(0))))] = kSuSp;
    outKExpl[((int)((int)(get_global_id(0))))] = kExpl;
    outDomegaEff[((int)((int)(get_global_id(0))))] = domEff;
    outDkEff[((int)((int)(get_global_id(0))))] = dkEff;
  }
}

)kernel";
const char* k_opencl_navatala_cfd_mu_eff_from_rho_nu_eff = R"kernel(
__kernel void navatala_cfd_mu_eff_from_rho_nu_eff(__global const float* rho, __global const float* nuEff, __global const int* params, __global float* outMuEff) {
  int gid0 = (int)get_global_id(0);
  if (((int)((int)(get_global_id(0)))) >= params[0]) {
    return;
  } else {
    float r = rho[((int)((int)(get_global_id(0))))];
    float nu = nuEff[((int)((int)(get_global_id(0))))];
    outMuEff[((int)((int)(get_global_id(0))))] = (r * nu);
  }
}

)kernel";
const char* k_opencl_navatala_cfd_mu_eff_from_rho_nut_and_nu_mix = R"kernel(
__kernel void navatala_cfd_mu_eff_from_rho_nut_and_nu_mix(__global const float* rho, __global const float* nut, __global const float* nuMix, __global const uint* params, __global float* outMu) {
  int gid0 = (int)get_global_id(0);
  if ((int)(get_global_id(0)) >= ((int)(params[0]))) {
    return;
  } else {
    outMu[(int)(get_global_id(0))] = (rho[(int)(get_global_id(0))] * (nuMix[(int)(get_global_id(0))] + nut[(int)(get_global_id(0))]));
  }
}

)kernel";
const char* k_opencl_navatala_cfd_turbulence_add_explicit_source_device = R"kernel(
__kernel void navatala_cfd_turbulence_add_explicit_source_device(__global const float* explicitSrc, __global const float* vol, __global const uint* counts, __global float* outVals) {
  int gid0 = (int)get_global_id(0);
  const int nSafeMax = (((int)(counts[0])) > 0 ? ((int)(counts[0])) - 1 : 0);
  const int safeIdx = (gid0 < nSafeMax ? gid0 : nSafeMax);
  if (gid0 >= ((int)(counts[0]))) return;
  if ((int)(get_global_id(0)) >= ((int)(counts[0]))) {
    return;
  } else {
    outVals[(int)(get_global_id(0))] = (explicitSrc[(int)(get_global_id(0))] * vol[(int)(get_global_id(0))]);
  }
}

)kernel";
const char* k_opencl_navatala_cfd_turbulence_dirichlet_cell_source = R"kernel(
__kernel void navatala_cfd_turbulence_dirichlet_cell_source(__global const int* constrainedMask, __global const float* constrainedValue, __global const float* diag, __global const int* nCells, __global float* source) {
  int gid0 = (int)get_global_id(0);
  if (((int)((int)(get_global_id(0)))) >= nCells[0]) {
    return;
  } else {
    int mask = constrainedMask[((int)((int)(get_global_id(0))))];
    if (mask == 0) {
      return;
    } else {
      source[((int)((int)(get_global_id(0))))] = (diag[((int)((int)(get_global_id(0))))] * constrainedValue[((int)((int)(get_global_id(0))))]);
    }
  }
}

)kernel";
const char* k_opencl_navatala_cfd_turbulence_dirichlet_face_elimination = R"kernel(
__kernel void navatala_cfd_turbulence_dirichlet_face_elimination(__global const int* owner, __global const int* neighbour, __global const int* constrainedMask, __global const float* constrainedValue, __global const int* nIntFaces, __global float* upper, __global float* lower, __global float* source) {
  int gid0 = (int)get_global_id(0);
  if (((int)((int)(get_global_id(0)))) >= nIntFaces[0]) {
    return;
  } else {
    int o = owner[((int)((int)(get_global_id(0))))];
    int n = neighbour[((int)((int)(get_global_id(0))))];
    int oC = constrainedMask[o];
    int nC = constrainedMask[n];
    if ((oC | nC) == 0) {
      return;
    } else {
      float up = upper[((int)((int)(get_global_id(0))))];
      float lo = lower[((int)((int)(get_global_id(0))))];
      if (oC != 0) {
        float cv = constrainedValue[o];
        atomic_add(&source[n], (-(lo * cv)));
      }
      if (nC != 0) {
        float cv = constrainedValue[n];
        atomic_add(&source[o], (-(up * cv)));
      }
      upper[((int)((int)(get_global_id(0))))] = as_float(0x00000000u);
      lower[((int)((int)(get_global_id(0))))] = as_float(0x00000000u);
    }
  }
}

)kernel";
const char* k_opencl_navatala_cfd_two_mag_sqr_symm_grad_u = R"kernel(
__kernel void navatala_cfd_two_mag_sqr_symm_grad_u(__global const float* ufX, __global const float* ufY, __global const float* ufZ, __global const float* sfX, __global const float* sfY, __global const float* sfZ, __global const int* offsets, __global const int* faceIdx, __global const float* sign, __global const float* vol, __global const int* counts, __global float* outS2) {
  int gid0 = (int)get_global_id(0);
  const int nSafeMax = (((int)(counts[0])) > 0 ? ((int)(counts[0])) - 1 : 0);
  const int safeIdx = (gid0 < nSafeMax ? gid0 : nSafeMax);
  if (gid0 >= ((int)(counts[0]))) return;
  if (((int)((int)(get_global_id(0)))) >= counts[0]) {
    return;
  } else {
    int beg = offsets[((int)((int)(get_global_id(0))))];
    int c1 = (((int)((int)(get_global_id(0)))) + 1);
    int end = offsets[c1];
    int len = (end - beg);
    float g00 = as_float(0x00000000u);
    float g01 = as_float(0x00000000u);
    float g02 = as_float(0x00000000u);
    float g10 = as_float(0x00000000u);
    float g11 = as_float(0x00000000u);
    float g12 = as_float(0x00000000u);
    float g20 = as_float(0x00000000u);
    float g21 = as_float(0x00000000u);
    float g22 = as_float(0x00000000u);
    for (int t = 0; t < (int)(len); ++t) {
      int k = (beg + t);
      int f = faceIdx[k];
      float s = sign[k];
      float sx = (s * sfX[f]);
      float sy = (s * sfY[f]);
      float sz = (s * sfZ[f]);
      float ux = ufX[f];
      float uy = ufY[f];
      float uz = ufZ[f];
      g00 = (g00 + (sx * ux));
      g01 = (g01 + (sx * uy));
      g02 = (g02 + (sx * uz));
      g10 = (g10 + (sy * ux));
      g11 = (g11 + (sy * uy));
      g12 = (g12 + (sy * uz));
      g20 = (g20 + (sz * ux));
      g21 = (g21 + (sz * uy));
      g22 = (g22 + (sz * uz));
    }
    float v = vol[((int)((int)(get_global_id(0))))];
    float invV = as_float(0x00000000u);
    if (v != as_float(0x00000000u)) {
      invV = (as_float(0x3f800000u) / v);
    }
    g00 = (g00 * invV);
    g01 = (g01 * invV);
    g02 = (g02 * invV);
    g10 = (g10 * invV);
    g11 = (g11 * invV);
    g12 = (g12 * invV);
    g20 = (g20 * invV);
    g21 = (g21 * invV);
    g22 = (g22 * invV);
    float s00 = g00;
    float s11 = g11;
    float s22 = g22;
    float s01 = (as_float(0x3f000000u) * (g01 + g10));
    float s02 = (as_float(0x3f000000u) * (g02 + g20));
    float s12 = (as_float(0x3f000000u) * (g12 + g21));
    float d0 = (s00 * s00);
    float d1 = (s11 * s11);
    float d2 = (s22 * s22);
    float o0 = (s01 * s01);
    float o1 = (s02 * s02);
    float o2 = (s12 * s12);
    float diag = ((d0 + d1) + d2);
    float off = ((o0 + o1) + o2);
    outS2[((int)((int)(get_global_id(0))))] = ((as_float(0x40000000u) * diag) + (as_float(0x40800000u) * off));
  }
}

)kernel";
const char* k_opencl_navatala_cfd_attention_row_softmax = R"kernel(
__kernel void navatala_cfd_attention_row_softmax(__global const float* scores, __global const int* numQ, __global const int* numK, __global float* weights) {
  int gid0 = (int)get_global_id(0);
  int qi = (int)(get_global_id(0));
  int nq = numQ[0];
  int nk = numK[0];
  if (qi < nq) {
    int rowBase = (qi * nk);
    float maxScore = as_float(0xf149f2cau);
    for (int j = 0; j < (int)(nk); ++j) {
      int idx = (rowBase + j);
      float s = scores[idx];
      maxScore = (((s > maxScore)) ? (s) : (maxScore));
    }
    float sumExp = as_float(0x00000000u);
    for (int j = 0; j < (int)(nk); ++j) {
      int idx = (rowBase + j);
      float s = scores[idx];
      float e = exp((s - maxScore));
      sumExp = (sumExp + e);
    }
    float denom = (sumExp + as_float(0x3089705fu));
    for (int j = 0; j < (int)(nk); ++j) {
      int idx = (rowBase + j);
      float s = scores[idx];
      float e = exp((s - maxScore));
      float w = (e / denom);
      weights[idx] = w;
    }
  }
}

)kernel";
const char* k_opencl_navatala_cfd_attention_value_projection = R"kernel(
__kernel void navatala_cfd_attention_value_projection(__global const float* weights, __global const float* v, __global const int* numQ, __global const int* numK, __global const int* headDim, __global float* _out) {
  int gid0 = (int)get_global_id(0);
  int gid = (int)(get_global_id(0));
  int nq = numQ[0];
  int nk = numK[0];
  int hd = headDim[0];
  int total = (nq * hd);
  if (gid < total) {
    int qi = (gid / hd);
    int d = (gid % hd);
    float acc = as_float(0x00000000u);
    for (int j = 0; j < (int)(nk); ++j) {
      int wIdx = ((qi * nk) + j);
      int vIdx = ((j * hd) + d);
      float w = weights[wIdx];
      float vv = v[vIdx];
      acc = (acc + (w * vv));
    }
    _out[gid] = acc;
  }
}

)kernel";
const char* k_opencl_navatala_cfd_gather_mean = R"kernel(
__kernel void navatala_cfd_gather_mean(__global const float* features, __global const int* neighbors, __global const int* neighborCounts, __global const int* numNodes, __global const int* featureDim, __global float* _out) {
  int gid0 = (int)get_global_id(0);
  int gid = (int)(get_global_id(0));
  int nNodes = numNodes[0];
  if (gid < nNodes) {
    int cnt = neighborCounts[gid];
    int fd = featureDim[0];
    for (int f = 0; f < (int)(fd); ++f) {
      float acc = as_float(0x00000000u);
      for (int j = 0; j < (int)(cnt); ++j) {
        int nbrFlat = ((gid * 8) + j);
        int nbr = neighbors[nbrFlat];
        int featFlat = ((nbr * fd) + f);
        float v = features[featFlat];
        acc = (acc + v);
      }
      float cntF = ((float)(cnt));
      float mean = (((cnt > 0)) ? ((acc / cntF)) : (as_float(0x00000000u)));
      int outFlat = ((gid * fd) + f);
      _out[outFlat] = mean;
    }
  }
}

)kernel";
const char* k_opencl_navatala_cfd_layer_norm_forward = R"kernel(
__kernel void navatala_cfd_layer_norm_forward(__global const float* x, __global const int* numRows, __global const float* gamma, __global const float* beta, __global float* y) {
  int gid0 = (int)get_global_id(0);
  int row = (int)(get_global_id(0));
  int nRows = numRows[0];
  if (row < nRows) {
    int hd = 16;
    int rowBase = (row * hd);
    float sum = as_float(0x00000000u);
    for (int j = 0; j < (int)(hd); ++j) {
      int idx = (rowBase + j);
      float xv = x[idx];
      sum = (sum + xv);
    }
    float hdF = ((float)(hd));
    float mean = (sum / hdF);
    float varSum = as_float(0x00000000u);
    for (int j = 0; j < (int)(hd); ++j) {
      int idx = (rowBase + j);
      float xv = x[idx];
      float diff = (xv - mean);
      varSum = (varSum + (diff * diff));
    }
    float var = (varSum / hdF);
    float invStd = sqrt((as_float(0x3f800000u) / (var + as_float(0x3727c5acu))));
    for (int j = 0; j < (int)(hd); ++j) {
      int idx = (rowBase + j);
      float xv = x[idx];
      float xn = ((xv - mean) * invStd);
      float g = gamma[j];
      float b = beta[j];
      float yv = ((xn * g) + b);
      y[idx] = yv;
    }
  }
}

)kernel";
const char* k_opencl_navatala_cfd_radius_count = R"kernel(
__kernel void navatala_cfd_radius_count(__global const float* query, __global const float* ref, __global const int* numQuery, __global const int* numRef, __global const float* radiusIn, __global int* counts) {
  int gid0 = (int)get_global_id(0);
  int qid = (int)(get_global_id(0));
  int nq = numQuery[0];
  int nr = numRef[0];
  if (qid < nq) {
    float qx = query[((qid)*3 + 0)];
    float qy = query[((qid)*3 + 1)];
    float qz = query[((qid)*3 + 2)];
    float rIn = radiusIn[0];
    float rCmp = rIn;
    int cnt = 0;
    for (int rid = 0; rid < (int)(nr); ++rid) {
      float rx = ref[((rid)*3 + 0)];
      float ry = ref[((rid)*3 + 1)];
      float rz = ref[((rid)*3 + 2)];
      float dx = (qx - rx);
      float dy = (qy - ry);
      float dz = (qz - rz);
      float d2 = (((dx * dx) + (dy * dy)) + (dz * dz));
      if (d2 <= rCmp) {
        cnt = (cnt + 1);
      }
    }
    counts[qid] = cnt;
  }
}

)kernel";
const char* k_opencl_navatala_cfd_scaled_dot_product = R"kernel(
__kernel void navatala_cfd_scaled_dot_product(__global const float* q, __global const float* k, __global const int* numQ, __global const int* numK, __global const int* headDim, __global float* scores) {
  int gid0 = (int)get_global_id(0);
  int gid = (int)(get_global_id(0));
  int nq = numQ[0];
  int nk = numK[0];
  int hd = headDim[0];
  int total = (nq * nk);
  if (gid < total) {
    int qi = (gid / nk);
    int kj = (gid % nk);
    float acc = as_float(0x00000000u);
    for (int d = 0; d < (int)(hd); ++d) {
      int qFlat = ((qi * hd) + d);
      int kFlat = ((kj * hd) + d);
      float qv = q[qFlat];
      float kv = k[kFlat];
      acc = (acc + (qv * kv));
    }
    scores[gid] = (acc * as_float(0x3e800000u));
  }
}

)kernel";
const char* k_opencl_navatala_cfd_scatter_add_atomic = R"kernel(
__kernel void navatala_cfd_scatter_add_atomic(__global const int* values, __global const int* indices, __global const int* numEdges, __global const int* featureDim, __global int* dst) {
  int gid0 = (int)get_global_id(0);
  int gid = (int)(get_global_id(0));
  int nEdges = numEdges[0];
  if (gid < nEdges) {
    int dstIdx = indices[gid];
    int fd = featureDim[0];
    for (int f = 0; f < (int)(fd); ++f) {
      int valFlat = ((gid * fd) + f);
      int dstFlat = ((dstIdx * fd) + f);
      int v = values[valFlat];
      int _ignored = atomic_add(&(dst[dstFlat]), v);
    }
  }
}

)kernel";
const char* k_opencl_navatala_cfd_spectral_complex_mul = R"kernel(
__kernel void navatala_cfd_spectral_complex_mul(__global const float* x, __global const float* w, __global const int* n, __global float* y) {
  int gid0 = (int)get_global_id(0);
  int gid = (int)(get_global_id(0));
  int count = n[0];
  if (gid < count) {
    float xr = x[((gid)*2 + 0)];
    float xi = x[((gid)*2 + 1)];
    float wr = w[((gid)*2 + 0)];
    float wi = w[((gid)*2 + 1)];
    float yr = ((xr * wr) - (xi * wi));
    float yi = ((xr * wi) + (xi * wr));
    y[((gid)*2 + 0)] = yr;
    y[((gid)*2 + 1)] = yi;
  }
}

)kernel";
const char* k_opencl_navatala_cfd_spectral_conv1d_mode_sliced = R"kernel(
__kernel void navatala_cfd_spectral_conv1d_mode_sliced(__global const float* xFt, __global const float* wPos, __global const float* wNeg, __global const int* numBatches, __global const int* ftW, __global float* yFt) {
  int gid0 = (int)get_global_id(0);
  int gid = (int)(get_global_id(0));
  int nb = numBatches[0];
  int ftw = ftW[0];
  int outCh = 2;
  int inCh = 2;
  int m1 = 8;
  int total = ((nb * outCh) * ftw);
  if (gid < total) {
    int boc = (gid / ftw);
    int w = (gid % ftw);
    int b = (boc / outCh);
    int oc = (boc % outCh);
    float accR = as_float(0x00000000u);
    float accI = as_float(0x00000000u);
    if (w < m1) {
      for (int ic = 0; ic < (int)(inCh); ++ic) {
        int xBase = ((((b * inCh) + ic) * ftw) + w);
        int wBase = ((((oc * inCh) + ic) * m1) + w);
        float xr = xFt[((xBase)*2 + 0)];
        float xi = xFt[((xBase)*2 + 1)];
        float wr = wPos[((wBase)*2 + 0)];
        float wi = wPos[((wBase)*2 + 1)];
        accR = (accR + ((xr * wr) - (xi * wi)));
        accI = (accI + ((xr * wi) + (xi * wr)));
      }
    } else {
      int negStart = (ftw - m1);
      if (w >= negStart) {
        int wLocal = (w - negStart);
        for (int ic = 0; ic < (int)(inCh); ++ic) {
          int xBase = ((((b * inCh) + ic) * ftw) + w);
          int wBase = ((((oc * inCh) + ic) * m1) + wLocal);
          float xr = xFt[((xBase)*2 + 0)];
          float xi = xFt[((xBase)*2 + 1)];
          float wr = wNeg[((wBase)*2 + 0)];
          float wi = wNeg[((wBase)*2 + 1)];
          accR = (accR + ((xr * wr) - (xi * wi)));
          accI = (accI + ((xr * wi) + (xi * wr)));
        }
      }
    }
    yFt[((gid)*2 + 0)] = accR;
    yFt[((gid)*2 + 1)] = accI;
  }
}

)kernel";
const char* k_opencl_navatala_cfd_spectral_conv2d_mode_sliced = R"kernel(
__kernel void navatala_cfd_spectral_conv2d_mode_sliced(__global const float* xFt, __global const float* wPos, __global const float* wNeg, __global const int* numBatches, __global const int* ftH, __global const int* ftW, __global float* yFt) {
  int gid0 = (int)get_global_id(0);
  int gid = (int)(get_global_id(0));
  int nb = numBatches[0];
  int fth = ftH[0];
  int ftw = ftW[0];
  int outCh = 2;
  int inCh = 2;
  int m1 = 8;
  int m2 = 8;
  int plane = (fth * ftw);
  int total = ((nb * outCh) * plane);
  if (gid < total) {
    int boc = (gid / plane);
    int hwFlat = (gid % plane);
    int b = (boc / outCh);
    int oc = (boc % outCh);
    int h = (hwFlat / ftw);
    int w = (hwFlat % ftw);
    float accR = as_float(0x00000000u);
    float accI = as_float(0x00000000u);
    if ((h < m1) && (w < m2)) {
      for (int ic = 0; ic < (int)(inCh); ++ic) {
        int xBase = ((((((b * inCh) + ic) * fth) + h) * ftw) + w);
        int wBase = ((((((oc * inCh) + ic) * m1) + h) * m2) + w);
        float xr = xFt[((xBase)*2 + 0)];
        float xi = xFt[((xBase)*2 + 1)];
        float wr = wPos[((wBase)*2 + 0)];
        float wi = wPos[((wBase)*2 + 1)];
        accR = (accR + ((xr * wr) - (xi * wi)));
        accI = (accI + ((xr * wi) + (xi * wr)));
      }
    } else {
      int negStart = (fth - m1);
      if ((h >= negStart) && (w < m2)) {
        int hLocal = (h - negStart);
        for (int ic = 0; ic < (int)(inCh); ++ic) {
          int xBase = ((((((b * inCh) + ic) * fth) + h) * ftw) + w);
          int wBase = ((((((oc * inCh) + ic) * m1) + hLocal) * m2) + w);
          float xr = xFt[((xBase)*2 + 0)];
          float xi = xFt[((xBase)*2 + 1)];
          float wr = wNeg[((wBase)*2 + 0)];
          float wi = wNeg[((wBase)*2 + 1)];
          accR = (accR + ((xr * wr) - (xi * wi)));
          accI = (accI + ((xr * wi) + (xi * wr)));
        }
      }
    }
    yFt[((gid)*2 + 0)] = accR;
    yFt[((gid)*2 + 1)] = accI;
  }
}

)kernel";
const char* k_opencl_navatala_cfd_spectral_conv3d_mode_sliced = R"kernel(
__kernel void navatala_cfd_spectral_conv3d_mode_sliced(__global const float* xFt, __global const float* wPP, __global const float* wNP, __global const float* wPN, __global const float* wNN, __global const int* numBatches, __global const int* ftD, __global const int* ftH, __global const int* ftW, __global float* yFt) {
  int gid0 = (int)get_global_id(0);
  int gid = (int)(get_global_id(0));
  int nb = numBatches[0];
  int ftd = ftD[0];
  int fth = ftH[0];
  int ftw = ftW[0];
  int outCh = 2;
  int inCh = 2;
  int m1 = 4;
  int m2 = 4;
  int m3 = 4;
  int plane = ((ftd * fth) * ftw);
  int total = ((nb * outCh) * plane);
  if (gid < total) {
    int boc = (gid / plane);
    int dhwFlat = (gid % plane);
    int b = (boc / outCh);
    int oc = (boc % outCh);
    int d = (dhwFlat / (fth * ftw));
    int hwFlat = (dhwFlat % (fth * ftw));
    int h = (hwFlat / ftw);
    int w = (hwFlat % ftw);
    int negStartD = (ftd - m1);
    int negStartH = (fth - m2);
    float accR = as_float(0x00000000u);
    float accI = as_float(0x00000000u);
    if (((d < m1) && (h < m2)) && (w < m3)) {
      for (int ic = 0; ic < (int)(inCh); ++ic) {
        int xBase = ((((((((b * inCh) + ic) * ftd) + d) * fth) + h) * ftw) + w);
        int wBase = ((((((((oc * inCh) + ic) * m1) + d) * m2) + h) * m3) + w);
        float xr = xFt[((xBase)*2 + 0)];
        float xi = xFt[((xBase)*2 + 1)];
        float wr = wPP[((wBase)*2 + 0)];
        float wi = wPP[((wBase)*2 + 1)];
        accR = (accR + ((xr * wr) - (xi * wi)));
        accI = (accI + ((xr * wi) + (xi * wr)));
      }
    } else {
      if (((d >= negStartD) && (h < m2)) && (w < m3)) {
        for (int ic = 0; ic < (int)(inCh); ++ic) {
          int xBase = ((((((((b * inCh) + ic) * ftd) + d) * fth) + h) * ftw) + w);
          int wBase = ((((((((oc * inCh) + ic) * m1) + (d - negStartD)) * m2) + h) * m3) + w);
          float xr = xFt[((xBase)*2 + 0)];
          float xi = xFt[((xBase)*2 + 1)];
          float wr = wNP[((wBase)*2 + 0)];
          float wi = wNP[((wBase)*2 + 1)];
          accR = (accR + ((xr * wr) - (xi * wi)));
          accI = (accI + ((xr * wi) + (xi * wr)));
        }
      } else {
        if (((d < m1) && (h >= negStartH)) && (w < m3)) {
          for (int ic = 0; ic < (int)(inCh); ++ic) {
            int xBase = ((((((((b * inCh) + ic) * ftd) + d) * fth) + h) * ftw) + w);
            int wBase = ((((((((oc * inCh) + ic) * m1) + d) * m2) + (h - negStartH)) * m3) + w);
            float xr = xFt[((xBase)*2 + 0)];
            float xi = xFt[((xBase)*2 + 1)];
            float wr = wPN[((wBase)*2 + 0)];
            float wi = wPN[((wBase)*2 + 1)];
            accR = (accR + ((xr * wr) - (xi * wi)));
            accI = (accI + ((xr * wi) + (xi * wr)));
          }
        } else {
          if (((d >= negStartD) && (h >= negStartH)) && (w < m3)) {
            for (int ic = 0; ic < (int)(inCh); ++ic) {
              int xBase = ((((((((b * inCh) + ic) * ftd) + d) * fth) + h) * ftw) + w);
              int wBase = ((((((((oc * inCh) + ic) * m1) + (d - negStartD)) * m2) + (h - negStartH)) * m3) + w);
              float xr = xFt[((xBase)*2 + 0)];
              float xi = xFt[((xBase)*2 + 1)];
              float wr = wNN[((wBase)*2 + 0)];
              float wi = wNN[((wBase)*2 + 1)];
              accR = (accR + ((xr * wr) - (xi * wi)));
              accI = (accI + ((xr * wi) + (xi * wr)));
            }
          }
        }
      }
    }
    yFt[((gid)*2 + 0)] = accR;
    yFt[((gid)*2 + 1)] = accI;
  }
}

)kernel";
const char* k_opencl_navatala_cfd_spectral_conv4d_mode_sliced = R"kernel(
__kernel void navatala_cfd_spectral_conv4d_mode_sliced(__global const float* xFt, __global const float* wPPP, __global const float* wNPP, __global const float* wPNP, __global const float* wNNP, __global const float* wPPN, __global const float* wNPN, __global const float* wPNN, __global const float* wNNN, __global const int* numBatches, __global const int* ftT, __global const int* ftD, __global const int* ftH, __global const int* ftW, __global float* yFt) {
  int gid0 = (int)get_global_id(0);
  int gid = (int)(get_global_id(0));
  int nb = numBatches[0];
  int ftt = ftT[0];
  int ftd = ftD[0];
  int fth = ftH[0];
  int ftw = ftW[0];
  int outCh = 2;
  int inCh = 2;
  int m1 = 2;
  int m2 = 2;
  int m3 = 2;
  int m4 = 2;
  int hyper = (((ftt * ftd) * fth) * ftw);
  int total = ((nb * outCh) * hyper);
  if (gid < total) {
    int boc = (gid / hyper);
    int tdhwFlat = (gid % hyper);
    int b = (boc / outCh);
    int oc = (boc % outCh);
    int t = (tdhwFlat / ((ftd * fth) * ftw));
    int dhwFlat = (tdhwFlat % ((ftd * fth) * ftw));
    int d = (dhwFlat / (fth * ftw));
    int hwFlat = (dhwFlat % (fth * ftw));
    int h = (hwFlat / ftw);
    int w = (hwFlat % ftw);
    int negStartT = (ftt - m1);
    int negStartD = (ftd - m2);
    int negStartH = (fth - m3);
    float accR = as_float(0x00000000u);
    float accI = as_float(0x00000000u);
    if ((((t < m1) && (d < m2)) && (h < m3)) && (w < m4)) {
      for (int ic = 0; ic < (int)(inCh); ++ic) {
        int xBase = ((((((((((b * inCh) + ic) * ftt) + t) * ftd) + d) * fth) + h) * ftw) + w);
        int wBase = ((((((((((((oc * inCh) + ic) * m1) + t) * m2) + d) * m3) + h) * m4) + w) * 1) + 0);
        float xr = xFt[((xBase)*2 + 0)];
        float xi = xFt[((xBase)*2 + 1)];
        float wr = wPPP[((wBase)*2 + 0)];
        float wi = wPPP[((wBase)*2 + 1)];
        accR = (accR + ((xr * wr) - (xi * wi)));
        accI = (accI + ((xr * wi) + (xi * wr)));
      }
    } else {
      if ((((t >= negStartT) && (d < m2)) && (h < m3)) && (w < m4)) {
        for (int ic = 0; ic < (int)(inCh); ++ic) {
          int xBase = ((((((((((b * inCh) + ic) * ftt) + t) * ftd) + d) * fth) + h) * ftw) + w);
          int wBase = ((((((((((((oc * inCh) + ic) * m1) + (t - negStartT)) * m2) + d) * m3) + h) * m4) + w) * 1) + 0);
          float xr = xFt[((xBase)*2 + 0)];
          float xi = xFt[((xBase)*2 + 1)];
          float wr = wNPP[((wBase)*2 + 0)];
          float wi = wNPP[((wBase)*2 + 1)];
          accR = (accR + ((xr * wr) - (xi * wi)));
          accI = (accI + ((xr * wi) + (xi * wr)));
        }
      } else {
        if ((((t < m1) && (d >= negStartD)) && (h < m3)) && (w < m4)) {
          for (int ic = 0; ic < (int)(inCh); ++ic) {
            int xBase = ((((((((((b * inCh) + ic) * ftt) + t) * ftd) + d) * fth) + h) * ftw) + w);
            int wBase = ((((((((((((oc * inCh) + ic) * m1) + t) * m2) + (d - negStartD)) * m3) + h) * m4) + w) * 1) + 0);
            float xr = xFt[((xBase)*2 + 0)];
            float xi = xFt[((xBase)*2 + 1)];
            float wr = wPNP[((wBase)*2 + 0)];
            float wi = wPNP[((wBase)*2 + 1)];
            accR = (accR + ((xr * wr) - (xi * wi)));
            accI = (accI + ((xr * wi) + (xi * wr)));
          }
        } else {
          if ((((t >= negStartT) && (d >= negStartD)) && (h < m3)) && (w < m4)) {
            for (int ic = 0; ic < (int)(inCh); ++ic) {
              int xBase = ((((((((((b * inCh) + ic) * ftt) + t) * ftd) + d) * fth) + h) * ftw) + w);
              int wBase = ((((((((((((oc * inCh) + ic) * m1) + (t - negStartT)) * m2) + (d - negStartD)) * m3) + h) * m4) + w) * 1) + 0);
              float xr = xFt[((xBase)*2 + 0)];
              float xi = xFt[((xBase)*2 + 1)];
              float wr = wNNP[((wBase)*2 + 0)];
              float wi = wNNP[((wBase)*2 + 1)];
              accR = (accR + ((xr * wr) - (xi * wi)));
              accI = (accI + ((xr * wi) + (xi * wr)));
            }
          } else {
            if ((((t < m1) && (d < m2)) && (h >= negStartH)) && (w < m4)) {
              for (int ic = 0; ic < (int)(inCh); ++ic) {
                int xBase = ((((((((((b * inCh) + ic) * ftt) + t) * ftd) + d) * fth) + h) * ftw) + w);
                int wBase = ((((((((((((oc * inCh) + ic) * m1) + t) * m2) + d) * m3) + (h - negStartH)) * m4) + w) * 1) + 0);
                float xr = xFt[((xBase)*2 + 0)];
                float xi = xFt[((xBase)*2 + 1)];
                float wr = wPPN[((wBase)*2 + 0)];
                float wi = wPPN[((wBase)*2 + 1)];
                accR = (accR + ((xr * wr) - (xi * wi)));
                accI = (accI + ((xr * wi) + (xi * wr)));
              }
            } else {
              if ((((t >= negStartT) && (d < m2)) && (h >= negStartH)) && (w < m4)) {
                for (int ic = 0; ic < (int)(inCh); ++ic) {
                  int xBase = ((((((((((b * inCh) + ic) * ftt) + t) * ftd) + d) * fth) + h) * ftw) + w);
                  int wBase = ((((((((((((oc * inCh) + ic) * m1) + (t - negStartT)) * m2) + d) * m3) + (h - negStartH)) * m4) + w) * 1) + 0);
                  float xr = xFt[((xBase)*2 + 0)];
                  float xi = xFt[((xBase)*2 + 1)];
                  float wr = wNPN[((wBase)*2 + 0)];
                  float wi = wNPN[((wBase)*2 + 1)];
                  accR = (accR + ((xr * wr) - (xi * wi)));
                  accI = (accI + ((xr * wi) + (xi * wr)));
                }
              } else {
                if ((((t < m1) && (d >= negStartD)) && (h >= negStartH)) && (w < m4)) {
                  for (int ic = 0; ic < (int)(inCh); ++ic) {
                    int xBase = ((((((((((b * inCh) + ic) * ftt) + t) * ftd) + d) * fth) + h) * ftw) + w);
                    int wBase = ((((((((((((oc * inCh) + ic) * m1) + t) * m2) + (d - negStartD)) * m3) + (h - negStartH)) * m4) + w) * 1) + 0);
                    float xr = xFt[((xBase)*2 + 0)];
                    float xi = xFt[((xBase)*2 + 1)];
                    float wr = wPNN[((wBase)*2 + 0)];
                    float wi = wPNN[((wBase)*2 + 1)];
                    accR = (accR + ((xr * wr) - (xi * wi)));
                    accI = (accI + ((xr * wi) + (xi * wr)));
                  }
                } else {
                  if ((((t >= negStartT) && (d >= negStartD)) && (h >= negStartH)) && (w < m4)) {
                    for (int ic = 0; ic < (int)(inCh); ++ic) {
                      int xBase = ((((((((((b * inCh) + ic) * ftt) + t) * ftd) + d) * fth) + h) * ftw) + w);
                      int wBase = ((((((((((((oc * inCh) + ic) * m1) + (t - negStartT)) * m2) + (d - negStartD)) * m3) + (h - negStartH)) * m4) + w) * 1) + 0);
                      float xr = xFt[((xBase)*2 + 0)];
                      float xi = xFt[((xBase)*2 + 1)];
                      float wr = wNNN[((wBase)*2 + 0)];
                      float wi = wNNN[((wBase)*2 + 1)];
                      accR = (accR + ((xr * wr) - (xi * wi)));
                      accI = (accI + ((xr * wi) + (xi * wr)));
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
    yFt[((gid)*2 + 0)] = accR;
    yFt[((gid)*2 + 1)] = accI;
  }
}

)kernel";
const char* k_opencl_navatala_cfd_compute_face_area_pair_weights_f32 = R"kernel(
__kernel void navatala_cfd_compute_face_area_pair_weights_f32(__global const float* sfX, __global const float* sfY, __global const float* sfZ, __global const float* magSf, __global const uint* nFaces, __global float* faceWeight) {
  int gid0 = (int)get_global_id(0);
  int f = (int)(get_global_id(0));
  int N = ((int)(nFaces[0]));
  if (f < N) {
    float sx = sfX[f];
    float sy = sfY[f];
    float sz = sfZ[f];
    float msf = magSf[f];
    float invSqrt = (as_float(0x3f800000u) / sqrt(msf));
    float wx = (sx * invSqrt);
    float wy = ((sy * as_float(0x3f8147aeu)) * invSqrt);
    float wz = ((sz * as_float(0x3f828f5cu)) * invSqrt);
    float mag2 = (((wx * wx) + (wy * wy)) + (wz * wz));
    faceWeight[f] = sqrt(mag2);
  }
}

)kernel";
const char* k_opencl_navatala_cfd_restrict_by_agglomeration_gather_f32 = R"kernel(
__kernel void navatala_cfd_restrict_by_agglomeration_gather_f32(__global const float* fineField, __global const uint* coarseToFineOffsets, __global const uint* coarseToFineList, __global const uint* nCoarseCells, __global float* coarseField) {
  int gid0 = (int)get_global_id(0);
  int c = (int)(get_global_id(0));
  int N = ((int)(nCoarseCells[0]));
  if (c < N) {
    int begin = ((int)(coarseToFineOffsets[c]));
    int end = ((int)(coarseToFineOffsets[(c + 1)]));
    float sum = as_float(0x00000000u);
    for (int j = 0; j < (int)((end - begin)); ++j) {
      int pos = (begin + j);
      int fineIdx = ((int)(coarseToFineList[pos]));
      float v = fineField[fineIdx];
      sum = (sum + v);
    }
    coarseField[c] = sum;
  }
}

)kernel";
const char* k_opencl_navatala_cfd_prolongate_by_agglomeration_f32 = R"kernel(
__kernel void navatala_cfd_prolongate_by_agglomeration_f32(__global const float* coarseField, __global const uint* fineToCoarse, __global const uint* nFineCells, __global float* fineField) {
  int gid0 = (int)get_global_id(0);
  int i = (int)(get_global_id(0));
  int N = ((int)(nFineCells[0]));
  if (i < N) {
    int coarseIdx = ((int)(fineToCoarse[i]));
    float val = coarseField[coarseIdx];
    float cur = fineField[i];
    fineField[i] = (cur + val);
  }
}

)kernel";
const char* k_opencl_navatala_cfd_diag_inv_apply_f32 = R"kernel(
__kernel void navatala_cfd_diag_inv_apply_f32(__global const float* r, __global const float* diagInv, __global const uint* n, __global float* z) {
  int gid0 = (int)get_global_id(0);
  int i = (int)(get_global_id(0));
  int N = ((int)(n[0]));
  if (i < N) {
    float ri = r[i];
    float di = diagInv[i];
    z[i] = (di * ri);
  }
}

)kernel";
const char* k_opencl_navatala_cfd_csr_sp_m_v_mixed_f32_f64 = R"kernel(
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
__kernel void navatala_cfd_csr_sp_m_v_mixed_f32_f64(__global const uint* rowPtr, __global const uint* colIdx, __global const float* values, __global const float* x, __global const uint* nRows, __global float* y) {
  int gid0 = (int)get_global_id(0);
  int row = (int)(get_global_id(0));
  int N = ((int)(nRows[0]));
  if (row < N) {
    int start = ((int)(rowPtr[row]));
    int end = ((int)(rowPtr[(row + 1)]));
    double acc = as_double(0x0000000000000000ul);
    for (int j = 0; j < (int)((end - start)); ++j) {
      int idx = (start + j);
      float val = values[idx];
      int col = ((int)(colIdx[idx]));
      float xi = x[col];
      acc = (acc + (((double)(val)) * ((double)(xi))));
    }
    y[row] = ((float)(acc));
  }
}

)kernel";
const char* k_opencl_navatala_cfd_chebyshev_sweep_mixed_f32_f64 = R"kernel(
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
__kernel void navatala_cfd_chebyshev_sweep_mixed_f32_f64(__global const float* r, __global const float* d, __global const double* coeffA, __global const double* coeffB, __global const uint* n, __global float* x, __global float* dNew) {
  int gid0 = (int)get_global_id(0);
  int i = (int)(get_global_id(0));
  int N = ((int)(n[0]));
  if (i < N) {
    double cA = coeffA[0];
    double cB = coeffB[0];
    double ri = ((double)(r[i]));
    double di = ((double)(d[i]));
    double xi = ((double)(x[i]));
    double corr = ((cA * ri) + (cB * di));
    x[i] = ((float)((xi + corr)));
    dNew[i] = ((float)(corr));
  }
}

)kernel";
const char* k_opencl_navatala_cfd_ell_sp_m_v_mixed_f32_f64 = R"kernel(
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
__kernel void navatala_cfd_ell_sp_m_v_mixed_f32_f64(__global const uint* colIdx, __global const float* values, __global const float* x, __global const uint* nRows, __global const uint* maxWidth, __global float* y) {
  int gid0 = (int)get_global_id(0);
  int row = (int)(get_global_id(0));
  int N = ((int)(nRows[0]));
  if (row < N) {
    int W = ((int)(maxWidth[0]));
    double acc = as_double(0x0000000000000000ul);
    for (int col = 0; col < (int)(W); ++col) {
      int idx = (row + (col * N));
      int c = ((int)(colIdx[idx]));
      float v = values[idx];
      float xi = x[c];
      acc = (acc + (((double)(v)) * ((double)(xi))));
    }
    y[row] = ((float)(acc));
  }
}

)kernel";
const char* k_opencl_navatala_cfd_chebyshev_smooth_post_fused_mixed = R"kernel(
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
__kernel void navatala_cfd_chebyshev_smooth_post_fused_mixed(__global const float* Az, __global const float* r, __global const float* diagInv, __global const float* dPrev, __global const double* coeffA, __global const double* coeffB, __global const uint* n, __global float* z, __global float* dNew) {
  int gid0 = (int)get_global_id(0);
  int i = (int)(get_global_id(0));
  int N = ((int)(n[0]));
  if (i < N) {
    double cA = coeffA[0];
    double cB = coeffB[0];
    double az_i = ((double)(Az[i]));
    double r_i = ((double)(r[i]));
    double di_i = ((double)(diagInv[i]));
    double dp_i = ((double)(dPrev[i]));
    double z_i = ((double)(z[i]));
    double res = (r_i - az_i);
    double dinvr = (di_i * res);
    double corr = ((cA * dinvr) + (cB * dp_i));
    z[i] = ((float)((z_i + corr)));
    dNew[i] = ((float)(corr));
  }
}

)kernel";
const char* k_opencl_navatala_cfd_gamg_proc_pack_f32 = R"kernel(
__kernel void navatala_cfd_gamg_proc_pack_f32(__global const float* x, __global const uint* localCells, __global const uint* count, __global float* sendVals) {
  int gid0 = (int)get_global_id(0);
  int i = (int)(get_global_id(0));
  int N = ((int)(count[0]));
  if (i < N) {
    int cell = ((int)(localCells[i]));
    float v = x[cell];
    sendVals[i] = v;
  }
}

)kernel";
const char* k_opencl_navatala_cfd_gamg_proc_add_gather_f32 = R"kernel(
__kernel void navatala_cfd_gamg_proc_add_gather_f32(__global const uint* entryOffsets, __global const uint* entryIndices, __global const float* coeffs, __global const float* recvVals, __global const uint* nCells, __global float* y) {
  int gid0 = (int)get_global_id(0);
  int c = (int)(get_global_id(0));
  int N = ((int)(nCells[0]));
  if (c < N) {
    int begin = ((int)(entryOffsets[c]));
    int end = ((int)(entryOffsets[(c + 1)]));
    float acc = y[c];
    for (int j = 0; j < (int)((end - begin)); ++j) {
      int pos = (begin + j);
      int entry = ((int)(entryIndices[pos]));
      float co = coeffs[entry];
      float rv = recvVals[entry];
      acc = (acc + (co * rv));
    }
    y[c] = acc;
  }
}

)kernel";

namespace NavatalaRegistry {

const KernelArgumentInfo kAbiArgs_opencl_navatala_cfd_boundary_force_partials[] = {
  { "pAllFaces", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "sfComponent", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "counts", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 8, 8, 256, nullptr, 0, 0 },
  { "outPartials", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_cfd_boundary_force_partials = {
  1,
  "navatala_cfd_boundary_force_partials",
  "opencl",
  "navatala_cfd_boundary_force_partials",
  "kernel:opencl:navatala_cfd_boundary_force_partials",
  "abi-r1:opencl:navatala_cfd_boundary_force_partials",
  "abi-r1:opencl:navatala_cfd_boundary_force_partials",
  4,
  kAbiArgs_opencl_navatala_cfd_boundary_force_partials
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_cfd_pack_owner_values[] = {
  { "x", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "owner", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "procFaces", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "counts", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "outSend", 4, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_cfd_pack_owner_values = {
  1,
  "navatala_cfd_pack_owner_values",
  "opencl",
  "navatala_cfd_pack_owner_values",
  "kernel:opencl:navatala_cfd_pack_owner_values",
  "abi-r1:opencl:navatala_cfd_pack_owner_values",
  "abi-r1:opencl:navatala_cfd_pack_owner_values",
  5,
  kAbiArgs_opencl_navatala_cfd_pack_owner_values
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_cfd_scatter_values_and_mask[] = {
  { "procFaces", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "recv", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "counts", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "bcVal", 3, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "bcMask", 4, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_cfd_scatter_values_and_mask = {
  1,
  "navatala_cfd_scatter_values_and_mask",
  "opencl",
  "navatala_cfd_scatter_values_and_mask",
  "kernel:opencl:navatala_cfd_scatter_values_and_mask",
  "abi-r1:opencl:navatala_cfd_scatter_values_and_mask",
  "abi-r1:opencl:navatala_cfd_scatter_values_and_mask",
  5,
  kAbiArgs_opencl_navatala_cfd_scatter_values_and_mask
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_cfd_pack_mu_grad_owner[] = {
  { "muCell", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "gXX", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "gXY", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "gXZ", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "gYX", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "gYY", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "gYZ", 6, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "gZX", 7, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "gZY", 8, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "gZZ", 9, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "owner", 10, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "procFaces", 11, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "counts", 12, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "outSend", 13, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_cfd_pack_mu_grad_owner = {
  1,
  "navatala_cfd_pack_mu_grad_owner",
  "opencl",
  "navatala_cfd_pack_mu_grad_owner",
  "kernel:opencl:navatala_cfd_pack_mu_grad_owner",
  "abi-r1:opencl:navatala_cfd_pack_mu_grad_owner",
  "abi-r1:opencl:navatala_cfd_pack_mu_grad_owner",
  14,
  kAbiArgs_opencl_navatala_cfd_pack_mu_grad_owner
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_cfd_pack_owner_rho_vec3[] = {
  { "rhoCell", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "ux", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "uy", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "uz", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "owner", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "procFaces", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "counts", 6, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "outSend", 7, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_cfd_pack_owner_rho_vec3 = {
  1,
  "navatala_cfd_pack_owner_rho_vec3",
  "opencl",
  "navatala_cfd_pack_owner_rho_vec3",
  "kernel:opencl:navatala_cfd_pack_owner_rho_vec3",
  "abi-r1:opencl:navatala_cfd_pack_owner_rho_vec3",
  "abi-r1:opencl:navatala_cfd_pack_owner_rho_vec3",
  8,
  kAbiArgs_opencl_navatala_cfd_pack_owner_rho_vec3
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_cfd_pack_owner_vec3[] = {
  { "x", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "y", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "z", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "owner", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "procFaces", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "counts", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "outSend", 6, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_cfd_pack_owner_vec3 = {
  1,
  "navatala_cfd_pack_owner_vec3",
  "opencl",
  "navatala_cfd_pack_owner_vec3",
  "kernel:opencl:navatala_cfd_pack_owner_vec3",
  "abi-r1:opencl:navatala_cfd_pack_owner_vec3",
  "abi-r1:opencl:navatala_cfd_pack_owner_vec3",
  7,
  kAbiArgs_opencl_navatala_cfd_pack_owner_vec3
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_cfd_pack_owner_vec3_mu[] = {
  { "ux", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "uy", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "uz", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "muCell", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "owner", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "procFaces", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "counts", 6, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "outSend", 7, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_cfd_pack_owner_vec3_mu = {
  1,
  "navatala_cfd_pack_owner_vec3_mu",
  "opencl",
  "navatala_cfd_pack_owner_vec3_mu",
  "kernel:opencl:navatala_cfd_pack_owner_vec3_mu",
  "abi-r1:opencl:navatala_cfd_pack_owner_vec3_mu",
  "abi-r1:opencl:navatala_cfd_pack_owner_vec3_mu",
  8,
  kAbiArgs_opencl_navatala_cfd_pack_owner_vec3_mu
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_cfd_scatter_mu_grad_ghost[] = {
  { "recvVals", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "counts", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "muCell", 2, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "gXX", 3, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "gXY", 4, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "gXZ", 5, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "gYX", 6, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "gYY", 7, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "gYZ", 8, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "gZX", 9, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "gZY", 10, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "gZZ", 11, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_cfd_scatter_mu_grad_ghost = {
  1,
  "navatala_cfd_scatter_mu_grad_ghost",
  "opencl",
  "navatala_cfd_scatter_mu_grad_ghost",
  "kernel:opencl:navatala_cfd_scatter_mu_grad_ghost",
  "abi-r1:opencl:navatala_cfd_scatter_mu_grad_ghost",
  "abi-r1:opencl:navatala_cfd_scatter_mu_grad_ghost",
  12,
  kAbiArgs_opencl_navatala_cfd_scatter_mu_grad_ghost
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_cfd_scatter_rho_vec3_and_mask[] = {
  { "procFaces", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "recvVals", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "counts", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "rhoBcVal", 3, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "uBcX", 4, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "uBcY", 5, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "uBcZ", 6, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "rhoBcMask", 7, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "uBcMask", 8, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_cfd_scatter_rho_vec3_and_mask = {
  1,
  "navatala_cfd_scatter_rho_vec3_and_mask",
  "opencl",
  "navatala_cfd_scatter_rho_vec3_and_mask",
  "kernel:opencl:navatala_cfd_scatter_rho_vec3_and_mask",
  "abi-r1:opencl:navatala_cfd_scatter_rho_vec3_and_mask",
  "abi-r1:opencl:navatala_cfd_scatter_rho_vec3_and_mask",
  9,
  kAbiArgs_opencl_navatala_cfd_scatter_rho_vec3_and_mask
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_cfd_scatter_vec3_and_mask[] = {
  { "procFaces", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "recvVals", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "counts", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "bcX", 3, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "bcY", 4, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "bcZ", 5, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "bcMask", 6, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_cfd_scatter_vec3_and_mask = {
  1,
  "navatala_cfd_scatter_vec3_and_mask",
  "opencl",
  "navatala_cfd_scatter_vec3_and_mask",
  "kernel:opencl:navatala_cfd_scatter_vec3_and_mask",
  "abi-r1:opencl:navatala_cfd_scatter_vec3_and_mask",
  "abi-r1:opencl:navatala_cfd_scatter_vec3_and_mask",
  7,
  kAbiArgs_opencl_navatala_cfd_scatter_vec3_and_mask
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_cfd_scatter_vec3_mu_and_mask[] = {
  { "procFaces", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "recvVals", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "counts", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "bcX", 3, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "bcY", 4, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "bcZ", 5, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "muBc", 6, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "bcMask", 7, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_cfd_scatter_vec3_mu_and_mask = {
  1,
  "navatala_cfd_scatter_vec3_mu_and_mask",
  "opencl",
  "navatala_cfd_scatter_vec3_mu_and_mask",
  "kernel:opencl:navatala_cfd_scatter_vec3_mu_and_mask",
  "abi-r1:opencl:navatala_cfd_scatter_vec3_mu_and_mask",
  "abi-r1:opencl:navatala_cfd_scatter_vec3_mu_and_mask",
  8,
  kAbiArgs_opencl_navatala_cfd_scatter_vec3_mu_and_mask
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_cfd_cf_mesh_volume_optimizer_gradient_f32[] = {
  { "pointX", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "pointY", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "pointZ", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "tetConn", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "tetOffsets", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "tetCounts", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "fixedMask", 6, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "kPerStencil", 7, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "counts", 8, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "outGradX", 9, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "outGradY", 10, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "outGradZ", 11, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_cfd_cf_mesh_volume_optimizer_gradient_f32 = {
  1,
  "navatala_cfd_cf_mesh_volume_optimizer_gradient_f32",
  "opencl",
  "navatala_cfd_cf_mesh_volume_optimizer_gradient_f32",
  "kernel:opencl:navatala_cfd_cf_mesh_volume_optimizer_gradient_f32",
  "abi-r1:opencl:navatala_cfd_cf_mesh_volume_optimizer_gradient_f32",
  "abi-r1:opencl:navatala_cfd_cf_mesh_volume_optimizer_gradient_f32",
  12,
  kAbiArgs_opencl_navatala_cfd_cf_mesh_volume_optimizer_gradient_f32
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_cfd_cf_mesh_volume_optimizer_gradient_f64[] = {
  { "pointX", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "pointY", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "pointZ", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "tetConn", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "tetOffsets", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "tetCounts", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "fixedMask", 6, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "kPerStencil", 7, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "counts", 8, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "outGradX", 9, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "outGradY", 10, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "outGradZ", 11, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_cfd_cf_mesh_volume_optimizer_gradient_f64 = {
  1,
  "navatala_cfd_cf_mesh_volume_optimizer_gradient_f64",
  "opencl",
  "navatala_cfd_cf_mesh_volume_optimizer_gradient_f64",
  "kernel:opencl:navatala_cfd_cf_mesh_volume_optimizer_gradient_f64",
  "abi-r1:opencl:navatala_cfd_cf_mesh_volume_optimizer_gradient_f64",
  "abi-r1:opencl:navatala_cfd_cf_mesh_volume_optimizer_gradient_f64",
  12,
  kAbiArgs_opencl_navatala_cfd_cf_mesh_volume_optimizer_gradient_f64
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_cfd_cf_mesh_volume_optimizer_k_prepass_f32[] = {
  { "pointX", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "pointY", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "pointZ", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "tetConn", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "tetOffsets", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "tetCounts", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "fixedMask", 6, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "params", 7, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "counts", 8, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "outK", 9, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_cfd_cf_mesh_volume_optimizer_k_prepass_f32 = {
  1,
  "navatala_cfd_cf_mesh_volume_optimizer_k_prepass_f32",
  "opencl",
  "navatala_cfd_cf_mesh_volume_optimizer_k_prepass_f32",
  "kernel:opencl:navatala_cfd_cf_mesh_volume_optimizer_k_prepass_f32",
  "abi-r1:opencl:navatala_cfd_cf_mesh_volume_optimizer_k_prepass_f32",
  "abi-r1:opencl:navatala_cfd_cf_mesh_volume_optimizer_k_prepass_f32",
  10,
  kAbiArgs_opencl_navatala_cfd_cf_mesh_volume_optimizer_k_prepass_f32
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_cfd_cf_mesh_volume_optimizer_k_prepass_f64[] = {
  { "pointX", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "pointY", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "pointZ", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "tetConn", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "tetOffsets", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "tetCounts", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "fixedMask", 6, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "params", 7, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 8, 8, 256, nullptr, 0, 0 },
  { "counts", 8, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "outK", 9, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_cfd_cf_mesh_volume_optimizer_k_prepass_f64 = {
  1,
  "navatala_cfd_cf_mesh_volume_optimizer_k_prepass_f64",
  "opencl",
  "navatala_cfd_cf_mesh_volume_optimizer_k_prepass_f64",
  "kernel:opencl:navatala_cfd_cf_mesh_volume_optimizer_k_prepass_f64",
  "abi-r1:opencl:navatala_cfd_cf_mesh_volume_optimizer_k_prepass_f64",
  "abi-r1:opencl:navatala_cfd_cf_mesh_volume_optimizer_k_prepass_f64",
  10,
  kAbiArgs_opencl_navatala_cfd_cf_mesh_volume_optimizer_k_prepass_f64
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_cfd_cf_mesh_volume_optimizer_objective_f32[] = {
  { "pointX", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "pointY", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "pointZ", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "tetConn", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "tetOffsets", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "tetCounts", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "fixedMask", 6, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "kPerStencil", 7, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "counts", 8, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "outObjective", 9, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_cfd_cf_mesh_volume_optimizer_objective_f32 = {
  1,
  "navatala_cfd_cf_mesh_volume_optimizer_objective_f32",
  "opencl",
  "navatala_cfd_cf_mesh_volume_optimizer_objective_f32",
  "kernel:opencl:navatala_cfd_cf_mesh_volume_optimizer_objective_f32",
  "abi-r1:opencl:navatala_cfd_cf_mesh_volume_optimizer_objective_f32",
  "abi-r1:opencl:navatala_cfd_cf_mesh_volume_optimizer_objective_f32",
  10,
  kAbiArgs_opencl_navatala_cfd_cf_mesh_volume_optimizer_objective_f32
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_cfd_cf_mesh_volume_optimizer_objective_f64[] = {
  { "pointX", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "pointY", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "pointZ", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "tetConn", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "tetOffsets", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "tetCounts", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "fixedMask", 6, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "kPerStencil", 7, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "counts", 8, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "outObjective", 9, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_cfd_cf_mesh_volume_optimizer_objective_f64 = {
  1,
  "navatala_cfd_cf_mesh_volume_optimizer_objective_f64",
  "opencl",
  "navatala_cfd_cf_mesh_volume_optimizer_objective_f64",
  "kernel:opencl:navatala_cfd_cf_mesh_volume_optimizer_objective_f64",
  "abi-r1:opencl:navatala_cfd_cf_mesh_volume_optimizer_objective_f64",
  "abi-r1:opencl:navatala_cfd_cf_mesh_volume_optimizer_objective_f64",
  10,
  kAbiArgs_opencl_navatala_cfd_cf_mesh_volume_optimizer_objective_f64
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_cfd_add_vol_vector[] = {
  { "ax", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "ay", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "az", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "bx", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "by_", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "bz", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "params", 6, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "outX", 7, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "outY", 8, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "outZ", 9, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_cfd_add_vol_vector = {
  1,
  "navatala_cfd_add_vol_vector",
  "opencl",
  "navatala_cfd_add_vol_vector",
  "kernel:opencl:navatala_cfd_add_vol_vector",
  "abi-r1:opencl:navatala_cfd_add_vol_vector",
  "abi-r1:opencl:navatala_cfd_add_vol_vector",
  10,
  kAbiArgs_opencl_navatala_cfd_add_vol_vector
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_cfd_div_rho_phi_u_explicit[] = {
  { "phi", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "rhoCell", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "ux", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "uy", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "uz", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "owner", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "neighbour", 6, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "weights", 7, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "offsets", 8, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "faceIdx", 9, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "sign", 10, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "vol", 11, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "rhoBcVal", 12, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "rhoBcMask", 13, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "uBcX", 14, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "uBcY", 15, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "uBcZ", 16, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "uBcMask", 17, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "params", 18, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 12, 12, 256, nullptr, 0, 0 },
  { "outX", 19, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "outY", 20, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "outZ", 21, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_cfd_div_rho_phi_u_explicit = {
  1,
  "navatala_cfd_div_rho_phi_u_explicit",
  "opencl",
  "navatala_cfd_div_rho_phi_u_explicit",
  "kernel:opencl:navatala_cfd_div_rho_phi_u_explicit",
  "abi-r1:opencl:navatala_cfd_div_rho_phi_u_explicit",
  "abi-r1:opencl:navatala_cfd_div_rho_phi_u_explicit",
  22,
  kAbiArgs_opencl_navatala_cfd_div_rho_phi_u_explicit
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_cfd_hby_a_from_u_and_terms[] = {
  { "ux", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "uy", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "uz", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "lapx", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "lapy", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "lapz", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "divx", 6, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "divy", 7, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "divz", 8, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "extrax", 9, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "extray", 10, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "extraz", 11, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "rau", 12, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "counts", 13, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "modeBuf", 14, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "outx", 15, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "outy", 16, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "outz", 17, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_cfd_hby_a_from_u_and_terms = {
  1,
  "navatala_cfd_hby_a_from_u_and_terms",
  "opencl",
  "navatala_cfd_hby_a_from_u_and_terms",
  "kernel:opencl:navatala_cfd_hby_a_from_u_and_terms",
  "abi-r1:opencl:navatala_cfd_hby_a_from_u_and_terms",
  "abi-r1:opencl:navatala_cfd_hby_a_from_u_and_terms",
  18,
  kAbiArgs_opencl_navatala_cfd_hby_a_from_u_and_terms
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_cfd_apply_sp_to_hby_a[] = {
  { "hx", 0, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "hy", 1, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "hz", 2, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "ux", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "uy", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "uz", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "rAU", 6, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "sp", 7, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "counts", 8, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_cfd_apply_sp_to_hby_a = {
  1,
  "navatala_cfd_apply_sp_to_hby_a",
  "opencl",
  "navatala_cfd_apply_sp_to_hby_a",
  "kernel:opencl:navatala_cfd_apply_sp_to_hby_a",
  "abi-r1:opencl:navatala_cfd_apply_sp_to_hby_a",
  "abi-r1:opencl:navatala_cfd_apply_sp_to_hby_a",
  9,
  kAbiArgs_opencl_navatala_cfd_apply_sp_to_hby_a
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_cfd_laplacian_scalar_explicit[] = {
  { "muCell", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "phiCell", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "owner", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "neighbour", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "weights", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "magSf", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "deltaCoeffs", 6, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "offsets", 7, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "faceIdx", 8, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "sign", 9, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "vol", 10, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "bcMu", 11, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "bcMuMask", 12, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "bcPhi", 13, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "bcPhiMask", 14, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "params", 15, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 12, 12, 256, nullptr, 0, 0 },
  { "outL", 16, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_cfd_laplacian_scalar_explicit = {
  1,
  "navatala_cfd_laplacian_scalar_explicit",
  "opencl",
  "navatala_cfd_laplacian_scalar_explicit",
  "kernel:opencl:navatala_cfd_laplacian_scalar_explicit",
  "abi-r1:opencl:navatala_cfd_laplacian_scalar_explicit",
  "abi-r1:opencl:navatala_cfd_laplacian_scalar_explicit",
  17,
  kAbiArgs_opencl_navatala_cfd_laplacian_scalar_explicit
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_cfd_laplacian_vector_explicit[] = {
  { "muCell", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "muBc", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "ux", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "uy", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "uz", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "owner", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "neighbour", 6, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "weights", 7, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "magSf", 8, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "deltaCoeffs", 9, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "corrX", 10, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "corrY", 11, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "corrZ", 12, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "gradRow0", 13, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "gradRow1", 14, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "gradRow2", 15, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "offsets", 16, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "faceIdx", 17, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "sign", 18, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "vol", 19, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "bcX", 20, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "bcY", 21, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "bcZ", 22, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "bcMask", 23, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "params", 24, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 12, 12, 256, nullptr, 0, 0 },
  { "outX", 25, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "outY", 26, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "outZ", 27, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_cfd_laplacian_vector_explicit = {
  1,
  "navatala_cfd_laplacian_vector_explicit",
  "opencl",
  "navatala_cfd_laplacian_vector_explicit",
  "kernel:opencl:navatala_cfd_laplacian_vector_explicit",
  "abi-r1:opencl:navatala_cfd_laplacian_vector_explicit",
  "abi-r1:opencl:navatala_cfd_laplacian_vector_explicit",
  28,
  kAbiArgs_opencl_navatala_cfd_laplacian_vector_explicit
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_cfd_predictor_hby_a_from_source[] = {
  { "rAU", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "sourceX", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "sourceY", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "sourceZ", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "counts1", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "outHbyAX", 5, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "outHbyAY", 6, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "outHbyAZ", 7, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_cfd_predictor_hby_a_from_source = {
  1,
  "navatala_cfd_predictor_hby_a_from_source",
  "opencl",
  "navatala_cfd_predictor_hby_a_from_source",
  "kernel:opencl:navatala_cfd_predictor_hby_a_from_source",
  "abi-r1:opencl:navatala_cfd_predictor_hby_a_from_source",
  "abi-r1:opencl:navatala_cfd_predictor_hby_a_from_source",
  8,
  kAbiArgs_opencl_navatala_cfd_predictor_hby_a_from_source
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_cfd_predictor_inverse_diag[] = {
  { "inputDiag", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "counts1", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "outRAU", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_cfd_predictor_inverse_diag = {
  1,
  "navatala_cfd_predictor_inverse_diag",
  "opencl",
  "navatala_cfd_predictor_inverse_diag",
  "kernel:opencl:navatala_cfd_predictor_inverse_diag",
  "abi-r1:opencl:navatala_cfd_predictor_inverse_diag",
  "abi-r1:opencl:navatala_cfd_predictor_inverse_diag",
  3,
  kAbiArgs_opencl_navatala_cfd_predictor_inverse_diag
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_cfd_predictor_phi_hby_a_fuse[] = {
  { "fluxHbyA", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "rhoRauFace", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "ddtCorr", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "counts1", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "outPhiHbyA", 4, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_cfd_predictor_phi_hby_a_fuse = {
  1,
  "navatala_cfd_predictor_phi_hby_a_fuse",
  "opencl",
  "navatala_cfd_predictor_phi_hby_a_fuse",
  "kernel:opencl:navatala_cfd_predictor_phi_hby_a_fuse",
  "abi-r1:opencl:navatala_cfd_predictor_phi_hby_a_fuse",
  "abi-r1:opencl:navatala_cfd_predictor_phi_hby_a_fuse",
  5,
  kAbiArgs_opencl_navatala_cfd_predictor_phi_hby_a_fuse
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_cfd_axpy_cells[] = {
  { "x", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "counts", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "paramsF", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "y", 3, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_cfd_axpy_cells = {
  1,
  "navatala_cfd_axpy_cells",
  "opencl",
  "navatala_cfd_axpy_cells",
  "kernel:opencl:navatala_cfd_axpy_cells",
  "abi-r1:opencl:navatala_cfd_axpy_cells",
  "abi-r1:opencl:navatala_cfd_axpy_cells",
  4,
  kAbiArgs_opencl_navatala_cfd_axpy_cells
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_cfd_ddt_diagonal[] = {
  { "rho", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "vol", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "counts", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "paramsF", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "outDiag", 4, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_cfd_ddt_diagonal = {
  1,
  "navatala_cfd_ddt_diagonal",
  "opencl",
  "navatala_cfd_ddt_diagonal",
  "kernel:opencl:navatala_cfd_ddt_diagonal",
  "abi-r1:opencl:navatala_cfd_ddt_diagonal",
  "abi-r1:opencl:navatala_cfd_ddt_diagonal",
  5,
  kAbiArgs_opencl_navatala_cfd_ddt_diagonal
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_cfd_div_upwind_coeffs[] = {
  { "weights", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "flux", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "counts", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "outLower", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "outUpper", 4, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_cfd_div_upwind_coeffs = {
  1,
  "navatala_cfd_div_upwind_coeffs",
  "opencl",
  "navatala_cfd_div_upwind_coeffs",
  "kernel:opencl:navatala_cfd_div_upwind_coeffs",
  "abi-r1:opencl:navatala_cfd_div_upwind_coeffs",
  "abi-r1:opencl:navatala_cfd_div_upwind_coeffs",
  5,
  kAbiArgs_opencl_navatala_cfd_div_upwind_coeffs
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_cfd_fvm_sp[] = {
  { "sp", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "vol", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "counts", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "outDiag", 3, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_cfd_fvm_sp = {
  1,
  "navatala_cfd_fvm_sp",
  "opencl",
  "navatala_cfd_fvm_sp",
  "kernel:opencl:navatala_cfd_fvm_sp",
  "abi-r1:opencl:navatala_cfd_fvm_sp",
  "abi-r1:opencl:navatala_cfd_fvm_sp",
  4,
  kAbiArgs_opencl_navatala_cfd_fvm_sp
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_cfd_fvm_su_sp[] = {
  { "sp", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "psi", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "vol", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "counts", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "outDiag", 4, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "outSource", 5, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_cfd_fvm_su_sp = {
  1,
  "navatala_cfd_fvm_su_sp",
  "opencl",
  "navatala_cfd_fvm_su_sp",
  "kernel:opencl:navatala_cfd_fvm_su_sp",
  "abi-r1:opencl:navatala_cfd_fvm_su_sp",
  "abi-r1:opencl:navatala_cfd_fvm_su_sp",
  6,
  kAbiArgs_opencl_navatala_cfd_fvm_su_sp
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_cfd_laplacian_coeffs[] = {
  { "gamma", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "owner", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "neighbour", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "weights", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "deltaCoeffs", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "magSf", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "counts", 6, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "outUpper", 7, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "outLower", 8, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_cfd_laplacian_coeffs = {
  1,
  "navatala_cfd_laplacian_coeffs",
  "opencl",
  "navatala_cfd_laplacian_coeffs",
  "kernel:opencl:navatala_cfd_laplacian_coeffs",
  "abi-r1:opencl:navatala_cfd_laplacian_coeffs",
  "abi-r1:opencl:navatala_cfd_laplacian_coeffs",
  9,
  kAbiArgs_opencl_navatala_cfd_laplacian_coeffs
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_cfd_laplacian_coeffs_accum[] = {
  { "gamma", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "deltaCoeffs", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "magSf", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "counts", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "outUpper", 4, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "outLower", 5, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_cfd_laplacian_coeffs_accum = {
  1,
  "navatala_cfd_laplacian_coeffs_accum",
  "opencl",
  "navatala_cfd_laplacian_coeffs_accum",
  "kernel:opencl:navatala_cfd_laplacian_coeffs_accum",
  "abi-r1:opencl:navatala_cfd_laplacian_coeffs_accum",
  "abi-r1:opencl:navatala_cfd_laplacian_coeffs_accum",
  6,
  kAbiArgs_opencl_navatala_cfd_laplacian_coeffs_accum
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_cfd_linear_upwind_correction_source[] = {
  { "luFaceVals", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "flux", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "owner", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "neighbour", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "psi", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "offsets", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "faceIdx", 6, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "signF", 7, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "counts", 8, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 8, 8, 256, nullptr, 0, 0 },
  { "outSource", 9, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_cfd_linear_upwind_correction_source = {
  1,
  "navatala_cfd_linear_upwind_correction_source",
  "opencl",
  "navatala_cfd_linear_upwind_correction_source",
  "kernel:opencl:navatala_cfd_linear_upwind_correction_source",
  "abi-r1:opencl:navatala_cfd_linear_upwind_correction_source",
  "abi-r1:opencl:navatala_cfd_linear_upwind_correction_source",
  10,
  kAbiArgs_opencl_navatala_cfd_linear_upwind_correction_source
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_cfd_matrix_relax[] = {
  { "diagOld", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "sumOff", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "psi", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "counts", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "paramsF", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "diag", 5, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "source", 6, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_cfd_matrix_relax = {
  1,
  "navatala_cfd_matrix_relax",
  "opencl",
  "navatala_cfd_matrix_relax",
  "kernel:opencl:navatala_cfd_matrix_relax",
  "abi-r1:opencl:navatala_cfd_matrix_relax",
  "abi-r1:opencl:navatala_cfd_matrix_relax",
  7,
  kAbiArgs_opencl_navatala_cfd_matrix_relax
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_cfd_neg_sum_diag[] = {
  { "upper", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "lower", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "offsets", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "faceIdx", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "sign", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "counts", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 8, 8, 256, nullptr, 0, 0 },
  { "outDiag", 6, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_cfd_neg_sum_diag = {
  1,
  "navatala_cfd_neg_sum_diag",
  "opencl",
  "navatala_cfd_neg_sum_diag",
  "kernel:opencl:navatala_cfd_neg_sum_diag",
  "abi-r1:opencl:navatala_cfd_neg_sum_diag",
  "abi-r1:opencl:navatala_cfd_neg_sum_diag",
  7,
  kAbiArgs_opencl_navatala_cfd_neg_sum_diag
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_cfd_pack_boundary_adjacent_cells[] = {
  { "cellField", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "indexMap", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "counts", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "outPacked", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_cfd_pack_boundary_adjacent_cells = {
  1,
  "navatala_cfd_pack_boundary_adjacent_cells",
  "opencl",
  "navatala_cfd_pack_boundary_adjacent_cells",
  "kernel:opencl:navatala_cfd_pack_boundary_adjacent_cells",
  "abi-r1:opencl:navatala_cfd_pack_boundary_adjacent_cells",
  "abi-r1:opencl:navatala_cfd_pack_boundary_adjacent_cells",
  4,
  kAbiArgs_opencl_navatala_cfd_pack_boundary_adjacent_cells
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_cfd_scalar_jacobi_clamp_min[] = {
  { "lowerBound", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "counts", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "x", 2, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_cfd_scalar_jacobi_clamp_min = {
  1,
  "navatala_cfd_scalar_jacobi_clamp_min",
  "opencl",
  "navatala_cfd_scalar_jacobi_clamp_min",
  "kernel:opencl:navatala_cfd_scalar_jacobi_clamp_min",
  "abi-r1:opencl:navatala_cfd_scalar_jacobi_clamp_min",
  "abi-r1:opencl:navatala_cfd_scalar_jacobi_clamp_min",
  3,
  kAbiArgs_opencl_navatala_cfd_scalar_jacobi_clamp_min
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_cfd_scalar_jacobi_update[] = {
  { "ax", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "diag", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "rhs", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "params", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "counts", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "x", 5, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_cfd_scalar_jacobi_update = {
  1,
  "navatala_cfd_scalar_jacobi_update",
  "opencl",
  "navatala_cfd_scalar_jacobi_update",
  "kernel:opencl:navatala_cfd_scalar_jacobi_update",
  "abi-r1:opencl:navatala_cfd_scalar_jacobi_update",
  "abi-r1:opencl:navatala_cfd_scalar_jacobi_update",
  6,
  kAbiArgs_opencl_navatala_cfd_scalar_jacobi_update
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_cfd_scalar_ldu_mat_vec[] = {
  { "diag", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "upper", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "lower", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "x", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "owner", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "neighbour", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "offsets", 6, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "faceIdx", 7, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "sign", 8, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "counts", 9, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "ax", 10, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_cfd_scalar_ldu_mat_vec = {
  1,
  "navatala_cfd_scalar_ldu_mat_vec",
  "opencl",
  "navatala_cfd_scalar_ldu_mat_vec",
  "kernel:opencl:navatala_cfd_scalar_ldu_mat_vec",
  "abi-r1:opencl:navatala_cfd_scalar_ldu_mat_vec",
  "abi-r1:opencl:navatala_cfd_scalar_ldu_mat_vec",
  11,
  kAbiArgs_opencl_navatala_cfd_scalar_ldu_mat_vec
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_cfd_scalar_ldu_multi_color_gs_sweep[] = {
  { "diag", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "upper", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "lower", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "rhs", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "owner", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "neighbour", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "offsets", 6, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "faceIdx", 7, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "sign", 8, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "colorOrder", 9, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "colorRange", 10, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "params", 11, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "counts", 12, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "x", 13, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_cfd_scalar_ldu_multi_color_gs_sweep = {
  1,
  "navatala_cfd_scalar_ldu_multi_color_gs_sweep",
  "opencl",
  "navatala_cfd_scalar_ldu_multi_color_gs_sweep",
  "kernel:opencl:navatala_cfd_scalar_ldu_multi_color_gs_sweep",
  "abi-r1:opencl:navatala_cfd_scalar_ldu_multi_color_gs_sweep",
  "abi-r1:opencl:navatala_cfd_scalar_ldu_multi_color_gs_sweep",
  14,
  kAbiArgs_opencl_navatala_cfd_scalar_ldu_multi_color_gs_sweep
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_cfd_sum_mag_off_diag[] = {
  { "upper", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "lower", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "offsets", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "faceIdx", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "sign", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "counts", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 8, 8, 256, nullptr, 0, 0 },
  { "outSumOff", 6, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_cfd_sum_mag_off_diag = {
  1,
  "navatala_cfd_sum_mag_off_diag",
  "opencl",
  "navatala_cfd_sum_mag_off_diag",
  "kernel:opencl:navatala_cfd_sum_mag_off_diag",
  "abi-r1:opencl:navatala_cfd_sum_mag_off_diag",
  "abi-r1:opencl:navatala_cfd_sum_mag_off_diag",
  7,
  kAbiArgs_opencl_navatala_cfd_sum_mag_off_diag
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_cfd_gamg_axpy_in_place[] = {
  { "x", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "alpha", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "nRows", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "y", 3, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_cfd_gamg_axpy_in_place = {
  1,
  "navatala_cfd_gamg_axpy_in_place",
  "opencl",
  "navatala_cfd_gamg_axpy_in_place",
  "kernel:opencl:navatala_cfd_gamg_axpy_in_place",
  "abi-r1:opencl:navatala_cfd_gamg_axpy_in_place",
  "abi-r1:opencl:navatala_cfd_gamg_axpy_in_place",
  4,
  kAbiArgs_opencl_navatala_cfd_gamg_axpy_in_place
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_cfd_gamg_compute_scale_factor[] = {
  { "num", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "den", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "sf", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_cfd_gamg_compute_scale_factor = {
  1,
  "navatala_cfd_gamg_compute_scale_factor",
  "opencl",
  "navatala_cfd_gamg_compute_scale_factor",
  "kernel:opencl:navatala_cfd_gamg_compute_scale_factor",
  "abi-r1:opencl:navatala_cfd_gamg_compute_scale_factor",
  "abi-r1:opencl:navatala_cfd_gamg_compute_scale_factor",
  3,
  kAbiArgs_opencl_navatala_cfd_gamg_compute_scale_factor
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_cfd_gamg_dot_finalize[] = {
  { "partials", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "result", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_cfd_gamg_dot_finalize = {
  1,
  "navatala_cfd_gamg_dot_finalize",
  "opencl",
  "navatala_cfd_gamg_dot_finalize",
  "kernel:opencl:navatala_cfd_gamg_dot_finalize",
  "abi-r1:opencl:navatala_cfd_gamg_dot_finalize",
  "abi-r1:opencl:navatala_cfd_gamg_dot_finalize",
  3,
  kAbiArgs_opencl_navatala_cfd_gamg_dot_finalize
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_cfd_gamg_negate_scalar[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 1, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_cfd_gamg_negate_scalar = {
  1,
  "navatala_cfd_gamg_negate_scalar",
  "opencl",
  "navatala_cfd_gamg_negate_scalar",
  "kernel:opencl:navatala_cfd_gamg_negate_scalar",
  "abi-r1:opencl:navatala_cfd_gamg_negate_scalar",
  "abi-r1:opencl:navatala_cfd_gamg_negate_scalar",
  2,
  kAbiArgs_opencl_navatala_cfd_gamg_negate_scalar
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_cfd_gamg_scale_correction[] = {
  { "z", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "r", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "Az", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "diagInv", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "sfPtr", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "nRows", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "corr", 6, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_cfd_gamg_scale_correction = {
  1,
  "navatala_cfd_gamg_scale_correction",
  "opencl",
  "navatala_cfd_gamg_scale_correction",
  "kernel:opencl:navatala_cfd_gamg_scale_correction",
  "abi-r1:opencl:navatala_cfd_gamg_scale_correction",
  "abi-r1:opencl:navatala_cfd_gamg_scale_correction",
  7,
  kAbiArgs_opencl_navatala_cfd_gamg_scale_correction
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_cfd_mg_dic_build[] = {
  { "diag", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "edgeU", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "edgeV", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "edgeCf", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "counts", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "rD", 5, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_cfd_mg_dic_build = {
  1,
  "navatala_cfd_mg_dic_build",
  "opencl",
  "navatala_cfd_mg_dic_build",
  "kernel:opencl:navatala_cfd_mg_dic_build",
  "abi-r1:opencl:navatala_cfd_mg_dic_build",
  "abi-r1:opencl:navatala_cfd_mg_dic_build",
  6,
  kAbiArgs_opencl_navatala_cfd_mg_dic_build
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_cfd_mg_dic_smooth[] = {
  { "b", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "ax", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "rD", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "edgeU", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "edgeV", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "edgeCf", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "counts", 6, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "x", 7, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "z", 8, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_cfd_mg_dic_smooth = {
  1,
  "navatala_cfd_mg_dic_smooth",
  "opencl",
  "navatala_cfd_mg_dic_smooth",
  "kernel:opencl:navatala_cfd_mg_dic_smooth",
  "abi-r1:opencl:navatala_cfd_mg_dic_smooth",
  "abi-r1:opencl:navatala_cfd_mg_dic_smooth",
  9,
  kAbiArgs_opencl_navatala_cfd_mg_dic_smooth
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_cfd_mg_coarse_jacobi[] = {
  { "x", 0, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "r", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "diag", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "mgCounts", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_cfd_mg_coarse_jacobi = {
  1,
  "navatala_cfd_mg_coarse_jacobi",
  "opencl",
  "navatala_cfd_mg_coarse_jacobi",
  "kernel:opencl:navatala_cfd_mg_coarse_jacobi",
  "abi-r1:opencl:navatala_cfd_mg_coarse_jacobi",
  "abi-r1:opencl:navatala_cfd_mg_coarse_jacobi",
  4,
  kAbiArgs_opencl_navatala_cfd_mg_coarse_jacobi
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_cfd_mg_coarse_matvec[] = {
  { "edgeU", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "edgeV", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "edgeCf", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "diag", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "x", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "outAx", 5, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "mgCounts", 6, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_cfd_mg_coarse_matvec = {
  1,
  "navatala_cfd_mg_coarse_matvec",
  "opencl",
  "navatala_cfd_mg_coarse_matvec",
  "kernel:opencl:navatala_cfd_mg_coarse_matvec",
  "abi-r1:opencl:navatala_cfd_mg_coarse_matvec",
  "abi-r1:opencl:navatala_cfd_mg_coarse_matvec",
  7,
  kAbiArgs_opencl_navatala_cfd_mg_coarse_matvec
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_cfd_mg_coarse_rhs[] = {
  { "b", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "ax", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "r", 2, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "mgCounts", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_cfd_mg_coarse_rhs = {
  1,
  "navatala_cfd_mg_coarse_rhs",
  "opencl",
  "navatala_cfd_mg_coarse_rhs",
  "kernel:opencl:navatala_cfd_mg_coarse_rhs",
  "abi-r1:opencl:navatala_cfd_mg_coarse_rhs",
  "abi-r1:opencl:navatala_cfd_mg_coarse_rhs",
  4,
  kAbiArgs_opencl_navatala_cfd_mg_coarse_rhs
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_cfd_mg_find_best_match[] = {
  { "csrOffsets", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "csrFaceIdx", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "csrSign", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "strength", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "owner", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "nei", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "bestNeighbor", 6, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "counts", 7, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_cfd_mg_find_best_match = {
  1,
  "navatala_cfd_mg_find_best_match",
  "opencl",
  "navatala_cfd_mg_find_best_match",
  "kernel:opencl:navatala_cfd_mg_find_best_match",
  "abi-r1:opencl:navatala_cfd_mg_find_best_match",
  "abi-r1:opencl:navatala_cfd_mg_find_best_match",
  8,
  kAbiArgs_opencl_navatala_cfd_mg_find_best_match
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_cfd_mg_pair_match[] = {
  { "bestNeighbor", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "aggMap", 1, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "counts", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_cfd_mg_pair_match = {
  1,
  "navatala_cfd_mg_pair_match",
  "opencl",
  "navatala_cfd_mg_pair_match",
  "kernel:opencl:navatala_cfd_mg_pair_match",
  "abi-r1:opencl:navatala_cfd_mg_pair_match",
  "abi-r1:opencl:navatala_cfd_mg_pair_match",
  3,
  kAbiArgs_opencl_navatala_cfd_mg_pair_match
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_cfd_mg_prolong_add[] = {
  { "xCoarse", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "prolongRowOffsets", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "prolongCols", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "prolongVals", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "zFine", 4, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "mgCounts", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_cfd_mg_prolong_add = {
  1,
  "navatala_cfd_mg_prolong_add",
  "opencl",
  "navatala_cfd_mg_prolong_add",
  "kernel:opencl:navatala_cfd_mg_prolong_add",
  "abi-r1:opencl:navatala_cfd_mg_prolong_add",
  "abi-r1:opencl:navatala_cfd_mg_prolong_add",
  6,
  kAbiArgs_opencl_navatala_cfd_mg_prolong_add
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_cfd_mg_renumber_aggregates[] = {
  { "aggMap", 0, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "counter", 1, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "counts", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_cfd_mg_renumber_aggregates = {
  1,
  "navatala_cfd_mg_renumber_aggregates",
  "opencl",
  "navatala_cfd_mg_renumber_aggregates",
  "kernel:opencl:navatala_cfd_mg_renumber_aggregates",
  "abi-r1:opencl:navatala_cfd_mg_renumber_aggregates",
  "abi-r1:opencl:navatala_cfd_mg_renumber_aggregates",
  3,
  kAbiArgs_opencl_navatala_cfd_mg_renumber_aggregates
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_cfd_mg_renumber_propagate[] = {
  { "aggMap", 0, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "counts", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_cfd_mg_renumber_propagate = {
  1,
  "navatala_cfd_mg_renumber_propagate",
  "opencl",
  "navatala_cfd_mg_renumber_propagate",
  "kernel:opencl:navatala_cfd_mg_renumber_propagate",
  "abi-r1:opencl:navatala_cfd_mg_renumber_propagate",
  "abi-r1:opencl:navatala_cfd_mg_renumber_propagate",
  2,
  kAbiArgs_opencl_navatala_cfd_mg_renumber_propagate
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_cfd_mg_resolve_chains[] = {
  { "aggMap", 0, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "counts", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_cfd_mg_resolve_chains = {
  1,
  "navatala_cfd_mg_resolve_chains",
  "opencl",
  "navatala_cfd_mg_resolve_chains",
  "kernel:opencl:navatala_cfd_mg_resolve_chains",
  "abi-r1:opencl:navatala_cfd_mg_resolve_chains",
  "abi-r1:opencl:navatala_cfd_mg_resolve_chains",
  2,
  kAbiArgs_opencl_navatala_cfd_mg_resolve_chains
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_cfd_mg_restrict_sum[] = {
  { "rFine", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "prolongRowOffsets", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "prolongCols", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "prolongVals", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "rCoarse", 4, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "mgCounts", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_cfd_mg_restrict_sum = {
  1,
  "navatala_cfd_mg_restrict_sum",
  "opencl",
  "navatala_cfd_mg_restrict_sum",
  "kernel:opencl:navatala_cfd_mg_restrict_sum",
  "abi-r1:opencl:navatala_cfd_mg_restrict_sum",
  "abi-r1:opencl:navatala_cfd_mg_restrict_sum",
  6,
  kAbiArgs_opencl_navatala_cfd_mg_restrict_sum
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_cfd_mg_strength_of_connection[] = {
  { "diag", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "upper", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "owner", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "nei", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "strength", 4, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "counts", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_cfd_mg_strength_of_connection = {
  1,
  "navatala_cfd_mg_strength_of_connection",
  "opencl",
  "navatala_cfd_mg_strength_of_connection",
  "kernel:opencl:navatala_cfd_mg_strength_of_connection",
  "abi-r1:opencl:navatala_cfd_mg_strength_of_connection",
  "abi-r1:opencl:navatala_cfd_mg_strength_of_connection",
  6,
  kAbiArgs_opencl_navatala_cfd_mg_strength_of_connection
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_cfd_mg_zero_coarse[] = {
  { "x", 0, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "mgCounts", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_cfd_mg_zero_coarse = {
  1,
  "navatala_cfd_mg_zero_coarse",
  "opencl",
  "navatala_cfd_mg_zero_coarse",
  "kernel:opencl:navatala_cfd_mg_zero_coarse",
  "abi-r1:opencl:navatala_cfd_mg_zero_coarse",
  "abi-r1:opencl:navatala_cfd_mg_zero_coarse",
  2,
  kAbiArgs_opencl_navatala_cfd_mg_zero_coarse
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_cfd_phi_correct[] = {
  { "phiIn", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "pCell", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "owner", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "neighbour", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "magSf", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "deltaCoeffs", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "coeff", 6, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "bcVal", 7, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "bcMask", 8, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "bcSnGrad", 9, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "bcSnGradMask", 10, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "faceFluxCorrection", 11, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "counts", 12, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 8, 8, 256, nullptr, 0, 0 },
  { "paramsF", 13, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "phiOut", 14, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_cfd_phi_correct = {
  1,
  "navatala_cfd_phi_correct",
  "opencl",
  "navatala_cfd_phi_correct",
  "kernel:opencl:navatala_cfd_phi_correct",
  "abi-r1:opencl:navatala_cfd_phi_correct",
  "abi-r1:opencl:navatala_cfd_phi_correct",
  15,
  kAbiArgs_opencl_navatala_cfd_phi_correct
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_cfd_phi_correct_boundary[] = {
  { "phiIn", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "pCell", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "owner", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "magSf", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "deltaCoeffs", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "coeff", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "bcVal", 6, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "bcMask", 7, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "bcSnGrad", 8, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "bcSnGradMask", 9, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "faceFluxCorrection", 10, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "counts", 11, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 8, 8, 256, nullptr, 0, 0 },
  { "paramsF", 12, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "phiOut", 13, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_cfd_phi_correct_boundary = {
  1,
  "navatala_cfd_phi_correct_boundary",
  "opencl",
  "navatala_cfd_phi_correct_boundary",
  "kernel:opencl:navatala_cfd_phi_correct_boundary",
  "abi-r1:opencl:navatala_cfd_phi_correct_boundary",
  "abi-r1:opencl:navatala_cfd_phi_correct_boundary",
  14,
  kAbiArgs_opencl_navatala_cfd_phi_correct_boundary
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_cfd_phi_correct_internal[] = {
  { "phiIn", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "pCell", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "owner", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "neighbour", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "magSf", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "deltaCoeffs", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "coeff", 6, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "faceFluxCorrection", 7, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "counts", 8, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 8, 8, 256, nullptr, 0, 0 },
  { "paramsF", 9, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "phiOut", 10, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_cfd_phi_correct_internal = {
  1,
  "navatala_cfd_phi_correct_internal",
  "opencl",
  "navatala_cfd_phi_correct_internal",
  "kernel:opencl:navatala_cfd_phi_correct_internal",
  "abi-r1:opencl:navatala_cfd_phi_correct_internal",
  "abi-r1:opencl:navatala_cfd_phi_correct_internal",
  11,
  kAbiArgs_opencl_navatala_cfd_phi_correct_internal
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_cfd_axpy_in_place[] = {
  { "x", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "y", 1, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "counts", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 12, 12, 256, nullptr, 0, 0 },
  { "paramsF", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_cfd_axpy_in_place = {
  1,
  "navatala_cfd_axpy_in_place",
  "opencl",
  "navatala_cfd_axpy_in_place",
  "kernel:opencl:navatala_cfd_axpy_in_place",
  "abi-r1:opencl:navatala_cfd_axpy_in_place",
  "abi-r1:opencl:navatala_cfd_axpy_in_place",
  4,
  kAbiArgs_opencl_navatala_cfd_axpy_in_place
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_cfd_bc_dirichlet_face_flux[] = {
  { "cf", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "bcVal", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "bcMask", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "counts", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 12, 12, 256, nullptr, 0, 0 },
  { "outFlux", 4, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_cfd_bc_dirichlet_face_flux = {
  1,
  "navatala_cfd_bc_dirichlet_face_flux",
  "opencl",
  "navatala_cfd_bc_dirichlet_face_flux",
  "kernel:opencl:navatala_cfd_bc_dirichlet_face_flux",
  "abi-r1:opencl:navatala_cfd_bc_dirichlet_face_flux",
  "abi-r1:opencl:navatala_cfd_bc_dirichlet_face_flux",
  5,
  kAbiArgs_opencl_navatala_cfd_bc_dirichlet_face_flux
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_cfd_bc_sn_grad_face_flux[] = {
  { "cf", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "delta", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "bcSnGrad", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "bcSnGradMask", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "counts", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 12, 12, 256, nullptr, 0, 0 },
  { "outFlux", 5, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_cfd_bc_sn_grad_face_flux = {
  1,
  "navatala_cfd_bc_sn_grad_face_flux",
  "opencl",
  "navatala_cfd_bc_sn_grad_face_flux",
  "kernel:opencl:navatala_cfd_bc_sn_grad_face_flux",
  "abi-r1:opencl:navatala_cfd_bc_sn_grad_face_flux",
  "abi-r1:opencl:navatala_cfd_bc_sn_grad_face_flux",
  6,
  kAbiArgs_opencl_navatala_cfd_bc_sn_grad_face_flux
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_cfd_cast_f32_to_f64[] = {
  { "src", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "dst", 1, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "counts", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 12, 12, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_cfd_cast_f32_to_f64 = {
  1,
  "navatala_cfd_cast_f32_to_f64",
  "opencl",
  "navatala_cfd_cast_f32_to_f64",
  "kernel:opencl:navatala_cfd_cast_f32_to_f64",
  "abi-r1:opencl:navatala_cfd_cast_f32_to_f64",
  "abi-r1:opencl:navatala_cfd_cast_f32_to_f64",
  3,
  kAbiArgs_opencl_navatala_cfd_cast_f32_to_f64
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_cfd_cast_f64_to_f32[] = {
  { "src", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "dst", 1, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "counts", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 12, 12, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_cfd_cast_f64_to_f32 = {
  1,
  "navatala_cfd_cast_f64_to_f32",
  "opencl",
  "navatala_cfd_cast_f64_to_f32",
  "kernel:opencl:navatala_cfd_cast_f64_to_f32",
  "abi-r1:opencl:navatala_cfd_cast_f64_to_f32",
  "abi-r1:opencl:navatala_cfd_cast_f64_to_f32",
  3,
  kAbiArgs_opencl_navatala_cfd_cast_f64_to_f32
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_cfd_coeff_to_cf_in_place[] = {
  { "cf", 0, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "magSf", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "delta", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "counts", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 12, 12, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_cfd_coeff_to_cf_in_place = {
  1,
  "navatala_cfd_coeff_to_cf_in_place",
  "opencl",
  "navatala_cfd_coeff_to_cf_in_place",
  "kernel:opencl:navatala_cfd_coeff_to_cf_in_place",
  "abi-r1:opencl:navatala_cfd_coeff_to_cf_in_place",
  "abi-r1:opencl:navatala_cfd_coeff_to_cf_in_place",
  4,
  kAbiArgs_opencl_navatala_cfd_coeff_to_cf_in_place
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_cfd_diag_from_cf[] = {
  { "cf", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "offsets", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "faceIdx", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "vol", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "counts", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 12, 12, 256, nullptr, 0, 0 },
  { "diag", 5, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_cfd_diag_from_cf = {
  1,
  "navatala_cfd_diag_from_cf",
  "opencl",
  "navatala_cfd_diag_from_cf",
  "kernel:opencl:navatala_cfd_diag_from_cf",
  "abi-r1:opencl:navatala_cfd_diag_from_cf",
  "abi-r1:opencl:navatala_cfd_diag_from_cf",
  6,
  kAbiArgs_opencl_navatala_cfd_diag_from_cf
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_cfd_dic_apply[] = {
  { "r", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "rD", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "upper", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "owner", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "neighbour", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "counts", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 12, 12, 256, nullptr, 0, 0 },
  { "z", 6, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_cfd_dic_apply = {
  1,
  "navatala_cfd_dic_apply",
  "opencl",
  "navatala_cfd_dic_apply",
  "kernel:opencl:navatala_cfd_dic_apply",
  "abi-r1:opencl:navatala_cfd_dic_apply",
  "abi-r1:opencl:navatala_cfd_dic_apply",
  7,
  kAbiArgs_opencl_navatala_cfd_dic_apply
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_cfd_dic_build_reciprocal_d[] = {
  { "diag", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "upper", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "owner", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "neighbour", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "counts", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 12, 12, 256, nullptr, 0, 0 },
  { "rD", 5, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_cfd_dic_build_reciprocal_d = {
  1,
  "navatala_cfd_dic_build_reciprocal_d",
  "opencl",
  "navatala_cfd_dic_build_reciprocal_d",
  "kernel:opencl:navatala_cfd_dic_build_reciprocal_d",
  "abi-r1:opencl:navatala_cfd_dic_build_reciprocal_d",
  "abi-r1:opencl:navatala_cfd_dic_build_reciprocal_d",
  6,
  kAbiArgs_opencl_navatala_cfd_dic_build_reciprocal_d
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_cfd_dot_partials[] = {
  { "a", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "b", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "counts", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 12, 12, 256, nullptr, 0, 0 },
  { "outPartials", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_cfd_dot_partials = {
  1,
  "navatala_cfd_dot_partials",
  "opencl",
  "navatala_cfd_dot_partials",
  "kernel:opencl:navatala_cfd_dot_partials",
  "abi-r1:opencl:navatala_cfd_dot_partials",
  "abi-r1:opencl:navatala_cfd_dot_partials",
  4,
  kAbiArgs_opencl_navatala_cfd_dot_partials
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_cfd_face_flux[] = {
  { "x", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "owner", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "neighbour", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "cf", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "bcVal", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "bcMask", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "counts", 6, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 12, 12, 256, nullptr, 0, 0 },
  { "outFlux", 7, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_cfd_face_flux = {
  1,
  "navatala_cfd_face_flux",
  "opencl",
  "navatala_cfd_face_flux",
  "kernel:opencl:navatala_cfd_face_flux",
  "abi-r1:opencl:navatala_cfd_face_flux",
  "abi-r1:opencl:navatala_cfd_face_flux",
  8,
  kAbiArgs_opencl_navatala_cfd_face_flux
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_cfd_inv_diag[] = {
  { "r", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "diag", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "counts", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 12, 12, 256, nullptr, 0, 0 },
  { "z", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_cfd_inv_diag = {
  1,
  "navatala_cfd_inv_diag",
  "opencl",
  "navatala_cfd_inv_diag",
  "kernel:opencl:navatala_cfd_inv_diag",
  "abi-r1:opencl:navatala_cfd_inv_diag",
  "abi-r1:opencl:navatala_cfd_inv_diag",
  4,
  kAbiArgs_opencl_navatala_cfd_inv_diag
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_cfd_laplacian_from_face_flux[] = {
  { "faceFlux", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "offsets", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "faceIdx", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "signF", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "vol", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "counts", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 12, 12, 256, nullptr, 0, 0 },
  { "outLap", 6, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_cfd_laplacian_from_face_flux = {
  1,
  "navatala_cfd_laplacian_from_face_flux",
  "opencl",
  "navatala_cfd_laplacian_from_face_flux",
  "kernel:opencl:navatala_cfd_laplacian_from_face_flux",
  "abi-r1:opencl:navatala_cfd_laplacian_from_face_flux",
  "abi-r1:opencl:navatala_cfd_laplacian_from_face_flux",
  7,
  kAbiArgs_opencl_navatala_cfd_laplacian_from_face_flux
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_cfd_mul_by_vol_in_place[] = {
  { "b", 0, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "vol", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "counts", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 12, 12, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_cfd_mul_by_vol_in_place = {
  1,
  "navatala_cfd_mul_by_vol_in_place",
  "opencl",
  "navatala_cfd_mul_by_vol_in_place",
  "kernel:opencl:navatala_cfd_mul_by_vol_in_place",
  "abi-r1:opencl:navatala_cfd_mul_by_vol_in_place",
  "abi-r1:opencl:navatala_cfd_mul_by_vol_in_place",
  3,
  kAbiArgs_opencl_navatala_cfd_mul_by_vol_in_place
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_cfd_ref_add_ax[] = {
  { "diag", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "x", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "outAx", 2, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "refCell", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_cfd_ref_add_ax = {
  1,
  "navatala_cfd_ref_add_ax",
  "opencl",
  "navatala_cfd_ref_add_ax",
  "kernel:opencl:navatala_cfd_ref_add_ax",
  "abi-r1:opencl:navatala_cfd_ref_add_ax",
  "abi-r1:opencl:navatala_cfd_ref_add_ax",
  4,
  kAbiArgs_opencl_navatala_cfd_ref_add_ax
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_cfd_ref_add_b[] = {
  { "diag", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "b", 1, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "refCell", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "refVal", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_cfd_ref_add_b = {
  1,
  "navatala_cfd_ref_add_b",
  "opencl",
  "navatala_cfd_ref_add_b",
  "kernel:opencl:navatala_cfd_ref_add_b",
  "abi-r1:opencl:navatala_cfd_ref_add_b",
  "abi-r1:opencl:navatala_cfd_ref_add_b",
  4,
  kAbiArgs_opencl_navatala_cfd_ref_add_b
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_cfd_rhs_sub[] = {
  { "b", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "ax", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "counts", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 12, 12, 256, nullptr, 0, 0 },
  { "r", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_cfd_rhs_sub = {
  1,
  "navatala_cfd_rhs_sub",
  "opencl",
  "navatala_cfd_rhs_sub",
  "kernel:opencl:navatala_cfd_rhs_sub",
  "abi-r1:opencl:navatala_cfd_rhs_sub",
  "abi-r1:opencl:navatala_cfd_rhs_sub",
  4,
  kAbiArgs_opencl_navatala_cfd_rhs_sub
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_cfd_shift_in_place[] = {
  { "x", 0, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "counts", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 12, 12, 256, nullptr, 0, 0 },
  { "paramsF", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_cfd_shift_in_place = {
  1,
  "navatala_cfd_shift_in_place",
  "opencl",
  "navatala_cfd_shift_in_place",
  "kernel:opencl:navatala_cfd_shift_in_place",
  "abi-r1:opencl:navatala_cfd_shift_in_place",
  "abi-r1:opencl:navatala_cfd_shift_in_place",
  3,
  kAbiArgs_opencl_navatala_cfd_shift_in_place
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_cfd_sum_abs_partials[] = {
  { "a", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "counts", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 12, 12, 256, nullptr, 0, 0 },
  { "outPartials", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_cfd_sum_abs_partials = {
  1,
  "navatala_cfd_sum_abs_partials",
  "opencl",
  "navatala_cfd_sum_abs_partials",
  "kernel:opencl:navatala_cfd_sum_abs_partials",
  "abi-r1:opencl:navatala_cfd_sum_abs_partials",
  "abi-r1:opencl:navatala_cfd_sum_abs_partials",
  3,
  kAbiArgs_opencl_navatala_cfd_sum_abs_partials
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_cfd_upper_from_cf[] = {
  { "cf", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "counts", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 12, 12, 256, nullptr, 0, 0 },
  { "upper", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_cfd_upper_from_cf = {
  1,
  "navatala_cfd_upper_from_cf",
  "opencl",
  "navatala_cfd_upper_from_cf",
  "kernel:opencl:navatala_cfd_upper_from_cf",
  "abi-r1:opencl:navatala_cfd_upper_from_cf",
  "abi-r1:opencl:navatala_cfd_upper_from_cf",
  3,
  kAbiArgs_opencl_navatala_cfd_upper_from_cf
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_cfd_xpay_in_place[] = {
  { "z", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "pIn", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "pOut", 2, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "counts", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 12, 12, 256, nullptr, 0, 0 },
  { "paramsF", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_cfd_xpay_in_place = {
  1,
  "navatala_cfd_xpay_in_place",
  "opencl",
  "navatala_cfd_xpay_in_place",
  "kernel:opencl:navatala_cfd_xpay_in_place",
  "abi-r1:opencl:navatala_cfd_xpay_in_place",
  "abi-r1:opencl:navatala_cfd_xpay_in_place",
  5,
  kAbiArgs_opencl_navatala_cfd_xpay_in_place
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_cfd_dot_finalize_partials[] = {
  { "partials", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "result", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 8, 8, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_cfd_dot_finalize_partials = {
  1,
  "navatala_cfd_dot_finalize_partials",
  "opencl",
  "navatala_cfd_dot_finalize_partials",
  "kernel:opencl:navatala_cfd_dot_finalize_partials",
  "abi-r1:opencl:navatala_cfd_dot_finalize_partials",
  "abi-r1:opencl:navatala_cfd_dot_finalize_partials",
  3,
  kAbiArgs_opencl_navatala_cfd_dot_finalize_partials
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_cfd_negate_scalar[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 1, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_cfd_negate_scalar = {
  1,
  "navatala_cfd_negate_scalar",
  "opencl",
  "navatala_cfd_negate_scalar",
  "kernel:opencl:navatala_cfd_negate_scalar",
  "abi-r1:opencl:navatala_cfd_negate_scalar",
  "abi-r1:opencl:navatala_cfd_negate_scalar",
  2,
  kAbiArgs_opencl_navatala_cfd_negate_scalar
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_cfd_pcg_compute_alpha[] = {
  { "rz", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 8, 8, 256, nullptr, 0, 0 },
  { "pAp", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 8, 8, 256, nullptr, 0, 0 },
  { "alpha", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_cfd_pcg_compute_alpha = {
  1,
  "navatala_cfd_pcg_compute_alpha",
  "opencl",
  "navatala_cfd_pcg_compute_alpha",
  "kernel:opencl:navatala_cfd_pcg_compute_alpha",
  "abi-r1:opencl:navatala_cfd_pcg_compute_alpha",
  "abi-r1:opencl:navatala_cfd_pcg_compute_alpha",
  3,
  kAbiArgs_opencl_navatala_cfd_pcg_compute_alpha
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_cfd_pcg_compute_beta[] = {
  { "rzNew", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 8, 8, 256, nullptr, 0, 0 },
  { "rzOld", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 8, 8, 256, nullptr, 0, 0 },
  { "beta", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_cfd_pcg_compute_beta = {
  1,
  "navatala_cfd_pcg_compute_beta",
  "opencl",
  "navatala_cfd_pcg_compute_beta",
  "kernel:opencl:navatala_cfd_pcg_compute_beta",
  "abi-r1:opencl:navatala_cfd_pcg_compute_beta",
  "abi-r1:opencl:navatala_cfd_pcg_compute_beta",
  3,
  kAbiArgs_opencl_navatala_cfd_pcg_compute_beta
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_cfd_exact_mat_vec[] = {
  { "x", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "owner", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "neighbour", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "offsets", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "faceIdx", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "sign", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "upper", 6, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "lower", 7, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "diag", 8, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "counts", 9, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "outAx", 10, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_cfd_exact_mat_vec = {
  1,
  "navatala_cfd_exact_mat_vec",
  "opencl",
  "navatala_cfd_exact_mat_vec",
  "kernel:opencl:navatala_cfd_exact_mat_vec",
  "abi-r1:opencl:navatala_cfd_exact_mat_vec",
  "abi-r1:opencl:navatala_cfd_exact_mat_vec",
  11,
  kAbiArgs_opencl_navatala_cfd_exact_mat_vec
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_cfd_exact_mat_vec_f64[] = {
  { "x", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "owner", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "neighbour", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "offsets", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "faceIdx", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "sign", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "upper", 6, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "lower", 7, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "diag", 8, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "counts", 9, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "outAx", 10, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_cfd_exact_mat_vec_f64 = {
  1,
  "navatala_cfd_exact_mat_vec_f64",
  "opencl",
  "navatala_cfd_exact_mat_vec_f64",
  "kernel:opencl:navatala_cfd_exact_mat_vec_f64",
  "abi-r1:opencl:navatala_cfd_exact_mat_vec_f64",
  "abi-r1:opencl:navatala_cfd_exact_mat_vec_f64",
  11,
  kAbiArgs_opencl_navatala_cfd_exact_mat_vec_f64
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_cfd_face_flux_boundary[] = {
  { "x", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "owner", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "cf", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "bcVal", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "bcMask", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "params", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "faceFlux", 6, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_cfd_face_flux_boundary = {
  1,
  "navatala_cfd_face_flux_boundary",
  "opencl",
  "navatala_cfd_face_flux_boundary",
  "kernel:opencl:navatala_cfd_face_flux_boundary",
  "abi-r1:opencl:navatala_cfd_face_flux_boundary",
  "abi-r1:opencl:navatala_cfd_face_flux_boundary",
  7,
  kAbiArgs_opencl_navatala_cfd_face_flux_boundary
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_cfd_face_flux_internal[] = {
  { "x", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "owner", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "neighbour", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "cf", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "params", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "faceFlux", 5, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_cfd_face_flux_internal = {
  1,
  "navatala_cfd_face_flux_internal",
  "opencl",
  "navatala_cfd_face_flux_internal",
  "kernel:opencl:navatala_cfd_face_flux_internal",
  "abi-r1:opencl:navatala_cfd_face_flux_internal",
  "abi-r1:opencl:navatala_cfd_face_flux_internal",
  6,
  kAbiArgs_opencl_navatala_cfd_face_flux_internal
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_cfd_pressure_face_flux_correction_corrected[] = {
  { "owner", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "neighbour", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "weights", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "gradX", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "gradY", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "gradZ", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "magSf", 6, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "dCorrX", 7, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "dCorrY", 8, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "dCorrZ", 9, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "rAUf", 10, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "params", 11, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "outFaceFlux", 12, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_cfd_pressure_face_flux_correction_corrected = {
  1,
  "navatala_cfd_pressure_face_flux_correction_corrected",
  "opencl",
  "navatala_cfd_pressure_face_flux_correction_corrected",
  "kernel:opencl:navatala_cfd_pressure_face_flux_correction_corrected",
  "abi-r1:opencl:navatala_cfd_pressure_face_flux_correction_corrected",
  "abi-r1:opencl:navatala_cfd_pressure_face_flux_correction_corrected",
  13,
  kAbiArgs_opencl_navatala_cfd_pressure_face_flux_correction_corrected
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_cfd_pressure_sn_grad[] = {
  { "phiHbyA", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "ownerAll", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "sfX", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "sfY", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "sfZ", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "magSf", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "ux", 6, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "uy", 7, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "uz", 8, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "uBcX", 9, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "uBcY", 10, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "uBcZ", 11, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "uBcMask", 12, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "rAUf", 13, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "snGradMask", 14, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "params", 15, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "outSnGrad", 16, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_cfd_pressure_sn_grad = {
  1,
  "navatala_cfd_pressure_sn_grad",
  "opencl",
  "navatala_cfd_pressure_sn_grad",
  "kernel:opencl:navatala_cfd_pressure_sn_grad",
  "abi-r1:opencl:navatala_cfd_pressure_sn_grad",
  "abi-r1:opencl:navatala_cfd_pressure_sn_grad",
  17,
  kAbiArgs_opencl_navatala_cfd_pressure_sn_grad
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_cfd_u_correct[] = {
  { "pCell", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "owner", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "neighbour", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "sfX", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "sfY", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "sfZ", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "bcVal", 6, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "bcMask", 7, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "offsets", 8, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "faceIdx", 9, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "sign", 10, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "vol", 11, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "rAU", 12, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "ux", 13, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "uy", 14, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "uz", 15, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "counts", 16, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 12, 12, 256, nullptr, 0, 0 },
  { "paramsF", 17, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "outX", 18, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "outY", 19, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "outZ", 20, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_cfd_u_correct = {
  1,
  "navatala_cfd_u_correct",
  "opencl",
  "navatala_cfd_u_correct",
  "kernel:opencl:navatala_cfd_u_correct",
  "abi-r1:opencl:navatala_cfd_u_correct",
  "abi-r1:opencl:navatala_cfd_u_correct",
  21,
  kAbiArgs_opencl_navatala_cfd_u_correct
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_cfd_u_correct_reconstruct[] = {
  { "pCell", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "rAU", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "hbx", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "hby", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "hbz", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "rAUf", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "phig", 6, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "owner", 7, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "neighbour", 8, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "sfX", 9, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "sfY", 10, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "sfZ", 11, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "magSf", 12, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "deltaCoeffs", 13, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "offsets", 14, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "faceIdx", 15, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "sign", 16, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "vol", 17, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "bcVal", 18, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "bcMask", 19, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "bcSnGrad", 20, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "bcSnGradMask", 21, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "faceFluxCorrection", 22, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "counts", 23, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 12, 12, 256, nullptr, 0, 0 },
  { "paramsF", 24, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "outX", 25, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "outY", 26, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "outZ", 27, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_cfd_u_correct_reconstruct = {
  1,
  "navatala_cfd_u_correct_reconstruct",
  "opencl",
  "navatala_cfd_u_correct_reconstruct",
  "kernel:opencl:navatala_cfd_u_correct_reconstruct",
  "abi-r1:opencl:navatala_cfd_u_correct_reconstruct",
  "abi-r1:opencl:navatala_cfd_u_correct_reconstruct",
  28,
  kAbiArgs_opencl_navatala_cfd_u_correct_reconstruct
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_cfd_grad_alpha_cell[] = {
  { "alphaF", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "sfX", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "sfY", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "sfZ", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "offsets", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "faceIdx", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "sign", 6, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "vol", 7, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "counts", 8, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "gx", 9, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "gy", 10, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "gz", 11, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_cfd_grad_alpha_cell = {
  1,
  "navatala_cfd_grad_alpha_cell",
  "opencl",
  "navatala_cfd_grad_alpha_cell",
  "kernel:opencl:navatala_cfd_grad_alpha_cell",
  "abi-r1:opencl:navatala_cfd_grad_alpha_cell",
  "abi-r1:opencl:navatala_cfd_grad_alpha_cell",
  12,
  kAbiArgs_opencl_navatala_cfd_grad_alpha_cell
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_cfd_grad_alpha_cell_f64[] = {
  { "alphaF", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "sfX", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "sfY", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "sfZ", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "offsets", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "faceIdx", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "sign", 6, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "vol", 7, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "counts", 8, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "gx", 9, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "gy", 10, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "gz", 11, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_cfd_grad_alpha_cell_f64 = {
  1,
  "navatala_cfd_grad_alpha_cell_f64",
  "opencl",
  "navatala_cfd_grad_alpha_cell_f64",
  "kernel:opencl:navatala_cfd_grad_alpha_cell_f64",
  "abi-r1:opencl:navatala_cfd_grad_alpha_cell_f64",
  "abi-r1:opencl:navatala_cfd_grad_alpha_cell_f64",
  12,
  kAbiArgs_opencl_navatala_cfd_grad_alpha_cell_f64
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_cfd_n_hatf_all[] = {
  { "gx", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "gy", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "gz", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "sfX", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "sfY", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "sfZ", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "owner", 6, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "nei", 7, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "weights", 8, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "theta", 9, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "thetaMask", 10, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "paramsF", 11, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "counts", 12, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 12, 12, 256, nullptr, 0, 0 },
  { "contactGrad", 13, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "nHatf", 14, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_cfd_n_hatf_all = {
  1,
  "navatala_cfd_n_hatf_all",
  "opencl",
  "navatala_cfd_n_hatf_all",
  "kernel:opencl:navatala_cfd_n_hatf_all",
  "abi-r1:opencl:navatala_cfd_n_hatf_all",
  "abi-r1:opencl:navatala_cfd_n_hatf_all",
  15,
  kAbiArgs_opencl_navatala_cfd_n_hatf_all
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_cfd_n_hatf_all_f64[] = {
  { "gx", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "gy", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "gz", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "sfX", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "sfY", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "sfZ", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "owner", 6, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "nei", 7, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "weights", 8, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "theta", 9, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "thetaMask", 10, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "paramsF", 11, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 8, 8, 256, nullptr, 0, 0 },
  { "counts", 12, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 12, 12, 256, nullptr, 0, 0 },
  { "contactGrad", 13, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "nHatf", 14, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_cfd_n_hatf_all_f64 = {
  1,
  "navatala_cfd_n_hatf_all_f64",
  "opencl",
  "navatala_cfd_n_hatf_all_f64",
  "kernel:opencl:navatala_cfd_n_hatf_all_f64",
  "abi-r1:opencl:navatala_cfd_n_hatf_all_f64",
  "abi-r1:opencl:navatala_cfd_n_hatf_all_f64",
  15,
  kAbiArgs_opencl_navatala_cfd_n_hatf_all_f64
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_cfd_stf_all_faces[] = {
  { "divN", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "alpha", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "alphaF", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "delta", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "owner", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "nei", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "weights", 6, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "contactGrad", 7, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "thetaMask", 8, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "paramsF", 9, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "counts", 10, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 12, 12, 256, nullptr, 0, 0 },
  { "stf", 11, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_cfd_stf_all_faces = {
  1,
  "navatala_cfd_stf_all_faces",
  "opencl",
  "navatala_cfd_stf_all_faces",
  "kernel:opencl:navatala_cfd_stf_all_faces",
  "abi-r1:opencl:navatala_cfd_stf_all_faces",
  "abi-r1:opencl:navatala_cfd_stf_all_faces",
  12,
  kAbiArgs_opencl_navatala_cfd_stf_all_faces
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_cfd_stf_all_faces_f64[] = {
  { "divN", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "alpha", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "alphaF", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "delta", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "owner", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "nei", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "weights", 6, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "contactGrad", 7, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "thetaMask", 8, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "paramsF", 9, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "counts", 10, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 12, 12, 256, nullptr, 0, 0 },
  { "stf", 11, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_cfd_stf_all_faces_f64 = {
  1,
  "navatala_cfd_stf_all_faces_f64",
  "opencl",
  "navatala_cfd_stf_all_faces_f64",
  "kernel:opencl:navatala_cfd_stf_all_faces_f64",
  "abi-r1:opencl:navatala_cfd_stf_all_faces_f64",
  "abi-r1:opencl:navatala_cfd_stf_all_faces_f64",
  12,
  kAbiArgs_opencl_navatala_cfd_stf_all_faces_f64
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_cfd_k_omega_s_s_t_blending[] = {
  { "k", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "omega", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "nut", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "nu", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "y", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "gradKx", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "gradKy", 6, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "gradKz", 7, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "gradOmegax", 8, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "gradOmegay", 9, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "gradOmegaz", 10, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "counts", 11, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "outF1", 12, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "outF23", 13, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "outCDkOmega", 14, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_cfd_k_omega_s_s_t_blending = {
  1,
  "navatala_cfd_k_omega_s_s_t_blending",
  "opencl",
  "navatala_cfd_k_omega_s_s_t_blending",
  "kernel:opencl:navatala_cfd_k_omega_s_s_t_blending",
  "abi-r1:opencl:navatala_cfd_k_omega_s_s_t_blending",
  "abi-r1:opencl:navatala_cfd_k_omega_s_s_t_blending",
  15,
  kAbiArgs_opencl_navatala_cfd_k_omega_s_s_t_blending
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_cfd_k_omega_s_s_t_nut_compute[] = {
  { "kVals", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "omegaVals", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "f23Vals", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "s2Vals", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "counts", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "params", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "outNut", 6, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_cfd_k_omega_s_s_t_nut_compute = {
  1,
  "navatala_cfd_k_omega_s_s_t_nut_compute",
  "opencl",
  "navatala_cfd_k_omega_s_s_t_nut_compute",
  "kernel:opencl:navatala_cfd_k_omega_s_s_t_nut_compute",
  "abi-r1:opencl:navatala_cfd_k_omega_s_s_t_nut_compute",
  "abi-r1:opencl:navatala_cfd_k_omega_s_s_t_nut_compute",
  7,
  kAbiArgs_opencl_navatala_cfd_k_omega_s_s_t_nut_compute
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_cfd_k_omega_s_s_t_sources[] = {
  { "k", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "omega", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "nut", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "nu", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "alpha", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "rho", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "F1", 6, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "F23", 7, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "CDkOmega", 8, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "S2", 9, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "divU", 10, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "counts", 11, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "outOmegaSp", 12, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "outOmegaSuSp1", 13, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "outOmegaSuSp2", 14, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "outOmegaExpl", 15, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "outKSp", 16, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "outKSuSp", 17, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "outKExpl", 18, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "outDomegaEff", 19, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "outDkEff", 20, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_cfd_k_omega_s_s_t_sources = {
  1,
  "navatala_cfd_k_omega_s_s_t_sources",
  "opencl",
  "navatala_cfd_k_omega_s_s_t_sources",
  "kernel:opencl:navatala_cfd_k_omega_s_s_t_sources",
  "abi-r1:opencl:navatala_cfd_k_omega_s_s_t_sources",
  "abi-r1:opencl:navatala_cfd_k_omega_s_s_t_sources",
  21,
  kAbiArgs_opencl_navatala_cfd_k_omega_s_s_t_sources
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_cfd_mu_eff_from_rho_nu_eff[] = {
  { "rho", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "nuEff", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "params", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "outMuEff", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_cfd_mu_eff_from_rho_nu_eff = {
  1,
  "navatala_cfd_mu_eff_from_rho_nu_eff",
  "opencl",
  "navatala_cfd_mu_eff_from_rho_nu_eff",
  "kernel:opencl:navatala_cfd_mu_eff_from_rho_nu_eff",
  "abi-r1:opencl:navatala_cfd_mu_eff_from_rho_nu_eff",
  "abi-r1:opencl:navatala_cfd_mu_eff_from_rho_nu_eff",
  4,
  kAbiArgs_opencl_navatala_cfd_mu_eff_from_rho_nu_eff
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_cfd_mu_eff_from_rho_nut_and_nu_mix[] = {
  { "rho", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "nut", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "nuMix", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "params", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "outMu", 4, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_cfd_mu_eff_from_rho_nut_and_nu_mix = {
  1,
  "navatala_cfd_mu_eff_from_rho_nut_and_nu_mix",
  "opencl",
  "navatala_cfd_mu_eff_from_rho_nut_and_nu_mix",
  "kernel:opencl:navatala_cfd_mu_eff_from_rho_nut_and_nu_mix",
  "abi-r1:opencl:navatala_cfd_mu_eff_from_rho_nut_and_nu_mix",
  "abi-r1:opencl:navatala_cfd_mu_eff_from_rho_nut_and_nu_mix",
  5,
  kAbiArgs_opencl_navatala_cfd_mu_eff_from_rho_nut_and_nu_mix
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_cfd_turbulence_add_explicit_source_device[] = {
  { "explicitSrc", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "vol", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "counts", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "outVals", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_cfd_turbulence_add_explicit_source_device = {
  1,
  "navatala_cfd_turbulence_add_explicit_source_device",
  "opencl",
  "navatala_cfd_turbulence_add_explicit_source_device",
  "kernel:opencl:navatala_cfd_turbulence_add_explicit_source_device",
  "abi-r1:opencl:navatala_cfd_turbulence_add_explicit_source_device",
  "abi-r1:opencl:navatala_cfd_turbulence_add_explicit_source_device",
  4,
  kAbiArgs_opencl_navatala_cfd_turbulence_add_explicit_source_device
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_cfd_turbulence_dirichlet_cell_source[] = {
  { "constrainedMask", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "constrainedValue", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "diag", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "nCells", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "source", 4, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_cfd_turbulence_dirichlet_cell_source = {
  1,
  "navatala_cfd_turbulence_dirichlet_cell_source",
  "opencl",
  "navatala_cfd_turbulence_dirichlet_cell_source",
  "kernel:opencl:navatala_cfd_turbulence_dirichlet_cell_source",
  "abi-r1:opencl:navatala_cfd_turbulence_dirichlet_cell_source",
  "abi-r1:opencl:navatala_cfd_turbulence_dirichlet_cell_source",
  5,
  kAbiArgs_opencl_navatala_cfd_turbulence_dirichlet_cell_source
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_cfd_turbulence_dirichlet_face_elimination[] = {
  { "owner", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "neighbour", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "constrainedMask", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "constrainedValue", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "nIntFaces", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "upper", 5, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "lower", 6, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "source", 7, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_cfd_turbulence_dirichlet_face_elimination = {
  1,
  "navatala_cfd_turbulence_dirichlet_face_elimination",
  "opencl",
  "navatala_cfd_turbulence_dirichlet_face_elimination",
  "kernel:opencl:navatala_cfd_turbulence_dirichlet_face_elimination",
  "abi-r1:opencl:navatala_cfd_turbulence_dirichlet_face_elimination",
  "abi-r1:opencl:navatala_cfd_turbulence_dirichlet_face_elimination",
  8,
  kAbiArgs_opencl_navatala_cfd_turbulence_dirichlet_face_elimination
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_cfd_two_mag_sqr_symm_grad_u[] = {
  { "ufX", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "ufY", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "ufZ", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "sfX", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "sfY", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "sfZ", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "offsets", 6, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "faceIdx", 7, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "sign", 8, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "vol", 9, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "counts", 10, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "outS2", 11, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_cfd_two_mag_sqr_symm_grad_u = {
  1,
  "navatala_cfd_two_mag_sqr_symm_grad_u",
  "opencl",
  "navatala_cfd_two_mag_sqr_symm_grad_u",
  "kernel:opencl:navatala_cfd_two_mag_sqr_symm_grad_u",
  "abi-r1:opencl:navatala_cfd_two_mag_sqr_symm_grad_u",
  "abi-r1:opencl:navatala_cfd_two_mag_sqr_symm_grad_u",
  12,
  kAbiArgs_opencl_navatala_cfd_two_mag_sqr_symm_grad_u
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_cfd_attention_row_softmax[] = {
  { "scores", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "numQ", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "numK", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "weights", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_cfd_attention_row_softmax = {
  1,
  "navatala_cfd_attention_row_softmax",
  "opencl",
  "navatala_cfd_attention_row_softmax",
  "kernel:opencl:navatala_cfd_attention_row_softmax",
  "abi-r1:opencl:navatala_cfd_attention_row_softmax",
  "abi-r1:opencl:navatala_cfd_attention_row_softmax",
  4,
  kAbiArgs_opencl_navatala_cfd_attention_row_softmax
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_cfd_attention_value_projection[] = {
  { "weights", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "v", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "numQ", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "numK", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "headDim", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "out", 5, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_cfd_attention_value_projection = {
  1,
  "navatala_cfd_attention_value_projection",
  "opencl",
  "navatala_cfd_attention_value_projection",
  "kernel:opencl:navatala_cfd_attention_value_projection",
  "abi-r1:opencl:navatala_cfd_attention_value_projection",
  "abi-r1:opencl:navatala_cfd_attention_value_projection",
  6,
  kAbiArgs_opencl_navatala_cfd_attention_value_projection
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_cfd_gather_mean[] = {
  { "features", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "neighbors", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "neighborCounts", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "numNodes", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "featureDim", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "out", 5, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_cfd_gather_mean = {
  1,
  "navatala_cfd_gather_mean",
  "opencl",
  "navatala_cfd_gather_mean",
  "kernel:opencl:navatala_cfd_gather_mean",
  "abi-r1:opencl:navatala_cfd_gather_mean",
  "abi-r1:opencl:navatala_cfd_gather_mean",
  6,
  kAbiArgs_opencl_navatala_cfd_gather_mean
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_cfd_layer_norm_forward[] = {
  { "x", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "numRows", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "gamma", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "beta", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "y", 4, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_cfd_layer_norm_forward = {
  1,
  "navatala_cfd_layer_norm_forward",
  "opencl",
  "navatala_cfd_layer_norm_forward",
  "kernel:opencl:navatala_cfd_layer_norm_forward",
  "abi-r1:opencl:navatala_cfd_layer_norm_forward",
  "abi-r1:opencl:navatala_cfd_layer_norm_forward",
  5,
  kAbiArgs_opencl_navatala_cfd_layer_norm_forward
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_cfd_radius_count[] = {
  { "query", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "ref", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "numQuery", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "numRef", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "radiusIn", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "counts", 5, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_cfd_radius_count = {
  1,
  "navatala_cfd_radius_count",
  "opencl",
  "navatala_cfd_radius_count",
  "kernel:opencl:navatala_cfd_radius_count",
  "abi-r1:opencl:navatala_cfd_radius_count",
  "abi-r1:opencl:navatala_cfd_radius_count",
  6,
  kAbiArgs_opencl_navatala_cfd_radius_count
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_cfd_scaled_dot_product[] = {
  { "q", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "k", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "numQ", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "numK", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "headDim", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "scores", 5, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_cfd_scaled_dot_product = {
  1,
  "navatala_cfd_scaled_dot_product",
  "opencl",
  "navatala_cfd_scaled_dot_product",
  "kernel:opencl:navatala_cfd_scaled_dot_product",
  "abi-r1:opencl:navatala_cfd_scaled_dot_product",
  "abi-r1:opencl:navatala_cfd_scaled_dot_product",
  6,
  kAbiArgs_opencl_navatala_cfd_scaled_dot_product
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_cfd_scatter_add_atomic[] = {
  { "values", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "indices", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "numEdges", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "featureDim", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "dst", 4, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_cfd_scatter_add_atomic = {
  1,
  "navatala_cfd_scatter_add_atomic",
  "opencl",
  "navatala_cfd_scatter_add_atomic",
  "kernel:opencl:navatala_cfd_scatter_add_atomic",
  "abi-r1:opencl:navatala_cfd_scatter_add_atomic",
  "abi-r1:opencl:navatala_cfd_scatter_add_atomic",
  5,
  kAbiArgs_opencl_navatala_cfd_scatter_add_atomic
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_cfd_spectral_complex_mul[] = {
  { "x", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "w", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "y", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_cfd_spectral_complex_mul = {
  1,
  "navatala_cfd_spectral_complex_mul",
  "opencl",
  "navatala_cfd_spectral_complex_mul",
  "kernel:opencl:navatala_cfd_spectral_complex_mul",
  "abi-r1:opencl:navatala_cfd_spectral_complex_mul",
  "abi-r1:opencl:navatala_cfd_spectral_complex_mul",
  4,
  kAbiArgs_opencl_navatala_cfd_spectral_complex_mul
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_cfd_spectral_conv1d_mode_sliced[] = {
  { "xFt", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "wPos", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "wNeg", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "numBatches", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "ftW", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "yFt", 5, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_cfd_spectral_conv1d_mode_sliced = {
  1,
  "navatala_cfd_spectral_conv1d_mode_sliced",
  "opencl",
  "navatala_cfd_spectral_conv1d_mode_sliced",
  "kernel:opencl:navatala_cfd_spectral_conv1d_mode_sliced",
  "abi-r1:opencl:navatala_cfd_spectral_conv1d_mode_sliced",
  "abi-r1:opencl:navatala_cfd_spectral_conv1d_mode_sliced",
  6,
  kAbiArgs_opencl_navatala_cfd_spectral_conv1d_mode_sliced
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_cfd_spectral_conv2d_mode_sliced[] = {
  { "xFt", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "wPos", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "wNeg", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "numBatches", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "ftH", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "ftW", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "yFt", 6, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_cfd_spectral_conv2d_mode_sliced = {
  1,
  "navatala_cfd_spectral_conv2d_mode_sliced",
  "opencl",
  "navatala_cfd_spectral_conv2d_mode_sliced",
  "kernel:opencl:navatala_cfd_spectral_conv2d_mode_sliced",
  "abi-r1:opencl:navatala_cfd_spectral_conv2d_mode_sliced",
  "abi-r1:opencl:navatala_cfd_spectral_conv2d_mode_sliced",
  7,
  kAbiArgs_opencl_navatala_cfd_spectral_conv2d_mode_sliced
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_cfd_spectral_conv3d_mode_sliced[] = {
  { "xFt", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "wPP", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "wNP", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "wPN", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "wNN", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "numBatches", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "ftD", 6, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "ftH", 7, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "ftW", 8, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "yFt", 9, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_cfd_spectral_conv3d_mode_sliced = {
  1,
  "navatala_cfd_spectral_conv3d_mode_sliced",
  "opencl",
  "navatala_cfd_spectral_conv3d_mode_sliced",
  "kernel:opencl:navatala_cfd_spectral_conv3d_mode_sliced",
  "abi-r1:opencl:navatala_cfd_spectral_conv3d_mode_sliced",
  "abi-r1:opencl:navatala_cfd_spectral_conv3d_mode_sliced",
  10,
  kAbiArgs_opencl_navatala_cfd_spectral_conv3d_mode_sliced
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_cfd_spectral_conv4d_mode_sliced[] = {
  { "xFt", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "wPPP", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "wNPP", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "wPNP", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "wNNP", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "wPPN", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "wNPN", 6, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "wPNN", 7, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "wNNN", 8, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "numBatches", 9, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "ftT", 10, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "ftD", 11, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "ftH", 12, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "ftW", 13, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "yFt", 14, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_cfd_spectral_conv4d_mode_sliced = {
  1,
  "navatala_cfd_spectral_conv4d_mode_sliced",
  "opencl",
  "navatala_cfd_spectral_conv4d_mode_sliced",
  "kernel:opencl:navatala_cfd_spectral_conv4d_mode_sliced",
  "abi-r1:opencl:navatala_cfd_spectral_conv4d_mode_sliced",
  "abi-r1:opencl:navatala_cfd_spectral_conv4d_mode_sliced",
  15,
  kAbiArgs_opencl_navatala_cfd_spectral_conv4d_mode_sliced
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_cfd_compute_face_area_pair_weights_f32[] = {
  { "sfX", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "sfY", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "sfZ", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "magSf", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "nFaces", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "faceWeight", 5, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_cfd_compute_face_area_pair_weights_f32 = {
  1,
  "navatala_cfd_compute_face_area_pair_weights_f32",
  "opencl",
  "navatala_cfd_compute_face_area_pair_weights_f32",
  "kernel:opencl:navatala_cfd_compute_face_area_pair_weights_f32",
  "abi-r1:opencl:navatala_cfd_compute_face_area_pair_weights_f32",
  "abi-r1:opencl:navatala_cfd_compute_face_area_pair_weights_f32",
  6,
  kAbiArgs_opencl_navatala_cfd_compute_face_area_pair_weights_f32
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_cfd_restrict_by_agglomeration_gather_f32[] = {
  { "fineField", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "coarseToFineOffsets", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "coarseToFineList", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "nCoarseCells", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "coarseField", 4, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_cfd_restrict_by_agglomeration_gather_f32 = {
  1,
  "navatala_cfd_restrict_by_agglomeration_gather_f32",
  "opencl",
  "navatala_cfd_restrict_by_agglomeration_gather_f32",
  "kernel:opencl:navatala_cfd_restrict_by_agglomeration_gather_f32",
  "abi-r1:opencl:navatala_cfd_restrict_by_agglomeration_gather_f32",
  "abi-r1:opencl:navatala_cfd_restrict_by_agglomeration_gather_f32",
  5,
  kAbiArgs_opencl_navatala_cfd_restrict_by_agglomeration_gather_f32
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_cfd_prolongate_by_agglomeration_f32[] = {
  { "coarseField", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "fineToCoarse", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "nFineCells", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "fineField", 3, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_cfd_prolongate_by_agglomeration_f32 = {
  1,
  "navatala_cfd_prolongate_by_agglomeration_f32",
  "opencl",
  "navatala_cfd_prolongate_by_agglomeration_f32",
  "kernel:opencl:navatala_cfd_prolongate_by_agglomeration_f32",
  "abi-r1:opencl:navatala_cfd_prolongate_by_agglomeration_f32",
  "abi-r1:opencl:navatala_cfd_prolongate_by_agglomeration_f32",
  4,
  kAbiArgs_opencl_navatala_cfd_prolongate_by_agglomeration_f32
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_cfd_diag_inv_apply_f32[] = {
  { "r", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "diagInv", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "z", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_cfd_diag_inv_apply_f32 = {
  1,
  "navatala_cfd_diag_inv_apply_f32",
  "opencl",
  "navatala_cfd_diag_inv_apply_f32",
  "kernel:opencl:navatala_cfd_diag_inv_apply_f32",
  "abi-r1:opencl:navatala_cfd_diag_inv_apply_f32",
  "abi-r1:opencl:navatala_cfd_diag_inv_apply_f32",
  4,
  kAbiArgs_opencl_navatala_cfd_diag_inv_apply_f32
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_cfd_csr_sp_m_v_mixed_f32_f64[] = {
  { "rowPtr", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "colIdx", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "values", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "x", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "nRows", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "y", 5, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_cfd_csr_sp_m_v_mixed_f32_f64 = {
  1,
  "navatala_cfd_csr_sp_m_v_mixed_f32_f64",
  "opencl",
  "navatala_cfd_csr_sp_m_v_mixed_f32_f64",
  "kernel:opencl:navatala_cfd_csr_sp_m_v_mixed_f32_f64",
  "abi-r1:opencl:navatala_cfd_csr_sp_m_v_mixed_f32_f64",
  "abi-r1:opencl:navatala_cfd_csr_sp_m_v_mixed_f32_f64",
  6,
  kAbiArgs_opencl_navatala_cfd_csr_sp_m_v_mixed_f32_f64
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_cfd_chebyshev_sweep_mixed_f32_f64[] = {
  { "r", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "d", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "coeffA", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 8, 8, 256, nullptr, 0, 0 },
  { "coeffB", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 8, 8, 256, nullptr, 0, 0 },
  { "n", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "x", 5, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "dNew", 6, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_cfd_chebyshev_sweep_mixed_f32_f64 = {
  1,
  "navatala_cfd_chebyshev_sweep_mixed_f32_f64",
  "opencl",
  "navatala_cfd_chebyshev_sweep_mixed_f32_f64",
  "kernel:opencl:navatala_cfd_chebyshev_sweep_mixed_f32_f64",
  "abi-r1:opencl:navatala_cfd_chebyshev_sweep_mixed_f32_f64",
  "abi-r1:opencl:navatala_cfd_chebyshev_sweep_mixed_f32_f64",
  7,
  kAbiArgs_opencl_navatala_cfd_chebyshev_sweep_mixed_f32_f64
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_cfd_ell_sp_m_v_mixed_f32_f64[] = {
  { "colIdx", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "values", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "x", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "nRows", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "maxWidth", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "y", 5, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_cfd_ell_sp_m_v_mixed_f32_f64 = {
  1,
  "navatala_cfd_ell_sp_m_v_mixed_f32_f64",
  "opencl",
  "navatala_cfd_ell_sp_m_v_mixed_f32_f64",
  "kernel:opencl:navatala_cfd_ell_sp_m_v_mixed_f32_f64",
  "abi-r1:opencl:navatala_cfd_ell_sp_m_v_mixed_f32_f64",
  "abi-r1:opencl:navatala_cfd_ell_sp_m_v_mixed_f32_f64",
  6,
  kAbiArgs_opencl_navatala_cfd_ell_sp_m_v_mixed_f32_f64
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_cfd_chebyshev_smooth_post_fused_mixed[] = {
  { "Az", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "r", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "diagInv", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "dPrev", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "coeffA", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 8, 8, 256, nullptr, 0, 0 },
  { "coeffB", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 8, 8, 256, nullptr, 0, 0 },
  { "n", 6, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "z", 7, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "dNew", 8, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_cfd_chebyshev_smooth_post_fused_mixed = {
  1,
  "navatala_cfd_chebyshev_smooth_post_fused_mixed",
  "opencl",
  "navatala_cfd_chebyshev_smooth_post_fused_mixed",
  "kernel:opencl:navatala_cfd_chebyshev_smooth_post_fused_mixed",
  "abi-r1:opencl:navatala_cfd_chebyshev_smooth_post_fused_mixed",
  "abi-r1:opencl:navatala_cfd_chebyshev_smooth_post_fused_mixed",
  9,
  kAbiArgs_opencl_navatala_cfd_chebyshev_smooth_post_fused_mixed
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_cfd_gamg_proc_pack_f32[] = {
  { "x", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "localCells", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "sendVals", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_cfd_gamg_proc_pack_f32 = {
  1,
  "navatala_cfd_gamg_proc_pack_f32",
  "opencl",
  "navatala_cfd_gamg_proc_pack_f32",
  "kernel:opencl:navatala_cfd_gamg_proc_pack_f32",
  "abi-r1:opencl:navatala_cfd_gamg_proc_pack_f32",
  "abi-r1:opencl:navatala_cfd_gamg_proc_pack_f32",
  4,
  kAbiArgs_opencl_navatala_cfd_gamg_proc_pack_f32
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_cfd_gamg_proc_add_gather_f32[] = {
  { "entryOffsets", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "entryIndices", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "coeffs", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "recvVals", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "nCells", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "y", 5, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_cfd_gamg_proc_add_gather_f32 = {
  1,
  "navatala_cfd_gamg_proc_add_gather_f32",
  "opencl",
  "navatala_cfd_gamg_proc_add_gather_f32",
  "kernel:opencl:navatala_cfd_gamg_proc_add_gather_f32",
  "abi-r1:opencl:navatala_cfd_gamg_proc_add_gather_f32",
  "abi-r1:opencl:navatala_cfd_gamg_proc_add_gather_f32",
  6,
  kAbiArgs_opencl_navatala_cfd_gamg_proc_add_gather_f32
};

bool tryGetKernelAbiManifest_opencl_cfd(const std::string& backend, const std::string& kernelName, const KernelAbiManifestInfo*& out) {
  if (backend == "opencl" && kernelName == "navatala_cfd_boundary_force_partials") {
    out = &kAbiManifest_opencl_navatala_cfd_boundary_force_partials;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_cfd_pack_owner_values") {
    out = &kAbiManifest_opencl_navatala_cfd_pack_owner_values;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_cfd_scatter_values_and_mask") {
    out = &kAbiManifest_opencl_navatala_cfd_scatter_values_and_mask;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_cfd_pack_mu_grad_owner") {
    out = &kAbiManifest_opencl_navatala_cfd_pack_mu_grad_owner;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_cfd_pack_owner_rho_vec3") {
    out = &kAbiManifest_opencl_navatala_cfd_pack_owner_rho_vec3;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_cfd_pack_owner_vec3") {
    out = &kAbiManifest_opencl_navatala_cfd_pack_owner_vec3;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_cfd_pack_owner_vec3_mu") {
    out = &kAbiManifest_opencl_navatala_cfd_pack_owner_vec3_mu;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_cfd_scatter_mu_grad_ghost") {
    out = &kAbiManifest_opencl_navatala_cfd_scatter_mu_grad_ghost;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_cfd_scatter_rho_vec3_and_mask") {
    out = &kAbiManifest_opencl_navatala_cfd_scatter_rho_vec3_and_mask;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_cfd_scatter_vec3_and_mask") {
    out = &kAbiManifest_opencl_navatala_cfd_scatter_vec3_and_mask;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_cfd_scatter_vec3_mu_and_mask") {
    out = &kAbiManifest_opencl_navatala_cfd_scatter_vec3_mu_and_mask;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_cfd_cf_mesh_volume_optimizer_gradient_f32") {
    out = &kAbiManifest_opencl_navatala_cfd_cf_mesh_volume_optimizer_gradient_f32;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_cfd_cf_mesh_volume_optimizer_gradient_f64") {
    out = &kAbiManifest_opencl_navatala_cfd_cf_mesh_volume_optimizer_gradient_f64;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_cfd_cf_mesh_volume_optimizer_k_prepass_f32") {
    out = &kAbiManifest_opencl_navatala_cfd_cf_mesh_volume_optimizer_k_prepass_f32;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_cfd_cf_mesh_volume_optimizer_k_prepass_f64") {
    out = &kAbiManifest_opencl_navatala_cfd_cf_mesh_volume_optimizer_k_prepass_f64;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_cfd_cf_mesh_volume_optimizer_objective_f32") {
    out = &kAbiManifest_opencl_navatala_cfd_cf_mesh_volume_optimizer_objective_f32;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_cfd_cf_mesh_volume_optimizer_objective_f64") {
    out = &kAbiManifest_opencl_navatala_cfd_cf_mesh_volume_optimizer_objective_f64;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_cfd_add_vol_vector") {
    out = &kAbiManifest_opencl_navatala_cfd_add_vol_vector;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_cfd_div_rho_phi_u_explicit") {
    out = &kAbiManifest_opencl_navatala_cfd_div_rho_phi_u_explicit;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_cfd_hby_a_from_u_and_terms") {
    out = &kAbiManifest_opencl_navatala_cfd_hby_a_from_u_and_terms;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_cfd_apply_sp_to_hby_a") {
    out = &kAbiManifest_opencl_navatala_cfd_apply_sp_to_hby_a;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_cfd_laplacian_scalar_explicit") {
    out = &kAbiManifest_opencl_navatala_cfd_laplacian_scalar_explicit;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_cfd_laplacian_vector_explicit") {
    out = &kAbiManifest_opencl_navatala_cfd_laplacian_vector_explicit;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_cfd_predictor_hby_a_from_source") {
    out = &kAbiManifest_opencl_navatala_cfd_predictor_hby_a_from_source;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_cfd_predictor_inverse_diag") {
    out = &kAbiManifest_opencl_navatala_cfd_predictor_inverse_diag;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_cfd_predictor_phi_hby_a_fuse") {
    out = &kAbiManifest_opencl_navatala_cfd_predictor_phi_hby_a_fuse;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_cfd_axpy_cells") {
    out = &kAbiManifest_opencl_navatala_cfd_axpy_cells;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_cfd_ddt_diagonal") {
    out = &kAbiManifest_opencl_navatala_cfd_ddt_diagonal;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_cfd_div_upwind_coeffs") {
    out = &kAbiManifest_opencl_navatala_cfd_div_upwind_coeffs;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_cfd_fvm_sp") {
    out = &kAbiManifest_opencl_navatala_cfd_fvm_sp;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_cfd_fvm_su_sp") {
    out = &kAbiManifest_opencl_navatala_cfd_fvm_su_sp;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_cfd_laplacian_coeffs") {
    out = &kAbiManifest_opencl_navatala_cfd_laplacian_coeffs;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_cfd_laplacian_coeffs_accum") {
    out = &kAbiManifest_opencl_navatala_cfd_laplacian_coeffs_accum;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_cfd_linear_upwind_correction_source") {
    out = &kAbiManifest_opencl_navatala_cfd_linear_upwind_correction_source;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_cfd_matrix_relax") {
    out = &kAbiManifest_opencl_navatala_cfd_matrix_relax;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_cfd_neg_sum_diag") {
    out = &kAbiManifest_opencl_navatala_cfd_neg_sum_diag;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_cfd_pack_boundary_adjacent_cells") {
    out = &kAbiManifest_opencl_navatala_cfd_pack_boundary_adjacent_cells;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_cfd_scalar_jacobi_clamp_min") {
    out = &kAbiManifest_opencl_navatala_cfd_scalar_jacobi_clamp_min;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_cfd_scalar_jacobi_update") {
    out = &kAbiManifest_opencl_navatala_cfd_scalar_jacobi_update;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_cfd_scalar_ldu_mat_vec") {
    out = &kAbiManifest_opencl_navatala_cfd_scalar_ldu_mat_vec;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_cfd_scalar_ldu_multi_color_gs_sweep") {
    out = &kAbiManifest_opencl_navatala_cfd_scalar_ldu_multi_color_gs_sweep;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_cfd_sum_mag_off_diag") {
    out = &kAbiManifest_opencl_navatala_cfd_sum_mag_off_diag;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_cfd_gamg_axpy_in_place") {
    out = &kAbiManifest_opencl_navatala_cfd_gamg_axpy_in_place;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_cfd_gamg_compute_scale_factor") {
    out = &kAbiManifest_opencl_navatala_cfd_gamg_compute_scale_factor;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_cfd_gamg_dot_finalize") {
    out = &kAbiManifest_opencl_navatala_cfd_gamg_dot_finalize;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_cfd_gamg_negate_scalar") {
    out = &kAbiManifest_opencl_navatala_cfd_gamg_negate_scalar;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_cfd_gamg_scale_correction") {
    out = &kAbiManifest_opencl_navatala_cfd_gamg_scale_correction;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_cfd_mg_dic_build") {
    out = &kAbiManifest_opencl_navatala_cfd_mg_dic_build;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_cfd_mg_dic_smooth") {
    out = &kAbiManifest_opencl_navatala_cfd_mg_dic_smooth;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_cfd_mg_coarse_jacobi") {
    out = &kAbiManifest_opencl_navatala_cfd_mg_coarse_jacobi;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_cfd_mg_coarse_matvec") {
    out = &kAbiManifest_opencl_navatala_cfd_mg_coarse_matvec;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_cfd_mg_coarse_rhs") {
    out = &kAbiManifest_opencl_navatala_cfd_mg_coarse_rhs;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_cfd_mg_find_best_match") {
    out = &kAbiManifest_opencl_navatala_cfd_mg_find_best_match;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_cfd_mg_pair_match") {
    out = &kAbiManifest_opencl_navatala_cfd_mg_pair_match;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_cfd_mg_prolong_add") {
    out = &kAbiManifest_opencl_navatala_cfd_mg_prolong_add;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_cfd_mg_renumber_aggregates") {
    out = &kAbiManifest_opencl_navatala_cfd_mg_renumber_aggregates;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_cfd_mg_renumber_propagate") {
    out = &kAbiManifest_opencl_navatala_cfd_mg_renumber_propagate;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_cfd_mg_resolve_chains") {
    out = &kAbiManifest_opencl_navatala_cfd_mg_resolve_chains;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_cfd_mg_restrict_sum") {
    out = &kAbiManifest_opencl_navatala_cfd_mg_restrict_sum;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_cfd_mg_strength_of_connection") {
    out = &kAbiManifest_opencl_navatala_cfd_mg_strength_of_connection;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_cfd_mg_zero_coarse") {
    out = &kAbiManifest_opencl_navatala_cfd_mg_zero_coarse;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_cfd_phi_correct") {
    out = &kAbiManifest_opencl_navatala_cfd_phi_correct;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_cfd_phi_correct_boundary") {
    out = &kAbiManifest_opencl_navatala_cfd_phi_correct_boundary;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_cfd_phi_correct_internal") {
    out = &kAbiManifest_opencl_navatala_cfd_phi_correct_internal;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_cfd_axpy_in_place") {
    out = &kAbiManifest_opencl_navatala_cfd_axpy_in_place;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_cfd_bc_dirichlet_face_flux") {
    out = &kAbiManifest_opencl_navatala_cfd_bc_dirichlet_face_flux;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_cfd_bc_sn_grad_face_flux") {
    out = &kAbiManifest_opencl_navatala_cfd_bc_sn_grad_face_flux;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_cfd_cast_f32_to_f64") {
    out = &kAbiManifest_opencl_navatala_cfd_cast_f32_to_f64;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_cfd_cast_f64_to_f32") {
    out = &kAbiManifest_opencl_navatala_cfd_cast_f64_to_f32;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_cfd_coeff_to_cf_in_place") {
    out = &kAbiManifest_opencl_navatala_cfd_coeff_to_cf_in_place;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_cfd_diag_from_cf") {
    out = &kAbiManifest_opencl_navatala_cfd_diag_from_cf;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_cfd_dic_apply") {
    out = &kAbiManifest_opencl_navatala_cfd_dic_apply;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_cfd_dic_build_reciprocal_d") {
    out = &kAbiManifest_opencl_navatala_cfd_dic_build_reciprocal_d;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_cfd_dot_partials") {
    out = &kAbiManifest_opencl_navatala_cfd_dot_partials;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_cfd_face_flux") {
    out = &kAbiManifest_opencl_navatala_cfd_face_flux;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_cfd_inv_diag") {
    out = &kAbiManifest_opencl_navatala_cfd_inv_diag;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_cfd_laplacian_from_face_flux") {
    out = &kAbiManifest_opencl_navatala_cfd_laplacian_from_face_flux;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_cfd_mul_by_vol_in_place") {
    out = &kAbiManifest_opencl_navatala_cfd_mul_by_vol_in_place;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_cfd_ref_add_ax") {
    out = &kAbiManifest_opencl_navatala_cfd_ref_add_ax;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_cfd_ref_add_b") {
    out = &kAbiManifest_opencl_navatala_cfd_ref_add_b;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_cfd_rhs_sub") {
    out = &kAbiManifest_opencl_navatala_cfd_rhs_sub;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_cfd_shift_in_place") {
    out = &kAbiManifest_opencl_navatala_cfd_shift_in_place;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_cfd_sum_abs_partials") {
    out = &kAbiManifest_opencl_navatala_cfd_sum_abs_partials;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_cfd_upper_from_cf") {
    out = &kAbiManifest_opencl_navatala_cfd_upper_from_cf;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_cfd_xpay_in_place") {
    out = &kAbiManifest_opencl_navatala_cfd_xpay_in_place;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_cfd_dot_finalize_partials") {
    out = &kAbiManifest_opencl_navatala_cfd_dot_finalize_partials;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_cfd_negate_scalar") {
    out = &kAbiManifest_opencl_navatala_cfd_negate_scalar;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_cfd_pcg_compute_alpha") {
    out = &kAbiManifest_opencl_navatala_cfd_pcg_compute_alpha;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_cfd_pcg_compute_beta") {
    out = &kAbiManifest_opencl_navatala_cfd_pcg_compute_beta;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_cfd_exact_mat_vec") {
    out = &kAbiManifest_opencl_navatala_cfd_exact_mat_vec;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_cfd_exact_mat_vec_f64") {
    out = &kAbiManifest_opencl_navatala_cfd_exact_mat_vec_f64;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_cfd_face_flux_boundary") {
    out = &kAbiManifest_opencl_navatala_cfd_face_flux_boundary;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_cfd_face_flux_internal") {
    out = &kAbiManifest_opencl_navatala_cfd_face_flux_internal;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_cfd_pressure_face_flux_correction_corrected") {
    out = &kAbiManifest_opencl_navatala_cfd_pressure_face_flux_correction_corrected;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_cfd_pressure_sn_grad") {
    out = &kAbiManifest_opencl_navatala_cfd_pressure_sn_grad;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_cfd_u_correct") {
    out = &kAbiManifest_opencl_navatala_cfd_u_correct;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_cfd_u_correct_reconstruct") {
    out = &kAbiManifest_opencl_navatala_cfd_u_correct_reconstruct;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_cfd_grad_alpha_cell") {
    out = &kAbiManifest_opencl_navatala_cfd_grad_alpha_cell;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_cfd_grad_alpha_cell_f64") {
    out = &kAbiManifest_opencl_navatala_cfd_grad_alpha_cell_f64;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_cfd_n_hatf_all") {
    out = &kAbiManifest_opencl_navatala_cfd_n_hatf_all;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_cfd_n_hatf_all_f64") {
    out = &kAbiManifest_opencl_navatala_cfd_n_hatf_all_f64;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_cfd_stf_all_faces") {
    out = &kAbiManifest_opencl_navatala_cfd_stf_all_faces;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_cfd_stf_all_faces_f64") {
    out = &kAbiManifest_opencl_navatala_cfd_stf_all_faces_f64;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_cfd_k_omega_s_s_t_blending") {
    out = &kAbiManifest_opencl_navatala_cfd_k_omega_s_s_t_blending;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_cfd_k_omega_s_s_t_nut_compute") {
    out = &kAbiManifest_opencl_navatala_cfd_k_omega_s_s_t_nut_compute;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_cfd_k_omega_s_s_t_sources") {
    out = &kAbiManifest_opencl_navatala_cfd_k_omega_s_s_t_sources;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_cfd_mu_eff_from_rho_nu_eff") {
    out = &kAbiManifest_opencl_navatala_cfd_mu_eff_from_rho_nu_eff;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_cfd_mu_eff_from_rho_nut_and_nu_mix") {
    out = &kAbiManifest_opencl_navatala_cfd_mu_eff_from_rho_nut_and_nu_mix;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_cfd_turbulence_add_explicit_source_device") {
    out = &kAbiManifest_opencl_navatala_cfd_turbulence_add_explicit_source_device;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_cfd_turbulence_dirichlet_cell_source") {
    out = &kAbiManifest_opencl_navatala_cfd_turbulence_dirichlet_cell_source;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_cfd_turbulence_dirichlet_face_elimination") {
    out = &kAbiManifest_opencl_navatala_cfd_turbulence_dirichlet_face_elimination;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_cfd_two_mag_sqr_symm_grad_u") {
    out = &kAbiManifest_opencl_navatala_cfd_two_mag_sqr_symm_grad_u;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_cfd_attention_row_softmax") {
    out = &kAbiManifest_opencl_navatala_cfd_attention_row_softmax;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_cfd_attention_value_projection") {
    out = &kAbiManifest_opencl_navatala_cfd_attention_value_projection;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_cfd_gather_mean") {
    out = &kAbiManifest_opencl_navatala_cfd_gather_mean;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_cfd_layer_norm_forward") {
    out = &kAbiManifest_opencl_navatala_cfd_layer_norm_forward;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_cfd_radius_count") {
    out = &kAbiManifest_opencl_navatala_cfd_radius_count;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_cfd_scaled_dot_product") {
    out = &kAbiManifest_opencl_navatala_cfd_scaled_dot_product;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_cfd_scatter_add_atomic") {
    out = &kAbiManifest_opencl_navatala_cfd_scatter_add_atomic;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_cfd_spectral_complex_mul") {
    out = &kAbiManifest_opencl_navatala_cfd_spectral_complex_mul;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_cfd_spectral_conv1d_mode_sliced") {
    out = &kAbiManifest_opencl_navatala_cfd_spectral_conv1d_mode_sliced;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_cfd_spectral_conv2d_mode_sliced") {
    out = &kAbiManifest_opencl_navatala_cfd_spectral_conv2d_mode_sliced;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_cfd_spectral_conv3d_mode_sliced") {
    out = &kAbiManifest_opencl_navatala_cfd_spectral_conv3d_mode_sliced;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_cfd_spectral_conv4d_mode_sliced") {
    out = &kAbiManifest_opencl_navatala_cfd_spectral_conv4d_mode_sliced;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_cfd_compute_face_area_pair_weights_f32") {
    out = &kAbiManifest_opencl_navatala_cfd_compute_face_area_pair_weights_f32;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_cfd_restrict_by_agglomeration_gather_f32") {
    out = &kAbiManifest_opencl_navatala_cfd_restrict_by_agglomeration_gather_f32;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_cfd_prolongate_by_agglomeration_f32") {
    out = &kAbiManifest_opencl_navatala_cfd_prolongate_by_agglomeration_f32;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_cfd_diag_inv_apply_f32") {
    out = &kAbiManifest_opencl_navatala_cfd_diag_inv_apply_f32;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_cfd_csr_sp_m_v_mixed_f32_f64") {
    out = &kAbiManifest_opencl_navatala_cfd_csr_sp_m_v_mixed_f32_f64;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_cfd_chebyshev_sweep_mixed_f32_f64") {
    out = &kAbiManifest_opencl_navatala_cfd_chebyshev_sweep_mixed_f32_f64;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_cfd_ell_sp_m_v_mixed_f32_f64") {
    out = &kAbiManifest_opencl_navatala_cfd_ell_sp_m_v_mixed_f32_f64;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_cfd_chebyshev_smooth_post_fused_mixed") {
    out = &kAbiManifest_opencl_navatala_cfd_chebyshev_smooth_post_fused_mixed;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_cfd_gamg_proc_pack_f32") {
    out = &kAbiManifest_opencl_navatala_cfd_gamg_proc_pack_f32;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_cfd_gamg_proc_add_gather_f32") {
    out = &kAbiManifest_opencl_navatala_cfd_gamg_proc_add_gather_f32;
    return true;
  }
  out = nullptr;
  return false;
}

bool tryGetKernelSource_opencl_cfd(const std::string& backend, const std::string& kernelName, GpuRuntime::ProgramSource& out) {
  if (backend == "opencl" && kernelName == "navatala_cfd_boundary_force_partials") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_cfd_boundary_force_partials";
    std::string_view sv(k_opencl_navatala_cfd_boundary_force_partials);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_cfd_pack_owner_values") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_cfd_pack_owner_values";
    std::string_view sv(k_opencl_navatala_cfd_pack_owner_values);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_cfd_scatter_values_and_mask") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_cfd_scatter_values_and_mask";
    std::string_view sv(k_opencl_navatala_cfd_scatter_values_and_mask);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_cfd_pack_mu_grad_owner") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_cfd_pack_mu_grad_owner";
    std::string_view sv(k_opencl_navatala_cfd_pack_mu_grad_owner);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_cfd_pack_owner_rho_vec3") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_cfd_pack_owner_rho_vec3";
    std::string_view sv(k_opencl_navatala_cfd_pack_owner_rho_vec3);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_cfd_pack_owner_vec3") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_cfd_pack_owner_vec3";
    std::string_view sv(k_opencl_navatala_cfd_pack_owner_vec3);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_cfd_pack_owner_vec3_mu") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_cfd_pack_owner_vec3_mu";
    std::string_view sv(k_opencl_navatala_cfd_pack_owner_vec3_mu);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_cfd_scatter_mu_grad_ghost") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_cfd_scatter_mu_grad_ghost";
    std::string_view sv(k_opencl_navatala_cfd_scatter_mu_grad_ghost);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_cfd_scatter_rho_vec3_and_mask") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_cfd_scatter_rho_vec3_and_mask";
    std::string_view sv(k_opencl_navatala_cfd_scatter_rho_vec3_and_mask);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_cfd_scatter_vec3_and_mask") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_cfd_scatter_vec3_and_mask";
    std::string_view sv(k_opencl_navatala_cfd_scatter_vec3_and_mask);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_cfd_scatter_vec3_mu_and_mask") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_cfd_scatter_vec3_mu_and_mask";
    std::string_view sv(k_opencl_navatala_cfd_scatter_vec3_mu_and_mask);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_cfd_cf_mesh_volume_optimizer_gradient_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_cfd_cf_mesh_volume_optimizer_gradient_f32";
    std::string_view sv(k_opencl_navatala_cfd_cf_mesh_volume_optimizer_gradient_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_cfd_cf_mesh_volume_optimizer_gradient_f64") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_cfd_cf_mesh_volume_optimizer_gradient_f64";
    std::string_view sv(k_opencl_navatala_cfd_cf_mesh_volume_optimizer_gradient_f64);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_cfd_cf_mesh_volume_optimizer_k_prepass_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_cfd_cf_mesh_volume_optimizer_k_prepass_f32";
    std::string_view sv(k_opencl_navatala_cfd_cf_mesh_volume_optimizer_k_prepass_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_cfd_cf_mesh_volume_optimizer_k_prepass_f64") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_cfd_cf_mesh_volume_optimizer_k_prepass_f64";
    std::string_view sv(k_opencl_navatala_cfd_cf_mesh_volume_optimizer_k_prepass_f64);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_cfd_cf_mesh_volume_optimizer_objective_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_cfd_cf_mesh_volume_optimizer_objective_f32";
    std::string_view sv(k_opencl_navatala_cfd_cf_mesh_volume_optimizer_objective_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_cfd_cf_mesh_volume_optimizer_objective_f64") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_cfd_cf_mesh_volume_optimizer_objective_f64";
    std::string_view sv(k_opencl_navatala_cfd_cf_mesh_volume_optimizer_objective_f64);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_cfd_add_vol_vector") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_cfd_add_vol_vector";
    std::string_view sv(k_opencl_navatala_cfd_add_vol_vector);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_cfd_div_rho_phi_u_explicit") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_cfd_div_rho_phi_u_explicit";
    std::string_view sv(k_opencl_navatala_cfd_div_rho_phi_u_explicit);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_cfd_hby_a_from_u_and_terms") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_cfd_hby_a_from_u_and_terms";
    std::string_view sv(k_opencl_navatala_cfd_hby_a_from_u_and_terms);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_cfd_apply_sp_to_hby_a") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_cfd_apply_sp_to_hby_a";
    std::string_view sv(k_opencl_navatala_cfd_apply_sp_to_hby_a);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_cfd_laplacian_scalar_explicit") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_cfd_laplacian_scalar_explicit";
    std::string_view sv(k_opencl_navatala_cfd_laplacian_scalar_explicit);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_cfd_laplacian_vector_explicit") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_cfd_laplacian_vector_explicit";
    std::string_view sv(k_opencl_navatala_cfd_laplacian_vector_explicit);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_cfd_predictor_hby_a_from_source") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_cfd_predictor_hby_a_from_source";
    std::string_view sv(k_opencl_navatala_cfd_predictor_hby_a_from_source);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_cfd_predictor_inverse_diag") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_cfd_predictor_inverse_diag";
    std::string_view sv(k_opencl_navatala_cfd_predictor_inverse_diag);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_cfd_predictor_phi_hby_a_fuse") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_cfd_predictor_phi_hby_a_fuse";
    std::string_view sv(k_opencl_navatala_cfd_predictor_phi_hby_a_fuse);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_cfd_axpy_cells") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_cfd_axpy_cells";
    std::string_view sv(k_opencl_navatala_cfd_axpy_cells);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_cfd_ddt_diagonal") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_cfd_ddt_diagonal";
    std::string_view sv(k_opencl_navatala_cfd_ddt_diagonal);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_cfd_div_upwind_coeffs") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_cfd_div_upwind_coeffs";
    std::string_view sv(k_opencl_navatala_cfd_div_upwind_coeffs);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_cfd_fvm_sp") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_cfd_fvm_sp";
    std::string_view sv(k_opencl_navatala_cfd_fvm_sp);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_cfd_fvm_su_sp") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_cfd_fvm_su_sp";
    std::string_view sv(k_opencl_navatala_cfd_fvm_su_sp);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_cfd_laplacian_coeffs") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_cfd_laplacian_coeffs";
    std::string_view sv(k_opencl_navatala_cfd_laplacian_coeffs);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_cfd_laplacian_coeffs_accum") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_cfd_laplacian_coeffs_accum";
    std::string_view sv(k_opencl_navatala_cfd_laplacian_coeffs_accum);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_cfd_linear_upwind_correction_source") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_cfd_linear_upwind_correction_source";
    std::string_view sv(k_opencl_navatala_cfd_linear_upwind_correction_source);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_cfd_matrix_relax") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_cfd_matrix_relax";
    std::string_view sv(k_opencl_navatala_cfd_matrix_relax);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_cfd_neg_sum_diag") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_cfd_neg_sum_diag";
    std::string_view sv(k_opencl_navatala_cfd_neg_sum_diag);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_cfd_pack_boundary_adjacent_cells") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_cfd_pack_boundary_adjacent_cells";
    std::string_view sv(k_opencl_navatala_cfd_pack_boundary_adjacent_cells);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_cfd_scalar_jacobi_clamp_min") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_cfd_scalar_jacobi_clamp_min";
    std::string_view sv(k_opencl_navatala_cfd_scalar_jacobi_clamp_min);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_cfd_scalar_jacobi_update") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_cfd_scalar_jacobi_update";
    std::string_view sv(k_opencl_navatala_cfd_scalar_jacobi_update);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_cfd_scalar_ldu_mat_vec") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_cfd_scalar_ldu_mat_vec";
    std::string_view sv(k_opencl_navatala_cfd_scalar_ldu_mat_vec);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_cfd_scalar_ldu_multi_color_gs_sweep") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_cfd_scalar_ldu_multi_color_gs_sweep";
    std::string_view sv(k_opencl_navatala_cfd_scalar_ldu_multi_color_gs_sweep);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_cfd_sum_mag_off_diag") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_cfd_sum_mag_off_diag";
    std::string_view sv(k_opencl_navatala_cfd_sum_mag_off_diag);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_cfd_gamg_axpy_in_place") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_cfd_gamg_axpy_in_place";
    std::string_view sv(k_opencl_navatala_cfd_gamg_axpy_in_place);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_cfd_gamg_compute_scale_factor") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_cfd_gamg_compute_scale_factor";
    std::string_view sv(k_opencl_navatala_cfd_gamg_compute_scale_factor);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_cfd_gamg_dot_finalize") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_cfd_gamg_dot_finalize";
    std::string_view sv(k_opencl_navatala_cfd_gamg_dot_finalize);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_cfd_gamg_negate_scalar") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_cfd_gamg_negate_scalar";
    std::string_view sv(k_opencl_navatala_cfd_gamg_negate_scalar);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_cfd_gamg_scale_correction") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_cfd_gamg_scale_correction";
    std::string_view sv(k_opencl_navatala_cfd_gamg_scale_correction);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_cfd_mg_dic_build") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_cfd_mg_dic_build";
    std::string_view sv(k_opencl_navatala_cfd_mg_dic_build);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_cfd_mg_dic_smooth") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_cfd_mg_dic_smooth";
    std::string_view sv(k_opencl_navatala_cfd_mg_dic_smooth);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_cfd_mg_coarse_jacobi") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_cfd_mg_coarse_jacobi";
    std::string_view sv(k_opencl_navatala_cfd_mg_coarse_jacobi);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_cfd_mg_coarse_matvec") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_cfd_mg_coarse_matvec";
    std::string_view sv(k_opencl_navatala_cfd_mg_coarse_matvec);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_cfd_mg_coarse_rhs") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_cfd_mg_coarse_rhs";
    std::string_view sv(k_opencl_navatala_cfd_mg_coarse_rhs);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_cfd_mg_find_best_match") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_cfd_mg_find_best_match";
    std::string_view sv(k_opencl_navatala_cfd_mg_find_best_match);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_cfd_mg_pair_match") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_cfd_mg_pair_match";
    std::string_view sv(k_opencl_navatala_cfd_mg_pair_match);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_cfd_mg_prolong_add") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_cfd_mg_prolong_add";
    std::string_view sv(k_opencl_navatala_cfd_mg_prolong_add);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_cfd_mg_renumber_aggregates") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_cfd_mg_renumber_aggregates";
    std::string_view sv(k_opencl_navatala_cfd_mg_renumber_aggregates);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_cfd_mg_renumber_propagate") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_cfd_mg_renumber_propagate";
    std::string_view sv(k_opencl_navatala_cfd_mg_renumber_propagate);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_cfd_mg_resolve_chains") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_cfd_mg_resolve_chains";
    std::string_view sv(k_opencl_navatala_cfd_mg_resolve_chains);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_cfd_mg_restrict_sum") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_cfd_mg_restrict_sum";
    std::string_view sv(k_opencl_navatala_cfd_mg_restrict_sum);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_cfd_mg_strength_of_connection") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_cfd_mg_strength_of_connection";
    std::string_view sv(k_opencl_navatala_cfd_mg_strength_of_connection);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_cfd_mg_zero_coarse") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_cfd_mg_zero_coarse";
    std::string_view sv(k_opencl_navatala_cfd_mg_zero_coarse);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_cfd_phi_correct") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_cfd_phi_correct";
    std::string_view sv(k_opencl_navatala_cfd_phi_correct);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_cfd_phi_correct_boundary") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_cfd_phi_correct_boundary";
    std::string_view sv(k_opencl_navatala_cfd_phi_correct_boundary);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_cfd_phi_correct_internal") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_cfd_phi_correct_internal";
    std::string_view sv(k_opencl_navatala_cfd_phi_correct_internal);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_cfd_axpy_in_place") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_cfd_axpy_in_place";
    std::string_view sv(k_opencl_navatala_cfd_axpy_in_place);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_cfd_bc_dirichlet_face_flux") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_cfd_bc_dirichlet_face_flux";
    std::string_view sv(k_opencl_navatala_cfd_bc_dirichlet_face_flux);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_cfd_bc_sn_grad_face_flux") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_cfd_bc_sn_grad_face_flux";
    std::string_view sv(k_opencl_navatala_cfd_bc_sn_grad_face_flux);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_cfd_cast_f32_to_f64") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_cfd_cast_f32_to_f64";
    std::string_view sv(k_opencl_navatala_cfd_cast_f32_to_f64);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_cfd_cast_f64_to_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_cfd_cast_f64_to_f32";
    std::string_view sv(k_opencl_navatala_cfd_cast_f64_to_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_cfd_coeff_to_cf_in_place") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_cfd_coeff_to_cf_in_place";
    std::string_view sv(k_opencl_navatala_cfd_coeff_to_cf_in_place);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_cfd_diag_from_cf") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_cfd_diag_from_cf";
    std::string_view sv(k_opencl_navatala_cfd_diag_from_cf);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_cfd_dic_apply") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_cfd_dic_apply";
    std::string_view sv(k_opencl_navatala_cfd_dic_apply);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_cfd_dic_build_reciprocal_d") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_cfd_dic_build_reciprocal_d";
    std::string_view sv(k_opencl_navatala_cfd_dic_build_reciprocal_d);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_cfd_dot_partials") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_cfd_dot_partials";
    std::string_view sv(k_opencl_navatala_cfd_dot_partials);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_cfd_face_flux") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_cfd_face_flux";
    std::string_view sv(k_opencl_navatala_cfd_face_flux);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_cfd_inv_diag") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_cfd_inv_diag";
    std::string_view sv(k_opencl_navatala_cfd_inv_diag);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_cfd_laplacian_from_face_flux") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_cfd_laplacian_from_face_flux";
    std::string_view sv(k_opencl_navatala_cfd_laplacian_from_face_flux);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_cfd_mul_by_vol_in_place") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_cfd_mul_by_vol_in_place";
    std::string_view sv(k_opencl_navatala_cfd_mul_by_vol_in_place);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_cfd_ref_add_ax") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_cfd_ref_add_ax";
    std::string_view sv(k_opencl_navatala_cfd_ref_add_ax);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_cfd_ref_add_b") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_cfd_ref_add_b";
    std::string_view sv(k_opencl_navatala_cfd_ref_add_b);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_cfd_rhs_sub") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_cfd_rhs_sub";
    std::string_view sv(k_opencl_navatala_cfd_rhs_sub);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_cfd_shift_in_place") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_cfd_shift_in_place";
    std::string_view sv(k_opencl_navatala_cfd_shift_in_place);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_cfd_sum_abs_partials") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_cfd_sum_abs_partials";
    std::string_view sv(k_opencl_navatala_cfd_sum_abs_partials);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_cfd_upper_from_cf") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_cfd_upper_from_cf";
    std::string_view sv(k_opencl_navatala_cfd_upper_from_cf);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_cfd_xpay_in_place") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_cfd_xpay_in_place";
    std::string_view sv(k_opencl_navatala_cfd_xpay_in_place);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_cfd_dot_finalize_partials") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_cfd_dot_finalize_partials";
    std::string_view sv(k_opencl_navatala_cfd_dot_finalize_partials);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_cfd_negate_scalar") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_cfd_negate_scalar";
    std::string_view sv(k_opencl_navatala_cfd_negate_scalar);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_cfd_pcg_compute_alpha") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_cfd_pcg_compute_alpha";
    std::string_view sv(k_opencl_navatala_cfd_pcg_compute_alpha);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_cfd_pcg_compute_beta") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_cfd_pcg_compute_beta";
    std::string_view sv(k_opencl_navatala_cfd_pcg_compute_beta);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_cfd_exact_mat_vec") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_cfd_exact_mat_vec";
    std::string_view sv(k_opencl_navatala_cfd_exact_mat_vec);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_cfd_exact_mat_vec_f64") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_cfd_exact_mat_vec_f64";
    std::string_view sv(k_opencl_navatala_cfd_exact_mat_vec_f64);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_cfd_face_flux_boundary") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_cfd_face_flux_boundary";
    std::string_view sv(k_opencl_navatala_cfd_face_flux_boundary);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_cfd_face_flux_internal") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_cfd_face_flux_internal";
    std::string_view sv(k_opencl_navatala_cfd_face_flux_internal);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_cfd_pressure_face_flux_correction_corrected") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_cfd_pressure_face_flux_correction_corrected";
    std::string_view sv(k_opencl_navatala_cfd_pressure_face_flux_correction_corrected);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_cfd_pressure_sn_grad") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_cfd_pressure_sn_grad";
    std::string_view sv(k_opencl_navatala_cfd_pressure_sn_grad);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_cfd_u_correct") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_cfd_u_correct";
    std::string_view sv(k_opencl_navatala_cfd_u_correct);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_cfd_u_correct_reconstruct") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_cfd_u_correct_reconstruct";
    std::string_view sv(k_opencl_navatala_cfd_u_correct_reconstruct);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_cfd_grad_alpha_cell") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_cfd_grad_alpha_cell";
    std::string_view sv(k_opencl_navatala_cfd_grad_alpha_cell);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_cfd_grad_alpha_cell_f64") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_cfd_grad_alpha_cell_f64";
    std::string_view sv(k_opencl_navatala_cfd_grad_alpha_cell_f64);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_cfd_n_hatf_all") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_cfd_n_hatf_all";
    std::string_view sv(k_opencl_navatala_cfd_n_hatf_all);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_cfd_n_hatf_all_f64") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_cfd_n_hatf_all_f64";
    std::string_view sv(k_opencl_navatala_cfd_n_hatf_all_f64);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_cfd_stf_all_faces") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_cfd_stf_all_faces";
    std::string_view sv(k_opencl_navatala_cfd_stf_all_faces);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_cfd_stf_all_faces_f64") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_cfd_stf_all_faces_f64";
    std::string_view sv(k_opencl_navatala_cfd_stf_all_faces_f64);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_cfd_k_omega_s_s_t_blending") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_cfd_k_omega_s_s_t_blending";
    std::string_view sv(k_opencl_navatala_cfd_k_omega_s_s_t_blending);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_cfd_k_omega_s_s_t_nut_compute") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_cfd_k_omega_s_s_t_nut_compute";
    std::string_view sv(k_opencl_navatala_cfd_k_omega_s_s_t_nut_compute);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_cfd_k_omega_s_s_t_sources") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_cfd_k_omega_s_s_t_sources";
    std::string_view sv(k_opencl_navatala_cfd_k_omega_s_s_t_sources);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_cfd_mu_eff_from_rho_nu_eff") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_cfd_mu_eff_from_rho_nu_eff";
    std::string_view sv(k_opencl_navatala_cfd_mu_eff_from_rho_nu_eff);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_cfd_mu_eff_from_rho_nut_and_nu_mix") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_cfd_mu_eff_from_rho_nut_and_nu_mix";
    std::string_view sv(k_opencl_navatala_cfd_mu_eff_from_rho_nut_and_nu_mix);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_cfd_turbulence_add_explicit_source_device") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_cfd_turbulence_add_explicit_source_device";
    std::string_view sv(k_opencl_navatala_cfd_turbulence_add_explicit_source_device);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_cfd_turbulence_dirichlet_cell_source") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_cfd_turbulence_dirichlet_cell_source";
    std::string_view sv(k_opencl_navatala_cfd_turbulence_dirichlet_cell_source);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_cfd_turbulence_dirichlet_face_elimination") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_cfd_turbulence_dirichlet_face_elimination";
    std::string_view sv(k_opencl_navatala_cfd_turbulence_dirichlet_face_elimination);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_cfd_two_mag_sqr_symm_grad_u") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_cfd_two_mag_sqr_symm_grad_u";
    std::string_view sv(k_opencl_navatala_cfd_two_mag_sqr_symm_grad_u);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_cfd_attention_row_softmax") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_cfd_attention_row_softmax";
    std::string_view sv(k_opencl_navatala_cfd_attention_row_softmax);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_cfd_attention_value_projection") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_cfd_attention_value_projection";
    std::string_view sv(k_opencl_navatala_cfd_attention_value_projection);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_cfd_gather_mean") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_cfd_gather_mean";
    std::string_view sv(k_opencl_navatala_cfd_gather_mean);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_cfd_layer_norm_forward") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_cfd_layer_norm_forward";
    std::string_view sv(k_opencl_navatala_cfd_layer_norm_forward);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_cfd_radius_count") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_cfd_radius_count";
    std::string_view sv(k_opencl_navatala_cfd_radius_count);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_cfd_scaled_dot_product") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_cfd_scaled_dot_product";
    std::string_view sv(k_opencl_navatala_cfd_scaled_dot_product);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_cfd_scatter_add_atomic") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_cfd_scatter_add_atomic";
    std::string_view sv(k_opencl_navatala_cfd_scatter_add_atomic);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_cfd_spectral_complex_mul") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_cfd_spectral_complex_mul";
    std::string_view sv(k_opencl_navatala_cfd_spectral_complex_mul);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_cfd_spectral_conv1d_mode_sliced") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_cfd_spectral_conv1d_mode_sliced";
    std::string_view sv(k_opencl_navatala_cfd_spectral_conv1d_mode_sliced);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_cfd_spectral_conv2d_mode_sliced") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_cfd_spectral_conv2d_mode_sliced";
    std::string_view sv(k_opencl_navatala_cfd_spectral_conv2d_mode_sliced);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_cfd_spectral_conv3d_mode_sliced") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_cfd_spectral_conv3d_mode_sliced";
    std::string_view sv(k_opencl_navatala_cfd_spectral_conv3d_mode_sliced);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_cfd_spectral_conv4d_mode_sliced") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_cfd_spectral_conv4d_mode_sliced";
    std::string_view sv(k_opencl_navatala_cfd_spectral_conv4d_mode_sliced);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_cfd_compute_face_area_pair_weights_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_cfd_compute_face_area_pair_weights_f32";
    std::string_view sv(k_opencl_navatala_cfd_compute_face_area_pair_weights_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_cfd_restrict_by_agglomeration_gather_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_cfd_restrict_by_agglomeration_gather_f32";
    std::string_view sv(k_opencl_navatala_cfd_restrict_by_agglomeration_gather_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_cfd_prolongate_by_agglomeration_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_cfd_prolongate_by_agglomeration_f32";
    std::string_view sv(k_opencl_navatala_cfd_prolongate_by_agglomeration_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_cfd_diag_inv_apply_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_cfd_diag_inv_apply_f32";
    std::string_view sv(k_opencl_navatala_cfd_diag_inv_apply_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_cfd_csr_sp_m_v_mixed_f32_f64") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_cfd_csr_sp_m_v_mixed_f32_f64";
    std::string_view sv(k_opencl_navatala_cfd_csr_sp_m_v_mixed_f32_f64);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_cfd_chebyshev_sweep_mixed_f32_f64") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_cfd_chebyshev_sweep_mixed_f32_f64";
    std::string_view sv(k_opencl_navatala_cfd_chebyshev_sweep_mixed_f32_f64);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_cfd_ell_sp_m_v_mixed_f32_f64") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_cfd_ell_sp_m_v_mixed_f32_f64";
    std::string_view sv(k_opencl_navatala_cfd_ell_sp_m_v_mixed_f32_f64);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_cfd_chebyshev_smooth_post_fused_mixed") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_cfd_chebyshev_smooth_post_fused_mixed";
    std::string_view sv(k_opencl_navatala_cfd_chebyshev_smooth_post_fused_mixed);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_cfd_gamg_proc_pack_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_cfd_gamg_proc_pack_f32";
    std::string_view sv(k_opencl_navatala_cfd_gamg_proc_pack_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_cfd_gamg_proc_add_gather_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_cfd_gamg_proc_add_gather_f32";
    std::string_view sv(k_opencl_navatala_cfd_gamg_proc_add_gather_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  return false;
}

} // namespace NavatalaRegistry

