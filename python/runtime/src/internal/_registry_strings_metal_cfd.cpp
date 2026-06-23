// SPDX-License-Identifier: Apache-2.0
// Kernel registry partition for the `metal_cfd` bucket. Compiles
// against the declarations in `navatala/_registry.hpp` and links
// against the trampoline in `_registry.cpp`. Partitioning keeps each
// translation unit well under the 50K-line threshold so a typical
// build can parallelise across the buckets.
#include "navatala/_registry.hpp"
#include <cstdint>
#include <string>
#include <string_view>

const char* k_metal_navatala_cfd_boundary_force_partials = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_cfd_boundary_force_partials(device const float* pAllFaces [[buffer(0)]], device const float* sfComponent [[buffer(1)]], device const int* counts [[buffer(2)]], device float* outPartials [[buffer(3)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  threadgroup float tmp[256];
  float v = as_type<float>(0x00000000u);
  if (((int)(int(__gid.x))) < counts[0]) {
    int faceIdx = (counts[1] + ((int)(int(__gid.x))));
    float pVal = pAllFaces[faceIdx];
    float sfVal = sfComponent[faceIdx];
    v = (pVal * sfVal);
  }
  tmp[((int)(int(__tid.x)))] = v;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (((int)(int(__tid.x))) < 128) {
    tmp[((int)(int(__tid.x)))] = (tmp[((int)(int(__tid.x)))] + tmp[(((int)(int(__tid.x))) + 128)]);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (((int)(int(__tid.x))) < 64) {
    tmp[((int)(int(__tid.x)))] = (tmp[((int)(int(__tid.x)))] + tmp[(((int)(int(__tid.x))) + 64)]);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (((int)(int(__tid.x))) < 32) {
    tmp[((int)(int(__tid.x)))] = (tmp[((int)(int(__tid.x)))] + tmp[(((int)(int(__tid.x))) + 32)]);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (((int)(int(__tid.x))) < 16) {
    tmp[((int)(int(__tid.x)))] = (tmp[((int)(int(__tid.x)))] + tmp[(((int)(int(__tid.x))) + 16)]);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (((int)(int(__tid.x))) < 8) {
    tmp[((int)(int(__tid.x)))] = (tmp[((int)(int(__tid.x)))] + tmp[(((int)(int(__tid.x))) + 8)]);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (((int)(int(__tid.x))) < 4) {
    tmp[((int)(int(__tid.x)))] = (tmp[((int)(int(__tid.x)))] + tmp[(((int)(int(__tid.x))) + 4)]);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (((int)(int(__tid.x))) < 2) {
    tmp[((int)(int(__tid.x)))] = (tmp[((int)(int(__tid.x)))] + tmp[(((int)(int(__tid.x))) + 2)]);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (((int)(int(__tid.x))) < 1) {
    tmp[((int)(int(__tid.x)))] = (tmp[((int)(int(__tid.x)))] + tmp[(((int)(int(__tid.x))) + 1)]);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (((int)(int(__tid.x))) == 0) {
    outPartials[((int)(int(__tgid.x)))] = tmp[0];
  }
}

)kernel";
const char* k_metal_navatala_cfd_pack_owner_values = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_cfd_pack_owner_values(device const float* x [[buffer(0)]], device const int* owner [[buffer(1)]], device const int* procFaces [[buffer(2)]], device const int* counts [[buffer(3)]], device float* outSend [[buffer(4)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  if (int(__gid.x) >= counts[0]) {
    return;
  } else {
    int face = procFaces[int(__gid.x)];
    int o = owner[face];
    float v = x[o];
    outSend[int(__gid.x)] = v;
  }
}

)kernel";
const char* k_metal_navatala_cfd_scatter_values_and_mask = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_cfd_scatter_values_and_mask(device const int* procFaces [[buffer(0)]], device const float* recv [[buffer(1)]], device const int* counts [[buffer(2)]], device float* bcVal [[buffer(3)]], device int* bcMask [[buffer(4)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  if (int(__gid.x) >= counts[0]) {
    return;
  } else {
    int face = procFaces[int(__gid.x)];
    bcVal[face] = recv[int(__gid.x)];
    bcMask[face] = 1;
  }
}

)kernel";
const char* k_metal_navatala_cfd_pack_mu_grad_owner = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_cfd_pack_mu_grad_owner(device const float* muCell [[buffer(0)]], device const float* gXX [[buffer(1)]], device const float* gXY [[buffer(2)]], device const float* gXZ [[buffer(3)]], device const float* gYX [[buffer(4)]], device const float* gYY [[buffer(5)]], device const float* gYZ [[buffer(6)]], device const float* gZX [[buffer(7)]], device const float* gZY [[buffer(8)]], device const float* gZZ [[buffer(9)]], device const int* owner [[buffer(10)]], device const int* procFaces [[buffer(11)]], device const int* counts [[buffer(12)]], device float* outSend [[buffer(13)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  if (int(__gid.x) >= ((int)(counts[0]))) {
    return;
  } else {
    int f = procFaces[int(__gid.x)];
    int o = owner[f];
    int base = (int(__gid.x) * 10);
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
const char* k_metal_navatala_cfd_pack_owner_rho_vec3 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_cfd_pack_owner_rho_vec3(device const float* rhoCell [[buffer(0)]], device const float* ux [[buffer(1)]], device const float* uy [[buffer(2)]], device const float* uz [[buffer(3)]], device const int* owner [[buffer(4)]], device const int* procFaces [[buffer(5)]], device const int* counts [[buffer(6)]], device float* outSend [[buffer(7)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  if (int(__gid.x) >= ((int)(counts[0]))) {
    return;
  } else {
    int f = procFaces[int(__gid.x)];
    int o = owner[f];
    int base = (int(__gid.x) * 4);
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
const char* k_metal_navatala_cfd_pack_owner_vec3 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_cfd_pack_owner_vec3(device const float* x [[buffer(0)]], device const float* y [[buffer(1)]], device const float* z [[buffer(2)]], device const int* owner [[buffer(3)]], device const int* procFaces [[buffer(4)]], device const int* counts [[buffer(5)]], device float* outSend [[buffer(6)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  if (int(__gid.x) >= ((int)(counts[0]))) {
    return;
  } else {
    int f = procFaces[int(__gid.x)];
    int o = owner[f];
    int base = (int(__gid.x) * 3);
    outSend[base] = x[o];
    int b1 = (base + 1);
    outSend[b1] = y[o];
    int b2 = (base + 2);
    outSend[b2] = z[o];
  }
}

)kernel";
const char* k_metal_navatala_cfd_pack_owner_vec3_mu = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_cfd_pack_owner_vec3_mu(device const float* ux [[buffer(0)]], device const float* uy [[buffer(1)]], device const float* uz [[buffer(2)]], device const float* muCell [[buffer(3)]], device const int* owner [[buffer(4)]], device const int* procFaces [[buffer(5)]], device const int* counts [[buffer(6)]], device float* outSend [[buffer(7)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  if (int(__gid.x) >= ((int)(counts[0]))) {
    return;
  } else {
    int f = procFaces[int(__gid.x)];
    int o = owner[f];
    int base = (int(__gid.x) * 4);
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
const char* k_metal_navatala_cfd_scatter_mu_grad_ghost = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_cfd_scatter_mu_grad_ghost(device const float* recvVals [[buffer(0)]], device const int* counts [[buffer(1)]], device float* muCell [[buffer(2)]], device float* gXX [[buffer(3)]], device float* gXY [[buffer(4)]], device float* gXZ [[buffer(5)]], device float* gYX [[buffer(6)]], device float* gYY [[buffer(7)]], device float* gYZ [[buffer(8)]], device float* gZX [[buffer(9)]], device float* gZY [[buffer(10)]], device float* gZZ [[buffer(11)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  if (int(__gid.x) >= ((int)(counts[0]))) {
    return;
  } else {
    int idx = (((int)(counts[1])) + int(__gid.x));
    int base = (int(__gid.x) * 10);
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
const char* k_metal_navatala_cfd_scatter_rho_vec3_and_mask = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_cfd_scatter_rho_vec3_and_mask(device const int* procFaces [[buffer(0)]], device const float* recvVals [[buffer(1)]], device const int* counts [[buffer(2)]], device float* rhoBcVal [[buffer(3)]], device float* uBcX [[buffer(4)]], device float* uBcY [[buffer(5)]], device float* uBcZ [[buffer(6)]], device uint* rhoBcMask [[buffer(7)]], device uint* uBcMask [[buffer(8)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  if (int(__gid.x) >= ((int)(counts[0]))) {
    return;
  } else {
    int f = procFaces[int(__gid.x)];
    int base = (int(__gid.x) * 4);
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
const char* k_metal_navatala_cfd_scatter_vec3_and_mask = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_cfd_scatter_vec3_and_mask(device const int* procFaces [[buffer(0)]], device const float* recvVals [[buffer(1)]], device const int* counts [[buffer(2)]], device float* bcX [[buffer(3)]], device float* bcY [[buffer(4)]], device float* bcZ [[buffer(5)]], device uint* bcMask [[buffer(6)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  if (int(__gid.x) >= ((int)(counts[0]))) {
    return;
  } else {
    int f = procFaces[int(__gid.x)];
    int base = (int(__gid.x) * 3);
    int b1 = (base + 1);
    int b2 = (base + 2);
    bcX[f] = recvVals[base];
    bcY[f] = recvVals[b1];
    bcZ[f] = recvVals[b2];
    bcMask[f] = ((uint)(1));
  }
}

)kernel";
const char* k_metal_navatala_cfd_scatter_vec3_mu_and_mask = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_cfd_scatter_vec3_mu_and_mask(device const int* procFaces [[buffer(0)]], device const float* recvVals [[buffer(1)]], device const int* counts [[buffer(2)]], device float* bcX [[buffer(3)]], device float* bcY [[buffer(4)]], device float* bcZ [[buffer(5)]], device float* muBc [[buffer(6)]], device uint* bcMask [[buffer(7)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  if (int(__gid.x) >= ((int)(counts[0]))) {
    return;
  } else {
    int f = procFaces[int(__gid.x)];
    int base = (int(__gid.x) * 4);
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
const char* k_metal_navatala_cfd_add_vol_vector = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_cfd_add_vol_vector(device const float* ax [[buffer(0)]], device const float* ay [[buffer(1)]], device const float* az [[buffer(2)]], device const float* bx [[buffer(3)]], device const float* by_ [[buffer(4)]], device const float* bz [[buffer(5)]], device const int* params [[buffer(6)]], device float* outX [[buffer(7)]], device float* outY [[buffer(8)]], device float* outZ [[buffer(9)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  if (((int)(int(__gid.x))) >= params[0]) {
    return;
  } else {
    outX[((int)(int(__gid.x)))] = (ax[((int)(int(__gid.x)))] + bx[((int)(int(__gid.x)))]);
    outY[((int)(int(__gid.x)))] = (ay[((int)(int(__gid.x)))] + by_[((int)(int(__gid.x)))]);
    outZ[((int)(int(__gid.x)))] = (az[((int)(int(__gid.x)))] + bz[((int)(int(__gid.x)))]);
  }
}

)kernel";
const char* k_metal_navatala_cfd_div_rho_phi_u_explicit = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_cfd_div_rho_phi_u_explicit(device const float* phi [[buffer(0)]], device const float* rhoCell [[buffer(1)]], device const float* ux [[buffer(2)]], device const float* uy [[buffer(3)]], device const float* uz [[buffer(4)]], device const int* owner [[buffer(5)]], device const int* neighbour [[buffer(6)]], device const float* weights [[buffer(7)]], device const int* offsets [[buffer(8)]], device const int* faceIdx [[buffer(9)]], device const float* sign [[buffer(10)]], device const float* vol [[buffer(11)]], device const float* rhoBcVal [[buffer(12)]], device const uint* rhoBcMask [[buffer(13)]], device const float* uBcX [[buffer(14)]], device const float* uBcY [[buffer(15)]], device const float* uBcZ [[buffer(16)]], device const uint* uBcMask [[buffer(17)]], device const int* params [[buffer(18)]], device float* outX [[buffer(19)]], device float* outY [[buffer(20)]], device float* outZ [[buffer(21)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  if (((int)(int(__gid.x))) >= params[0]) {
    return;
  } else {
    float sumX = as_type<float>(0x00000000u);
    float sumY = as_type<float>(0x00000000u);
    float sumZ = as_type<float>(0x00000000u);
    int beg = offsets[((int)(int(__gid.x)))];
    int c1 = (((int)(int(__gid.x))) + 1);
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
          float iw = (as_type<float>(0x3f800000u) - w0);
          rhoF = ((w0 * rhoCell[o]) + (iw * rhoCell[n]));
          ufx = ((w0 * ux[o]) + (iw * ux[n]));
          ufy = ((w0 * uy[o]) + (iw * uy[n]));
          ufz = ((w0 * uz[o]) + (iw * uz[n]));
        } else {
          uint rm = rhoBcMask[f];
          if (rm == 1u) {
            float w0 = weights[f];
            float iw = (as_type<float>(0x3f800000u) - w0);
            rhoF = ((w0 * rhoCell[o]) + (iw * rhoBcVal[f]));
          }
          if (rm == 2u) {
            rhoF = rhoBcVal[f];
          }
          uint um = uBcMask[f];
          if (um == 1u) {
            float w0 = weights[f];
            float iw = (as_type<float>(0x3f800000u) - w0);
            ufx = ((w0 * ux[o]) + (iw * uBcX[f]));
            ufy = ((w0 * uy[o]) + (iw * uBcY[f]));
            ufz = ((w0 * uz[o]) + (iw * uBcZ[f]));
          }
          if (um == 2u) {
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
    float v = vol[((int)(int(__gid.x)))];
    float invV = as_type<float>(0x00000000u);
    if (v != as_type<float>(0x00000000u)) {
      invV = (as_type<float>(0x3f800000u) / v);
    }
    outX[((int)(int(__gid.x)))] = (sumX * invV);
    outY[((int)(int(__gid.x)))] = (sumY * invV);
    outZ[((int)(int(__gid.x)))] = (sumZ * invV);
  }
}

)kernel";
const char* k_metal_navatala_cfd_hby_a_from_u_and_terms = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_cfd_hby_a_from_u_and_terms(device const float* ux [[buffer(0)]], device const float* uy [[buffer(1)]], device const float* uz [[buffer(2)]], device const float* lapx [[buffer(3)]], device const float* lapy [[buffer(4)]], device const float* lapz [[buffer(5)]], device const float* divx [[buffer(6)]], device const float* divy [[buffer(7)]], device const float* divz [[buffer(8)]], device const float* extrax [[buffer(9)]], device const float* extray [[buffer(10)]], device const float* extraz [[buffer(11)]], device const float* rau [[buffer(12)]], device const int* counts [[buffer(13)]], device const int* modeBuf [[buffer(14)]], device float* outx [[buffer(15)]], device float* outy [[buffer(16)]], device float* outz [[buffer(17)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  if (((int)(int(__gid.x))) >= counts[0]) {
    return;
  } else {
    float tx = lapx[((int)(int(__gid.x)))];
    float ty = lapy[((int)(int(__gid.x)))];
    float tz = lapz[((int)(int(__gid.x)))];
    if ((modeBuf[0] == 1) || (modeBuf[0] == 3)) {
      tx = (tx - divx[((int)(int(__gid.x)))]);
      ty = (ty - divy[((int)(int(__gid.x)))]);
      tz = (tz - divz[((int)(int(__gid.x)))]);
    }
    if ((modeBuf[0] == 2) || (modeBuf[0] == 3)) {
      tx = (tx + extrax[((int)(int(__gid.x)))]);
      ty = (ty + extray[((int)(int(__gid.x)))]);
      tz = (tz + extraz[((int)(int(__gid.x)))]);
    }
    float a = rau[((int)(int(__gid.x)))];
    outx[((int)(int(__gid.x)))] = (ux[((int)(int(__gid.x)))] + (a * tx));
    outy[((int)(int(__gid.x)))] = (uy[((int)(int(__gid.x)))] + (a * ty));
    outz[((int)(int(__gid.x)))] = (uz[((int)(int(__gid.x)))] + (a * tz));
  }
}

)kernel";
const char* k_metal_navatala_cfd_apply_sp_to_hby_a = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_cfd_apply_sp_to_hby_a(device float* hx [[buffer(0)]], device float* hy [[buffer(1)]], device float* hz [[buffer(2)]], device const float* ux [[buffer(3)]], device const float* uy [[buffer(4)]], device const float* uz [[buffer(5)]], device const float* rAU [[buffer(6)]], device const float* sp [[buffer(7)]], device const int* counts [[buffer(8)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  if (((int)(int(__gid.x))) >= counts[0]) {
    return;
  } else {
    float fac = (sp[((int)(int(__gid.x)))] * rAU[((int)(int(__gid.x)))]);
    hx[((int)(int(__gid.x)))] = (hx[((int)(int(__gid.x)))] - (fac * ux[((int)(int(__gid.x)))]));
    hy[((int)(int(__gid.x)))] = (hy[((int)(int(__gid.x)))] - (fac * uy[((int)(int(__gid.x)))]));
    hz[((int)(int(__gid.x)))] = (hz[((int)(int(__gid.x)))] - (fac * uz[((int)(int(__gid.x)))]));
  }
}

)kernel";
const char* k_metal_navatala_cfd_laplacian_scalar_explicit = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_cfd_laplacian_scalar_explicit(device const float* muCell [[buffer(0)]], device const float* phiCell [[buffer(1)]], device const int* owner [[buffer(2)]], device const int* neighbour [[buffer(3)]], device const float* weights [[buffer(4)]], device const float* magSf [[buffer(5)]], device const float* deltaCoeffs [[buffer(6)]], device const int* offsets [[buffer(7)]], device const int* faceIdx [[buffer(8)]], device const int* sign [[buffer(9)]], device const float* vol [[buffer(10)]], device const float* bcMu [[buffer(11)]], device const uint* bcMuMask [[buffer(12)]], device const float* bcPhi [[buffer(13)]], device const uint* bcPhiMask [[buffer(14)]], device const int* params [[buffer(15)]], device float* outL [[buffer(16)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  if (((int)(int(__gid.x))) >= params[0]) {
    return;
  } else {
    float phiC = phiCell[((int)(int(__gid.x)))];
    float muC = muCell[((int)(int(__gid.x)))];
    float sum = as_type<float>(0x00000000u);
    int beg = offsets[((int)(int(__gid.x)))];
    int c1 = (((int)(int(__gid.x))) + 1);
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
            float iw = (as_type<float>(0x3f800000u) - w0);
            muF = ((w0 * muC) + (iw * muCell[n]));
          } else {
            if (bcPhiMask[f] != 0u) {
              phiOther = bcPhi[f];
            }
            if (bcMuMask[f] != 0u) {
              muF = bcMu[f];
            }
          }
        } else {
          float w0 = weights[f];
          float iw = (as_type<float>(0x3f800000u) - w0);
          muF = ((w0 * muCell[o]) + (iw * muC));
        }
        float coeff = (muF * (magSf[f] * deltaCoeffs[f]));
        sum = (sum + (coeff * (phiOther - phiC)));
      }
    }
    float v = vol[((int)(int(__gid.x)))];
    float invV = as_type<float>(0x00000000u);
    if (v != as_type<float>(0x00000000u)) {
      invV = (as_type<float>(0x3f800000u) / v);
    }
    outL[((int)(int(__gid.x)))] = (sum * invV);
  }
}

)kernel";
const char* k_metal_navatala_cfd_laplacian_vector_explicit = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_cfd_laplacian_vector_explicit(device const float* muCell [[buffer(0)]], device const float* muBc [[buffer(1)]], device const float* ux [[buffer(2)]], device const float* uy [[buffer(3)]], device const float* uz [[buffer(4)]], device const int* owner [[buffer(5)]], device const int* neighbour [[buffer(6)]], device const float* weights [[buffer(7)]], device const float* magSf [[buffer(8)]], device const float* deltaCoeffs [[buffer(9)]], device const float* corrX [[buffer(10)]], device const float* corrY [[buffer(11)]], device const float* corrZ [[buffer(12)]], device const float* gradRow0 [[buffer(13)]], device const float* gradRow1 [[buffer(14)]], device const float* gradRow2 [[buffer(15)]], device const int* offsets [[buffer(16)]], device const int* faceIdx [[buffer(17)]], device const int* sign [[buffer(18)]], device const float* vol [[buffer(19)]], device const float* bcX [[buffer(20)]], device const float* bcY [[buffer(21)]], device const float* bcZ [[buffer(22)]], device const int* bcMask [[buffer(23)]], device const int* params [[buffer(24)]], device float* outX [[buffer(25)]], device float* outY [[buffer(26)]], device float* outZ [[buffer(27)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  if (((int)(int(__gid.x))) >= params[0]) {
    return;
  } else {
    float ucx = ux[((int)(int(__gid.x)))];
    float ucy = uy[((int)(int(__gid.x)))];
    float ucz = uz[((int)(int(__gid.x)))];
    float muC = muCell[((int)(int(__gid.x)))];
    float sumX = as_type<float>(0x00000000u);
    float sumY = as_type<float>(0x00000000u);
    float sumZ = as_type<float>(0x00000000u);
    int beg = offsets[((int)(int(__gid.x)))];
    int c1 = (((int)(int(__gid.x))) + 1);
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
            float iw = (as_type<float>(0x3f800000u) - w0);
            muF = ((w0 * muC) + (iw * muCell[n]));
          } else {
            muF = muBc[f];
            if (bcMask[f] == 1) {
              float w0 = weights[f];
              float iw = (as_type<float>(0x3f800000u) - w0);
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
          float iw = (as_type<float>(0x3f800000u) - w0);
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
          float iw1 = (as_type<float>(0x3f800000u) - w1);
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
    float v = vol[((int)(int(__gid.x)))];
    float invV = as_type<float>(0x00000000u);
    if (v != as_type<float>(0x00000000u)) {
      invV = (as_type<float>(0x3f800000u) / v);
    }
    outX[((int)(int(__gid.x)))] = (sumX * invV);
    outY[((int)(int(__gid.x)))] = (sumY * invV);
    outZ[((int)(int(__gid.x)))] = (sumZ * invV);
  }
}

)kernel";
const char* k_metal_navatala_cfd_predictor_hby_a_from_source = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_cfd_predictor_hby_a_from_source(device const float* rAU [[buffer(0)]], device const float* sourceX [[buffer(1)]], device const float* sourceY [[buffer(2)]], device const float* sourceZ [[buffer(3)]], device const int* counts1 [[buffer(4)]], device float* outHbyAX [[buffer(5)]], device float* outHbyAY [[buffer(6)]], device float* outHbyAZ [[buffer(7)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  if (((int)(int(__gid.x))) >= counts1[0]) {
    return;
  } else {
    float r = rAU[((int)(int(__gid.x)))];
    outHbyAX[((int)(int(__gid.x)))] = (r * sourceX[((int)(int(__gid.x)))]);
    outHbyAY[((int)(int(__gid.x)))] = (r * sourceY[((int)(int(__gid.x)))]);
    outHbyAZ[((int)(int(__gid.x)))] = (r * sourceZ[((int)(int(__gid.x)))]);
  }
}

)kernel";
const char* k_metal_navatala_cfd_predictor_inverse_diag = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_cfd_predictor_inverse_diag(device const float* inputDiag [[buffer(0)]], device const int* counts1 [[buffer(1)]], device float* outRAU [[buffer(2)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  if (((int)(int(__gid.x))) >= counts1[0]) {
    return;
  } else {
    float a = inputDiag[((int)(int(__gid.x)))];
    if (a != as_type<float>(0x00000000u)) {
      outRAU[((int)(int(__gid.x)))] = (as_type<float>(0x3f800000u) / a);
    } else {
      outRAU[((int)(int(__gid.x)))] = as_type<float>(0x00000000u);
    }
  }
}

)kernel";
const char* k_metal_navatala_cfd_predictor_phi_hby_a_fuse = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_cfd_predictor_phi_hby_a_fuse(device const float* fluxHbyA [[buffer(0)]], device const float* rhoRauFace [[buffer(1)]], device const float* ddtCorr [[buffer(2)]], device const int* counts1 [[buffer(3)]], device float* outPhiHbyA [[buffer(4)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  if (((int)(int(__gid.x))) >= counts1[0]) {
    return;
  } else {
    float f = fluxHbyA[((int)(int(__gid.x)))];
    float r = rhoRauFace[((int)(int(__gid.x)))];
    float d = ddtCorr[((int)(int(__gid.x)))];
    outPhiHbyA[((int)(int(__gid.x)))] = (f + (r * d));
  }
}

)kernel";
const char* k_metal_navatala_cfd_axpy_cells = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_cfd_axpy_cells(device const float* x [[buffer(0)]], device const int* counts [[buffer(1)]], device const float* paramsF [[buffer(2)]], device float* y [[buffer(3)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  if (((int)(int(__gid.x))) >= counts[0]) {
    return;
  } else {
    float xVal = x[((int)(int(__gid.x)))];
    float yVal = y[((int)(int(__gid.x)))];
    float result = (yVal + (paramsF[0] * xVal));
    y[((int)(int(__gid.x)))] = result;
  }
}

)kernel";
const char* k_metal_navatala_cfd_ddt_diagonal = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_cfd_ddt_diagonal(device const float* rho [[buffer(0)]], device const float* vol [[buffer(1)]], device const int* counts [[buffer(2)]], device const float* paramsF [[buffer(3)]], device float* outDiag [[buffer(4)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  if (((int)(int(__gid.x))) >= counts[0]) {
    return;
  } else {
    float rhoVal = rho[((int)(int(__gid.x)))];
    float volVal = vol[((int)(int(__gid.x)))];
    float diag = ((rhoVal * volVal) * paramsF[0]);
    outDiag[((int)(int(__gid.x)))] = diag;
  }
}

)kernel";
const char* k_metal_navatala_cfd_div_upwind_coeffs = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_cfd_div_upwind_coeffs(device const float* weights [[buffer(0)]], device const float* flux [[buffer(1)]], device const int* counts [[buffer(2)]], device float* outLower [[buffer(3)]], device float* outUpper [[buffer(4)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  if (((int)(int(__gid.x))) >= counts[0]) {
    return;
  } else {
    float w = weights[((int)(int(__gid.x)))];
    float phi = flux[((int)(int(__gid.x)))];
    float low = ((as_type<float>(0x00000000u) - w) * phi);
    float up = (low + phi);
    outLower[((int)(int(__gid.x)))] = low;
    outUpper[((int)(int(__gid.x)))] = up;
  }
}

)kernel";
const char* k_metal_navatala_cfd_fvm_sp = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_cfd_fvm_sp(device const float* sp [[buffer(0)]], device const float* vol [[buffer(1)]], device const int* counts [[buffer(2)]], device float* outDiag [[buffer(3)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  if (((int)(int(__gid.x))) >= counts[0]) {
    return;
  } else {
    float spVal = sp[((int)(int(__gid.x)))];
    float volVal = vol[((int)(int(__gid.x)))];
    float contrib = (spVal * volVal);
    float prev = outDiag[((int)(int(__gid.x)))];
    float newDiag = (prev + contrib);
    outDiag[((int)(int(__gid.x)))] = newDiag;
  }
}

)kernel";
const char* k_metal_navatala_cfd_fvm_su_sp = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_cfd_fvm_su_sp(device const float* sp [[buffer(0)]], device const float* psi [[buffer(1)]], device const float* vol [[buffer(2)]], device const int* counts [[buffer(3)]], device float* outDiag [[buffer(4)]], device float* outSource [[buffer(5)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  if (((int)(int(__gid.x))) >= counts[0]) {
    return;
  } else {
    float spVal = sp[((int)(int(__gid.x)))];
    float psiVal = psi[((int)(int(__gid.x)))];
    float volVal = vol[((int)(int(__gid.x)))];
    float spVol = (spVal * volVal);
    if (spVal >= as_type<float>(0x00000000u)) {
      float prevDiag = outDiag[((int)(int(__gid.x)))];
      float newDiag = (prevDiag + spVol);
      outDiag[((int)(int(__gid.x)))] = newDiag;
    } else {
      float prevSrc = outSource[((int)(int(__gid.x)))];
      float contrib = (spVol * psiVal);
      float newSrc = (prevSrc - contrib);
      outSource[((int)(int(__gid.x)))] = newSrc;
    }
  }
}

)kernel";
const char* k_metal_navatala_cfd_laplacian_coeffs = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_cfd_laplacian_coeffs(device const float* gamma [[buffer(0)]], device const int* owner [[buffer(1)]], device const int* neighbour [[buffer(2)]], device const float* weights [[buffer(3)]], device const float* deltaCoeffs [[buffer(4)]], device const float* magSf [[buffer(5)]], device const int* counts [[buffer(6)]], device float* outUpper [[buffer(7)]], device float* outLower [[buffer(8)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  if (((int)(int(__gid.x))) >= counts[0]) {
    return;
  } else {
    float gammaF = gamma[((int)(int(__gid.x)))];
    float coeff = (deltaCoeffs[((int)(int(__gid.x)))] * (gammaF * magSf[((int)(int(__gid.x)))]));
    outUpper[((int)(int(__gid.x)))] = coeff;
    outLower[((int)(int(__gid.x)))] = coeff;
  }
}

)kernel";
const char* k_metal_navatala_cfd_laplacian_coeffs_accum = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_cfd_laplacian_coeffs_accum(device const float* gamma [[buffer(0)]], device const float* deltaCoeffs [[buffer(1)]], device const float* magSf [[buffer(2)]], device const int* counts [[buffer(3)]], device float* outUpper [[buffer(4)]], device float* outLower [[buffer(5)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  if (((int)(int(__gid.x))) >= counts[0]) {
    return;
  } else {
    float coeff = (deltaCoeffs[((int)(int(__gid.x)))] * (gamma[((int)(int(__gid.x)))] * magSf[((int)(int(__gid.x)))]));
    float prevUpper = outUpper[((int)(int(__gid.x)))];
    float prevLower = outLower[((int)(int(__gid.x)))];
    outUpper[((int)(int(__gid.x)))] = (prevUpper + coeff);
    outLower[((int)(int(__gid.x)))] = (prevLower + coeff);
  }
}

)kernel";
const char* k_metal_navatala_cfd_linear_upwind_correction_source = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_cfd_linear_upwind_correction_source(device const float* luFaceVals [[buffer(0)]], device const float* flux [[buffer(1)]], device const int* owner [[buffer(2)]], device const int* neighbour [[buffer(3)]], device const float* psi [[buffer(4)]], device const int* offsets [[buffer(5)]], device const int* faceIdx [[buffer(6)]], device const float* signF [[buffer(7)]], device const int* counts [[buffer(8)]], device float* outSource [[buffer(9)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  if (((int)(int(__gid.x))) >= counts[0]) {
    return;
  } else {
    float sum = as_type<float>(0x00000000u);
    int beg = offsets[((int)(int(__gid.x)))];
    int c1 = (((int)(int(__gid.x))) + 1);
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
    float prev = outSource[((int)(int(__gid.x)))];
    outSource[((int)(int(__gid.x)))] = (prev - sum);
  }
}

)kernel";
const char* k_metal_navatala_cfd_matrix_relax = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_cfd_matrix_relax(device const float* diagOld [[buffer(0)]], device const float* sumOff [[buffer(1)]], device const float* psi [[buffer(2)]], device const int* counts [[buffer(3)]], device const float* paramsF [[buffer(4)]], device float* diag [[buffer(5)]], device float* source [[buffer(6)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  if (((int)(int(__gid.x))) >= counts[0]) {
    return;
  } else {
    float d0 = diagOld[((int)(int(__gid.x)))];
    float sOff = sumOff[((int)(int(__gid.x)))];
    float x = psi[((int)(int(__gid.x)))];
    float negD0 = (as_type<float>(0x00000000u) - d0);
    float absD0 = (((d0 >= as_type<float>(0x00000000u))) ? (d0) : (negD0));
    float maxVal = (((absD0 >= sOff)) ? (absD0) : (sOff));
    float newDiag = (maxVal / paramsF[0]);
    float srcPrev = source[((int)(int(__gid.x)))];
    float delta = (newDiag - d0);
    float srcNew = (srcPrev + (delta * x));
    diag[((int)(int(__gid.x)))] = newDiag;
    source[((int)(int(__gid.x)))] = srcNew;
  }
}

)kernel";
const char* k_metal_navatala_cfd_neg_sum_diag = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_cfd_neg_sum_diag(device const float* upper [[buffer(0)]], device const float* lower [[buffer(1)]], device const int* offsets [[buffer(2)]], device const int* faceIdx [[buffer(3)]], device const int* sign [[buffer(4)]], device const int* counts [[buffer(5)]], device float* outDiag [[buffer(6)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  if (((int)(int(__gid.x))) >= counts[0]) {
    return;
  } else {
    float sum = as_type<float>(0x00000000u);
    int beg = offsets[((int)(int(__gid.x)))];
    int c1 = (((int)(int(__gid.x))) + 1);
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
    float prev = outDiag[((int)(int(__gid.x)))];
    outDiag[((int)(int(__gid.x)))] = (prev + sum);
  }
}

)kernel";
const char* k_metal_navatala_cfd_pack_boundary_adjacent_cells = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_cfd_pack_boundary_adjacent_cells(device const float* cellField [[buffer(0)]], device const int* indexMap [[buffer(1)]], device const int* counts [[buffer(2)]], device float* outPacked [[buffer(3)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  if (((int)(int(__gid.x))) >= counts[0]) {
    return;
  } else {
    int cellIdx = indexMap[((int)(int(__gid.x)))];
    float val = cellField[cellIdx];
    outPacked[((int)(int(__gid.x)))] = val;
  }
}

)kernel";
const char* k_metal_navatala_cfd_scalar_jacobi_clamp_min = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_cfd_scalar_jacobi_clamp_min(device const float* lowerBound [[buffer(0)]], device const uint* counts [[buffer(1)]], device float* x [[buffer(2)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  if (int(__gid.x) >= ((int)(counts[0]))) {
    return;
  } else {
    float lb = lowerBound[0];
    float xi = x[int(__gid.x)];
    if (xi > lb) {
      x[int(__gid.x)] = xi;
    } else {
      x[int(__gid.x)] = lb;
    }
  }
}

)kernel";
const char* k_metal_navatala_cfd_scalar_jacobi_update = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_cfd_scalar_jacobi_update(device const float* ax [[buffer(0)]], device const float* diag [[buffer(1)]], device const float* rhs [[buffer(2)]], device const float* params [[buffer(3)]], device const uint* counts [[buffer(4)]], device float* x [[buffer(5)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  if (int(__gid.x) >= ((int)(counts[0]))) {
    return;
  } else {
    float smallDiag = params[0];
    float omega = params[1];
    float d = diag[int(__gid.x)];
    float absD = abs(d);
    if (absD < smallDiag) {
      if (d >= as_type<float>(0x00000000u)) {
        d = smallDiag;
      } else {
        d = (-smallDiag);
      }
    }
    float xi = x[int(__gid.x)];
    float correction = (omega * ((rhs[int(__gid.x)] - ax[int(__gid.x)]) / d));
    x[int(__gid.x)] = (xi + correction);
  }
}

)kernel";
const char* k_metal_navatala_cfd_scalar_ldu_mat_vec = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_cfd_scalar_ldu_mat_vec(device const float* diag [[buffer(0)]], device const float* upper [[buffer(1)]], device const float* lower [[buffer(2)]], device const float* x [[buffer(3)]], device const int* owner [[buffer(4)]], device const int* neighbour [[buffer(5)]], device const uint* offsets [[buffer(6)]], device const uint* faceIdx [[buffer(7)]], device const int* sign [[buffer(8)]], device const uint* counts [[buffer(9)]], device float* ax [[buffer(10)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  if (int(__gid.x) >= ((int)(counts[0]))) {
    return;
  } else {
    float s = (diag[int(__gid.x)] * x[int(__gid.x)]);
    int beg = ((int)(offsets[int(__gid.x)]));
    int c1 = (int(__gid.x) + 1);
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
    ax[int(__gid.x)] = s;
  }
}

)kernel";
const char* k_metal_navatala_cfd_scalar_ldu_multi_color_gs_sweep = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_cfd_scalar_ldu_multi_color_gs_sweep(device const float* diag [[buffer(0)]], device const float* upper [[buffer(1)]], device const float* lower [[buffer(2)]], device const float* rhs [[buffer(3)]], device const int* owner [[buffer(4)]], device const int* neighbour [[buffer(5)]], device const uint* offsets [[buffer(6)]], device const uint* faceIdx [[buffer(7)]], device const int* sign [[buffer(8)]], device const uint* colorOrder [[buffer(9)]], device const uint* colorRange [[buffer(10)]], device const float* params [[buffer(11)]], device const uint* counts [[buffer(12)]], device float* x [[buffer(13)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  int colorSize = (((int)(colorRange[1])) - ((int)(colorRange[0])));
  if (int(__gid.x) >= colorSize) {
    return;
  } else {
    int cellIdx = (((int)(colorRange[0])) + int(__gid.x));
    uint celli = colorOrder[cellIdx];
    int ci = ((int)(celli));
    float smallDiag = params[0];
    float omega = params[1];
    float d = diag[ci];
    float absD = abs(d);
    if (absD < smallDiag) {
      if (d >= as_type<float>(0x00000000u)) {
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
const char* k_metal_navatala_cfd_sum_mag_off_diag = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_cfd_sum_mag_off_diag(device const float* upper [[buffer(0)]], device const float* lower [[buffer(1)]], device const int* offsets [[buffer(2)]], device const int* faceIdx [[buffer(3)]], device const int* sign [[buffer(4)]], device const int* counts [[buffer(5)]], device float* outSumOff [[buffer(6)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  if (((int)(int(__gid.x))) >= counts[0]) {
    return;
  } else {
    float sum = as_type<float>(0x00000000u);
    int beg = offsets[((int)(int(__gid.x)))];
    int c1 = (((int)(int(__gid.x))) + 1);
    int end = offsets[c1];
    int len = (end - beg);
    for (int t = 0; t < (int)(len); ++t) {
      int k = (beg + t);
      int f = faceIdx[k];
      if (f < counts[1]) {
        int sgn = sign[k];
        if (sgn > 0) {
          float val = upper[f];
          float negVal = (as_type<float>(0x00000000u) - val);
          float absVal = (((val >= as_type<float>(0x00000000u))) ? (val) : (negVal));
          sum = (sum + absVal);
        } else {
          float val = lower[f];
          float negVal = (as_type<float>(0x00000000u) - val);
          float absVal = (((val >= as_type<float>(0x00000000u))) ? (val) : (negVal));
          sum = (sum + absVal);
        }
      }
    }
    outSumOff[((int)(int(__gid.x)))] = sum;
  }
}

)kernel";
const char* k_metal_navatala_cfd_gamg_axpy_in_place = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_cfd_gamg_axpy_in_place(device const float* x [[buffer(0)]], device const float* alpha [[buffer(1)]], device const uint* nRows [[buffer(2)]], device float* y [[buffer(3)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  if (int(__gid.x) >= ((int)(nRows[0]))) {
    return;
  } else {
    float a = alpha[0];
    float xi = x[int(__gid.x)];
    float yi = y[int(__gid.x)];
    y[int(__gid.x)] = (yi + (a * xi));
  }
}

)kernel";
const char* k_metal_navatala_cfd_gamg_compute_scale_factor = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_cfd_gamg_compute_scale_factor(device const float* num [[buffer(0)]], device const float* den [[buffer(1)]], device float* sf [[buffer(2)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  if (int(__gid.x) >= 1) {
    return;
  } else {
    float n = num[0];
    float d = den[0];
    float dAbs = abs(d);
    if (dAbs > as_type<float>(0x0da24260u)) {
      sf[0] = (n / d);
    } else {
      sf[0] = as_type<float>(0x3f800000u);
    }
  }
}

)kernel";
const char* k_metal_navatala_cfd_gamg_dot_finalize = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_cfd_gamg_dot_finalize(device const float* partials [[buffer(0)]], device const uint* count [[buffer(1)]], device float* result [[buffer(2)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  if (int(__gid.x) >= 1) {
    return;
  } else {
    float s = as_type<float>(0x00000000u);
    for (int i = 0; i < (int)(((int)(count[0]))); ++i) {
      s = (s + partials[i]);
    }
    result[0] = s;
  }
}

)kernel";
const char* k_metal_navatala_cfd_gamg_negate_scalar = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_cfd_gamg_negate_scalar(device const float* _input [[buffer(0)]], device float* _output [[buffer(1)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  if (int(__gid.x) >= 1) {
    return;
  } else {
    float v = _input[0];
    _output[0] = (-v);
  }
}

)kernel";
const char* k_metal_navatala_cfd_gamg_scale_correction = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_cfd_gamg_scale_correction(device const float* z [[buffer(0)]], device const float* r [[buffer(1)]], device const float* Az [[buffer(2)]], device const float* diagInv [[buffer(3)]], device const float* sfPtr [[buffer(4)]], device const uint* nRows [[buffer(5)]], device float* corr [[buffer(6)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  if (int(__gid.x) >= ((int)(nRows[0]))) {
    return;
  } else {
    float sf = sfPtr[0];
    float zi = z[int(__gid.x)];
    float ri = r[int(__gid.x)];
    float azi = Az[int(__gid.x)];
    float di = diagInv[int(__gid.x)];
    float sfz = (sf * zi);
    float sfaz = (sf * azi);
    float residual = (ri - sfaz);
    float corrVal = (sfz + (residual * di));
    corr[int(__gid.x)] = corrVal;
  }
}

)kernel";
const char* k_metal_navatala_cfd_mg_coarse_jacobi = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_cfd_mg_coarse_jacobi(device float* x [[buffer(0)]], device const float* r [[buffer(1)]], device const float* diag [[buffer(2)]], device const int* mgCounts [[buffer(3)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  if (int(__gid.x) >= ((int)(mgCounts[1]))) {
    return;
  } else {
    float d = diag[int(__gid.x)];
    if (d == as_type<float>(0x00000000u)) {
      return;
    } else {
      float omega = as_type<float>(0x3f4ccccdu);
      x[int(__gid.x)] = (x[int(__gid.x)] + (omega * (r[int(__gid.x)] / d)));
    }
  }
}

)kernel";
const char* k_metal_navatala_cfd_mg_coarse_matvec = R"kernel(
#include <metal_stdlib>
using namespace metal;

static inline int gpu_atomic_cas_int(device atomic_int* ptr, int expected, int desired) {
  int exp = expected;
  while (!atomic_compare_exchange_weak_explicit(ptr, &exp, desired,
      memory_order_relaxed, memory_order_relaxed) && exp == expected) {}
  return exp;
}

static inline uint gpu_atomic_cas_uint(device atomic_uint* ptr, uint expected, uint desired) {
  uint exp = expected;
  while (!atomic_compare_exchange_weak_explicit(ptr, &exp, desired,
      memory_order_relaxed, memory_order_relaxed) && exp == expected) {}
  return exp;
}

kernel void navatala_cfd_mg_coarse_matvec(device const int* edgeU [[buffer(0)]], device const int* edgeV [[buffer(1)]], device const float* edgeCf [[buffer(2)]], device const float* diag [[buffer(3)]], device const float* x [[buffer(4)]], device float* outAx [[buffer(5)]], device const int* mgCounts [[buffer(6)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  int total = (((int)(mgCounts[1])) + ((int)(mgCounts[2])));
  if (int(__gid.x) >= total) {
    return;
  } else {
    if (int(__gid.x) < ((int)(mgCounts[1]))) {
      atomic_fetch_add_explicit((device atomic_float*)(&outAx[int(__gid.x)]), (diag[int(__gid.x)] * x[int(__gid.x)]), memory_order_relaxed);
    } else {
      int e = (int(__gid.x) - ((int)(mgCounts[1])));
      int u = edgeU[e];
      int v = edgeV[e];
      float cf = edgeCf[e];
      atomic_fetch_add_explicit((device atomic_float*)(&outAx[u]), (cf * x[v]), memory_order_relaxed);
      atomic_fetch_add_explicit((device atomic_float*)(&outAx[v]), (cf * x[u]), memory_order_relaxed);
    }
  }
}

)kernel";
const char* k_metal_navatala_cfd_mg_coarse_rhs = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_cfd_mg_coarse_rhs(device const float* b [[buffer(0)]], device const float* ax [[buffer(1)]], device float* r [[buffer(2)]], device const int* mgCounts [[buffer(3)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  if (int(__gid.x) >= ((int)(mgCounts[1]))) {
    return;
  } else {
    r[int(__gid.x)] = (b[int(__gid.x)] - ax[int(__gid.x)]);
  }
}

)kernel";
const char* k_metal_navatala_cfd_mg_find_best_match = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_cfd_mg_find_best_match(device const uint* csrOffsets [[buffer(0)]], device const uint* csrFaceIdx [[buffer(1)]], device const int* csrSign [[buffer(2)]], device const float* strength [[buffer(3)]], device const int* owner [[buffer(4)]], device const int* nei [[buffer(5)]], device int* bestNeighbor [[buffer(6)]], device const int* counts [[buffer(7)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  if (int(__gid.x) >= ((int)(counts[0]))) {
    return;
  } else {
    float bestS = as_type<float>(0xbf800000u);
    int bestN = -1;
    uint beg = csrOffsets[int(__gid.x)];
    int c1 = (int(__gid.x) + 1);
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
    bestNeighbor[int(__gid.x)] = bestN;
  }
}

)kernel";
const char* k_metal_navatala_cfd_mg_pair_match = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_cfd_mg_pair_match(device const int* bestNeighbor [[buffer(0)]], device int* aggMap [[buffer(1)]], device const int* counts [[buffer(2)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  if (int(__gid.x) >= ((int)(counts[0]))) {
    return;
  } else {
    int n = bestNeighbor[int(__gid.x)];
    if (n >= 0) {
      if (int(__gid.x) < n) {
        aggMap[int(__gid.x)] = int(__gid.x);
      } else {
        aggMap[int(__gid.x)] = n;
      }
    } else {
      aggMap[int(__gid.x)] = int(__gid.x);
    }
  }
}

)kernel";
const char* k_metal_navatala_cfd_mg_prolong_add = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_cfd_mg_prolong_add(device const float* xCoarse [[buffer(0)]], device const int* prolongRowOffsets [[buffer(1)]], device const int* prolongCols [[buffer(2)]], device const float* prolongVals [[buffer(3)]], device float* zFine [[buffer(4)]], device const int* mgCounts [[buffer(5)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  if (int(__gid.x) >= ((int)(mgCounts[0]))) {
    return;
  } else {
    float accum = as_type<float>(0x00000000u);
    int beg = prolongRowOffsets[int(__gid.x)];
    int c1 = (int(__gid.x) + 1);
    int end = prolongRowOffsets[c1];
    int len = (end - beg);
    for (int t = 0; t < (int)(len); ++t) {
      int k = (beg + t);
      int col = prolongCols[k];
      accum = (accum + (prolongVals[k] * xCoarse[col]));
    }
    zFine[int(__gid.x)] = (zFine[int(__gid.x)] + accum);
  }
}

)kernel";
const char* k_metal_navatala_cfd_mg_renumber_aggregates = R"kernel(
#include <metal_stdlib>
using namespace metal;

static inline int gpu_atomic_cas_int(device atomic_int* ptr, int expected, int desired) {
  int exp = expected;
  while (!atomic_compare_exchange_weak_explicit(ptr, &exp, desired,
      memory_order_relaxed, memory_order_relaxed) && exp == expected) {}
  return exp;
}

static inline uint gpu_atomic_cas_uint(device atomic_uint* ptr, uint expected, uint desired) {
  uint exp = expected;
  while (!atomic_compare_exchange_weak_explicit(ptr, &exp, desired,
      memory_order_relaxed, memory_order_relaxed) && exp == expected) {}
  return exp;
}

kernel void navatala_cfd_mg_renumber_aggregates(device int* aggMap [[buffer(0)]], device atomic_int* counter [[buffer(1)]], device const int* counts [[buffer(2)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  if (int(__gid.x) >= ((int)(counts[0]))) {
    return;
  } else {
    if (aggMap[int(__gid.x)] == int(__gid.x)) {
      int oldCount = atomic_fetch_add_explicit(((device atomic_int*)(&(counter[0]))), 1, memory_order_relaxed);
      aggMap[int(__gid.x)] = (-(oldCount + 1));
    }
  }
}

)kernel";
const char* k_metal_navatala_cfd_mg_renumber_propagate = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_cfd_mg_renumber_propagate(device int* aggMap [[buffer(0)]], device const int* counts [[buffer(1)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  if (int(__gid.x) >= ((int)(counts[0]))) {
    return;
  } else {
    int root = aggMap[int(__gid.x)];
    if (root < 0) {
      aggMap[int(__gid.x)] = (-(root + 1));
    } else {
      int rootVal = aggMap[root];
      if (rootVal < 0) {
        aggMap[int(__gid.x)] = (-(rootVal + 1));
      } else {
        aggMap[int(__gid.x)] = rootVal;
      }
    }
  }
}

)kernel";
const char* k_metal_navatala_cfd_mg_resolve_chains = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_cfd_mg_resolve_chains(device int* aggMap [[buffer(0)]], device const int* counts [[buffer(1)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  if (int(__gid.x) >= ((int)(counts[0]))) {
    return;
  } else {
    int root = aggMap[int(__gid.x)];
    if (root >= 0) {
      if (root < ((int)(counts[0]))) {
        if (root != int(__gid.x)) {
          int next = aggMap[root];
          if (next >= 0) {
            if (next < ((int)(counts[0]))) {
              aggMap[int(__gid.x)] = next;
            }
          }
        }
      }
    }
  }
}

)kernel";
const char* k_metal_navatala_cfd_mg_restrict_sum = R"kernel(
#include <metal_stdlib>
using namespace metal;

static inline int gpu_atomic_cas_int(device atomic_int* ptr, int expected, int desired) {
  int exp = expected;
  while (!atomic_compare_exchange_weak_explicit(ptr, &exp, desired,
      memory_order_relaxed, memory_order_relaxed) && exp == expected) {}
  return exp;
}

static inline uint gpu_atomic_cas_uint(device atomic_uint* ptr, uint expected, uint desired) {
  uint exp = expected;
  while (!atomic_compare_exchange_weak_explicit(ptr, &exp, desired,
      memory_order_relaxed, memory_order_relaxed) && exp == expected) {}
  return exp;
}

kernel void navatala_cfd_mg_restrict_sum(device const float* rFine [[buffer(0)]], device const int* prolongRowOffsets [[buffer(1)]], device const int* prolongCols [[buffer(2)]], device const float* prolongVals [[buffer(3)]], device float* rCoarse [[buffer(4)]], device const int* mgCounts [[buffer(5)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  if (int(__gid.x) >= ((int)(mgCounts[0]))) {
    return;
  } else {
    int beg = prolongRowOffsets[int(__gid.x)];
    int c1 = (int(__gid.x) + 1);
    int end = prolongRowOffsets[c1];
    float v = rFine[int(__gid.x)];
    int len = (end - beg);
    for (int t = 0; t < (int)(len); ++t) {
      int k = (beg + t);
      int col = prolongCols[k];
      atomic_fetch_add_explicit((device atomic_float*)(&rCoarse[col]), (prolongVals[k] * v), memory_order_relaxed);
    }
  }
}

)kernel";
const char* k_metal_navatala_cfd_mg_strength_of_connection = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_cfd_mg_strength_of_connection(device const float* diag [[buffer(0)]], device const float* upper [[buffer(1)]], device const int* owner [[buffer(2)]], device const int* nei [[buffer(3)]], device float* strength [[buffer(4)]], device const int* counts [[buffer(5)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  if (int(__gid.x) >= ((int)(counts[1]))) {
    return;
  } else {
    int o = owner[int(__gid.x)];
    int n = nei[int(__gid.x)];
    float dO = abs(diag[o]);
    float dN = abs(diag[n]);
    float denomSq = (dO * dN);
    if (denomSq > as_type<float>(0x0da24260u)) {
      strength[int(__gid.x)] = (abs(upper[int(__gid.x)]) / sqrt(denomSq));
    } else {
      strength[int(__gid.x)] = as_type<float>(0x00000000u);
    }
  }
}

)kernel";
const char* k_metal_navatala_cfd_mg_zero_coarse = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_cfd_mg_zero_coarse(device float* x [[buffer(0)]], device const int* mgCounts [[buffer(1)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  if (int(__gid.x) >= ((int)(mgCounts[1]))) {
    return;
  } else {
    x[int(__gid.x)] = as_type<float>(0x00000000u);
  }
}

)kernel";
const char* k_metal_navatala_cfd_phi_correct = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_cfd_phi_correct(device const float* phiIn [[buffer(0)]], device const float* pCell [[buffer(1)]], device const int* owner [[buffer(2)]], device const int* neighbour [[buffer(3)]], device const float* magSf [[buffer(4)]], device const float* deltaCoeffs [[buffer(5)]], device const float* coeff [[buffer(6)]], device const float* bcVal [[buffer(7)]], device const int* bcMask [[buffer(8)]], device const float* bcSnGrad [[buffer(9)]], device const int* bcSnGradMask [[buffer(10)]], device const float* faceFluxCorrection [[buffer(11)]], device const int* counts [[buffer(12)]], device const float* paramsF [[buffer(13)]], device float* phiOut [[buffer(14)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  if (((int)(int(__gid.x))) >= counts[0]) {
    return;
  } else {
    int o = owner[((int)(int(__gid.x)))];
    float po = pCell[o];
    float other = po;
    float snGrad = as_type<float>(0x00000000u);
    if (((int)(int(__gid.x))) < counts[1]) {
      int n = neighbour[((int)(int(__gid.x)))];
      other = pCell[n];
    } else {
      if (bcMask[((int)(int(__gid.x)))] != 0) {
        other = bcVal[((int)(int(__gid.x)))];
      }
    }
    snGrad = (deltaCoeffs[((int)(int(__gid.x)))] * (other - po));
    if (bcSnGradMask[((int)(int(__gid.x)))] != 0) {
      snGrad = bcSnGrad[((int)(int(__gid.x)))];
    }
    float gradFace = (magSf[((int)(int(__gid.x)))] * snGrad);
    phiOut[((int)(int(__gid.x)))] = (phiIn[((int)(int(__gid.x)))] - ((paramsF[0] * (coeff[((int)(int(__gid.x)))] * gradFace)) + faceFluxCorrection[((int)(int(__gid.x)))]));
  }
}

)kernel";
const char* k_metal_navatala_cfd_phi_correct_boundary = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_cfd_phi_correct_boundary(device const float* phiIn [[buffer(0)]], device const float* pCell [[buffer(1)]], device const int* owner [[buffer(2)]], device const float* magSf [[buffer(3)]], device const float* deltaCoeffs [[buffer(4)]], device const float* coeff [[buffer(5)]], device const float* bcVal [[buffer(6)]], device const int* bcMask [[buffer(7)]], device const float* bcSnGrad [[buffer(8)]], device const int* bcSnGradMask [[buffer(9)]], device const float* faceFluxCorrection [[buffer(10)]], device const int* counts [[buffer(11)]], device const float* paramsF [[buffer(12)]], device float* phiOut [[buffer(13)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  int f = (counts[1] + ((int)(int(__gid.x))));
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
const char* k_metal_navatala_cfd_phi_correct_internal = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_cfd_phi_correct_internal(device const float* phiIn [[buffer(0)]], device const float* pCell [[buffer(1)]], device const int* owner [[buffer(2)]], device const int* neighbour [[buffer(3)]], device const float* magSf [[buffer(4)]], device const float* deltaCoeffs [[buffer(5)]], device const float* coeff [[buffer(6)]], device const float* faceFluxCorrection [[buffer(7)]], device const int* counts [[buffer(8)]], device const float* paramsF [[buffer(9)]], device float* phiOut [[buffer(10)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  if (((int)(int(__gid.x))) >= counts[1]) {
    return;
  } else {
    int o = owner[((int)(int(__gid.x)))];
    int n = neighbour[((int)(int(__gid.x)))];
    float po = pCell[o];
    float pn = pCell[n];
    float snGrad = (deltaCoeffs[((int)(int(__gid.x)))] * (pn - po));
    float gradFace = (magSf[((int)(int(__gid.x)))] * snGrad);
    phiOut[((int)(int(__gid.x)))] = (phiIn[((int)(int(__gid.x)))] - ((paramsF[0] * (coeff[((int)(int(__gid.x)))] * gradFace)) + faceFluxCorrection[((int)(int(__gid.x)))]));
  }
}

)kernel";
const char* k_metal_navatala_cfd_axpy_in_place = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_cfd_axpy_in_place(device const float* x [[buffer(0)]], device float* y [[buffer(1)]], device const int* counts [[buffer(2)]], device const float* paramsF [[buffer(3)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  if (int(__gid.x) >= counts[0]) {
    return;
  } else {
    float yi = y[int(__gid.x)];
    y[int(__gid.x)] = (yi + (paramsF[0] * x[int(__gid.x)]));
  }
}

)kernel";
const char* k_metal_navatala_cfd_bc_dirichlet_face_flux = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_cfd_bc_dirichlet_face_flux(device const float* cf [[buffer(0)]], device const float* bcVal [[buffer(1)]], device const int* bcMask [[buffer(2)]], device const int* counts [[buffer(3)]], device float* outFlux [[buffer(4)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  if (((int)(int(__gid.x))) >= counts[1]) {
    return;
  } else {
    float flux = as_type<float>(0x00000000u);
    if (bcMask[((int)(int(__gid.x)))] == 2) {
      flux = (cf[((int)(int(__gid.x)))] * bcVal[((int)(int(__gid.x)))]);
    }
    outFlux[((int)(int(__gid.x)))] = flux;
  }
}

)kernel";
const char* k_metal_navatala_cfd_bc_sn_grad_face_flux = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_cfd_bc_sn_grad_face_flux(device const float* cf [[buffer(0)]], device const float* delta [[buffer(1)]], device const float* bcSnGrad [[buffer(2)]], device const int* bcSnGradMask [[buffer(3)]], device const int* counts [[buffer(4)]], device float* outFlux [[buffer(5)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  if (((int)(int(__gid.x))) >= counts[1]) {
    return;
  } else {
    float d = delta[((int)(int(__gid.x)))];
    float sg = bcSnGrad[((int)(int(__gid.x)))];
    float flux = as_type<float>(0x00000000u);
    if (bcSnGradMask[((int)(int(__gid.x)))] != 0) {
      if (d != as_type<float>(0x00000000u)) {
        flux = (cf[((int)(int(__gid.x)))] * (sg / d));
      }
    }
    outFlux[((int)(int(__gid.x)))] = flux;
  }
}

)kernel";
const char* k_metal_navatala_cfd_coeff_to_cf_in_place = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_cfd_coeff_to_cf_in_place(device float* cf [[buffer(0)]], device const float* magSf [[buffer(1)]], device const float* delta [[buffer(2)]], device const int* counts [[buffer(3)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  if (((int)(int(__gid.x))) >= counts[1]) {
    return;
  } else {
    float v = cf[((int)(int(__gid.x)))];
    cf[((int)(int(__gid.x)))] = ((v * magSf[((int)(int(__gid.x)))]) * delta[((int)(int(__gid.x)))]);
  }
}

)kernel";
const char* k_metal_navatala_cfd_diag_from_cf = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_cfd_diag_from_cf(device const float* cf [[buffer(0)]], device const int* offsets [[buffer(1)]], device const int* faceIdx [[buffer(2)]], device const float* vol [[buffer(3)]], device const int* counts [[buffer(4)]], device float* diag [[buffer(5)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  if (int(__gid.x) >= counts[0]) {
    return;
  } else {
    int beg = offsets[int(__gid.x)];
    int c1 = (int(__gid.x) + 1);
    int end = offsets[c1];
    int len = (end - beg);
    float s = as_type<float>(0x00000000u);
    for (int t = 0; t < (int)(len); ++t) {
      int k = (beg + t);
      int f = faceIdx[k];
      s = (s + cf[f]);
    }
    diag[int(__gid.x)] = (as_type<float>(0x00000000u) - s);
  }
}

)kernel";
const char* k_metal_navatala_cfd_dic_apply = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_cfd_dic_apply(device const float* r [[buffer(0)]], device const float* rD [[buffer(1)]], device const float* upper [[buffer(2)]], device const int* owner [[buffer(3)]], device const int* neighbour [[buffer(4)]], device const int* counts [[buffer(5)]], device float* z [[buffer(6)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
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
const char* k_metal_navatala_cfd_dic_build_reciprocal_d = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_cfd_dic_build_reciprocal_d(device const float* diag [[buffer(0)]], device const float* upper [[buffer(1)]], device const int* owner [[buffer(2)]], device const int* neighbour [[buffer(3)]], device const int* counts [[buffer(4)]], device float* rD [[buffer(5)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
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
    if (v != as_type<float>(0x00000000u)) {
      rD[c2] = (as_type<float>(0x3f800000u) / v);
    } else {
      rD[c2] = as_type<float>(0x00000000u);
    }
  }
}

)kernel";
const char* k_metal_navatala_cfd_dot_partials = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_cfd_dot_partials(device const float* a [[buffer(0)]], device const float* b [[buffer(1)]], device const int* counts [[buffer(2)]], device float* outPartials [[buffer(3)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  threadgroup float tmp[256];
  float v = as_type<float>(0x00000000u);
  if (((int)(int(__gid.x))) < counts[0]) {
    v = (a[((int)(int(__gid.x)))] * b[((int)(int(__gid.x)))]);
  }
  tmp[((int)(int(__tid.x)))] = v;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (((int)(int(__tid.x))) < 128) {
    tmp[((int)(int(__tid.x)))] = (tmp[((int)(int(__tid.x)))] + tmp[(((int)(int(__tid.x))) + 128)]);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (((int)(int(__tid.x))) < 64) {
    tmp[((int)(int(__tid.x)))] = (tmp[((int)(int(__tid.x)))] + tmp[(((int)(int(__tid.x))) + 64)]);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (((int)(int(__tid.x))) < 32) {
    tmp[((int)(int(__tid.x)))] = (tmp[((int)(int(__tid.x)))] + tmp[(((int)(int(__tid.x))) + 32)]);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (((int)(int(__tid.x))) < 16) {
    tmp[((int)(int(__tid.x)))] = (tmp[((int)(int(__tid.x)))] + tmp[(((int)(int(__tid.x))) + 16)]);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (((int)(int(__tid.x))) < 8) {
    tmp[((int)(int(__tid.x)))] = (tmp[((int)(int(__tid.x)))] + tmp[(((int)(int(__tid.x))) + 8)]);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (((int)(int(__tid.x))) < 4) {
    tmp[((int)(int(__tid.x)))] = (tmp[((int)(int(__tid.x)))] + tmp[(((int)(int(__tid.x))) + 4)]);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (((int)(int(__tid.x))) < 2) {
    tmp[((int)(int(__tid.x)))] = (tmp[((int)(int(__tid.x)))] + tmp[(((int)(int(__tid.x))) + 2)]);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (((int)(int(__tid.x))) < 1) {
    tmp[((int)(int(__tid.x)))] = (tmp[((int)(int(__tid.x)))] + tmp[(((int)(int(__tid.x))) + 1)]);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (((int)(int(__tid.x))) == 0) {
    outPartials[((int)(int(__tgid.x)))] = tmp[0];
  }
}

)kernel";
const char* k_metal_navatala_cfd_face_flux = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_cfd_face_flux(device const float* x [[buffer(0)]], device const int* owner [[buffer(1)]], device const int* neighbour [[buffer(2)]], device const float* cf [[buffer(3)]], device const float* bcVal [[buffer(4)]], device const int* bcMask [[buffer(5)]], device const int* counts [[buffer(6)]], device float* outFlux [[buffer(7)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  if (((int)(int(__gid.x))) >= counts[1]) {
    return;
  } else {
    int o = owner[((int)(int(__gid.x)))];
    float xo = x[o];
    float other = xo;
    if (((int)(int(__gid.x))) < counts[2]) {
      int n = neighbour[((int)(int(__gid.x)))];
      other = x[n];
    } else {
      int m = bcMask[((int)(int(__gid.x)))];
      if ((m == 1) || (m == 3)) {
        other = bcVal[((int)(int(__gid.x)))];
      } else {
        if (m == 2) {
          other = as_type<float>(0x00000000u);
        } else {
          if (m == 4) {
            outFlux[((int)(int(__gid.x)))] = bcVal[((int)(int(__gid.x)))];
            return;
          }
        }
      }
    }
    outFlux[((int)(int(__gid.x)))] = (cf[((int)(int(__gid.x)))] * (other - xo));
  }
}

)kernel";
const char* k_metal_navatala_cfd_inv_diag = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_cfd_inv_diag(device const float* r [[buffer(0)]], device const float* diag [[buffer(1)]], device const int* counts [[buffer(2)]], device float* z [[buffer(3)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  if (int(__gid.x) >= counts[0]) {
    return;
  } else {
    float d = diag[int(__gid.x)];
    float v = r[int(__gid.x)];
    if (d != as_type<float>(0x00000000u)) {
      z[int(__gid.x)] = (v / d);
    } else {
      z[int(__gid.x)] = v;
    }
  }
}

)kernel";
const char* k_metal_navatala_cfd_laplacian_from_face_flux = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_cfd_laplacian_from_face_flux(device const float* faceFlux [[buffer(0)]], device const int* offsets [[buffer(1)]], device const int* faceIdx [[buffer(2)]], device const float* signF [[buffer(3)]], device const float* vol [[buffer(4)]], device const int* counts [[buffer(5)]], device float* outLap [[buffer(6)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  if (int(__gid.x) >= counts[0]) {
    return;
  } else {
    int beg = offsets[int(__gid.x)];
    int c1 = (int(__gid.x) + 1);
    int end = offsets[c1];
    int len = (end - beg);
    float s = as_type<float>(0x00000000u);
    for (int t = 0; t < (int)(len); ++t) {
      int k = (beg + t);
      int f = faceIdx[k];
      s = (s + (signF[k] * faceFlux[f]));
    }
    outLap[int(__gid.x)] = s;
  }
}

)kernel";
const char* k_metal_navatala_cfd_mul_by_vol_in_place = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_cfd_mul_by_vol_in_place(device float* b [[buffer(0)]], device const float* vol [[buffer(1)]], device const int* counts [[buffer(2)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  if (((int)(int(__gid.x))) >= counts[0]) {
    return;
  } else {
    float v = b[((int)(int(__gid.x)))];
    b[((int)(int(__gid.x)))] = (v * vol[((int)(int(__gid.x)))]);
  }
}

)kernel";
const char* k_metal_navatala_cfd_ref_add_ax = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_cfd_ref_add_ax(device const float* diag [[buffer(0)]], device const float* x [[buffer(1)]], device float* outAx [[buffer(2)]], device const int* refCell [[buffer(3)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  int c = refCell[0];
  if (c < 0) {
    return;
  } else {
    float cur = outAx[c];
    outAx[c] = (cur + (diag[c] * x[c]));
  }
}

)kernel";
const char* k_metal_navatala_cfd_ref_add_b = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_cfd_ref_add_b(device const float* diag [[buffer(0)]], device float* b [[buffer(1)]], device const int* refCell [[buffer(2)]], device const float* refVal [[buffer(3)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  int c = refCell[0];
  if (c < 0) {
    return;
  } else {
    float cur = b[c];
    b[c] = (cur + (diag[c] * refVal[0]));
  }
}

)kernel";
const char* k_metal_navatala_cfd_rhs_sub = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_cfd_rhs_sub(device const float* b [[buffer(0)]], device const float* ax [[buffer(1)]], device const int* counts [[buffer(2)]], device float* r [[buffer(3)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  if (int(__gid.x) >= counts[0]) {
    return;
  } else {
    r[int(__gid.x)] = (b[int(__gid.x)] - ax[int(__gid.x)]);
  }
}

)kernel";
const char* k_metal_navatala_cfd_shift_in_place = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_cfd_shift_in_place(device float* x [[buffer(0)]], device const int* counts [[buffer(1)]], device const float* paramsF [[buffer(2)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  if (((int)(int(__gid.x))) >= counts[0]) {
    return;
  } else {
    float xi = x[((int)(int(__gid.x)))];
    x[((int)(int(__gid.x)))] = (xi + paramsF[0]);
  }
}

)kernel";
const char* k_metal_navatala_cfd_sum_abs_partials = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_cfd_sum_abs_partials(device const float* a [[buffer(0)]], device const int* counts [[buffer(1)]], device float* outPartials [[buffer(2)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  threadgroup float tmp[256];
  float v = as_type<float>(0x00000000u);
  if (((int)(int(__gid.x))) < counts[0]) {
    float ai = a[((int)(int(__gid.x)))];
    if (ai >= as_type<float>(0x00000000u)) {
      v = ai;
    } else {
      v = (as_type<float>(0x00000000u) - ai);
    }
  }
  tmp[((int)(int(__tid.x)))] = v;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (((int)(int(__tid.x))) < 128) {
    tmp[((int)(int(__tid.x)))] = (tmp[((int)(int(__tid.x)))] + tmp[(((int)(int(__tid.x))) + 128)]);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (((int)(int(__tid.x))) < 64) {
    tmp[((int)(int(__tid.x)))] = (tmp[((int)(int(__tid.x)))] + tmp[(((int)(int(__tid.x))) + 64)]);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (((int)(int(__tid.x))) < 32) {
    tmp[((int)(int(__tid.x)))] = (tmp[((int)(int(__tid.x)))] + tmp[(((int)(int(__tid.x))) + 32)]);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (((int)(int(__tid.x))) < 16) {
    tmp[((int)(int(__tid.x)))] = (tmp[((int)(int(__tid.x)))] + tmp[(((int)(int(__tid.x))) + 16)]);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (((int)(int(__tid.x))) < 8) {
    tmp[((int)(int(__tid.x)))] = (tmp[((int)(int(__tid.x)))] + tmp[(((int)(int(__tid.x))) + 8)]);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (((int)(int(__tid.x))) < 4) {
    tmp[((int)(int(__tid.x)))] = (tmp[((int)(int(__tid.x)))] + tmp[(((int)(int(__tid.x))) + 4)]);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (((int)(int(__tid.x))) < 2) {
    tmp[((int)(int(__tid.x)))] = (tmp[((int)(int(__tid.x)))] + tmp[(((int)(int(__tid.x))) + 2)]);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (((int)(int(__tid.x))) < 1) {
    tmp[((int)(int(__tid.x)))] = (tmp[((int)(int(__tid.x)))] + tmp[(((int)(int(__tid.x))) + 1)]);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (((int)(int(__tid.x))) == 0) {
    outPartials[((int)(int(__tgid.x)))] = tmp[0];
  }
}

)kernel";
const char* k_metal_navatala_cfd_upper_from_cf = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_cfd_upper_from_cf(device const float* cf [[buffer(0)]], device const int* counts [[buffer(1)]], device float* upper [[buffer(2)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  if (((int)(int(__gid.x))) >= counts[2]) {
    return;
  } else {
    upper[((int)(int(__gid.x)))] = cf[((int)(int(__gid.x)))];
  }
}

)kernel";
const char* k_metal_navatala_cfd_xpay_in_place = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_cfd_xpay_in_place(device const float* z [[buffer(0)]], device const float* pIn [[buffer(1)]], device float* pOut [[buffer(2)]], device const int* counts [[buffer(3)]], device const float* paramsF [[buffer(4)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  if (int(__gid.x) >= counts[0]) {
    return;
  } else {
    pOut[int(__gid.x)] = (z[int(__gid.x)] + (paramsF[0] * pIn[int(__gid.x)]));
  }
}

)kernel";
const char* k_metal_navatala_cfd_negate_scalar = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_cfd_negate_scalar(device const float* _input [[buffer(0)]], device float* _output [[buffer(1)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  if (int(__gid.x) >= 1) {
    return;
  } else {
    float v = _input[0];
    _output[0] = (-v);
  }
}

)kernel";
const char* k_metal_navatala_cfd_exact_mat_vec = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_cfd_exact_mat_vec(device const float* x [[buffer(0)]], device const int* owner [[buffer(1)]], device const int* neighbour [[buffer(2)]], device const uint* offsets [[buffer(3)]], device const uint* faceIdx [[buffer(4)]], device const int* sign [[buffer(5)]], device const float* upper [[buffer(6)]], device const float* lower [[buffer(7)]], device const float* diag [[buffer(8)]], device const int* counts [[buffer(9)]], device float* outAx [[buffer(10)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  if (int(__gid.x) >= ((int)(counts[0]))) {
    return;
  } else {
    float s = (diag[int(__gid.x)] * x[int(__gid.x)]);
    uint beg = offsets[int(__gid.x)];
    int c1 = (int(__gid.x) + 1);
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
    outAx[int(__gid.x)] = s;
  }
}

)kernel";
const char* k_metal_navatala_cfd_face_flux_boundary = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_cfd_face_flux_boundary(device const float* x [[buffer(0)]], device const int* owner [[buffer(1)]], device const float* cf [[buffer(2)]], device const float* bcVal [[buffer(3)]], device const uint* bcMask [[buffer(4)]], device const uint* params [[buffer(5)]], device float* faceFlux [[buffer(6)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  if (int(__gid.x) >= ((int)(params[0]))) {
    return;
  } else {
    int gf = (((int)(params[1])) + int(__gid.x));
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
            other = as_type<float>(0x00000000u);
          }
        }
      }
      faceFlux[gf] = (cf[gf] * (other - xo));
    }
  }
}

)kernel";
const char* k_metal_navatala_cfd_face_flux_internal = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_cfd_face_flux_internal(device const float* x [[buffer(0)]], device const int* owner [[buffer(1)]], device const int* neighbour [[buffer(2)]], device const float* cf [[buffer(3)]], device const uint* params [[buffer(4)]], device float* faceFlux [[buffer(5)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  if (int(__gid.x) >= ((int)(params[0]))) {
    return;
  } else {
    int o = owner[int(__gid.x)];
    int n = neighbour[int(__gid.x)];
    faceFlux[int(__gid.x)] = (cf[int(__gid.x)] * (x[n] - x[o]));
  }
}

)kernel";
const char* k_metal_navatala_cfd_pressure_face_flux_correction_corrected = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_cfd_pressure_face_flux_correction_corrected(device const int* owner [[buffer(0)]], device const int* neighbour [[buffer(1)]], device const float* weights [[buffer(2)]], device const float* gradX [[buffer(3)]], device const float* gradY [[buffer(4)]], device const float* gradZ [[buffer(5)]], device const float* magSf [[buffer(6)]], device const float* dCorrX [[buffer(7)]], device const float* dCorrY [[buffer(8)]], device const float* dCorrZ [[buffer(9)]], device const float* rAUf [[buffer(10)]], device const uint* params [[buffer(11)]], device float* outFaceFlux [[buffer(12)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  if (int(__gid.x) >= ((int)(params[0]))) {
    return;
  } else {
    if (int(__gid.x) >= ((int)(params[1]))) {
      outFaceFlux[int(__gid.x)] = as_type<float>(0x00000000u);
    } else {
      int own = owner[int(__gid.x)];
      int nei = neighbour[int(__gid.x)];
      float w = weights[int(__gid.x)];
      float omw = (as_type<float>(0x3f800000u) - w);
      float gx = ((w * gradX[own]) + (omw * gradX[nei]));
      float gy = ((w * gradY[own]) + (omw * gradY[nei]));
      float gz = ((w * gradZ[own]) + (omw * gradZ[nei]));
      float corr = (((dCorrX[int(__gid.x)] * gx) + (dCorrY[int(__gid.x)] * gy)) + (dCorrZ[int(__gid.x)] * gz));
      outFaceFlux[int(__gid.x)] = (rAUf[int(__gid.x)] * (magSf[int(__gid.x)] * corr));
    }
  }
}

)kernel";
const char* k_metal_navatala_cfd_pressure_sn_grad = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_cfd_pressure_sn_grad(device const float* phiHbyA [[buffer(0)]], device const int* ownerAll [[buffer(1)]], device const float* sfX [[buffer(2)]], device const float* sfY [[buffer(3)]], device const float* sfZ [[buffer(4)]], device const float* magSf [[buffer(5)]], device const float* ux [[buffer(6)]], device const float* uy [[buffer(7)]], device const float* uz [[buffer(8)]], device const float* uBcX [[buffer(9)]], device const float* uBcY [[buffer(10)]], device const float* uBcZ [[buffer(11)]], device const uint* uBcMask [[buffer(12)]], device const float* rAUf [[buffer(13)]], device const uint* snGradMask [[buffer(14)]], device const uint* params [[buffer(15)]], device float* outSnGrad [[buffer(16)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  if (int(__gid.x) >= ((int)(params[1]))) {
    return;
  } else {
    uint mask = snGradMask[int(__gid.x)];
    if (((int)(mask)) == 0) {
      outSnGrad[int(__gid.x)] = as_type<float>(0x00000000u);
    } else {
      int o = ownerAll[int(__gid.x)];
      if (o < 0) {
        outSnGrad[int(__gid.x)] = as_type<float>(0x00000000u);
      } else {
        if (o >= ((int)(params[0]))) {
          outSnGrad[int(__gid.x)] = as_type<float>(0x00000000u);
        } else {
          float uxF = ux[o];
          float uyF = uy[o];
          float uzF = uz[o];
          uint bcm = uBcMask[int(__gid.x)];
          if (((int)(bcm)) == 2) {
            uxF = uBcX[int(__gid.x)];
            uyF = uBcY[int(__gid.x)];
            uzF = uBcZ[int(__gid.x)];
          }
          float phiU = (((sfX[int(__gid.x)] * uxF) + (sfY[int(__gid.x)] * uyF)) + (sfZ[int(__gid.x)] * uzF));
          float rf = rAUf[int(__gid.x)];
          float ms = magSf[int(__gid.x)];
          float denom = (rf * ms);
          float ph = phiHbyA[int(__gid.x)];
          if (abs(denom) < as_type<float>(0x1e3ce508u)) {
            outSnGrad[int(__gid.x)] = as_type<float>(0x00000000u);
          } else {
            outSnGrad[int(__gid.x)] = ((ph - (rf * phiU)) / denom);
          }
        }
      }
    }
  }
}

)kernel";
const char* k_metal_navatala_cfd_u_correct = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_cfd_u_correct(device const float* pCell [[buffer(0)]], device const int* owner [[buffer(1)]], device const int* neighbour [[buffer(2)]], device const float* sfX [[buffer(3)]], device const float* sfY [[buffer(4)]], device const float* sfZ [[buffer(5)]], device const float* bcVal [[buffer(6)]], device const int* bcMask [[buffer(7)]], device const int* offsets [[buffer(8)]], device const int* faceIdx [[buffer(9)]], device const float* sign [[buffer(10)]], device const float* vol [[buffer(11)]], device const float* rAU [[buffer(12)]], device const float* ux [[buffer(13)]], device const float* uy [[buffer(14)]], device const float* uz [[buffer(15)]], device const int* counts [[buffer(16)]], device const float* paramsF [[buffer(17)]], device float* outX [[buffer(18)]], device float* outY [[buffer(19)]], device float* outZ [[buffer(20)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  if (((int)(int(__gid.x))) >= counts[0]) {
    return;
  } else {
    int beg = offsets[((int)(int(__gid.x)))];
    int c1 = (((int)(int(__gid.x))) + 1);
    int end = offsets[c1];
    int len = (end - beg);
    float gx = as_type<float>(0x00000000u);
    float gy = as_type<float>(0x00000000u);
    float gz = as_type<float>(0x00000000u);
    for (int t = 0; t < (int)(len); ++t) {
      int k = (beg + t);
      int f = faceIdx[k];
      int o = owner[f];
      float pf = pCell[o];
      if (f < counts[2]) {
        int n = neighbour[f];
        pf = (as_type<float>(0x3f000000u) * (pCell[o] + pCell[n]));
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
    float invV = (as_type<float>(0x3f800000u) / vol[((int)(int(__gid.x)))]);
    float sC = (paramsF[0] * rAU[((int)(int(__gid.x)))]);
    float dUx = ((gx * invV) * sC);
    float dUy = ((gy * invV) * sC);
    float dUz = ((gz * invV) * sC);
    outX[((int)(int(__gid.x)))] = (ux[((int)(int(__gid.x)))] - dUx);
    outY[((int)(int(__gid.x)))] = (uy[((int)(int(__gid.x)))] - dUy);
    outZ[((int)(int(__gid.x)))] = (uz[((int)(int(__gid.x)))] - dUz);
  }
}

)kernel";
const char* k_metal_navatala_cfd_u_correct_reconstruct = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_cfd_u_correct_reconstruct(device const float* pCell [[buffer(0)]], device const float* rAU [[buffer(1)]], device const float* hbx [[buffer(2)]], device const float* hby [[buffer(3)]], device const float* hbz [[buffer(4)]], device const float* rAUf [[buffer(5)]], device const float* phig [[buffer(6)]], device const int* owner [[buffer(7)]], device const int* neighbour [[buffer(8)]], device const float* sfX [[buffer(9)]], device const float* sfY [[buffer(10)]], device const float* sfZ [[buffer(11)]], device const float* magSf [[buffer(12)]], device const float* deltaCoeffs [[buffer(13)]], device const int* offsets [[buffer(14)]], device const int* faceIdx [[buffer(15)]], device const float* sign [[buffer(16)]], device const float* vol [[buffer(17)]], device const float* bcVal [[buffer(18)]], device const int* bcMask [[buffer(19)]], device const float* bcSnGrad [[buffer(20)]], device const int* bcSnGradMask [[buffer(21)]], device const float* faceFluxCorrection [[buffer(22)]], device const int* counts [[buffer(23)]], device const float* paramsF [[buffer(24)]], device float* outX [[buffer(25)]], device float* outY [[buffer(26)]], device float* outZ [[buffer(27)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  if (((int)(int(__gid.x))) >= counts[0]) {
    return;
  } else {
    int beg = offsets[((int)(int(__gid.x)))];
    int c1 = (((int)(int(__gid.x))) + 1);
    int end = offsets[c1];
    int len = (end - beg);
    float a00 = as_type<float>(0x00000000u);
    float a01 = as_type<float>(0x00000000u);
    float a02 = as_type<float>(0x00000000u);
    float a10 = as_type<float>(0x00000000u);
    float a11 = as_type<float>(0x00000000u);
    float a12 = as_type<float>(0x00000000u);
    float a20 = as_type<float>(0x00000000u);
    float a21 = as_type<float>(0x00000000u);
    float a22 = as_type<float>(0x00000000u);
    float bx = as_type<float>(0x00000000u);
    float by = as_type<float>(0x00000000u);
    float bz = as_type<float>(0x00000000u);
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
        float phiTerm = as_type<float>(0x00000000u);
        if (bcSnGradMask[f] == 2) {
          phiTerm = bcSnGrad[f];
        } else {
          if (bcSnGradMask[f] == 1) {
            snGrad = bcSnGrad[f];
          }
          float gradFace = (magSf[f] * snGrad);
          float denom = rAUf[f];
          float pFlux = ((paramsF[0] * (denom * gradFace)) + faceFluxCorrection[f]);
          if (denom != as_type<float>(0x00000000u)) {
            phiTerm = ((phig[f] - pFlux) / denom);
          }
        }
        float ms = magSf[f];
        if (ms > as_type<float>(0x00000000u)) {
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
    float smallConst = as_type<float>(0x33d6bf95u);
    float rootVSmall = as_type<float>(0x1e3ce508u);
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
      w00 = (w00 + as_type<float>(0x3f800000u));
    }
    if (small_yy) {
      w11 = (w11 + as_type<float>(0x3f800000u));
    }
    if (small_zz) {
      w22 = (w22 + as_type<float>(0x3f800000u));
    }
    float det = (((((w00 * w11) * w22) + ((w01 * w12) * w20)) + ((w02 * w10) * w21)) - ((((w00 * w12) * w21) + ((w01 * w10) * w22)) + ((w02 * w11) * w20)));
    float rx = as_type<float>(0x00000000u);
    float ry = as_type<float>(0x00000000u);
    float rz = as_type<float>(0x00000000u);
    if (abs(det) >= rootVSmall) {
      float invDet = (as_type<float>(0x3f800000u) / det);
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
        i00 = (i00 - as_type<float>(0x3f800000u));
      }
      if (small_yy) {
        i11 = (i11 - as_type<float>(0x3f800000u));
      }
      if (small_zz) {
        i22 = (i22 - as_type<float>(0x3f800000u));
      }
      rx = (((i00 * bx) + (i01 * by)) + (i02 * bz));
      ry = (((i10 * bx) + (i11 * by)) + (i12 * bz));
      rz = (((i20 * bx) + (i21 * by)) + (i22 * bz));
    }
    float fac = rAU[((int)(int(__gid.x)))];
    outX[((int)(int(__gid.x)))] = (hbx[((int)(int(__gid.x)))] + (fac * rx));
    outY[((int)(int(__gid.x)))] = (hby[((int)(int(__gid.x)))] + (fac * ry));
    outZ[((int)(int(__gid.x)))] = (hbz[((int)(int(__gid.x)))] + (fac * rz));
  }
}

)kernel";
const char* k_metal_navatala_cfd_grad_alpha_cell = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_cfd_grad_alpha_cell(device const float* alphaF [[buffer(0)]], device const float* sfX [[buffer(1)]], device const float* sfY [[buffer(2)]], device const float* sfZ [[buffer(3)]], device const int* offsets [[buffer(4)]], device const int* faceIdx [[buffer(5)]], device const float* sign [[buffer(6)]], device const float* vol [[buffer(7)]], device const int* counts [[buffer(8)]], device float* gx [[buffer(9)]], device float* gy [[buffer(10)]], device float* gz [[buffer(11)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  if (((int)(int(__gid.x))) >= counts[0]) {
    return;
  } else {
    int beg = offsets[((int)(int(__gid.x)))];
    int c1 = (((int)(int(__gid.x))) + 1);
    int end = offsets[c1];
    int len = (end - beg);
    float sx = as_type<float>(0x00000000u);
    float sy = as_type<float>(0x00000000u);
    float sz = as_type<float>(0x00000000u);
    for (int t = 0; t < (int)(len); ++t) {
      int k = (beg + t);
      int f = faceIdx[k];
      float s = (sign[k] * alphaF[f]);
      sx = (sx + (s * sfX[f]));
      sy = (sy + (s * sfY[f]));
      sz = (sz + (s * sfZ[f]));
    }
    float invV = (as_type<float>(0x3f800000u) / vol[((int)(int(__gid.x)))]);
    gx[((int)(int(__gid.x)))] = (sx * invV);
    gy[((int)(int(__gid.x)))] = (sy * invV);
    gz[((int)(int(__gid.x)))] = (sz * invV);
  }
}

)kernel";
const char* k_metal_navatala_cfd_n_hatf_all = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_cfd_n_hatf_all(device const float* gx [[buffer(0)]], device const float* gy [[buffer(1)]], device const float* gz [[buffer(2)]], device const float* sfX [[buffer(3)]], device const float* sfY [[buffer(4)]], device const float* sfZ [[buffer(5)]], device const int* owner [[buffer(6)]], device const int* nei [[buffer(7)]], device const float* weights [[buffer(8)]], device const float* theta [[buffer(9)]], device const uint* thetaMask [[buffer(10)]], device const float* paramsF [[buffer(11)]], device const int* counts [[buffer(12)]], device float* contactGrad [[buffer(13)]], device float* nHatf [[buffer(14)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  if (((int)(int(__gid.x))) >= counts[1]) {
    return;
  } else {
    int o = owner[((int)(int(__gid.x)))];
    int n = nei[((int)(int(__gid.x)))];
    bool isBoundary = (((int)(int(__gid.x))) >= counts[2]);
    float w = weights[((int)(int(__gid.x)))];
    float wm = (as_type<float>(0x3f800000u) - w);
    float gfx = ((isBoundary) ? (gx[o]) : (((w * gx[o]) + (wm * gx[n]))));
    float gfy = ((isBoundary) ? (gy[o]) : (((w * gy[o]) + (wm * gy[n]))));
    float gfz = ((isBoundary) ? (gz[o]) : (((w * gz[o]) + (wm * gz[n]))));
    float magg = sqrt((((gfx * gfx) + (gfy * gfy)) + (gfz * gfz)));
    float inv = (as_type<float>(0x3f800000u) / (magg + paramsF[0]));
    float nx = (gfx * inv);
    float ny = (gfy * inv);
    float nz = (gfz * inv);
    contactGrad[((int)(int(__gid.x)))] = as_type<float>(0x00000000u);
    if (thetaMask[((int)(int(__gid.x)))] != 0u) {
      float sfx = sfX[((int)(int(__gid.x)))];
      float sfy = sfY[((int)(int(__gid.x)))];
      float sfz = sfZ[((int)(int(__gid.x)))];
      float magSf = sqrt((((sfx * sfx) + (sfy * sfy)) + (sfz * sfz)));
      if (magSf > as_type<float>(0x1e3ce508u)) {
        float nfx = (sfx / magSf);
        float nfy = (sfy / magSf);
        float nfz = (sfz / magSf);
        float a12raw = (((nx * nfx) + (ny * nfy)) + (nz * nfz));
        float a12 = a12raw;
        if (a12 < as_type<float>(0xbf800000u)) {
          a12 = as_type<float>(0xbf800000u);
        } else {
          if (a12 > as_type<float>(0x3f800000u)) {
            a12 = as_type<float>(0x3f800000u);
          }
        }
        float th = theta[((int)(int(__gid.x)))];
        float b1 = cos(th);
        float acosA12 = acos(a12);
        float b2 = cos((acosA12 - th));
        float det = (as_type<float>(0x3f800000u) - (a12 * a12));
        if (det < as_type<float>(0x358637bdu)) {
          det = as_type<float>(0x358637bdu);
        }
        float a = ((b1 - (a12 * b2)) / det);
        float b = ((b2 - (a12 * b1)) / det);
        float nx2 = ((a * nfx) + (b * nx));
        float ny2 = ((a * nfy) + (b * ny));
        float nz2 = ((a * nfz) + (b * nz));
        float magN = sqrt((((nx2 * nx2) + (ny2 * ny2)) + (nz2 * nz2)));
        float invN = (as_type<float>(0x3f800000u) / (magN + paramsF[0]));
        nx = (nx2 * invN);
        ny = (ny2 * invN);
        nz = (nz2 * invN);
        float dot = (((nx * nfx) + (ny * nfy)) + (nz * nfz));
        contactGrad[((int)(int(__gid.x)))] = (dot * magg);
      }
    }
    nHatf[((int)(int(__gid.x)))] = (((nx * sfX[((int)(int(__gid.x)))]) + (ny * sfY[((int)(int(__gid.x)))])) + (nz * sfZ[((int)(int(__gid.x)))]));
  }
}

)kernel";
const char* k_metal_navatala_cfd_stf_all_faces = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_cfd_stf_all_faces(device const float* divN [[buffer(0)]], device const float* alpha [[buffer(1)]], device const float* alphaF [[buffer(2)]], device const float* delta [[buffer(3)]], device const int* owner [[buffer(4)]], device const int* nei [[buffer(5)]], device const float* weights [[buffer(6)]], device const float* contactGrad [[buffer(7)]], device const uint* thetaMask [[buffer(8)]], device const float* paramsF [[buffer(9)]], device const int* counts [[buffer(10)]], device float* stf [[buffer(11)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  if (((int)(int(__gid.x))) >= counts[1]) {
    return;
  } else {
    int o = owner[((int)(int(__gid.x)))];
    int n = nei[((int)(int(__gid.x)))];
    float w = weights[((int)(int(__gid.x)))];
    float wm = (as_type<float>(0x3f800000u) - w);
    float divInterp = ((w * divN[o]) + (wm * divN[n]));
    float Kf = (as_type<float>(0xbf800000u) * divInterp);
    float snGrad = as_type<float>(0x00000000u);
    if (((int)(int(__gid.x))) < counts[2]) {
      snGrad = (delta[((int)(int(__gid.x)))] * (alpha[n] - alpha[o]));
    } else {
      if (thetaMask[((int)(int(__gid.x)))] != 0u) {
        snGrad = contactGrad[((int)(int(__gid.x)))];
      } else {
        snGrad = (delta[((int)(int(__gid.x)))] * (alphaF[((int)(int(__gid.x)))] - alpha[o]));
      }
    }
    stf[((int)(int(__gid.x)))] = ((paramsF[0] * Kf) * snGrad);
  }
}

)kernel";
const char* k_metal_navatala_cfd_k_omega_s_s_t_blending = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_cfd_k_omega_s_s_t_blending(device const float* k [[buffer(0)]], device const float* omega [[buffer(1)]], device const float* nut [[buffer(2)]], device const float* nu [[buffer(3)]], device const float* y [[buffer(4)]], device const float* gradKx [[buffer(5)]], device const float* gradKy [[buffer(6)]], device const float* gradKz [[buffer(7)]], device const float* gradOmegax [[buffer(8)]], device const float* gradOmegay [[buffer(9)]], device const float* gradOmegaz [[buffer(10)]], device const int* counts [[buffer(11)]], device float* outF1 [[buffer(12)]], device float* outF23 [[buffer(13)]], device float* outCDkOmega [[buffer(14)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  if (((int)(int(__gid.x))) >= counts[0]) {
    return;
  } else {
    float kC = k[((int)(int(__gid.x)))];
    float omC = omega[((int)(int(__gid.x)))];
    float nutC = nut[((int)(int(__gid.x)))];
    float nuC = nu[((int)(int(__gid.x)))];
    float yC = y[((int)(int(__gid.x)))];
    float gKx = gradKx[((int)(int(__gid.x)))];
    float gKy = gradKy[((int)(int(__gid.x)))];
    float gKz = gradKz[((int)(int(__gid.x)))];
    float gOx = gradOmegax[((int)(int(__gid.x)))];
    float gOy = gradOmegay[((int)(int(__gid.x)))];
    float gOz = gradOmegaz[((int)(int(__gid.x)))];
    float dotGrad = (((gKx * gOx) + (gKy * gOy)) + (gKz * gOz));
    float omSafe = (((omC > as_type<float>(0x2edbe6ffu))) ? (omC) : (as_type<float>(0x2edbe6ffu)));
    float cdkw = ((as_type<float>(0x3fdb22d1u) * dotGrad) / omSafe);
    float cdkwPlus = (((cdkw > as_type<float>(0x2edbe6ffu))) ? (cdkw) : (as_type<float>(0x2edbe6ffu)));
    float ySq = (yC * yC);
    float ySqSafe = (((ySq > as_type<float>(0x1e3ce508u))) ? (ySq) : (as_type<float>(0x1e3ce508u)));
    float kSafe = (((kC > as_type<float>(0x00000000u))) ? (kC) : (as_type<float>(0x00000000u)));
    float sqrtK = sqrt(kSafe);
    float denom1 = (as_type<float>(0x3db851ecu) * (omSafe * yC));
    float denom1Safe = (((denom1 > as_type<float>(0x2edbe6ffu))) ? (denom1) : (as_type<float>(0x2edbe6ffu)));
    float term1 = (sqrtK / denom1Safe);
    float denom2 = (ySqSafe * omSafe);
    float term2 = ((as_type<float>(0x43fa0000u) * nuC) / denom2);
    float maxT12 = (((term1 >= term2)) ? (term1) : (term2));
    float denom3 = (cdkwPlus * ySqSafe);
    float denom3Safe = (((denom3 > as_type<float>(0x2edbe6ffu))) ? (denom3) : (as_type<float>(0x2edbe6ffu)));
    float term3 = ((as_type<float>(0x405b22d1u) * kC) / denom3Safe);
    float minT3 = (((maxT12 <= term3)) ? (maxT12) : (term3));
    float arg1 = (((minT3 <= as_type<float>(0x41200000u))) ? (minT3) : (as_type<float>(0x41200000u)));
    float arg1sq = (arg1 * arg1);
    float arg1p4 = (arg1sq * arg1sq);
    float arg1p4Safe = (((arg1p4 <= as_type<float>(0x42200000u))) ? (arg1p4) : (as_type<float>(0x42200000u)));
    float f1Val = tanh(arg1p4Safe);
    float term2a = ((as_type<float>(0x40000000u) * sqrtK) / denom1Safe);
    float maxF2 = (((term2a >= term2)) ? (term2a) : (term2));
    float arg2 = (((maxF2 <= as_type<float>(0x42c80000u))) ? (maxF2) : (as_type<float>(0x42c80000u)));
    float arg2sq = (arg2 * arg2);
    float arg2sqSafe = (((arg2sq <= as_type<float>(0x42200000u))) ? (arg2sq) : (as_type<float>(0x42200000u)));
    float f23Val = tanh(arg2sqSafe);
    outF1[((int)(int(__gid.x)))] = f1Val;
    outF23[((int)(int(__gid.x)))] = f23Val;
    outCDkOmega[((int)(int(__gid.x)))] = cdkw;
  }
}

)kernel";
const char* k_metal_navatala_cfd_k_omega_s_s_t_nut_compute = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_cfd_k_omega_s_s_t_nut_compute(device const float* kVals [[buffer(0)]], device const float* omegaVals [[buffer(1)]], device const float* f23Vals [[buffer(2)]], device const float* s2Vals [[buffer(3)]], device const uint* counts [[buffer(4)]], device const float* params [[buffer(5)]], device float* outNut [[buffer(6)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  if (int(__gid.x) >= ((int)(counts[0]))) {
    return;
  } else {
    float a1 = params[0];
    float b1 = params[1];
    float k = kVals[int(__gid.x)];
    float om = omegaVals[int(__gid.x)];
    float f23 = f23Vals[int(__gid.x)];
    float s2raw = s2Vals[int(__gid.x)];
    float denomA = (a1 * om);
    float denomB = ((b1 * f23) * sqrt((s2raw * ((float)((s2raw > as_type<float>(0x00000000u)))))));
    float num = (a1 * k);
    if (denomA > denomB) {
      if (denomA > as_type<float>(0x00000000u)) {
        outNut[int(__gid.x)] = (num / denomA);
      } else {
        outNut[int(__gid.x)] = as_type<float>(0x00000000u);
      }
    } else {
      if (denomB > as_type<float>(0x00000000u)) {
        outNut[int(__gid.x)] = (num / denomB);
      } else {
        outNut[int(__gid.x)] = as_type<float>(0x00000000u);
      }
    }
  }
}

)kernel";
const char* k_metal_navatala_cfd_k_omega_s_s_t_sources = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_cfd_k_omega_s_s_t_sources(device const float* k [[buffer(0)]], device const float* omega [[buffer(1)]], device const float* nut [[buffer(2)]], device const float* nu [[buffer(3)]], device const float* alpha [[buffer(4)]], device const float* rho [[buffer(5)]], device const float* F1 [[buffer(6)]], device const float* F23 [[buffer(7)]], device const float* CDkOmega [[buffer(8)]], device const float* S2 [[buffer(9)]], device const float* divU [[buffer(10)]], device const int* counts [[buffer(11)]], device float* outOmegaSp [[buffer(12)]], device float* outOmegaSuSp1 [[buffer(13)]], device float* outOmegaSuSp2 [[buffer(14)]], device float* outOmegaExpl [[buffer(15)]], device float* outKSp [[buffer(16)]], device float* outKSuSp [[buffer(17)]], device float* outKExpl [[buffer(18)]], device float* outDomegaEff [[buffer(19)]], device float* outDkEff [[buffer(20)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  if (((int)(int(__gid.x))) >= counts[0]) {
    return;
  } else {
    float kC = k[((int)(int(__gid.x)))];
    float omC = omega[((int)(int(__gid.x)))];
    float nutC = nut[((int)(int(__gid.x)))];
    float nuC = nu[((int)(int(__gid.x)))];
    float alphaC = alpha[((int)(int(__gid.x)))];
    float rhoC = rho[((int)(int(__gid.x)))];
    float f1C = F1[((int)(int(__gid.x)))];
    float f23C = F23[((int)(int(__gid.x)))];
    float cdkwC = CDkOmega[((int)(int(__gid.x)))];
    float s2C = S2[((int)(int(__gid.x)))];
    float divUC = divU[((int)(int(__gid.x)))];
    float arC = (alphaC * rhoC);
    float gammaC = ((f1C * as_type<float>(0x3decbfb1u)) + as_type<float>(0x3ee147aeu));
    float betaC = (as_type<float>(0x3da9930cu) - (f1C * as_type<float>(0x3bff9724u)));
    float alphaKC = (as_type<float>(0x3f800000u) - (f1C * as_type<float>(0x3e19999au)));
    float alphaOmC = (as_type<float>(0x3f5b22d1u) - (f1C * as_type<float>(0x3eb645a2u)));
    float s2Safe = (((s2C > as_type<float>(0x00000000u))) ? (s2C) : (as_type<float>(0x00000000u)));
    float sqrtS2 = sqrt(s2Safe);
    float a1om = (as_type<float>(0x3e9eb852u) * omC);
    float f23s = (f23C * sqrtS2);
    float maxTerm = (((a1om >= f23s)) ? (a1om) : (f23s));
    float limRhs = (as_type<float>(0x4039ce07u) * (omC * maxTerm));
    float GbyNu = (((s2C <= limRhs)) ? (s2C) : (limRhs));
    float nutS2 = (nutC * s2C);
    float c1bsKOm = (as_type<float>(0x3f666666u) * (kC * omC));
    float pk = (((nutS2 <= c1bsKOm)) ? (nutS2) : (c1bsKOm));
    float omSp = (as_type<float>(0x00000000u) - (arC * (betaC * omC)));
    float omSuSp1 = (as_type<float>(0x00000000u) - (as_type<float>(0x3f2aacdau) * (arC * (gammaC * divUC))));
    float omSafe = (((omC > as_type<float>(0x26901d7du))) ? (omC) : (as_type<float>(0x26901d7du)));
    float f1m1 = (f1C - as_type<float>(0x3f800000u));
    float omSuSp2 = (as_type<float>(0x00000000u) - ((arC * (f1m1 * cdkwC)) / omSafe));
    float omExpl = (arC * (gammaC * GbyNu));
    float kSp = (as_type<float>(0x00000000u) - (arC * (as_type<float>(0x3db851ecu) * omC)));
    float kSuSp = (as_type<float>(0x00000000u) - (as_type<float>(0x3f2aacdau) * (arC * divUC)));
    float kExpl = (arC * pk);
    float domEff = (arC * ((alphaOmC * nutC) + nuC));
    float dkEff = (arC * ((alphaKC * nutC) + nuC));
    outOmegaSp[((int)(int(__gid.x)))] = omSp;
    outOmegaSuSp1[((int)(int(__gid.x)))] = omSuSp1;
    outOmegaSuSp2[((int)(int(__gid.x)))] = omSuSp2;
    outOmegaExpl[((int)(int(__gid.x)))] = omExpl;
    outKSp[((int)(int(__gid.x)))] = kSp;
    outKSuSp[((int)(int(__gid.x)))] = kSuSp;
    outKExpl[((int)(int(__gid.x)))] = kExpl;
    outDomegaEff[((int)(int(__gid.x)))] = domEff;
    outDkEff[((int)(int(__gid.x)))] = dkEff;
  }
}

)kernel";
const char* k_metal_navatala_cfd_mu_eff_from_rho_nu_eff = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_cfd_mu_eff_from_rho_nu_eff(device const float* rho [[buffer(0)]], device const float* nuEff [[buffer(1)]], device const int* params [[buffer(2)]], device float* outMuEff [[buffer(3)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  if (((int)(int(__gid.x))) >= params[0]) {
    return;
  } else {
    float r = rho[((int)(int(__gid.x)))];
    float nu = nuEff[((int)(int(__gid.x)))];
    outMuEff[((int)(int(__gid.x)))] = (r * nu);
  }
}

)kernel";
const char* k_metal_navatala_cfd_mu_eff_from_rho_nut_and_nu_mix = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_cfd_mu_eff_from_rho_nut_and_nu_mix(device const float* rho [[buffer(0)]], device const float* nut [[buffer(1)]], device const float* nuMix [[buffer(2)]], device const uint* params [[buffer(3)]], device float* outMu [[buffer(4)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  if (int(__gid.x) >= ((int)(params[0]))) {
    return;
  } else {
    outMu[int(__gid.x)] = (rho[int(__gid.x)] * (nuMix[int(__gid.x)] + nut[int(__gid.x)]));
  }
}

)kernel";
const char* k_metal_navatala_cfd_turbulence_add_explicit_source_device = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_cfd_turbulence_add_explicit_source_device(device const float* explicitSrc [[buffer(0)]], device const float* vol [[buffer(1)]], device const uint* counts [[buffer(2)]], device float* outVals [[buffer(3)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  if (int(__gid.x) >= ((int)(counts[0]))) {
    return;
  } else {
    outVals[int(__gid.x)] = (explicitSrc[int(__gid.x)] * vol[int(__gid.x)]);
  }
}

)kernel";
const char* k_metal_navatala_cfd_turbulence_dirichlet_cell_source = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_cfd_turbulence_dirichlet_cell_source(device const int* constrainedMask [[buffer(0)]], device const float* constrainedValue [[buffer(1)]], device const float* diag [[buffer(2)]], device const int* nCells [[buffer(3)]], device float* source [[buffer(4)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  if (((int)(int(__gid.x))) >= nCells[0]) {
    return;
  } else {
    int mask = constrainedMask[((int)(int(__gid.x)))];
    if (mask == 0) {
      return;
    } else {
      source[((int)(int(__gid.x)))] = (diag[((int)(int(__gid.x)))] * constrainedValue[((int)(int(__gid.x)))]);
    }
  }
}

)kernel";
const char* k_metal_navatala_cfd_turbulence_dirichlet_face_elimination = R"kernel(
#include <metal_stdlib>
using namespace metal;

static inline int gpu_atomic_cas_int(device atomic_int* ptr, int expected, int desired) {
  int exp = expected;
  while (!atomic_compare_exchange_weak_explicit(ptr, &exp, desired,
      memory_order_relaxed, memory_order_relaxed) && exp == expected) {}
  return exp;
}

static inline uint gpu_atomic_cas_uint(device atomic_uint* ptr, uint expected, uint desired) {
  uint exp = expected;
  while (!atomic_compare_exchange_weak_explicit(ptr, &exp, desired,
      memory_order_relaxed, memory_order_relaxed) && exp == expected) {}
  return exp;
}

kernel void navatala_cfd_turbulence_dirichlet_face_elimination(device const int* owner [[buffer(0)]], device const int* neighbour [[buffer(1)]], device const int* constrainedMask [[buffer(2)]], device const float* constrainedValue [[buffer(3)]], device const int* nIntFaces [[buffer(4)]], device float* upper [[buffer(5)]], device float* lower [[buffer(6)]], device float* source [[buffer(7)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  if (((int)(int(__gid.x))) >= nIntFaces[0]) {
    return;
  } else {
    int o = owner[((int)(int(__gid.x)))];
    int n = neighbour[((int)(int(__gid.x)))];
    int oC = constrainedMask[o];
    int nC = constrainedMask[n];
    if ((oC | nC) == 0) {
      return;
    } else {
      float up = upper[((int)(int(__gid.x)))];
      float lo = lower[((int)(int(__gid.x)))];
      if (oC != 0) {
        float cv = constrainedValue[o];
        atomic_fetch_add_explicit((device atomic_float*)(&source[n]), (-(lo * cv)), memory_order_relaxed);
      }
      if (nC != 0) {
        float cv = constrainedValue[n];
        atomic_fetch_add_explicit((device atomic_float*)(&source[o]), (-(up * cv)), memory_order_relaxed);
      }
      upper[((int)(int(__gid.x)))] = as_type<float>(0x00000000u);
      lower[((int)(int(__gid.x)))] = as_type<float>(0x00000000u);
    }
  }
}

)kernel";
const char* k_metal_navatala_cfd_two_mag_sqr_symm_grad_u = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_cfd_two_mag_sqr_symm_grad_u(device const float* ufX [[buffer(0)]], device const float* ufY [[buffer(1)]], device const float* ufZ [[buffer(2)]], device const float* sfX [[buffer(3)]], device const float* sfY [[buffer(4)]], device const float* sfZ [[buffer(5)]], device const int* offsets [[buffer(6)]], device const int* faceIdx [[buffer(7)]], device const float* sign [[buffer(8)]], device const float* vol [[buffer(9)]], device const int* counts [[buffer(10)]], device float* outS2 [[buffer(11)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  if (((int)(int(__gid.x))) >= counts[0]) {
    return;
  } else {
    int beg = offsets[((int)(int(__gid.x)))];
    int c1 = (((int)(int(__gid.x))) + 1);
    int end = offsets[c1];
    int len = (end - beg);
    float g00 = as_type<float>(0x00000000u);
    float g01 = as_type<float>(0x00000000u);
    float g02 = as_type<float>(0x00000000u);
    float g10 = as_type<float>(0x00000000u);
    float g11 = as_type<float>(0x00000000u);
    float g12 = as_type<float>(0x00000000u);
    float g20 = as_type<float>(0x00000000u);
    float g21 = as_type<float>(0x00000000u);
    float g22 = as_type<float>(0x00000000u);
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
    float v = vol[((int)(int(__gid.x)))];
    float invV = as_type<float>(0x00000000u);
    if (v != as_type<float>(0x00000000u)) {
      invV = (as_type<float>(0x3f800000u) / v);
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
    float s01 = (as_type<float>(0x3f000000u) * (g01 + g10));
    float s02 = (as_type<float>(0x3f000000u) * (g02 + g20));
    float s12 = (as_type<float>(0x3f000000u) * (g12 + g21));
    float d0 = (s00 * s00);
    float d1 = (s11 * s11);
    float d2 = (s22 * s22);
    float o0 = (s01 * s01);
    float o1 = (s02 * s02);
    float o2 = (s12 * s12);
    float diag = ((d0 + d1) + d2);
    float off = ((o0 + o1) + o2);
    outS2[((int)(int(__gid.x)))] = ((as_type<float>(0x40000000u) * diag) + (as_type<float>(0x40800000u) * off));
  }
}

)kernel";
const char* k_metal_navatala_cfd_attention_row_softmax = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_cfd_attention_row_softmax(device const float* scores [[buffer(0)]], device const int* numQ [[buffer(1)]], device const int* numK [[buffer(2)]], device float* weights [[buffer(3)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  int qi = int(__gid.x);
  int nq = numQ[0];
  int nk = numK[0];
  if (qi < nq) {
    int rowBase = (qi * nk);
    float maxScore = as_type<float>(0xf149f2cau);
    for (int j = 0; j < (int)(nk); ++j) {
      int idx = (rowBase + j);
      float s = scores[idx];
      maxScore = (((s > maxScore)) ? (s) : (maxScore));
    }
    float sumExp = as_type<float>(0x00000000u);
    for (int j = 0; j < (int)(nk); ++j) {
      int idx = (rowBase + j);
      float s = scores[idx];
      float e = exp((s - maxScore));
      sumExp = (sumExp + e);
    }
    float denom = (sumExp + as_type<float>(0x3089705fu));
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
const char* k_metal_navatala_cfd_attention_value_projection = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_cfd_attention_value_projection(device const float* weights [[buffer(0)]], device const float* v [[buffer(1)]], device const int* numQ [[buffer(2)]], device const int* numK [[buffer(3)]], device const int* headDim [[buffer(4)]], device float* _out [[buffer(5)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  int gid = int(__gid.x);
  int nq = numQ[0];
  int nk = numK[0];
  int hd = headDim[0];
  int total = (nq * hd);
  if (gid < total) {
    int qi = (gid / hd);
    int d = (gid % hd);
    float acc = as_type<float>(0x00000000u);
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
const char* k_metal_navatala_cfd_gather_mean = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_cfd_gather_mean(device const float* features [[buffer(0)]], device const int* neighbors [[buffer(1)]], device const int* neighborCounts [[buffer(2)]], device const int* numNodes [[buffer(3)]], device const int* featureDim [[buffer(4)]], device float* _out [[buffer(5)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  int gid = int(__gid.x);
  int nNodes = numNodes[0];
  if (gid < nNodes) {
    int cnt = neighborCounts[gid];
    int fd = featureDim[0];
    for (int f = 0; f < (int)(fd); ++f) {
      float acc = as_type<float>(0x00000000u);
      for (int j = 0; j < (int)(cnt); ++j) {
        int nbrFlat = ((gid * 8) + j);
        int nbr = neighbors[nbrFlat];
        int featFlat = ((nbr * fd) + f);
        float v = features[featFlat];
        acc = (acc + v);
      }
      float cntF = ((float)(cnt));
      float mean = (((cnt > 0)) ? ((acc / cntF)) : (as_type<float>(0x00000000u)));
      int outFlat = ((gid * fd) + f);
      _out[outFlat] = mean;
    }
  }
}

)kernel";
const char* k_metal_navatala_cfd_layer_norm_forward = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_cfd_layer_norm_forward(device const float* x [[buffer(0)]], device const int* numRows [[buffer(1)]], device const float* gamma [[buffer(2)]], device const float* beta [[buffer(3)]], device float* y [[buffer(4)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  int row = int(__gid.x);
  int nRows = numRows[0];
  if (row < nRows) {
    int hd = 16;
    int rowBase = (row * hd);
    float sum = as_type<float>(0x00000000u);
    for (int j = 0; j < (int)(hd); ++j) {
      int idx = (rowBase + j);
      float xv = x[idx];
      sum = (sum + xv);
    }
    float hdF = ((float)(hd));
    float mean = (sum / hdF);
    float varSum = as_type<float>(0x00000000u);
    for (int j = 0; j < (int)(hd); ++j) {
      int idx = (rowBase + j);
      float xv = x[idx];
      float diff = (xv - mean);
      varSum = (varSum + (diff * diff));
    }
    float var = (varSum / hdF);
    float invStd = sqrt((as_type<float>(0x3f800000u) / (var + as_type<float>(0x3727c5acu))));
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
const char* k_metal_navatala_cfd_radius_count = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_cfd_radius_count(device const float* query [[buffer(0)]], device const float* ref [[buffer(1)]], device const int* numQuery [[buffer(2)]], device const int* numRef [[buffer(3)]], device const float* radiusIn [[buffer(4)]], device int* counts [[buffer(5)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  int qid = int(__gid.x);
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
const char* k_metal_navatala_cfd_scaled_dot_product = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_cfd_scaled_dot_product(device const float* q [[buffer(0)]], device const float* k [[buffer(1)]], device const int* numQ [[buffer(2)]], device const int* numK [[buffer(3)]], device const int* headDim [[buffer(4)]], device float* scores [[buffer(5)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  int gid = int(__gid.x);
  int nq = numQ[0];
  int nk = numK[0];
  int hd = headDim[0];
  int total = (nq * nk);
  if (gid < total) {
    int qi = (gid / nk);
    int kj = (gid % nk);
    float acc = as_type<float>(0x00000000u);
    for (int d = 0; d < (int)(hd); ++d) {
      int qFlat = ((qi * hd) + d);
      int kFlat = ((kj * hd) + d);
      float qv = q[qFlat];
      float kv = k[kFlat];
      acc = (acc + (qv * kv));
    }
    scores[gid] = (acc * as_type<float>(0x3e800000u));
  }
}

)kernel";
const char* k_metal_navatala_cfd_scatter_add_atomic = R"kernel(
#include <metal_stdlib>
using namespace metal;

static inline int gpu_atomic_cas_int(device atomic_int* ptr, int expected, int desired) {
  int exp = expected;
  while (!atomic_compare_exchange_weak_explicit(ptr, &exp, desired,
      memory_order_relaxed, memory_order_relaxed) && exp == expected) {}
  return exp;
}

static inline uint gpu_atomic_cas_uint(device atomic_uint* ptr, uint expected, uint desired) {
  uint exp = expected;
  while (!atomic_compare_exchange_weak_explicit(ptr, &exp, desired,
      memory_order_relaxed, memory_order_relaxed) && exp == expected) {}
  return exp;
}

kernel void navatala_cfd_scatter_add_atomic(device const int* values [[buffer(0)]], device const int* indices [[buffer(1)]], device const int* numEdges [[buffer(2)]], device const int* featureDim [[buffer(3)]], device atomic_int* dst [[buffer(4)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  int gid = int(__gid.x);
  int nEdges = numEdges[0];
  if (gid < nEdges) {
    int dstIdx = indices[gid];
    int fd = featureDim[0];
    for (int f = 0; f < (int)(fd); ++f) {
      int valFlat = ((gid * fd) + f);
      int dstFlat = ((dstIdx * fd) + f);
      int v = values[valFlat];
      int _ignored = atomic_fetch_add_explicit((device atomic_int*)(&(dst[dstFlat])), v, memory_order_relaxed);
    }
  }
}

)kernel";
const char* k_metal_navatala_cfd_spectral_complex_mul = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_cfd_spectral_complex_mul(device const float* x [[buffer(0)]], device const float* w [[buffer(1)]], device const int* n [[buffer(2)]], device float* y [[buffer(3)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  int gid = int(__gid.x);
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
const char* k_metal_navatala_cfd_spectral_conv1d_mode_sliced = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_cfd_spectral_conv1d_mode_sliced(device const float* xFt [[buffer(0)]], device const float* wPos [[buffer(1)]], device const float* wNeg [[buffer(2)]], device const int* numBatches [[buffer(3)]], device const int* ftW [[buffer(4)]], device float* yFt [[buffer(5)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  int gid = int(__gid.x);
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
    float accR = as_type<float>(0x00000000u);
    float accI = as_type<float>(0x00000000u);
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
const char* k_metal_navatala_cfd_spectral_conv2d_mode_sliced = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_cfd_spectral_conv2d_mode_sliced(device const float* xFt [[buffer(0)]], device const float* wPos [[buffer(1)]], device const float* wNeg [[buffer(2)]], device const int* numBatches [[buffer(3)]], device const int* ftH [[buffer(4)]], device const int* ftW [[buffer(5)]], device float* yFt [[buffer(6)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  int gid = int(__gid.x);
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
    float accR = as_type<float>(0x00000000u);
    float accI = as_type<float>(0x00000000u);
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
const char* k_metal_navatala_cfd_spectral_conv3d_mode_sliced = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_cfd_spectral_conv3d_mode_sliced(device const float* xFt [[buffer(0)]], device const float* wPP [[buffer(1)]], device const float* wNP [[buffer(2)]], device const float* wPN [[buffer(3)]], device const float* wNN [[buffer(4)]], device const int* numBatches [[buffer(5)]], device const int* ftD [[buffer(6)]], device const int* ftH [[buffer(7)]], device const int* ftW [[buffer(8)]], device float* yFt [[buffer(9)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  int gid = int(__gid.x);
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
    float accR = as_type<float>(0x00000000u);
    float accI = as_type<float>(0x00000000u);
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
const char* k_metal_navatala_cfd_spectral_conv4d_mode_sliced = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_cfd_spectral_conv4d_mode_sliced(device const float* xFt [[buffer(0)]], device const float* wPPP [[buffer(1)]], device const float* wNPP [[buffer(2)]], device const float* wPNP [[buffer(3)]], device const float* wNNP [[buffer(4)]], device const float* wPPN [[buffer(5)]], device const float* wNPN [[buffer(6)]], device const float* wPNN [[buffer(7)]], device const float* wNNN [[buffer(8)]], device const int* numBatches [[buffer(9)]], device const int* ftT [[buffer(10)]], device const int* ftD [[buffer(11)]], device const int* ftH [[buffer(12)]], device const int* ftW [[buffer(13)]], device float* yFt [[buffer(14)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  int gid = int(__gid.x);
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
    float accR = as_type<float>(0x00000000u);
    float accI = as_type<float>(0x00000000u);
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
const char* k_metal_navatala_cfd_compute_face_area_pair_weights_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_cfd_compute_face_area_pair_weights_f32(device const float* sfX [[buffer(0)]], device const float* sfY [[buffer(1)]], device const float* sfZ [[buffer(2)]], device const float* magSf [[buffer(3)]], device const uint* nFaces [[buffer(4)]], device float* faceWeight [[buffer(5)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  int f = int(__gid.x);
  int N = ((int)(nFaces[0]));
  if (f < N) {
    float sx = sfX[f];
    float sy = sfY[f];
    float sz = sfZ[f];
    float msf = magSf[f];
    float invSqrt = (as_type<float>(0x3f800000u) / sqrt(msf));
    float wx = (sx * invSqrt);
    float wy = ((sy * as_type<float>(0x3f8147aeu)) * invSqrt);
    float wz = ((sz * as_type<float>(0x3f828f5cu)) * invSqrt);
    float mag2 = (((wx * wx) + (wy * wy)) + (wz * wz));
    faceWeight[f] = sqrt(mag2);
  }
}

)kernel";
const char* k_metal_navatala_cfd_restrict_by_agglomeration_gather_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_cfd_restrict_by_agglomeration_gather_f32(device const float* fineField [[buffer(0)]], device const uint* coarseToFineOffsets [[buffer(1)]], device const uint* coarseToFineList [[buffer(2)]], device const uint* nCoarseCells [[buffer(3)]], device float* coarseField [[buffer(4)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  int c = int(__gid.x);
  int N = ((int)(nCoarseCells[0]));
  if (c < N) {
    int begin = ((int)(coarseToFineOffsets[c]));
    int end = ((int)(coarseToFineOffsets[(c + 1)]));
    float sum = as_type<float>(0x00000000u);
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
const char* k_metal_navatala_cfd_prolongate_by_agglomeration_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_cfd_prolongate_by_agglomeration_f32(device const float* coarseField [[buffer(0)]], device const uint* fineToCoarse [[buffer(1)]], device const uint* nFineCells [[buffer(2)]], device float* fineField [[buffer(3)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  int i = int(__gid.x);
  int N = ((int)(nFineCells[0]));
  if (i < N) {
    int coarseIdx = ((int)(fineToCoarse[i]));
    float val = coarseField[coarseIdx];
    float cur = fineField[i];
    fineField[i] = (cur + val);
  }
}

)kernel";
const char* k_metal_navatala_cfd_diag_inv_apply_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_cfd_diag_inv_apply_f32(device const float* r [[buffer(0)]], device const float* diagInv [[buffer(1)]], device const uint* n [[buffer(2)]], device float* z [[buffer(3)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  int i = int(__gid.x);
  int N = ((int)(n[0]));
  if (i < N) {
    float ri = r[i];
    float di = diagInv[i];
    z[i] = (di * ri);
  }
}

)kernel";
const char* k_metal_navatala_cfd_gamg_proc_pack_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_cfd_gamg_proc_pack_f32(device const float* x [[buffer(0)]], device const uint* localCells [[buffer(1)]], device const uint* count [[buffer(2)]], device float* sendVals [[buffer(3)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  int i = int(__gid.x);
  int N = ((int)(count[0]));
  if (i < N) {
    int cell = ((int)(localCells[i]));
    float v = x[cell];
    sendVals[i] = v;
  }
}

)kernel";
const char* k_metal_navatala_cfd_gamg_proc_add_gather_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_cfd_gamg_proc_add_gather_f32(device const uint* entryOffsets [[buffer(0)]], device const uint* entryIndices [[buffer(1)]], device const float* coeffs [[buffer(2)]], device const float* recvVals [[buffer(3)]], device const uint* nCells [[buffer(4)]], device float* y [[buffer(5)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  int c = int(__gid.x);
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

const KernelArgumentInfo kAbiArgs_metal_navatala_cfd_boundary_force_partials[] = {
  { "pAllFaces", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "sfComponent", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "counts", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 8, 8, 256, nullptr, 0, 0 },
  { "outPartials", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_cfd_boundary_force_partials = {
  1,
  "navatala_cfd_boundary_force_partials",
  "metal",
  "navatala_cfd_boundary_force_partials",
  "kernel:metal:navatala_cfd_boundary_force_partials",
  "abi-r1:metal:navatala_cfd_boundary_force_partials",
  "abi-r1:metal:navatala_cfd_boundary_force_partials",
  4,
  kAbiArgs_metal_navatala_cfd_boundary_force_partials
};

const KernelArgumentInfo kAbiArgs_metal_navatala_cfd_pack_owner_values[] = {
  { "x", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "owner", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "procFaces", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "counts", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "outSend", 4, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_cfd_pack_owner_values = {
  1,
  "navatala_cfd_pack_owner_values",
  "metal",
  "navatala_cfd_pack_owner_values",
  "kernel:metal:navatala_cfd_pack_owner_values",
  "abi-r1:metal:navatala_cfd_pack_owner_values",
  "abi-r1:metal:navatala_cfd_pack_owner_values",
  5,
  kAbiArgs_metal_navatala_cfd_pack_owner_values
};

const KernelArgumentInfo kAbiArgs_metal_navatala_cfd_scatter_values_and_mask[] = {
  { "procFaces", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "recv", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "counts", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "bcVal", 3, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "bcMask", 4, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_cfd_scatter_values_and_mask = {
  1,
  "navatala_cfd_scatter_values_and_mask",
  "metal",
  "navatala_cfd_scatter_values_and_mask",
  "kernel:metal:navatala_cfd_scatter_values_and_mask",
  "abi-r1:metal:navatala_cfd_scatter_values_and_mask",
  "abi-r1:metal:navatala_cfd_scatter_values_and_mask",
  5,
  kAbiArgs_metal_navatala_cfd_scatter_values_and_mask
};

const KernelArgumentInfo kAbiArgs_metal_navatala_cfd_pack_mu_grad_owner[] = {
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
const KernelAbiManifestInfo kAbiManifest_metal_navatala_cfd_pack_mu_grad_owner = {
  1,
  "navatala_cfd_pack_mu_grad_owner",
  "metal",
  "navatala_cfd_pack_mu_grad_owner",
  "kernel:metal:navatala_cfd_pack_mu_grad_owner",
  "abi-r1:metal:navatala_cfd_pack_mu_grad_owner",
  "abi-r1:metal:navatala_cfd_pack_mu_grad_owner",
  14,
  kAbiArgs_metal_navatala_cfd_pack_mu_grad_owner
};

const KernelArgumentInfo kAbiArgs_metal_navatala_cfd_pack_owner_rho_vec3[] = {
  { "rhoCell", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "ux", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "uy", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "uz", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "owner", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "procFaces", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "counts", 6, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "outSend", 7, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_cfd_pack_owner_rho_vec3 = {
  1,
  "navatala_cfd_pack_owner_rho_vec3",
  "metal",
  "navatala_cfd_pack_owner_rho_vec3",
  "kernel:metal:navatala_cfd_pack_owner_rho_vec3",
  "abi-r1:metal:navatala_cfd_pack_owner_rho_vec3",
  "abi-r1:metal:navatala_cfd_pack_owner_rho_vec3",
  8,
  kAbiArgs_metal_navatala_cfd_pack_owner_rho_vec3
};

const KernelArgumentInfo kAbiArgs_metal_navatala_cfd_pack_owner_vec3[] = {
  { "x", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "y", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "z", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "owner", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "procFaces", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "counts", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "outSend", 6, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_cfd_pack_owner_vec3 = {
  1,
  "navatala_cfd_pack_owner_vec3",
  "metal",
  "navatala_cfd_pack_owner_vec3",
  "kernel:metal:navatala_cfd_pack_owner_vec3",
  "abi-r1:metal:navatala_cfd_pack_owner_vec3",
  "abi-r1:metal:navatala_cfd_pack_owner_vec3",
  7,
  kAbiArgs_metal_navatala_cfd_pack_owner_vec3
};

const KernelArgumentInfo kAbiArgs_metal_navatala_cfd_pack_owner_vec3_mu[] = {
  { "ux", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "uy", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "uz", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "muCell", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "owner", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "procFaces", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "counts", 6, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "outSend", 7, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_cfd_pack_owner_vec3_mu = {
  1,
  "navatala_cfd_pack_owner_vec3_mu",
  "metal",
  "navatala_cfd_pack_owner_vec3_mu",
  "kernel:metal:navatala_cfd_pack_owner_vec3_mu",
  "abi-r1:metal:navatala_cfd_pack_owner_vec3_mu",
  "abi-r1:metal:navatala_cfd_pack_owner_vec3_mu",
  8,
  kAbiArgs_metal_navatala_cfd_pack_owner_vec3_mu
};

const KernelArgumentInfo kAbiArgs_metal_navatala_cfd_scatter_mu_grad_ghost[] = {
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
const KernelAbiManifestInfo kAbiManifest_metal_navatala_cfd_scatter_mu_grad_ghost = {
  1,
  "navatala_cfd_scatter_mu_grad_ghost",
  "metal",
  "navatala_cfd_scatter_mu_grad_ghost",
  "kernel:metal:navatala_cfd_scatter_mu_grad_ghost",
  "abi-r1:metal:navatala_cfd_scatter_mu_grad_ghost",
  "abi-r1:metal:navatala_cfd_scatter_mu_grad_ghost",
  12,
  kAbiArgs_metal_navatala_cfd_scatter_mu_grad_ghost
};

const KernelArgumentInfo kAbiArgs_metal_navatala_cfd_scatter_rho_vec3_and_mask[] = {
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
const KernelAbiManifestInfo kAbiManifest_metal_navatala_cfd_scatter_rho_vec3_and_mask = {
  1,
  "navatala_cfd_scatter_rho_vec3_and_mask",
  "metal",
  "navatala_cfd_scatter_rho_vec3_and_mask",
  "kernel:metal:navatala_cfd_scatter_rho_vec3_and_mask",
  "abi-r1:metal:navatala_cfd_scatter_rho_vec3_and_mask",
  "abi-r1:metal:navatala_cfd_scatter_rho_vec3_and_mask",
  9,
  kAbiArgs_metal_navatala_cfd_scatter_rho_vec3_and_mask
};

const KernelArgumentInfo kAbiArgs_metal_navatala_cfd_scatter_vec3_and_mask[] = {
  { "procFaces", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "recvVals", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "counts", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "bcX", 3, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "bcY", 4, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "bcZ", 5, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "bcMask", 6, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_cfd_scatter_vec3_and_mask = {
  1,
  "navatala_cfd_scatter_vec3_and_mask",
  "metal",
  "navatala_cfd_scatter_vec3_and_mask",
  "kernel:metal:navatala_cfd_scatter_vec3_and_mask",
  "abi-r1:metal:navatala_cfd_scatter_vec3_and_mask",
  "abi-r1:metal:navatala_cfd_scatter_vec3_and_mask",
  7,
  kAbiArgs_metal_navatala_cfd_scatter_vec3_and_mask
};

const KernelArgumentInfo kAbiArgs_metal_navatala_cfd_scatter_vec3_mu_and_mask[] = {
  { "procFaces", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "recvVals", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "counts", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "bcX", 3, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "bcY", 4, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "bcZ", 5, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "muBc", 6, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "bcMask", 7, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_cfd_scatter_vec3_mu_and_mask = {
  1,
  "navatala_cfd_scatter_vec3_mu_and_mask",
  "metal",
  "navatala_cfd_scatter_vec3_mu_and_mask",
  "kernel:metal:navatala_cfd_scatter_vec3_mu_and_mask",
  "abi-r1:metal:navatala_cfd_scatter_vec3_mu_and_mask",
  "abi-r1:metal:navatala_cfd_scatter_vec3_mu_and_mask",
  8,
  kAbiArgs_metal_navatala_cfd_scatter_vec3_mu_and_mask
};

const KernelArgumentInfo kAbiArgs_metal_navatala_cfd_add_vol_vector[] = {
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
const KernelAbiManifestInfo kAbiManifest_metal_navatala_cfd_add_vol_vector = {
  1,
  "navatala_cfd_add_vol_vector",
  "metal",
  "navatala_cfd_add_vol_vector",
  "kernel:metal:navatala_cfd_add_vol_vector",
  "abi-r1:metal:navatala_cfd_add_vol_vector",
  "abi-r1:metal:navatala_cfd_add_vol_vector",
  10,
  kAbiArgs_metal_navatala_cfd_add_vol_vector
};

const KernelArgumentInfo kAbiArgs_metal_navatala_cfd_div_rho_phi_u_explicit[] = {
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
const KernelAbiManifestInfo kAbiManifest_metal_navatala_cfd_div_rho_phi_u_explicit = {
  1,
  "navatala_cfd_div_rho_phi_u_explicit",
  "metal",
  "navatala_cfd_div_rho_phi_u_explicit",
  "kernel:metal:navatala_cfd_div_rho_phi_u_explicit",
  "abi-r1:metal:navatala_cfd_div_rho_phi_u_explicit",
  "abi-r1:metal:navatala_cfd_div_rho_phi_u_explicit",
  22,
  kAbiArgs_metal_navatala_cfd_div_rho_phi_u_explicit
};

const KernelArgumentInfo kAbiArgs_metal_navatala_cfd_hby_a_from_u_and_terms[] = {
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
const KernelAbiManifestInfo kAbiManifest_metal_navatala_cfd_hby_a_from_u_and_terms = {
  1,
  "navatala_cfd_hby_a_from_u_and_terms",
  "metal",
  "navatala_cfd_hby_a_from_u_and_terms",
  "kernel:metal:navatala_cfd_hby_a_from_u_and_terms",
  "abi-r1:metal:navatala_cfd_hby_a_from_u_and_terms",
  "abi-r1:metal:navatala_cfd_hby_a_from_u_and_terms",
  18,
  kAbiArgs_metal_navatala_cfd_hby_a_from_u_and_terms
};

const KernelArgumentInfo kAbiArgs_metal_navatala_cfd_apply_sp_to_hby_a[] = {
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
const KernelAbiManifestInfo kAbiManifest_metal_navatala_cfd_apply_sp_to_hby_a = {
  1,
  "navatala_cfd_apply_sp_to_hby_a",
  "metal",
  "navatala_cfd_apply_sp_to_hby_a",
  "kernel:metal:navatala_cfd_apply_sp_to_hby_a",
  "abi-r1:metal:navatala_cfd_apply_sp_to_hby_a",
  "abi-r1:metal:navatala_cfd_apply_sp_to_hby_a",
  9,
  kAbiArgs_metal_navatala_cfd_apply_sp_to_hby_a
};

const KernelArgumentInfo kAbiArgs_metal_navatala_cfd_laplacian_scalar_explicit[] = {
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
const KernelAbiManifestInfo kAbiManifest_metal_navatala_cfd_laplacian_scalar_explicit = {
  1,
  "navatala_cfd_laplacian_scalar_explicit",
  "metal",
  "navatala_cfd_laplacian_scalar_explicit",
  "kernel:metal:navatala_cfd_laplacian_scalar_explicit",
  "abi-r1:metal:navatala_cfd_laplacian_scalar_explicit",
  "abi-r1:metal:navatala_cfd_laplacian_scalar_explicit",
  17,
  kAbiArgs_metal_navatala_cfd_laplacian_scalar_explicit
};

const KernelArgumentInfo kAbiArgs_metal_navatala_cfd_laplacian_vector_explicit[] = {
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
const KernelAbiManifestInfo kAbiManifest_metal_navatala_cfd_laplacian_vector_explicit = {
  1,
  "navatala_cfd_laplacian_vector_explicit",
  "metal",
  "navatala_cfd_laplacian_vector_explicit",
  "kernel:metal:navatala_cfd_laplacian_vector_explicit",
  "abi-r1:metal:navatala_cfd_laplacian_vector_explicit",
  "abi-r1:metal:navatala_cfd_laplacian_vector_explicit",
  28,
  kAbiArgs_metal_navatala_cfd_laplacian_vector_explicit
};

const KernelArgumentInfo kAbiArgs_metal_navatala_cfd_predictor_hby_a_from_source[] = {
  { "rAU", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "sourceX", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "sourceY", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "sourceZ", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "counts1", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "outHbyAX", 5, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "outHbyAY", 6, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "outHbyAZ", 7, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_cfd_predictor_hby_a_from_source = {
  1,
  "navatala_cfd_predictor_hby_a_from_source",
  "metal",
  "navatala_cfd_predictor_hby_a_from_source",
  "kernel:metal:navatala_cfd_predictor_hby_a_from_source",
  "abi-r1:metal:navatala_cfd_predictor_hby_a_from_source",
  "abi-r1:metal:navatala_cfd_predictor_hby_a_from_source",
  8,
  kAbiArgs_metal_navatala_cfd_predictor_hby_a_from_source
};

const KernelArgumentInfo kAbiArgs_metal_navatala_cfd_predictor_inverse_diag[] = {
  { "inputDiag", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "counts1", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "outRAU", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_cfd_predictor_inverse_diag = {
  1,
  "navatala_cfd_predictor_inverse_diag",
  "metal",
  "navatala_cfd_predictor_inverse_diag",
  "kernel:metal:navatala_cfd_predictor_inverse_diag",
  "abi-r1:metal:navatala_cfd_predictor_inverse_diag",
  "abi-r1:metal:navatala_cfd_predictor_inverse_diag",
  3,
  kAbiArgs_metal_navatala_cfd_predictor_inverse_diag
};

const KernelArgumentInfo kAbiArgs_metal_navatala_cfd_predictor_phi_hby_a_fuse[] = {
  { "fluxHbyA", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "rhoRauFace", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "ddtCorr", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "counts1", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "outPhiHbyA", 4, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_cfd_predictor_phi_hby_a_fuse = {
  1,
  "navatala_cfd_predictor_phi_hby_a_fuse",
  "metal",
  "navatala_cfd_predictor_phi_hby_a_fuse",
  "kernel:metal:navatala_cfd_predictor_phi_hby_a_fuse",
  "abi-r1:metal:navatala_cfd_predictor_phi_hby_a_fuse",
  "abi-r1:metal:navatala_cfd_predictor_phi_hby_a_fuse",
  5,
  kAbiArgs_metal_navatala_cfd_predictor_phi_hby_a_fuse
};

const KernelArgumentInfo kAbiArgs_metal_navatala_cfd_axpy_cells[] = {
  { "x", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "counts", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "paramsF", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "y", 3, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_cfd_axpy_cells = {
  1,
  "navatala_cfd_axpy_cells",
  "metal",
  "navatala_cfd_axpy_cells",
  "kernel:metal:navatala_cfd_axpy_cells",
  "abi-r1:metal:navatala_cfd_axpy_cells",
  "abi-r1:metal:navatala_cfd_axpy_cells",
  4,
  kAbiArgs_metal_navatala_cfd_axpy_cells
};

const KernelArgumentInfo kAbiArgs_metal_navatala_cfd_ddt_diagonal[] = {
  { "rho", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "vol", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "counts", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "paramsF", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "outDiag", 4, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_cfd_ddt_diagonal = {
  1,
  "navatala_cfd_ddt_diagonal",
  "metal",
  "navatala_cfd_ddt_diagonal",
  "kernel:metal:navatala_cfd_ddt_diagonal",
  "abi-r1:metal:navatala_cfd_ddt_diagonal",
  "abi-r1:metal:navatala_cfd_ddt_diagonal",
  5,
  kAbiArgs_metal_navatala_cfd_ddt_diagonal
};

const KernelArgumentInfo kAbiArgs_metal_navatala_cfd_div_upwind_coeffs[] = {
  { "weights", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "flux", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "counts", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "outLower", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "outUpper", 4, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_cfd_div_upwind_coeffs = {
  1,
  "navatala_cfd_div_upwind_coeffs",
  "metal",
  "navatala_cfd_div_upwind_coeffs",
  "kernel:metal:navatala_cfd_div_upwind_coeffs",
  "abi-r1:metal:navatala_cfd_div_upwind_coeffs",
  "abi-r1:metal:navatala_cfd_div_upwind_coeffs",
  5,
  kAbiArgs_metal_navatala_cfd_div_upwind_coeffs
};

const KernelArgumentInfo kAbiArgs_metal_navatala_cfd_fvm_sp[] = {
  { "sp", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "vol", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "counts", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "outDiag", 3, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_cfd_fvm_sp = {
  1,
  "navatala_cfd_fvm_sp",
  "metal",
  "navatala_cfd_fvm_sp",
  "kernel:metal:navatala_cfd_fvm_sp",
  "abi-r1:metal:navatala_cfd_fvm_sp",
  "abi-r1:metal:navatala_cfd_fvm_sp",
  4,
  kAbiArgs_metal_navatala_cfd_fvm_sp
};

const KernelArgumentInfo kAbiArgs_metal_navatala_cfd_fvm_su_sp[] = {
  { "sp", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "psi", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "vol", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "counts", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "outDiag", 4, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "outSource", 5, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_cfd_fvm_su_sp = {
  1,
  "navatala_cfd_fvm_su_sp",
  "metal",
  "navatala_cfd_fvm_su_sp",
  "kernel:metal:navatala_cfd_fvm_su_sp",
  "abi-r1:metal:navatala_cfd_fvm_su_sp",
  "abi-r1:metal:navatala_cfd_fvm_su_sp",
  6,
  kAbiArgs_metal_navatala_cfd_fvm_su_sp
};

const KernelArgumentInfo kAbiArgs_metal_navatala_cfd_laplacian_coeffs[] = {
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
const KernelAbiManifestInfo kAbiManifest_metal_navatala_cfd_laplacian_coeffs = {
  1,
  "navatala_cfd_laplacian_coeffs",
  "metal",
  "navatala_cfd_laplacian_coeffs",
  "kernel:metal:navatala_cfd_laplacian_coeffs",
  "abi-r1:metal:navatala_cfd_laplacian_coeffs",
  "abi-r1:metal:navatala_cfd_laplacian_coeffs",
  9,
  kAbiArgs_metal_navatala_cfd_laplacian_coeffs
};

const KernelArgumentInfo kAbiArgs_metal_navatala_cfd_laplacian_coeffs_accum[] = {
  { "gamma", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "deltaCoeffs", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "magSf", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "counts", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "outUpper", 4, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "outLower", 5, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_cfd_laplacian_coeffs_accum = {
  1,
  "navatala_cfd_laplacian_coeffs_accum",
  "metal",
  "navatala_cfd_laplacian_coeffs_accum",
  "kernel:metal:navatala_cfd_laplacian_coeffs_accum",
  "abi-r1:metal:navatala_cfd_laplacian_coeffs_accum",
  "abi-r1:metal:navatala_cfd_laplacian_coeffs_accum",
  6,
  kAbiArgs_metal_navatala_cfd_laplacian_coeffs_accum
};

const KernelArgumentInfo kAbiArgs_metal_navatala_cfd_linear_upwind_correction_source[] = {
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
const KernelAbiManifestInfo kAbiManifest_metal_navatala_cfd_linear_upwind_correction_source = {
  1,
  "navatala_cfd_linear_upwind_correction_source",
  "metal",
  "navatala_cfd_linear_upwind_correction_source",
  "kernel:metal:navatala_cfd_linear_upwind_correction_source",
  "abi-r1:metal:navatala_cfd_linear_upwind_correction_source",
  "abi-r1:metal:navatala_cfd_linear_upwind_correction_source",
  10,
  kAbiArgs_metal_navatala_cfd_linear_upwind_correction_source
};

const KernelArgumentInfo kAbiArgs_metal_navatala_cfd_matrix_relax[] = {
  { "diagOld", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "sumOff", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "psi", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "counts", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "paramsF", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "diag", 5, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "source", 6, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_cfd_matrix_relax = {
  1,
  "navatala_cfd_matrix_relax",
  "metal",
  "navatala_cfd_matrix_relax",
  "kernel:metal:navatala_cfd_matrix_relax",
  "abi-r1:metal:navatala_cfd_matrix_relax",
  "abi-r1:metal:navatala_cfd_matrix_relax",
  7,
  kAbiArgs_metal_navatala_cfd_matrix_relax
};

const KernelArgumentInfo kAbiArgs_metal_navatala_cfd_neg_sum_diag[] = {
  { "upper", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "lower", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "offsets", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "faceIdx", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "sign", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "counts", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 8, 8, 256, nullptr, 0, 0 },
  { "outDiag", 6, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_cfd_neg_sum_diag = {
  1,
  "navatala_cfd_neg_sum_diag",
  "metal",
  "navatala_cfd_neg_sum_diag",
  "kernel:metal:navatala_cfd_neg_sum_diag",
  "abi-r1:metal:navatala_cfd_neg_sum_diag",
  "abi-r1:metal:navatala_cfd_neg_sum_diag",
  7,
  kAbiArgs_metal_navatala_cfd_neg_sum_diag
};

const KernelArgumentInfo kAbiArgs_metal_navatala_cfd_pack_boundary_adjacent_cells[] = {
  { "cellField", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "indexMap", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "counts", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "outPacked", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_cfd_pack_boundary_adjacent_cells = {
  1,
  "navatala_cfd_pack_boundary_adjacent_cells",
  "metal",
  "navatala_cfd_pack_boundary_adjacent_cells",
  "kernel:metal:navatala_cfd_pack_boundary_adjacent_cells",
  "abi-r1:metal:navatala_cfd_pack_boundary_adjacent_cells",
  "abi-r1:metal:navatala_cfd_pack_boundary_adjacent_cells",
  4,
  kAbiArgs_metal_navatala_cfd_pack_boundary_adjacent_cells
};

const KernelArgumentInfo kAbiArgs_metal_navatala_cfd_scalar_jacobi_clamp_min[] = {
  { "lowerBound", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "counts", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "x", 2, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_cfd_scalar_jacobi_clamp_min = {
  1,
  "navatala_cfd_scalar_jacobi_clamp_min",
  "metal",
  "navatala_cfd_scalar_jacobi_clamp_min",
  "kernel:metal:navatala_cfd_scalar_jacobi_clamp_min",
  "abi-r1:metal:navatala_cfd_scalar_jacobi_clamp_min",
  "abi-r1:metal:navatala_cfd_scalar_jacobi_clamp_min",
  3,
  kAbiArgs_metal_navatala_cfd_scalar_jacobi_clamp_min
};

const KernelArgumentInfo kAbiArgs_metal_navatala_cfd_scalar_jacobi_update[] = {
  { "ax", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "diag", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "rhs", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "params", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "counts", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "x", 5, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_cfd_scalar_jacobi_update = {
  1,
  "navatala_cfd_scalar_jacobi_update",
  "metal",
  "navatala_cfd_scalar_jacobi_update",
  "kernel:metal:navatala_cfd_scalar_jacobi_update",
  "abi-r1:metal:navatala_cfd_scalar_jacobi_update",
  "abi-r1:metal:navatala_cfd_scalar_jacobi_update",
  6,
  kAbiArgs_metal_navatala_cfd_scalar_jacobi_update
};

const KernelArgumentInfo kAbiArgs_metal_navatala_cfd_scalar_ldu_mat_vec[] = {
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
const KernelAbiManifestInfo kAbiManifest_metal_navatala_cfd_scalar_ldu_mat_vec = {
  1,
  "navatala_cfd_scalar_ldu_mat_vec",
  "metal",
  "navatala_cfd_scalar_ldu_mat_vec",
  "kernel:metal:navatala_cfd_scalar_ldu_mat_vec",
  "abi-r1:metal:navatala_cfd_scalar_ldu_mat_vec",
  "abi-r1:metal:navatala_cfd_scalar_ldu_mat_vec",
  11,
  kAbiArgs_metal_navatala_cfd_scalar_ldu_mat_vec
};

const KernelArgumentInfo kAbiArgs_metal_navatala_cfd_scalar_ldu_multi_color_gs_sweep[] = {
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
const KernelAbiManifestInfo kAbiManifest_metal_navatala_cfd_scalar_ldu_multi_color_gs_sweep = {
  1,
  "navatala_cfd_scalar_ldu_multi_color_gs_sweep",
  "metal",
  "navatala_cfd_scalar_ldu_multi_color_gs_sweep",
  "kernel:metal:navatala_cfd_scalar_ldu_multi_color_gs_sweep",
  "abi-r1:metal:navatala_cfd_scalar_ldu_multi_color_gs_sweep",
  "abi-r1:metal:navatala_cfd_scalar_ldu_multi_color_gs_sweep",
  14,
  kAbiArgs_metal_navatala_cfd_scalar_ldu_multi_color_gs_sweep
};

const KernelArgumentInfo kAbiArgs_metal_navatala_cfd_sum_mag_off_diag[] = {
  { "upper", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "lower", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "offsets", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "faceIdx", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "sign", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "counts", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 8, 8, 256, nullptr, 0, 0 },
  { "outSumOff", 6, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_cfd_sum_mag_off_diag = {
  1,
  "navatala_cfd_sum_mag_off_diag",
  "metal",
  "navatala_cfd_sum_mag_off_diag",
  "kernel:metal:navatala_cfd_sum_mag_off_diag",
  "abi-r1:metal:navatala_cfd_sum_mag_off_diag",
  "abi-r1:metal:navatala_cfd_sum_mag_off_diag",
  7,
  kAbiArgs_metal_navatala_cfd_sum_mag_off_diag
};

const KernelArgumentInfo kAbiArgs_metal_navatala_cfd_gamg_axpy_in_place[] = {
  { "x", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "alpha", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "nRows", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "y", 3, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_cfd_gamg_axpy_in_place = {
  1,
  "navatala_cfd_gamg_axpy_in_place",
  "metal",
  "navatala_cfd_gamg_axpy_in_place",
  "kernel:metal:navatala_cfd_gamg_axpy_in_place",
  "abi-r1:metal:navatala_cfd_gamg_axpy_in_place",
  "abi-r1:metal:navatala_cfd_gamg_axpy_in_place",
  4,
  kAbiArgs_metal_navatala_cfd_gamg_axpy_in_place
};

const KernelArgumentInfo kAbiArgs_metal_navatala_cfd_gamg_compute_scale_factor[] = {
  { "num", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "den", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "sf", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_cfd_gamg_compute_scale_factor = {
  1,
  "navatala_cfd_gamg_compute_scale_factor",
  "metal",
  "navatala_cfd_gamg_compute_scale_factor",
  "kernel:metal:navatala_cfd_gamg_compute_scale_factor",
  "abi-r1:metal:navatala_cfd_gamg_compute_scale_factor",
  "abi-r1:metal:navatala_cfd_gamg_compute_scale_factor",
  3,
  kAbiArgs_metal_navatala_cfd_gamg_compute_scale_factor
};

const KernelArgumentInfo kAbiArgs_metal_navatala_cfd_gamg_dot_finalize[] = {
  { "partials", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "result", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_cfd_gamg_dot_finalize = {
  1,
  "navatala_cfd_gamg_dot_finalize",
  "metal",
  "navatala_cfd_gamg_dot_finalize",
  "kernel:metal:navatala_cfd_gamg_dot_finalize",
  "abi-r1:metal:navatala_cfd_gamg_dot_finalize",
  "abi-r1:metal:navatala_cfd_gamg_dot_finalize",
  3,
  kAbiArgs_metal_navatala_cfd_gamg_dot_finalize
};

const KernelArgumentInfo kAbiArgs_metal_navatala_cfd_gamg_negate_scalar[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 1, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_cfd_gamg_negate_scalar = {
  1,
  "navatala_cfd_gamg_negate_scalar",
  "metal",
  "navatala_cfd_gamg_negate_scalar",
  "kernel:metal:navatala_cfd_gamg_negate_scalar",
  "abi-r1:metal:navatala_cfd_gamg_negate_scalar",
  "abi-r1:metal:navatala_cfd_gamg_negate_scalar",
  2,
  kAbiArgs_metal_navatala_cfd_gamg_negate_scalar
};

const KernelArgumentInfo kAbiArgs_metal_navatala_cfd_gamg_scale_correction[] = {
  { "z", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "r", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "Az", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "diagInv", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "sfPtr", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "nRows", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "corr", 6, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_cfd_gamg_scale_correction = {
  1,
  "navatala_cfd_gamg_scale_correction",
  "metal",
  "navatala_cfd_gamg_scale_correction",
  "kernel:metal:navatala_cfd_gamg_scale_correction",
  "abi-r1:metal:navatala_cfd_gamg_scale_correction",
  "abi-r1:metal:navatala_cfd_gamg_scale_correction",
  7,
  kAbiArgs_metal_navatala_cfd_gamg_scale_correction
};

const KernelArgumentInfo kAbiArgs_metal_navatala_cfd_mg_coarse_jacobi[] = {
  { "x", 0, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "r", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "diag", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "mgCounts", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_cfd_mg_coarse_jacobi = {
  1,
  "navatala_cfd_mg_coarse_jacobi",
  "metal",
  "navatala_cfd_mg_coarse_jacobi",
  "kernel:metal:navatala_cfd_mg_coarse_jacobi",
  "abi-r1:metal:navatala_cfd_mg_coarse_jacobi",
  "abi-r1:metal:navatala_cfd_mg_coarse_jacobi",
  4,
  kAbiArgs_metal_navatala_cfd_mg_coarse_jacobi
};

const KernelArgumentInfo kAbiArgs_metal_navatala_cfd_mg_coarse_matvec[] = {
  { "edgeU", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "edgeV", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "edgeCf", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "diag", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "x", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "outAx", 5, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "mgCounts", 6, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_cfd_mg_coarse_matvec = {
  1,
  "navatala_cfd_mg_coarse_matvec",
  "metal",
  "navatala_cfd_mg_coarse_matvec",
  "kernel:metal:navatala_cfd_mg_coarse_matvec",
  "abi-r1:metal:navatala_cfd_mg_coarse_matvec",
  "abi-r1:metal:navatala_cfd_mg_coarse_matvec",
  7,
  kAbiArgs_metal_navatala_cfd_mg_coarse_matvec
};

const KernelArgumentInfo kAbiArgs_metal_navatala_cfd_mg_coarse_rhs[] = {
  { "b", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "ax", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "r", 2, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "mgCounts", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_cfd_mg_coarse_rhs = {
  1,
  "navatala_cfd_mg_coarse_rhs",
  "metal",
  "navatala_cfd_mg_coarse_rhs",
  "kernel:metal:navatala_cfd_mg_coarse_rhs",
  "abi-r1:metal:navatala_cfd_mg_coarse_rhs",
  "abi-r1:metal:navatala_cfd_mg_coarse_rhs",
  4,
  kAbiArgs_metal_navatala_cfd_mg_coarse_rhs
};

const KernelArgumentInfo kAbiArgs_metal_navatala_cfd_mg_find_best_match[] = {
  { "csrOffsets", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "csrFaceIdx", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "csrSign", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "strength", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "owner", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "nei", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "bestNeighbor", 6, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "counts", 7, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_cfd_mg_find_best_match = {
  1,
  "navatala_cfd_mg_find_best_match",
  "metal",
  "navatala_cfd_mg_find_best_match",
  "kernel:metal:navatala_cfd_mg_find_best_match",
  "abi-r1:metal:navatala_cfd_mg_find_best_match",
  "abi-r1:metal:navatala_cfd_mg_find_best_match",
  8,
  kAbiArgs_metal_navatala_cfd_mg_find_best_match
};

const KernelArgumentInfo kAbiArgs_metal_navatala_cfd_mg_pair_match[] = {
  { "bestNeighbor", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "aggMap", 1, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "counts", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_cfd_mg_pair_match = {
  1,
  "navatala_cfd_mg_pair_match",
  "metal",
  "navatala_cfd_mg_pair_match",
  "kernel:metal:navatala_cfd_mg_pair_match",
  "abi-r1:metal:navatala_cfd_mg_pair_match",
  "abi-r1:metal:navatala_cfd_mg_pair_match",
  3,
  kAbiArgs_metal_navatala_cfd_mg_pair_match
};

const KernelArgumentInfo kAbiArgs_metal_navatala_cfd_mg_prolong_add[] = {
  { "xCoarse", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "prolongRowOffsets", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "prolongCols", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "prolongVals", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "zFine", 4, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "mgCounts", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_cfd_mg_prolong_add = {
  1,
  "navatala_cfd_mg_prolong_add",
  "metal",
  "navatala_cfd_mg_prolong_add",
  "kernel:metal:navatala_cfd_mg_prolong_add",
  "abi-r1:metal:navatala_cfd_mg_prolong_add",
  "abi-r1:metal:navatala_cfd_mg_prolong_add",
  6,
  kAbiArgs_metal_navatala_cfd_mg_prolong_add
};

const KernelArgumentInfo kAbiArgs_metal_navatala_cfd_mg_renumber_aggregates[] = {
  { "aggMap", 0, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "counter", 1, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "counts", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_cfd_mg_renumber_aggregates = {
  1,
  "navatala_cfd_mg_renumber_aggregates",
  "metal",
  "navatala_cfd_mg_renumber_aggregates",
  "kernel:metal:navatala_cfd_mg_renumber_aggregates",
  "abi-r1:metal:navatala_cfd_mg_renumber_aggregates",
  "abi-r1:metal:navatala_cfd_mg_renumber_aggregates",
  3,
  kAbiArgs_metal_navatala_cfd_mg_renumber_aggregates
};

const KernelArgumentInfo kAbiArgs_metal_navatala_cfd_mg_renumber_propagate[] = {
  { "aggMap", 0, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "counts", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_cfd_mg_renumber_propagate = {
  1,
  "navatala_cfd_mg_renumber_propagate",
  "metal",
  "navatala_cfd_mg_renumber_propagate",
  "kernel:metal:navatala_cfd_mg_renumber_propagate",
  "abi-r1:metal:navatala_cfd_mg_renumber_propagate",
  "abi-r1:metal:navatala_cfd_mg_renumber_propagate",
  2,
  kAbiArgs_metal_navatala_cfd_mg_renumber_propagate
};

const KernelArgumentInfo kAbiArgs_metal_navatala_cfd_mg_resolve_chains[] = {
  { "aggMap", 0, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "counts", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_cfd_mg_resolve_chains = {
  1,
  "navatala_cfd_mg_resolve_chains",
  "metal",
  "navatala_cfd_mg_resolve_chains",
  "kernel:metal:navatala_cfd_mg_resolve_chains",
  "abi-r1:metal:navatala_cfd_mg_resolve_chains",
  "abi-r1:metal:navatala_cfd_mg_resolve_chains",
  2,
  kAbiArgs_metal_navatala_cfd_mg_resolve_chains
};

const KernelArgumentInfo kAbiArgs_metal_navatala_cfd_mg_restrict_sum[] = {
  { "rFine", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "prolongRowOffsets", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "prolongCols", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "prolongVals", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "rCoarse", 4, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "mgCounts", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_cfd_mg_restrict_sum = {
  1,
  "navatala_cfd_mg_restrict_sum",
  "metal",
  "navatala_cfd_mg_restrict_sum",
  "kernel:metal:navatala_cfd_mg_restrict_sum",
  "abi-r1:metal:navatala_cfd_mg_restrict_sum",
  "abi-r1:metal:navatala_cfd_mg_restrict_sum",
  6,
  kAbiArgs_metal_navatala_cfd_mg_restrict_sum
};

const KernelArgumentInfo kAbiArgs_metal_navatala_cfd_mg_strength_of_connection[] = {
  { "diag", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "upper", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "owner", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "nei", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "strength", 4, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "counts", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_cfd_mg_strength_of_connection = {
  1,
  "navatala_cfd_mg_strength_of_connection",
  "metal",
  "navatala_cfd_mg_strength_of_connection",
  "kernel:metal:navatala_cfd_mg_strength_of_connection",
  "abi-r1:metal:navatala_cfd_mg_strength_of_connection",
  "abi-r1:metal:navatala_cfd_mg_strength_of_connection",
  6,
  kAbiArgs_metal_navatala_cfd_mg_strength_of_connection
};

const KernelArgumentInfo kAbiArgs_metal_navatala_cfd_mg_zero_coarse[] = {
  { "x", 0, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "mgCounts", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_cfd_mg_zero_coarse = {
  1,
  "navatala_cfd_mg_zero_coarse",
  "metal",
  "navatala_cfd_mg_zero_coarse",
  "kernel:metal:navatala_cfd_mg_zero_coarse",
  "abi-r1:metal:navatala_cfd_mg_zero_coarse",
  "abi-r1:metal:navatala_cfd_mg_zero_coarse",
  2,
  kAbiArgs_metal_navatala_cfd_mg_zero_coarse
};

const KernelArgumentInfo kAbiArgs_metal_navatala_cfd_phi_correct[] = {
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
const KernelAbiManifestInfo kAbiManifest_metal_navatala_cfd_phi_correct = {
  1,
  "navatala_cfd_phi_correct",
  "metal",
  "navatala_cfd_phi_correct",
  "kernel:metal:navatala_cfd_phi_correct",
  "abi-r1:metal:navatala_cfd_phi_correct",
  "abi-r1:metal:navatala_cfd_phi_correct",
  15,
  kAbiArgs_metal_navatala_cfd_phi_correct
};

const KernelArgumentInfo kAbiArgs_metal_navatala_cfd_phi_correct_boundary[] = {
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
const KernelAbiManifestInfo kAbiManifest_metal_navatala_cfd_phi_correct_boundary = {
  1,
  "navatala_cfd_phi_correct_boundary",
  "metal",
  "navatala_cfd_phi_correct_boundary",
  "kernel:metal:navatala_cfd_phi_correct_boundary",
  "abi-r1:metal:navatala_cfd_phi_correct_boundary",
  "abi-r1:metal:navatala_cfd_phi_correct_boundary",
  14,
  kAbiArgs_metal_navatala_cfd_phi_correct_boundary
};

const KernelArgumentInfo kAbiArgs_metal_navatala_cfd_phi_correct_internal[] = {
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
const KernelAbiManifestInfo kAbiManifest_metal_navatala_cfd_phi_correct_internal = {
  1,
  "navatala_cfd_phi_correct_internal",
  "metal",
  "navatala_cfd_phi_correct_internal",
  "kernel:metal:navatala_cfd_phi_correct_internal",
  "abi-r1:metal:navatala_cfd_phi_correct_internal",
  "abi-r1:metal:navatala_cfd_phi_correct_internal",
  11,
  kAbiArgs_metal_navatala_cfd_phi_correct_internal
};

const KernelArgumentInfo kAbiArgs_metal_navatala_cfd_axpy_in_place[] = {
  { "x", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "y", 1, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "counts", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 12, 12, 256, nullptr, 0, 0 },
  { "paramsF", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_cfd_axpy_in_place = {
  1,
  "navatala_cfd_axpy_in_place",
  "metal",
  "navatala_cfd_axpy_in_place",
  "kernel:metal:navatala_cfd_axpy_in_place",
  "abi-r1:metal:navatala_cfd_axpy_in_place",
  "abi-r1:metal:navatala_cfd_axpy_in_place",
  4,
  kAbiArgs_metal_navatala_cfd_axpy_in_place
};

const KernelArgumentInfo kAbiArgs_metal_navatala_cfd_bc_dirichlet_face_flux[] = {
  { "cf", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "bcVal", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "bcMask", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "counts", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 12, 12, 256, nullptr, 0, 0 },
  { "outFlux", 4, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_cfd_bc_dirichlet_face_flux = {
  1,
  "navatala_cfd_bc_dirichlet_face_flux",
  "metal",
  "navatala_cfd_bc_dirichlet_face_flux",
  "kernel:metal:navatala_cfd_bc_dirichlet_face_flux",
  "abi-r1:metal:navatala_cfd_bc_dirichlet_face_flux",
  "abi-r1:metal:navatala_cfd_bc_dirichlet_face_flux",
  5,
  kAbiArgs_metal_navatala_cfd_bc_dirichlet_face_flux
};

const KernelArgumentInfo kAbiArgs_metal_navatala_cfd_bc_sn_grad_face_flux[] = {
  { "cf", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "delta", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "bcSnGrad", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "bcSnGradMask", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "counts", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 12, 12, 256, nullptr, 0, 0 },
  { "outFlux", 5, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_cfd_bc_sn_grad_face_flux = {
  1,
  "navatala_cfd_bc_sn_grad_face_flux",
  "metal",
  "navatala_cfd_bc_sn_grad_face_flux",
  "kernel:metal:navatala_cfd_bc_sn_grad_face_flux",
  "abi-r1:metal:navatala_cfd_bc_sn_grad_face_flux",
  "abi-r1:metal:navatala_cfd_bc_sn_grad_face_flux",
  6,
  kAbiArgs_metal_navatala_cfd_bc_sn_grad_face_flux
};

const KernelArgumentInfo kAbiArgs_metal_navatala_cfd_coeff_to_cf_in_place[] = {
  { "cf", 0, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "magSf", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "delta", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "counts", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 12, 12, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_cfd_coeff_to_cf_in_place = {
  1,
  "navatala_cfd_coeff_to_cf_in_place",
  "metal",
  "navatala_cfd_coeff_to_cf_in_place",
  "kernel:metal:navatala_cfd_coeff_to_cf_in_place",
  "abi-r1:metal:navatala_cfd_coeff_to_cf_in_place",
  "abi-r1:metal:navatala_cfd_coeff_to_cf_in_place",
  4,
  kAbiArgs_metal_navatala_cfd_coeff_to_cf_in_place
};

const KernelArgumentInfo kAbiArgs_metal_navatala_cfd_diag_from_cf[] = {
  { "cf", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "offsets", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "faceIdx", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "vol", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "counts", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 12, 12, 256, nullptr, 0, 0 },
  { "diag", 5, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_cfd_diag_from_cf = {
  1,
  "navatala_cfd_diag_from_cf",
  "metal",
  "navatala_cfd_diag_from_cf",
  "kernel:metal:navatala_cfd_diag_from_cf",
  "abi-r1:metal:navatala_cfd_diag_from_cf",
  "abi-r1:metal:navatala_cfd_diag_from_cf",
  6,
  kAbiArgs_metal_navatala_cfd_diag_from_cf
};

const KernelArgumentInfo kAbiArgs_metal_navatala_cfd_dic_apply[] = {
  { "r", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "rD", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "upper", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "owner", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "neighbour", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "counts", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 12, 12, 256, nullptr, 0, 0 },
  { "z", 6, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_cfd_dic_apply = {
  1,
  "navatala_cfd_dic_apply",
  "metal",
  "navatala_cfd_dic_apply",
  "kernel:metal:navatala_cfd_dic_apply",
  "abi-r1:metal:navatala_cfd_dic_apply",
  "abi-r1:metal:navatala_cfd_dic_apply",
  7,
  kAbiArgs_metal_navatala_cfd_dic_apply
};

const KernelArgumentInfo kAbiArgs_metal_navatala_cfd_dic_build_reciprocal_d[] = {
  { "diag", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "upper", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "owner", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "neighbour", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "counts", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 12, 12, 256, nullptr, 0, 0 },
  { "rD", 5, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_cfd_dic_build_reciprocal_d = {
  1,
  "navatala_cfd_dic_build_reciprocal_d",
  "metal",
  "navatala_cfd_dic_build_reciprocal_d",
  "kernel:metal:navatala_cfd_dic_build_reciprocal_d",
  "abi-r1:metal:navatala_cfd_dic_build_reciprocal_d",
  "abi-r1:metal:navatala_cfd_dic_build_reciprocal_d",
  6,
  kAbiArgs_metal_navatala_cfd_dic_build_reciprocal_d
};

const KernelArgumentInfo kAbiArgs_metal_navatala_cfd_dot_partials[] = {
  { "a", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "b", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "counts", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 12, 12, 256, nullptr, 0, 0 },
  { "outPartials", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_cfd_dot_partials = {
  1,
  "navatala_cfd_dot_partials",
  "metal",
  "navatala_cfd_dot_partials",
  "kernel:metal:navatala_cfd_dot_partials",
  "abi-r1:metal:navatala_cfd_dot_partials",
  "abi-r1:metal:navatala_cfd_dot_partials",
  4,
  kAbiArgs_metal_navatala_cfd_dot_partials
};

const KernelArgumentInfo kAbiArgs_metal_navatala_cfd_face_flux[] = {
  { "x", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "owner", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "neighbour", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "cf", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "bcVal", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "bcMask", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "counts", 6, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 12, 12, 256, nullptr, 0, 0 },
  { "outFlux", 7, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_cfd_face_flux = {
  1,
  "navatala_cfd_face_flux",
  "metal",
  "navatala_cfd_face_flux",
  "kernel:metal:navatala_cfd_face_flux",
  "abi-r1:metal:navatala_cfd_face_flux",
  "abi-r1:metal:navatala_cfd_face_flux",
  8,
  kAbiArgs_metal_navatala_cfd_face_flux
};

const KernelArgumentInfo kAbiArgs_metal_navatala_cfd_inv_diag[] = {
  { "r", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "diag", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "counts", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 12, 12, 256, nullptr, 0, 0 },
  { "z", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_cfd_inv_diag = {
  1,
  "navatala_cfd_inv_diag",
  "metal",
  "navatala_cfd_inv_diag",
  "kernel:metal:navatala_cfd_inv_diag",
  "abi-r1:metal:navatala_cfd_inv_diag",
  "abi-r1:metal:navatala_cfd_inv_diag",
  4,
  kAbiArgs_metal_navatala_cfd_inv_diag
};

const KernelArgumentInfo kAbiArgs_metal_navatala_cfd_laplacian_from_face_flux[] = {
  { "faceFlux", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "offsets", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "faceIdx", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "signF", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "vol", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "counts", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 12, 12, 256, nullptr, 0, 0 },
  { "outLap", 6, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_cfd_laplacian_from_face_flux = {
  1,
  "navatala_cfd_laplacian_from_face_flux",
  "metal",
  "navatala_cfd_laplacian_from_face_flux",
  "kernel:metal:navatala_cfd_laplacian_from_face_flux",
  "abi-r1:metal:navatala_cfd_laplacian_from_face_flux",
  "abi-r1:metal:navatala_cfd_laplacian_from_face_flux",
  7,
  kAbiArgs_metal_navatala_cfd_laplacian_from_face_flux
};

const KernelArgumentInfo kAbiArgs_metal_navatala_cfd_mul_by_vol_in_place[] = {
  { "b", 0, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "vol", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "counts", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 12, 12, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_cfd_mul_by_vol_in_place = {
  1,
  "navatala_cfd_mul_by_vol_in_place",
  "metal",
  "navatala_cfd_mul_by_vol_in_place",
  "kernel:metal:navatala_cfd_mul_by_vol_in_place",
  "abi-r1:metal:navatala_cfd_mul_by_vol_in_place",
  "abi-r1:metal:navatala_cfd_mul_by_vol_in_place",
  3,
  kAbiArgs_metal_navatala_cfd_mul_by_vol_in_place
};

const KernelArgumentInfo kAbiArgs_metal_navatala_cfd_ref_add_ax[] = {
  { "diag", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "x", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "outAx", 2, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "refCell", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_cfd_ref_add_ax = {
  1,
  "navatala_cfd_ref_add_ax",
  "metal",
  "navatala_cfd_ref_add_ax",
  "kernel:metal:navatala_cfd_ref_add_ax",
  "abi-r1:metal:navatala_cfd_ref_add_ax",
  "abi-r1:metal:navatala_cfd_ref_add_ax",
  4,
  kAbiArgs_metal_navatala_cfd_ref_add_ax
};

const KernelArgumentInfo kAbiArgs_metal_navatala_cfd_ref_add_b[] = {
  { "diag", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "b", 1, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "refCell", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "refVal", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_cfd_ref_add_b = {
  1,
  "navatala_cfd_ref_add_b",
  "metal",
  "navatala_cfd_ref_add_b",
  "kernel:metal:navatala_cfd_ref_add_b",
  "abi-r1:metal:navatala_cfd_ref_add_b",
  "abi-r1:metal:navatala_cfd_ref_add_b",
  4,
  kAbiArgs_metal_navatala_cfd_ref_add_b
};

const KernelArgumentInfo kAbiArgs_metal_navatala_cfd_rhs_sub[] = {
  { "b", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "ax", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "counts", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 12, 12, 256, nullptr, 0, 0 },
  { "r", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_cfd_rhs_sub = {
  1,
  "navatala_cfd_rhs_sub",
  "metal",
  "navatala_cfd_rhs_sub",
  "kernel:metal:navatala_cfd_rhs_sub",
  "abi-r1:metal:navatala_cfd_rhs_sub",
  "abi-r1:metal:navatala_cfd_rhs_sub",
  4,
  kAbiArgs_metal_navatala_cfd_rhs_sub
};

const KernelArgumentInfo kAbiArgs_metal_navatala_cfd_shift_in_place[] = {
  { "x", 0, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "counts", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 12, 12, 256, nullptr, 0, 0 },
  { "paramsF", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_cfd_shift_in_place = {
  1,
  "navatala_cfd_shift_in_place",
  "metal",
  "navatala_cfd_shift_in_place",
  "kernel:metal:navatala_cfd_shift_in_place",
  "abi-r1:metal:navatala_cfd_shift_in_place",
  "abi-r1:metal:navatala_cfd_shift_in_place",
  3,
  kAbiArgs_metal_navatala_cfd_shift_in_place
};

const KernelArgumentInfo kAbiArgs_metal_navatala_cfd_sum_abs_partials[] = {
  { "a", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "counts", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 12, 12, 256, nullptr, 0, 0 },
  { "outPartials", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_cfd_sum_abs_partials = {
  1,
  "navatala_cfd_sum_abs_partials",
  "metal",
  "navatala_cfd_sum_abs_partials",
  "kernel:metal:navatala_cfd_sum_abs_partials",
  "abi-r1:metal:navatala_cfd_sum_abs_partials",
  "abi-r1:metal:navatala_cfd_sum_abs_partials",
  3,
  kAbiArgs_metal_navatala_cfd_sum_abs_partials
};

const KernelArgumentInfo kAbiArgs_metal_navatala_cfd_upper_from_cf[] = {
  { "cf", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "counts", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 12, 12, 256, nullptr, 0, 0 },
  { "upper", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_cfd_upper_from_cf = {
  1,
  "navatala_cfd_upper_from_cf",
  "metal",
  "navatala_cfd_upper_from_cf",
  "kernel:metal:navatala_cfd_upper_from_cf",
  "abi-r1:metal:navatala_cfd_upper_from_cf",
  "abi-r1:metal:navatala_cfd_upper_from_cf",
  3,
  kAbiArgs_metal_navatala_cfd_upper_from_cf
};

const KernelArgumentInfo kAbiArgs_metal_navatala_cfd_xpay_in_place[] = {
  { "z", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "pIn", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "pOut", 2, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "counts", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 12, 12, 256, nullptr, 0, 0 },
  { "paramsF", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_cfd_xpay_in_place = {
  1,
  "navatala_cfd_xpay_in_place",
  "metal",
  "navatala_cfd_xpay_in_place",
  "kernel:metal:navatala_cfd_xpay_in_place",
  "abi-r1:metal:navatala_cfd_xpay_in_place",
  "abi-r1:metal:navatala_cfd_xpay_in_place",
  5,
  kAbiArgs_metal_navatala_cfd_xpay_in_place
};

const KernelArgumentInfo kAbiArgs_metal_navatala_cfd_negate_scalar[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "output", 1, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_cfd_negate_scalar = {
  1,
  "navatala_cfd_negate_scalar",
  "metal",
  "navatala_cfd_negate_scalar",
  "kernel:metal:navatala_cfd_negate_scalar",
  "abi-r1:metal:navatala_cfd_negate_scalar",
  "abi-r1:metal:navatala_cfd_negate_scalar",
  2,
  kAbiArgs_metal_navatala_cfd_negate_scalar
};

const KernelArgumentInfo kAbiArgs_metal_navatala_cfd_exact_mat_vec[] = {
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
const KernelAbiManifestInfo kAbiManifest_metal_navatala_cfd_exact_mat_vec = {
  1,
  "navatala_cfd_exact_mat_vec",
  "metal",
  "navatala_cfd_exact_mat_vec",
  "kernel:metal:navatala_cfd_exact_mat_vec",
  "abi-r1:metal:navatala_cfd_exact_mat_vec",
  "abi-r1:metal:navatala_cfd_exact_mat_vec",
  11,
  kAbiArgs_metal_navatala_cfd_exact_mat_vec
};

const KernelArgumentInfo kAbiArgs_metal_navatala_cfd_face_flux_boundary[] = {
  { "x", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "owner", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "cf", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "bcVal", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "bcMask", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "params", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "faceFlux", 6, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_cfd_face_flux_boundary = {
  1,
  "navatala_cfd_face_flux_boundary",
  "metal",
  "navatala_cfd_face_flux_boundary",
  "kernel:metal:navatala_cfd_face_flux_boundary",
  "abi-r1:metal:navatala_cfd_face_flux_boundary",
  "abi-r1:metal:navatala_cfd_face_flux_boundary",
  7,
  kAbiArgs_metal_navatala_cfd_face_flux_boundary
};

const KernelArgumentInfo kAbiArgs_metal_navatala_cfd_face_flux_internal[] = {
  { "x", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "owner", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "neighbour", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "cf", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "params", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "faceFlux", 5, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_cfd_face_flux_internal = {
  1,
  "navatala_cfd_face_flux_internal",
  "metal",
  "navatala_cfd_face_flux_internal",
  "kernel:metal:navatala_cfd_face_flux_internal",
  "abi-r1:metal:navatala_cfd_face_flux_internal",
  "abi-r1:metal:navatala_cfd_face_flux_internal",
  6,
  kAbiArgs_metal_navatala_cfd_face_flux_internal
};

const KernelArgumentInfo kAbiArgs_metal_navatala_cfd_pressure_face_flux_correction_corrected[] = {
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
const KernelAbiManifestInfo kAbiManifest_metal_navatala_cfd_pressure_face_flux_correction_corrected = {
  1,
  "navatala_cfd_pressure_face_flux_correction_corrected",
  "metal",
  "navatala_cfd_pressure_face_flux_correction_corrected",
  "kernel:metal:navatala_cfd_pressure_face_flux_correction_corrected",
  "abi-r1:metal:navatala_cfd_pressure_face_flux_correction_corrected",
  "abi-r1:metal:navatala_cfd_pressure_face_flux_correction_corrected",
  13,
  kAbiArgs_metal_navatala_cfd_pressure_face_flux_correction_corrected
};

const KernelArgumentInfo kAbiArgs_metal_navatala_cfd_pressure_sn_grad[] = {
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
const KernelAbiManifestInfo kAbiManifest_metal_navatala_cfd_pressure_sn_grad = {
  1,
  "navatala_cfd_pressure_sn_grad",
  "metal",
  "navatala_cfd_pressure_sn_grad",
  "kernel:metal:navatala_cfd_pressure_sn_grad",
  "abi-r1:metal:navatala_cfd_pressure_sn_grad",
  "abi-r1:metal:navatala_cfd_pressure_sn_grad",
  17,
  kAbiArgs_metal_navatala_cfd_pressure_sn_grad
};

const KernelArgumentInfo kAbiArgs_metal_navatala_cfd_u_correct[] = {
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
const KernelAbiManifestInfo kAbiManifest_metal_navatala_cfd_u_correct = {
  1,
  "navatala_cfd_u_correct",
  "metal",
  "navatala_cfd_u_correct",
  "kernel:metal:navatala_cfd_u_correct",
  "abi-r1:metal:navatala_cfd_u_correct",
  "abi-r1:metal:navatala_cfd_u_correct",
  21,
  kAbiArgs_metal_navatala_cfd_u_correct
};

const KernelArgumentInfo kAbiArgs_metal_navatala_cfd_u_correct_reconstruct[] = {
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
const KernelAbiManifestInfo kAbiManifest_metal_navatala_cfd_u_correct_reconstruct = {
  1,
  "navatala_cfd_u_correct_reconstruct",
  "metal",
  "navatala_cfd_u_correct_reconstruct",
  "kernel:metal:navatala_cfd_u_correct_reconstruct",
  "abi-r1:metal:navatala_cfd_u_correct_reconstruct",
  "abi-r1:metal:navatala_cfd_u_correct_reconstruct",
  28,
  kAbiArgs_metal_navatala_cfd_u_correct_reconstruct
};

const KernelArgumentInfo kAbiArgs_metal_navatala_cfd_grad_alpha_cell[] = {
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
const KernelAbiManifestInfo kAbiManifest_metal_navatala_cfd_grad_alpha_cell = {
  1,
  "navatala_cfd_grad_alpha_cell",
  "metal",
  "navatala_cfd_grad_alpha_cell",
  "kernel:metal:navatala_cfd_grad_alpha_cell",
  "abi-r1:metal:navatala_cfd_grad_alpha_cell",
  "abi-r1:metal:navatala_cfd_grad_alpha_cell",
  12,
  kAbiArgs_metal_navatala_cfd_grad_alpha_cell
};

const KernelArgumentInfo kAbiArgs_metal_navatala_cfd_n_hatf_all[] = {
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
const KernelAbiManifestInfo kAbiManifest_metal_navatala_cfd_n_hatf_all = {
  1,
  "navatala_cfd_n_hatf_all",
  "metal",
  "navatala_cfd_n_hatf_all",
  "kernel:metal:navatala_cfd_n_hatf_all",
  "abi-r1:metal:navatala_cfd_n_hatf_all",
  "abi-r1:metal:navatala_cfd_n_hatf_all",
  15,
  kAbiArgs_metal_navatala_cfd_n_hatf_all
};

const KernelArgumentInfo kAbiArgs_metal_navatala_cfd_stf_all_faces[] = {
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
const KernelAbiManifestInfo kAbiManifest_metal_navatala_cfd_stf_all_faces = {
  1,
  "navatala_cfd_stf_all_faces",
  "metal",
  "navatala_cfd_stf_all_faces",
  "kernel:metal:navatala_cfd_stf_all_faces",
  "abi-r1:metal:navatala_cfd_stf_all_faces",
  "abi-r1:metal:navatala_cfd_stf_all_faces",
  12,
  kAbiArgs_metal_navatala_cfd_stf_all_faces
};

const KernelArgumentInfo kAbiArgs_metal_navatala_cfd_k_omega_s_s_t_blending[] = {
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
const KernelAbiManifestInfo kAbiManifest_metal_navatala_cfd_k_omega_s_s_t_blending = {
  1,
  "navatala_cfd_k_omega_s_s_t_blending",
  "metal",
  "navatala_cfd_k_omega_s_s_t_blending",
  "kernel:metal:navatala_cfd_k_omega_s_s_t_blending",
  "abi-r1:metal:navatala_cfd_k_omega_s_s_t_blending",
  "abi-r1:metal:navatala_cfd_k_omega_s_s_t_blending",
  15,
  kAbiArgs_metal_navatala_cfd_k_omega_s_s_t_blending
};

const KernelArgumentInfo kAbiArgs_metal_navatala_cfd_k_omega_s_s_t_nut_compute[] = {
  { "kVals", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "omegaVals", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "f23Vals", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "s2Vals", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "counts", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "params", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "outNut", 6, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_cfd_k_omega_s_s_t_nut_compute = {
  1,
  "navatala_cfd_k_omega_s_s_t_nut_compute",
  "metal",
  "navatala_cfd_k_omega_s_s_t_nut_compute",
  "kernel:metal:navatala_cfd_k_omega_s_s_t_nut_compute",
  "abi-r1:metal:navatala_cfd_k_omega_s_s_t_nut_compute",
  "abi-r1:metal:navatala_cfd_k_omega_s_s_t_nut_compute",
  7,
  kAbiArgs_metal_navatala_cfd_k_omega_s_s_t_nut_compute
};

const KernelArgumentInfo kAbiArgs_metal_navatala_cfd_k_omega_s_s_t_sources[] = {
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
const KernelAbiManifestInfo kAbiManifest_metal_navatala_cfd_k_omega_s_s_t_sources = {
  1,
  "navatala_cfd_k_omega_s_s_t_sources",
  "metal",
  "navatala_cfd_k_omega_s_s_t_sources",
  "kernel:metal:navatala_cfd_k_omega_s_s_t_sources",
  "abi-r1:metal:navatala_cfd_k_omega_s_s_t_sources",
  "abi-r1:metal:navatala_cfd_k_omega_s_s_t_sources",
  21,
  kAbiArgs_metal_navatala_cfd_k_omega_s_s_t_sources
};

const KernelArgumentInfo kAbiArgs_metal_navatala_cfd_mu_eff_from_rho_nu_eff[] = {
  { "rho", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "nuEff", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "params", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "outMuEff", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_cfd_mu_eff_from_rho_nu_eff = {
  1,
  "navatala_cfd_mu_eff_from_rho_nu_eff",
  "metal",
  "navatala_cfd_mu_eff_from_rho_nu_eff",
  "kernel:metal:navatala_cfd_mu_eff_from_rho_nu_eff",
  "abi-r1:metal:navatala_cfd_mu_eff_from_rho_nu_eff",
  "abi-r1:metal:navatala_cfd_mu_eff_from_rho_nu_eff",
  4,
  kAbiArgs_metal_navatala_cfd_mu_eff_from_rho_nu_eff
};

const KernelArgumentInfo kAbiArgs_metal_navatala_cfd_mu_eff_from_rho_nut_and_nu_mix[] = {
  { "rho", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "nut", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "nuMix", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "params", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "outMu", 4, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_cfd_mu_eff_from_rho_nut_and_nu_mix = {
  1,
  "navatala_cfd_mu_eff_from_rho_nut_and_nu_mix",
  "metal",
  "navatala_cfd_mu_eff_from_rho_nut_and_nu_mix",
  "kernel:metal:navatala_cfd_mu_eff_from_rho_nut_and_nu_mix",
  "abi-r1:metal:navatala_cfd_mu_eff_from_rho_nut_and_nu_mix",
  "abi-r1:metal:navatala_cfd_mu_eff_from_rho_nut_and_nu_mix",
  5,
  kAbiArgs_metal_navatala_cfd_mu_eff_from_rho_nut_and_nu_mix
};

const KernelArgumentInfo kAbiArgs_metal_navatala_cfd_turbulence_add_explicit_source_device[] = {
  { "explicitSrc", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "vol", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "counts", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "outVals", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_cfd_turbulence_add_explicit_source_device = {
  1,
  "navatala_cfd_turbulence_add_explicit_source_device",
  "metal",
  "navatala_cfd_turbulence_add_explicit_source_device",
  "kernel:metal:navatala_cfd_turbulence_add_explicit_source_device",
  "abi-r1:metal:navatala_cfd_turbulence_add_explicit_source_device",
  "abi-r1:metal:navatala_cfd_turbulence_add_explicit_source_device",
  4,
  kAbiArgs_metal_navatala_cfd_turbulence_add_explicit_source_device
};

const KernelArgumentInfo kAbiArgs_metal_navatala_cfd_turbulence_dirichlet_cell_source[] = {
  { "constrainedMask", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "constrainedValue", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "diag", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "nCells", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "source", 4, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_cfd_turbulence_dirichlet_cell_source = {
  1,
  "navatala_cfd_turbulence_dirichlet_cell_source",
  "metal",
  "navatala_cfd_turbulence_dirichlet_cell_source",
  "kernel:metal:navatala_cfd_turbulence_dirichlet_cell_source",
  "abi-r1:metal:navatala_cfd_turbulence_dirichlet_cell_source",
  "abi-r1:metal:navatala_cfd_turbulence_dirichlet_cell_source",
  5,
  kAbiArgs_metal_navatala_cfd_turbulence_dirichlet_cell_source
};

const KernelArgumentInfo kAbiArgs_metal_navatala_cfd_turbulence_dirichlet_face_elimination[] = {
  { "owner", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "neighbour", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "constrainedMask", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "constrainedValue", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "nIntFaces", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "upper", 5, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "lower", 6, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "source", 7, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_cfd_turbulence_dirichlet_face_elimination = {
  1,
  "navatala_cfd_turbulence_dirichlet_face_elimination",
  "metal",
  "navatala_cfd_turbulence_dirichlet_face_elimination",
  "kernel:metal:navatala_cfd_turbulence_dirichlet_face_elimination",
  "abi-r1:metal:navatala_cfd_turbulence_dirichlet_face_elimination",
  "abi-r1:metal:navatala_cfd_turbulence_dirichlet_face_elimination",
  8,
  kAbiArgs_metal_navatala_cfd_turbulence_dirichlet_face_elimination
};

const KernelArgumentInfo kAbiArgs_metal_navatala_cfd_two_mag_sqr_symm_grad_u[] = {
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
const KernelAbiManifestInfo kAbiManifest_metal_navatala_cfd_two_mag_sqr_symm_grad_u = {
  1,
  "navatala_cfd_two_mag_sqr_symm_grad_u",
  "metal",
  "navatala_cfd_two_mag_sqr_symm_grad_u",
  "kernel:metal:navatala_cfd_two_mag_sqr_symm_grad_u",
  "abi-r1:metal:navatala_cfd_two_mag_sqr_symm_grad_u",
  "abi-r1:metal:navatala_cfd_two_mag_sqr_symm_grad_u",
  12,
  kAbiArgs_metal_navatala_cfd_two_mag_sqr_symm_grad_u
};

const KernelArgumentInfo kAbiArgs_metal_navatala_cfd_attention_row_softmax[] = {
  { "scores", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "numQ", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "numK", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "weights", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_cfd_attention_row_softmax = {
  1,
  "navatala_cfd_attention_row_softmax",
  "metal",
  "navatala_cfd_attention_row_softmax",
  "kernel:metal:navatala_cfd_attention_row_softmax",
  "abi-r1:metal:navatala_cfd_attention_row_softmax",
  "abi-r1:metal:navatala_cfd_attention_row_softmax",
  4,
  kAbiArgs_metal_navatala_cfd_attention_row_softmax
};

const KernelArgumentInfo kAbiArgs_metal_navatala_cfd_attention_value_projection[] = {
  { "weights", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "v", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "numQ", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "numK", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "headDim", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "out", 5, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_cfd_attention_value_projection = {
  1,
  "navatala_cfd_attention_value_projection",
  "metal",
  "navatala_cfd_attention_value_projection",
  "kernel:metal:navatala_cfd_attention_value_projection",
  "abi-r1:metal:navatala_cfd_attention_value_projection",
  "abi-r1:metal:navatala_cfd_attention_value_projection",
  6,
  kAbiArgs_metal_navatala_cfd_attention_value_projection
};

const KernelArgumentInfo kAbiArgs_metal_navatala_cfd_gather_mean[] = {
  { "features", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "neighbors", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "neighborCounts", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "numNodes", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "featureDim", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "out", 5, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_cfd_gather_mean = {
  1,
  "navatala_cfd_gather_mean",
  "metal",
  "navatala_cfd_gather_mean",
  "kernel:metal:navatala_cfd_gather_mean",
  "abi-r1:metal:navatala_cfd_gather_mean",
  "abi-r1:metal:navatala_cfd_gather_mean",
  6,
  kAbiArgs_metal_navatala_cfd_gather_mean
};

const KernelArgumentInfo kAbiArgs_metal_navatala_cfd_layer_norm_forward[] = {
  { "x", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "numRows", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "gamma", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "beta", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "y", 4, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_cfd_layer_norm_forward = {
  1,
  "navatala_cfd_layer_norm_forward",
  "metal",
  "navatala_cfd_layer_norm_forward",
  "kernel:metal:navatala_cfd_layer_norm_forward",
  "abi-r1:metal:navatala_cfd_layer_norm_forward",
  "abi-r1:metal:navatala_cfd_layer_norm_forward",
  5,
  kAbiArgs_metal_navatala_cfd_layer_norm_forward
};

const KernelArgumentInfo kAbiArgs_metal_navatala_cfd_radius_count[] = {
  { "query", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "ref", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "numQuery", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "numRef", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "radiusIn", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "counts", 5, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_cfd_radius_count = {
  1,
  "navatala_cfd_radius_count",
  "metal",
  "navatala_cfd_radius_count",
  "kernel:metal:navatala_cfd_radius_count",
  "abi-r1:metal:navatala_cfd_radius_count",
  "abi-r1:metal:navatala_cfd_radius_count",
  6,
  kAbiArgs_metal_navatala_cfd_radius_count
};

const KernelArgumentInfo kAbiArgs_metal_navatala_cfd_scaled_dot_product[] = {
  { "q", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "k", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "numQ", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "numK", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "headDim", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "scores", 5, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_cfd_scaled_dot_product = {
  1,
  "navatala_cfd_scaled_dot_product",
  "metal",
  "navatala_cfd_scaled_dot_product",
  "kernel:metal:navatala_cfd_scaled_dot_product",
  "abi-r1:metal:navatala_cfd_scaled_dot_product",
  "abi-r1:metal:navatala_cfd_scaled_dot_product",
  6,
  kAbiArgs_metal_navatala_cfd_scaled_dot_product
};

const KernelArgumentInfo kAbiArgs_metal_navatala_cfd_scatter_add_atomic[] = {
  { "values", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "indices", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "numEdges", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "featureDim", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "dst", 4, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_cfd_scatter_add_atomic = {
  1,
  "navatala_cfd_scatter_add_atomic",
  "metal",
  "navatala_cfd_scatter_add_atomic",
  "kernel:metal:navatala_cfd_scatter_add_atomic",
  "abi-r1:metal:navatala_cfd_scatter_add_atomic",
  "abi-r1:metal:navatala_cfd_scatter_add_atomic",
  5,
  kAbiArgs_metal_navatala_cfd_scatter_add_atomic
};

const KernelArgumentInfo kAbiArgs_metal_navatala_cfd_spectral_complex_mul[] = {
  { "x", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "w", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "y", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_cfd_spectral_complex_mul = {
  1,
  "navatala_cfd_spectral_complex_mul",
  "metal",
  "navatala_cfd_spectral_complex_mul",
  "kernel:metal:navatala_cfd_spectral_complex_mul",
  "abi-r1:metal:navatala_cfd_spectral_complex_mul",
  "abi-r1:metal:navatala_cfd_spectral_complex_mul",
  4,
  kAbiArgs_metal_navatala_cfd_spectral_complex_mul
};

const KernelArgumentInfo kAbiArgs_metal_navatala_cfd_spectral_conv1d_mode_sliced[] = {
  { "xFt", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "wPos", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "wNeg", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "numBatches", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "ftW", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "yFt", 5, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_cfd_spectral_conv1d_mode_sliced = {
  1,
  "navatala_cfd_spectral_conv1d_mode_sliced",
  "metal",
  "navatala_cfd_spectral_conv1d_mode_sliced",
  "kernel:metal:navatala_cfd_spectral_conv1d_mode_sliced",
  "abi-r1:metal:navatala_cfd_spectral_conv1d_mode_sliced",
  "abi-r1:metal:navatala_cfd_spectral_conv1d_mode_sliced",
  6,
  kAbiArgs_metal_navatala_cfd_spectral_conv1d_mode_sliced
};

const KernelArgumentInfo kAbiArgs_metal_navatala_cfd_spectral_conv2d_mode_sliced[] = {
  { "xFt", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "wPos", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "wNeg", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "numBatches", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "ftH", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "ftW", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "yFt", 6, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_cfd_spectral_conv2d_mode_sliced = {
  1,
  "navatala_cfd_spectral_conv2d_mode_sliced",
  "metal",
  "navatala_cfd_spectral_conv2d_mode_sliced",
  "kernel:metal:navatala_cfd_spectral_conv2d_mode_sliced",
  "abi-r1:metal:navatala_cfd_spectral_conv2d_mode_sliced",
  "abi-r1:metal:navatala_cfd_spectral_conv2d_mode_sliced",
  7,
  kAbiArgs_metal_navatala_cfd_spectral_conv2d_mode_sliced
};

const KernelArgumentInfo kAbiArgs_metal_navatala_cfd_spectral_conv3d_mode_sliced[] = {
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
const KernelAbiManifestInfo kAbiManifest_metal_navatala_cfd_spectral_conv3d_mode_sliced = {
  1,
  "navatala_cfd_spectral_conv3d_mode_sliced",
  "metal",
  "navatala_cfd_spectral_conv3d_mode_sliced",
  "kernel:metal:navatala_cfd_spectral_conv3d_mode_sliced",
  "abi-r1:metal:navatala_cfd_spectral_conv3d_mode_sliced",
  "abi-r1:metal:navatala_cfd_spectral_conv3d_mode_sliced",
  10,
  kAbiArgs_metal_navatala_cfd_spectral_conv3d_mode_sliced
};

const KernelArgumentInfo kAbiArgs_metal_navatala_cfd_spectral_conv4d_mode_sliced[] = {
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
const KernelAbiManifestInfo kAbiManifest_metal_navatala_cfd_spectral_conv4d_mode_sliced = {
  1,
  "navatala_cfd_spectral_conv4d_mode_sliced",
  "metal",
  "navatala_cfd_spectral_conv4d_mode_sliced",
  "kernel:metal:navatala_cfd_spectral_conv4d_mode_sliced",
  "abi-r1:metal:navatala_cfd_spectral_conv4d_mode_sliced",
  "abi-r1:metal:navatala_cfd_spectral_conv4d_mode_sliced",
  15,
  kAbiArgs_metal_navatala_cfd_spectral_conv4d_mode_sliced
};

const KernelArgumentInfo kAbiArgs_metal_navatala_cfd_compute_face_area_pair_weights_f32[] = {
  { "sfX", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "sfY", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "sfZ", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "magSf", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "nFaces", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "faceWeight", 5, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_cfd_compute_face_area_pair_weights_f32 = {
  1,
  "navatala_cfd_compute_face_area_pair_weights_f32",
  "metal",
  "navatala_cfd_compute_face_area_pair_weights_f32",
  "kernel:metal:navatala_cfd_compute_face_area_pair_weights_f32",
  "abi-r1:metal:navatala_cfd_compute_face_area_pair_weights_f32",
  "abi-r1:metal:navatala_cfd_compute_face_area_pair_weights_f32",
  6,
  kAbiArgs_metal_navatala_cfd_compute_face_area_pair_weights_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_cfd_restrict_by_agglomeration_gather_f32[] = {
  { "fineField", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "coarseToFineOffsets", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "coarseToFineList", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "nCoarseCells", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "coarseField", 4, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_cfd_restrict_by_agglomeration_gather_f32 = {
  1,
  "navatala_cfd_restrict_by_agglomeration_gather_f32",
  "metal",
  "navatala_cfd_restrict_by_agglomeration_gather_f32",
  "kernel:metal:navatala_cfd_restrict_by_agglomeration_gather_f32",
  "abi-r1:metal:navatala_cfd_restrict_by_agglomeration_gather_f32",
  "abi-r1:metal:navatala_cfd_restrict_by_agglomeration_gather_f32",
  5,
  kAbiArgs_metal_navatala_cfd_restrict_by_agglomeration_gather_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_cfd_prolongate_by_agglomeration_f32[] = {
  { "coarseField", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "fineToCoarse", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "nFineCells", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "fineField", 3, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_cfd_prolongate_by_agglomeration_f32 = {
  1,
  "navatala_cfd_prolongate_by_agglomeration_f32",
  "metal",
  "navatala_cfd_prolongate_by_agglomeration_f32",
  "kernel:metal:navatala_cfd_prolongate_by_agglomeration_f32",
  "abi-r1:metal:navatala_cfd_prolongate_by_agglomeration_f32",
  "abi-r1:metal:navatala_cfd_prolongate_by_agglomeration_f32",
  4,
  kAbiArgs_metal_navatala_cfd_prolongate_by_agglomeration_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_cfd_diag_inv_apply_f32[] = {
  { "r", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "diagInv", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "z", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_cfd_diag_inv_apply_f32 = {
  1,
  "navatala_cfd_diag_inv_apply_f32",
  "metal",
  "navatala_cfd_diag_inv_apply_f32",
  "kernel:metal:navatala_cfd_diag_inv_apply_f32",
  "abi-r1:metal:navatala_cfd_diag_inv_apply_f32",
  "abi-r1:metal:navatala_cfd_diag_inv_apply_f32",
  4,
  kAbiArgs_metal_navatala_cfd_diag_inv_apply_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_cfd_gamg_proc_pack_f32[] = {
  { "x", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "localCells", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "sendVals", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_cfd_gamg_proc_pack_f32 = {
  1,
  "navatala_cfd_gamg_proc_pack_f32",
  "metal",
  "navatala_cfd_gamg_proc_pack_f32",
  "kernel:metal:navatala_cfd_gamg_proc_pack_f32",
  "abi-r1:metal:navatala_cfd_gamg_proc_pack_f32",
  "abi-r1:metal:navatala_cfd_gamg_proc_pack_f32",
  4,
  kAbiArgs_metal_navatala_cfd_gamg_proc_pack_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_cfd_gamg_proc_add_gather_f32[] = {
  { "entryOffsets", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "entryIndices", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "coeffs", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "recvVals", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "nCells", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "y", 5, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_cfd_gamg_proc_add_gather_f32 = {
  1,
  "navatala_cfd_gamg_proc_add_gather_f32",
  "metal",
  "navatala_cfd_gamg_proc_add_gather_f32",
  "kernel:metal:navatala_cfd_gamg_proc_add_gather_f32",
  "abi-r1:metal:navatala_cfd_gamg_proc_add_gather_f32",
  "abi-r1:metal:navatala_cfd_gamg_proc_add_gather_f32",
  6,
  kAbiArgs_metal_navatala_cfd_gamg_proc_add_gather_f32
};

bool tryGetKernelAbiManifest_metal_cfd(const std::string& backend, const std::string& kernelName, const KernelAbiManifestInfo*& out) {
  if (backend == "metal" && kernelName == "navatala_cfd_boundary_force_partials") {
    out = &kAbiManifest_metal_navatala_cfd_boundary_force_partials;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_cfd_pack_owner_values") {
    out = &kAbiManifest_metal_navatala_cfd_pack_owner_values;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_cfd_scatter_values_and_mask") {
    out = &kAbiManifest_metal_navatala_cfd_scatter_values_and_mask;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_cfd_pack_mu_grad_owner") {
    out = &kAbiManifest_metal_navatala_cfd_pack_mu_grad_owner;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_cfd_pack_owner_rho_vec3") {
    out = &kAbiManifest_metal_navatala_cfd_pack_owner_rho_vec3;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_cfd_pack_owner_vec3") {
    out = &kAbiManifest_metal_navatala_cfd_pack_owner_vec3;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_cfd_pack_owner_vec3_mu") {
    out = &kAbiManifest_metal_navatala_cfd_pack_owner_vec3_mu;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_cfd_scatter_mu_grad_ghost") {
    out = &kAbiManifest_metal_navatala_cfd_scatter_mu_grad_ghost;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_cfd_scatter_rho_vec3_and_mask") {
    out = &kAbiManifest_metal_navatala_cfd_scatter_rho_vec3_and_mask;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_cfd_scatter_vec3_and_mask") {
    out = &kAbiManifest_metal_navatala_cfd_scatter_vec3_and_mask;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_cfd_scatter_vec3_mu_and_mask") {
    out = &kAbiManifest_metal_navatala_cfd_scatter_vec3_mu_and_mask;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_cfd_add_vol_vector") {
    out = &kAbiManifest_metal_navatala_cfd_add_vol_vector;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_cfd_div_rho_phi_u_explicit") {
    out = &kAbiManifest_metal_navatala_cfd_div_rho_phi_u_explicit;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_cfd_hby_a_from_u_and_terms") {
    out = &kAbiManifest_metal_navatala_cfd_hby_a_from_u_and_terms;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_cfd_apply_sp_to_hby_a") {
    out = &kAbiManifest_metal_navatala_cfd_apply_sp_to_hby_a;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_cfd_laplacian_scalar_explicit") {
    out = &kAbiManifest_metal_navatala_cfd_laplacian_scalar_explicit;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_cfd_laplacian_vector_explicit") {
    out = &kAbiManifest_metal_navatala_cfd_laplacian_vector_explicit;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_cfd_predictor_hby_a_from_source") {
    out = &kAbiManifest_metal_navatala_cfd_predictor_hby_a_from_source;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_cfd_predictor_inverse_diag") {
    out = &kAbiManifest_metal_navatala_cfd_predictor_inverse_diag;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_cfd_predictor_phi_hby_a_fuse") {
    out = &kAbiManifest_metal_navatala_cfd_predictor_phi_hby_a_fuse;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_cfd_axpy_cells") {
    out = &kAbiManifest_metal_navatala_cfd_axpy_cells;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_cfd_ddt_diagonal") {
    out = &kAbiManifest_metal_navatala_cfd_ddt_diagonal;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_cfd_div_upwind_coeffs") {
    out = &kAbiManifest_metal_navatala_cfd_div_upwind_coeffs;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_cfd_fvm_sp") {
    out = &kAbiManifest_metal_navatala_cfd_fvm_sp;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_cfd_fvm_su_sp") {
    out = &kAbiManifest_metal_navatala_cfd_fvm_su_sp;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_cfd_laplacian_coeffs") {
    out = &kAbiManifest_metal_navatala_cfd_laplacian_coeffs;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_cfd_laplacian_coeffs_accum") {
    out = &kAbiManifest_metal_navatala_cfd_laplacian_coeffs_accum;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_cfd_linear_upwind_correction_source") {
    out = &kAbiManifest_metal_navatala_cfd_linear_upwind_correction_source;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_cfd_matrix_relax") {
    out = &kAbiManifest_metal_navatala_cfd_matrix_relax;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_cfd_neg_sum_diag") {
    out = &kAbiManifest_metal_navatala_cfd_neg_sum_diag;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_cfd_pack_boundary_adjacent_cells") {
    out = &kAbiManifest_metal_navatala_cfd_pack_boundary_adjacent_cells;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_cfd_scalar_jacobi_clamp_min") {
    out = &kAbiManifest_metal_navatala_cfd_scalar_jacobi_clamp_min;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_cfd_scalar_jacobi_update") {
    out = &kAbiManifest_metal_navatala_cfd_scalar_jacobi_update;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_cfd_scalar_ldu_mat_vec") {
    out = &kAbiManifest_metal_navatala_cfd_scalar_ldu_mat_vec;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_cfd_scalar_ldu_multi_color_gs_sweep") {
    out = &kAbiManifest_metal_navatala_cfd_scalar_ldu_multi_color_gs_sweep;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_cfd_sum_mag_off_diag") {
    out = &kAbiManifest_metal_navatala_cfd_sum_mag_off_diag;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_cfd_gamg_axpy_in_place") {
    out = &kAbiManifest_metal_navatala_cfd_gamg_axpy_in_place;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_cfd_gamg_compute_scale_factor") {
    out = &kAbiManifest_metal_navatala_cfd_gamg_compute_scale_factor;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_cfd_gamg_dot_finalize") {
    out = &kAbiManifest_metal_navatala_cfd_gamg_dot_finalize;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_cfd_gamg_negate_scalar") {
    out = &kAbiManifest_metal_navatala_cfd_gamg_negate_scalar;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_cfd_gamg_scale_correction") {
    out = &kAbiManifest_metal_navatala_cfd_gamg_scale_correction;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_cfd_mg_coarse_jacobi") {
    out = &kAbiManifest_metal_navatala_cfd_mg_coarse_jacobi;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_cfd_mg_coarse_matvec") {
    out = &kAbiManifest_metal_navatala_cfd_mg_coarse_matvec;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_cfd_mg_coarse_rhs") {
    out = &kAbiManifest_metal_navatala_cfd_mg_coarse_rhs;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_cfd_mg_find_best_match") {
    out = &kAbiManifest_metal_navatala_cfd_mg_find_best_match;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_cfd_mg_pair_match") {
    out = &kAbiManifest_metal_navatala_cfd_mg_pair_match;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_cfd_mg_prolong_add") {
    out = &kAbiManifest_metal_navatala_cfd_mg_prolong_add;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_cfd_mg_renumber_aggregates") {
    out = &kAbiManifest_metal_navatala_cfd_mg_renumber_aggregates;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_cfd_mg_renumber_propagate") {
    out = &kAbiManifest_metal_navatala_cfd_mg_renumber_propagate;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_cfd_mg_resolve_chains") {
    out = &kAbiManifest_metal_navatala_cfd_mg_resolve_chains;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_cfd_mg_restrict_sum") {
    out = &kAbiManifest_metal_navatala_cfd_mg_restrict_sum;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_cfd_mg_strength_of_connection") {
    out = &kAbiManifest_metal_navatala_cfd_mg_strength_of_connection;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_cfd_mg_zero_coarse") {
    out = &kAbiManifest_metal_navatala_cfd_mg_zero_coarse;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_cfd_phi_correct") {
    out = &kAbiManifest_metal_navatala_cfd_phi_correct;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_cfd_phi_correct_boundary") {
    out = &kAbiManifest_metal_navatala_cfd_phi_correct_boundary;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_cfd_phi_correct_internal") {
    out = &kAbiManifest_metal_navatala_cfd_phi_correct_internal;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_cfd_axpy_in_place") {
    out = &kAbiManifest_metal_navatala_cfd_axpy_in_place;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_cfd_bc_dirichlet_face_flux") {
    out = &kAbiManifest_metal_navatala_cfd_bc_dirichlet_face_flux;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_cfd_bc_sn_grad_face_flux") {
    out = &kAbiManifest_metal_navatala_cfd_bc_sn_grad_face_flux;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_cfd_coeff_to_cf_in_place") {
    out = &kAbiManifest_metal_navatala_cfd_coeff_to_cf_in_place;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_cfd_diag_from_cf") {
    out = &kAbiManifest_metal_navatala_cfd_diag_from_cf;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_cfd_dic_apply") {
    out = &kAbiManifest_metal_navatala_cfd_dic_apply;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_cfd_dic_build_reciprocal_d") {
    out = &kAbiManifest_metal_navatala_cfd_dic_build_reciprocal_d;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_cfd_dot_partials") {
    out = &kAbiManifest_metal_navatala_cfd_dot_partials;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_cfd_face_flux") {
    out = &kAbiManifest_metal_navatala_cfd_face_flux;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_cfd_inv_diag") {
    out = &kAbiManifest_metal_navatala_cfd_inv_diag;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_cfd_laplacian_from_face_flux") {
    out = &kAbiManifest_metal_navatala_cfd_laplacian_from_face_flux;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_cfd_mul_by_vol_in_place") {
    out = &kAbiManifest_metal_navatala_cfd_mul_by_vol_in_place;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_cfd_ref_add_ax") {
    out = &kAbiManifest_metal_navatala_cfd_ref_add_ax;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_cfd_ref_add_b") {
    out = &kAbiManifest_metal_navatala_cfd_ref_add_b;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_cfd_rhs_sub") {
    out = &kAbiManifest_metal_navatala_cfd_rhs_sub;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_cfd_shift_in_place") {
    out = &kAbiManifest_metal_navatala_cfd_shift_in_place;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_cfd_sum_abs_partials") {
    out = &kAbiManifest_metal_navatala_cfd_sum_abs_partials;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_cfd_upper_from_cf") {
    out = &kAbiManifest_metal_navatala_cfd_upper_from_cf;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_cfd_xpay_in_place") {
    out = &kAbiManifest_metal_navatala_cfd_xpay_in_place;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_cfd_negate_scalar") {
    out = &kAbiManifest_metal_navatala_cfd_negate_scalar;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_cfd_exact_mat_vec") {
    out = &kAbiManifest_metal_navatala_cfd_exact_mat_vec;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_cfd_face_flux_boundary") {
    out = &kAbiManifest_metal_navatala_cfd_face_flux_boundary;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_cfd_face_flux_internal") {
    out = &kAbiManifest_metal_navatala_cfd_face_flux_internal;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_cfd_pressure_face_flux_correction_corrected") {
    out = &kAbiManifest_metal_navatala_cfd_pressure_face_flux_correction_corrected;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_cfd_pressure_sn_grad") {
    out = &kAbiManifest_metal_navatala_cfd_pressure_sn_grad;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_cfd_u_correct") {
    out = &kAbiManifest_metal_navatala_cfd_u_correct;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_cfd_u_correct_reconstruct") {
    out = &kAbiManifest_metal_navatala_cfd_u_correct_reconstruct;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_cfd_grad_alpha_cell") {
    out = &kAbiManifest_metal_navatala_cfd_grad_alpha_cell;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_cfd_n_hatf_all") {
    out = &kAbiManifest_metal_navatala_cfd_n_hatf_all;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_cfd_stf_all_faces") {
    out = &kAbiManifest_metal_navatala_cfd_stf_all_faces;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_cfd_k_omega_s_s_t_blending") {
    out = &kAbiManifest_metal_navatala_cfd_k_omega_s_s_t_blending;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_cfd_k_omega_s_s_t_nut_compute") {
    out = &kAbiManifest_metal_navatala_cfd_k_omega_s_s_t_nut_compute;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_cfd_k_omega_s_s_t_sources") {
    out = &kAbiManifest_metal_navatala_cfd_k_omega_s_s_t_sources;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_cfd_mu_eff_from_rho_nu_eff") {
    out = &kAbiManifest_metal_navatala_cfd_mu_eff_from_rho_nu_eff;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_cfd_mu_eff_from_rho_nut_and_nu_mix") {
    out = &kAbiManifest_metal_navatala_cfd_mu_eff_from_rho_nut_and_nu_mix;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_cfd_turbulence_add_explicit_source_device") {
    out = &kAbiManifest_metal_navatala_cfd_turbulence_add_explicit_source_device;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_cfd_turbulence_dirichlet_cell_source") {
    out = &kAbiManifest_metal_navatala_cfd_turbulence_dirichlet_cell_source;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_cfd_turbulence_dirichlet_face_elimination") {
    out = &kAbiManifest_metal_navatala_cfd_turbulence_dirichlet_face_elimination;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_cfd_two_mag_sqr_symm_grad_u") {
    out = &kAbiManifest_metal_navatala_cfd_two_mag_sqr_symm_grad_u;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_cfd_attention_row_softmax") {
    out = &kAbiManifest_metal_navatala_cfd_attention_row_softmax;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_cfd_attention_value_projection") {
    out = &kAbiManifest_metal_navatala_cfd_attention_value_projection;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_cfd_gather_mean") {
    out = &kAbiManifest_metal_navatala_cfd_gather_mean;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_cfd_layer_norm_forward") {
    out = &kAbiManifest_metal_navatala_cfd_layer_norm_forward;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_cfd_radius_count") {
    out = &kAbiManifest_metal_navatala_cfd_radius_count;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_cfd_scaled_dot_product") {
    out = &kAbiManifest_metal_navatala_cfd_scaled_dot_product;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_cfd_scatter_add_atomic") {
    out = &kAbiManifest_metal_navatala_cfd_scatter_add_atomic;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_cfd_spectral_complex_mul") {
    out = &kAbiManifest_metal_navatala_cfd_spectral_complex_mul;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_cfd_spectral_conv1d_mode_sliced") {
    out = &kAbiManifest_metal_navatala_cfd_spectral_conv1d_mode_sliced;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_cfd_spectral_conv2d_mode_sliced") {
    out = &kAbiManifest_metal_navatala_cfd_spectral_conv2d_mode_sliced;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_cfd_spectral_conv3d_mode_sliced") {
    out = &kAbiManifest_metal_navatala_cfd_spectral_conv3d_mode_sliced;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_cfd_spectral_conv4d_mode_sliced") {
    out = &kAbiManifest_metal_navatala_cfd_spectral_conv4d_mode_sliced;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_cfd_compute_face_area_pair_weights_f32") {
    out = &kAbiManifest_metal_navatala_cfd_compute_face_area_pair_weights_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_cfd_restrict_by_agglomeration_gather_f32") {
    out = &kAbiManifest_metal_navatala_cfd_restrict_by_agglomeration_gather_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_cfd_prolongate_by_agglomeration_f32") {
    out = &kAbiManifest_metal_navatala_cfd_prolongate_by_agglomeration_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_cfd_diag_inv_apply_f32") {
    out = &kAbiManifest_metal_navatala_cfd_diag_inv_apply_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_cfd_gamg_proc_pack_f32") {
    out = &kAbiManifest_metal_navatala_cfd_gamg_proc_pack_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_cfd_gamg_proc_add_gather_f32") {
    out = &kAbiManifest_metal_navatala_cfd_gamg_proc_add_gather_f32;
    return true;
  }
  out = nullptr;
  return false;
}

bool tryGetKernelSource_metal_cfd(const std::string& backend, const std::string& kernelName, GpuRuntime::ProgramSource& out) {
  if (backend == "metal" && kernelName == "navatala_cfd_boundary_force_partials") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_cfd_boundary_force_partials";
    std::string_view sv(k_metal_navatala_cfd_boundary_force_partials);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_cfd_pack_owner_values") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_cfd_pack_owner_values";
    std::string_view sv(k_metal_navatala_cfd_pack_owner_values);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_cfd_scatter_values_and_mask") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_cfd_scatter_values_and_mask";
    std::string_view sv(k_metal_navatala_cfd_scatter_values_and_mask);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_cfd_pack_mu_grad_owner") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_cfd_pack_mu_grad_owner";
    std::string_view sv(k_metal_navatala_cfd_pack_mu_grad_owner);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_cfd_pack_owner_rho_vec3") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_cfd_pack_owner_rho_vec3";
    std::string_view sv(k_metal_navatala_cfd_pack_owner_rho_vec3);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_cfd_pack_owner_vec3") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_cfd_pack_owner_vec3";
    std::string_view sv(k_metal_navatala_cfd_pack_owner_vec3);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_cfd_pack_owner_vec3_mu") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_cfd_pack_owner_vec3_mu";
    std::string_view sv(k_metal_navatala_cfd_pack_owner_vec3_mu);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_cfd_scatter_mu_grad_ghost") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_cfd_scatter_mu_grad_ghost";
    std::string_view sv(k_metal_navatala_cfd_scatter_mu_grad_ghost);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_cfd_scatter_rho_vec3_and_mask") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_cfd_scatter_rho_vec3_and_mask";
    std::string_view sv(k_metal_navatala_cfd_scatter_rho_vec3_and_mask);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_cfd_scatter_vec3_and_mask") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_cfd_scatter_vec3_and_mask";
    std::string_view sv(k_metal_navatala_cfd_scatter_vec3_and_mask);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_cfd_scatter_vec3_mu_and_mask") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_cfd_scatter_vec3_mu_and_mask";
    std::string_view sv(k_metal_navatala_cfd_scatter_vec3_mu_and_mask);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_cfd_add_vol_vector") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_cfd_add_vol_vector";
    std::string_view sv(k_metal_navatala_cfd_add_vol_vector);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_cfd_div_rho_phi_u_explicit") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_cfd_div_rho_phi_u_explicit";
    std::string_view sv(k_metal_navatala_cfd_div_rho_phi_u_explicit);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_cfd_hby_a_from_u_and_terms") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_cfd_hby_a_from_u_and_terms";
    std::string_view sv(k_metal_navatala_cfd_hby_a_from_u_and_terms);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_cfd_apply_sp_to_hby_a") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_cfd_apply_sp_to_hby_a";
    std::string_view sv(k_metal_navatala_cfd_apply_sp_to_hby_a);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_cfd_laplacian_scalar_explicit") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_cfd_laplacian_scalar_explicit";
    std::string_view sv(k_metal_navatala_cfd_laplacian_scalar_explicit);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_cfd_laplacian_vector_explicit") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_cfd_laplacian_vector_explicit";
    std::string_view sv(k_metal_navatala_cfd_laplacian_vector_explicit);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_cfd_predictor_hby_a_from_source") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_cfd_predictor_hby_a_from_source";
    std::string_view sv(k_metal_navatala_cfd_predictor_hby_a_from_source);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_cfd_predictor_inverse_diag") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_cfd_predictor_inverse_diag";
    std::string_view sv(k_metal_navatala_cfd_predictor_inverse_diag);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_cfd_predictor_phi_hby_a_fuse") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_cfd_predictor_phi_hby_a_fuse";
    std::string_view sv(k_metal_navatala_cfd_predictor_phi_hby_a_fuse);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_cfd_axpy_cells") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_cfd_axpy_cells";
    std::string_view sv(k_metal_navatala_cfd_axpy_cells);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_cfd_ddt_diagonal") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_cfd_ddt_diagonal";
    std::string_view sv(k_metal_navatala_cfd_ddt_diagonal);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_cfd_div_upwind_coeffs") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_cfd_div_upwind_coeffs";
    std::string_view sv(k_metal_navatala_cfd_div_upwind_coeffs);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_cfd_fvm_sp") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_cfd_fvm_sp";
    std::string_view sv(k_metal_navatala_cfd_fvm_sp);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_cfd_fvm_su_sp") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_cfd_fvm_su_sp";
    std::string_view sv(k_metal_navatala_cfd_fvm_su_sp);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_cfd_laplacian_coeffs") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_cfd_laplacian_coeffs";
    std::string_view sv(k_metal_navatala_cfd_laplacian_coeffs);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_cfd_laplacian_coeffs_accum") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_cfd_laplacian_coeffs_accum";
    std::string_view sv(k_metal_navatala_cfd_laplacian_coeffs_accum);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_cfd_linear_upwind_correction_source") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_cfd_linear_upwind_correction_source";
    std::string_view sv(k_metal_navatala_cfd_linear_upwind_correction_source);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_cfd_matrix_relax") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_cfd_matrix_relax";
    std::string_view sv(k_metal_navatala_cfd_matrix_relax);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_cfd_neg_sum_diag") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_cfd_neg_sum_diag";
    std::string_view sv(k_metal_navatala_cfd_neg_sum_diag);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_cfd_pack_boundary_adjacent_cells") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_cfd_pack_boundary_adjacent_cells";
    std::string_view sv(k_metal_navatala_cfd_pack_boundary_adjacent_cells);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_cfd_scalar_jacobi_clamp_min") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_cfd_scalar_jacobi_clamp_min";
    std::string_view sv(k_metal_navatala_cfd_scalar_jacobi_clamp_min);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_cfd_scalar_jacobi_update") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_cfd_scalar_jacobi_update";
    std::string_view sv(k_metal_navatala_cfd_scalar_jacobi_update);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_cfd_scalar_ldu_mat_vec") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_cfd_scalar_ldu_mat_vec";
    std::string_view sv(k_metal_navatala_cfd_scalar_ldu_mat_vec);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_cfd_scalar_ldu_multi_color_gs_sweep") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_cfd_scalar_ldu_multi_color_gs_sweep";
    std::string_view sv(k_metal_navatala_cfd_scalar_ldu_multi_color_gs_sweep);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_cfd_sum_mag_off_diag") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_cfd_sum_mag_off_diag";
    std::string_view sv(k_metal_navatala_cfd_sum_mag_off_diag);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_cfd_gamg_axpy_in_place") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_cfd_gamg_axpy_in_place";
    std::string_view sv(k_metal_navatala_cfd_gamg_axpy_in_place);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_cfd_gamg_compute_scale_factor") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_cfd_gamg_compute_scale_factor";
    std::string_view sv(k_metal_navatala_cfd_gamg_compute_scale_factor);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_cfd_gamg_dot_finalize") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_cfd_gamg_dot_finalize";
    std::string_view sv(k_metal_navatala_cfd_gamg_dot_finalize);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_cfd_gamg_negate_scalar") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_cfd_gamg_negate_scalar";
    std::string_view sv(k_metal_navatala_cfd_gamg_negate_scalar);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_cfd_gamg_scale_correction") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_cfd_gamg_scale_correction";
    std::string_view sv(k_metal_navatala_cfd_gamg_scale_correction);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_cfd_mg_coarse_jacobi") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_cfd_mg_coarse_jacobi";
    std::string_view sv(k_metal_navatala_cfd_mg_coarse_jacobi);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_cfd_mg_coarse_matvec") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_cfd_mg_coarse_matvec";
    std::string_view sv(k_metal_navatala_cfd_mg_coarse_matvec);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_cfd_mg_coarse_rhs") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_cfd_mg_coarse_rhs";
    std::string_view sv(k_metal_navatala_cfd_mg_coarse_rhs);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_cfd_mg_find_best_match") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_cfd_mg_find_best_match";
    std::string_view sv(k_metal_navatala_cfd_mg_find_best_match);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_cfd_mg_pair_match") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_cfd_mg_pair_match";
    std::string_view sv(k_metal_navatala_cfd_mg_pair_match);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_cfd_mg_prolong_add") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_cfd_mg_prolong_add";
    std::string_view sv(k_metal_navatala_cfd_mg_prolong_add);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_cfd_mg_renumber_aggregates") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_cfd_mg_renumber_aggregates";
    std::string_view sv(k_metal_navatala_cfd_mg_renumber_aggregates);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_cfd_mg_renumber_propagate") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_cfd_mg_renumber_propagate";
    std::string_view sv(k_metal_navatala_cfd_mg_renumber_propagate);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_cfd_mg_resolve_chains") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_cfd_mg_resolve_chains";
    std::string_view sv(k_metal_navatala_cfd_mg_resolve_chains);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_cfd_mg_restrict_sum") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_cfd_mg_restrict_sum";
    std::string_view sv(k_metal_navatala_cfd_mg_restrict_sum);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_cfd_mg_strength_of_connection") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_cfd_mg_strength_of_connection";
    std::string_view sv(k_metal_navatala_cfd_mg_strength_of_connection);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_cfd_mg_zero_coarse") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_cfd_mg_zero_coarse";
    std::string_view sv(k_metal_navatala_cfd_mg_zero_coarse);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_cfd_phi_correct") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_cfd_phi_correct";
    std::string_view sv(k_metal_navatala_cfd_phi_correct);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_cfd_phi_correct_boundary") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_cfd_phi_correct_boundary";
    std::string_view sv(k_metal_navatala_cfd_phi_correct_boundary);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_cfd_phi_correct_internal") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_cfd_phi_correct_internal";
    std::string_view sv(k_metal_navatala_cfd_phi_correct_internal);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_cfd_axpy_in_place") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_cfd_axpy_in_place";
    std::string_view sv(k_metal_navatala_cfd_axpy_in_place);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_cfd_bc_dirichlet_face_flux") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_cfd_bc_dirichlet_face_flux";
    std::string_view sv(k_metal_navatala_cfd_bc_dirichlet_face_flux);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_cfd_bc_sn_grad_face_flux") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_cfd_bc_sn_grad_face_flux";
    std::string_view sv(k_metal_navatala_cfd_bc_sn_grad_face_flux);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_cfd_coeff_to_cf_in_place") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_cfd_coeff_to_cf_in_place";
    std::string_view sv(k_metal_navatala_cfd_coeff_to_cf_in_place);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_cfd_diag_from_cf") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_cfd_diag_from_cf";
    std::string_view sv(k_metal_navatala_cfd_diag_from_cf);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_cfd_dic_apply") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_cfd_dic_apply";
    std::string_view sv(k_metal_navatala_cfd_dic_apply);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_cfd_dic_build_reciprocal_d") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_cfd_dic_build_reciprocal_d";
    std::string_view sv(k_metal_navatala_cfd_dic_build_reciprocal_d);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_cfd_dot_partials") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_cfd_dot_partials";
    std::string_view sv(k_metal_navatala_cfd_dot_partials);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_cfd_face_flux") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_cfd_face_flux";
    std::string_view sv(k_metal_navatala_cfd_face_flux);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_cfd_inv_diag") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_cfd_inv_diag";
    std::string_view sv(k_metal_navatala_cfd_inv_diag);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_cfd_laplacian_from_face_flux") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_cfd_laplacian_from_face_flux";
    std::string_view sv(k_metal_navatala_cfd_laplacian_from_face_flux);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_cfd_mul_by_vol_in_place") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_cfd_mul_by_vol_in_place";
    std::string_view sv(k_metal_navatala_cfd_mul_by_vol_in_place);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_cfd_ref_add_ax") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_cfd_ref_add_ax";
    std::string_view sv(k_metal_navatala_cfd_ref_add_ax);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_cfd_ref_add_b") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_cfd_ref_add_b";
    std::string_view sv(k_metal_navatala_cfd_ref_add_b);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_cfd_rhs_sub") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_cfd_rhs_sub";
    std::string_view sv(k_metal_navatala_cfd_rhs_sub);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_cfd_shift_in_place") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_cfd_shift_in_place";
    std::string_view sv(k_metal_navatala_cfd_shift_in_place);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_cfd_sum_abs_partials") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_cfd_sum_abs_partials";
    std::string_view sv(k_metal_navatala_cfd_sum_abs_partials);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_cfd_upper_from_cf") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_cfd_upper_from_cf";
    std::string_view sv(k_metal_navatala_cfd_upper_from_cf);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_cfd_xpay_in_place") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_cfd_xpay_in_place";
    std::string_view sv(k_metal_navatala_cfd_xpay_in_place);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_cfd_negate_scalar") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_cfd_negate_scalar";
    std::string_view sv(k_metal_navatala_cfd_negate_scalar);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_cfd_exact_mat_vec") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_cfd_exact_mat_vec";
    std::string_view sv(k_metal_navatala_cfd_exact_mat_vec);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_cfd_face_flux_boundary") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_cfd_face_flux_boundary";
    std::string_view sv(k_metal_navatala_cfd_face_flux_boundary);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_cfd_face_flux_internal") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_cfd_face_flux_internal";
    std::string_view sv(k_metal_navatala_cfd_face_flux_internal);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_cfd_pressure_face_flux_correction_corrected") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_cfd_pressure_face_flux_correction_corrected";
    std::string_view sv(k_metal_navatala_cfd_pressure_face_flux_correction_corrected);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_cfd_pressure_sn_grad") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_cfd_pressure_sn_grad";
    std::string_view sv(k_metal_navatala_cfd_pressure_sn_grad);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_cfd_u_correct") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_cfd_u_correct";
    std::string_view sv(k_metal_navatala_cfd_u_correct);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_cfd_u_correct_reconstruct") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_cfd_u_correct_reconstruct";
    std::string_view sv(k_metal_navatala_cfd_u_correct_reconstruct);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_cfd_grad_alpha_cell") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_cfd_grad_alpha_cell";
    std::string_view sv(k_metal_navatala_cfd_grad_alpha_cell);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_cfd_n_hatf_all") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_cfd_n_hatf_all";
    std::string_view sv(k_metal_navatala_cfd_n_hatf_all);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_cfd_stf_all_faces") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_cfd_stf_all_faces";
    std::string_view sv(k_metal_navatala_cfd_stf_all_faces);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_cfd_k_omega_s_s_t_blending") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_cfd_k_omega_s_s_t_blending";
    std::string_view sv(k_metal_navatala_cfd_k_omega_s_s_t_blending);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_cfd_k_omega_s_s_t_nut_compute") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_cfd_k_omega_s_s_t_nut_compute";
    std::string_view sv(k_metal_navatala_cfd_k_omega_s_s_t_nut_compute);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_cfd_k_omega_s_s_t_sources") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_cfd_k_omega_s_s_t_sources";
    std::string_view sv(k_metal_navatala_cfd_k_omega_s_s_t_sources);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_cfd_mu_eff_from_rho_nu_eff") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_cfd_mu_eff_from_rho_nu_eff";
    std::string_view sv(k_metal_navatala_cfd_mu_eff_from_rho_nu_eff);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_cfd_mu_eff_from_rho_nut_and_nu_mix") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_cfd_mu_eff_from_rho_nut_and_nu_mix";
    std::string_view sv(k_metal_navatala_cfd_mu_eff_from_rho_nut_and_nu_mix);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_cfd_turbulence_add_explicit_source_device") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_cfd_turbulence_add_explicit_source_device";
    std::string_view sv(k_metal_navatala_cfd_turbulence_add_explicit_source_device);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_cfd_turbulence_dirichlet_cell_source") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_cfd_turbulence_dirichlet_cell_source";
    std::string_view sv(k_metal_navatala_cfd_turbulence_dirichlet_cell_source);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_cfd_turbulence_dirichlet_face_elimination") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_cfd_turbulence_dirichlet_face_elimination";
    std::string_view sv(k_metal_navatala_cfd_turbulence_dirichlet_face_elimination);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_cfd_two_mag_sqr_symm_grad_u") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_cfd_two_mag_sqr_symm_grad_u";
    std::string_view sv(k_metal_navatala_cfd_two_mag_sqr_symm_grad_u);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_cfd_attention_row_softmax") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_cfd_attention_row_softmax";
    std::string_view sv(k_metal_navatala_cfd_attention_row_softmax);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_cfd_attention_value_projection") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_cfd_attention_value_projection";
    std::string_view sv(k_metal_navatala_cfd_attention_value_projection);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_cfd_gather_mean") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_cfd_gather_mean";
    std::string_view sv(k_metal_navatala_cfd_gather_mean);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_cfd_layer_norm_forward") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_cfd_layer_norm_forward";
    std::string_view sv(k_metal_navatala_cfd_layer_norm_forward);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_cfd_radius_count") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_cfd_radius_count";
    std::string_view sv(k_metal_navatala_cfd_radius_count);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_cfd_scaled_dot_product") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_cfd_scaled_dot_product";
    std::string_view sv(k_metal_navatala_cfd_scaled_dot_product);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_cfd_scatter_add_atomic") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_cfd_scatter_add_atomic";
    std::string_view sv(k_metal_navatala_cfd_scatter_add_atomic);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_cfd_spectral_complex_mul") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_cfd_spectral_complex_mul";
    std::string_view sv(k_metal_navatala_cfd_spectral_complex_mul);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_cfd_spectral_conv1d_mode_sliced") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_cfd_spectral_conv1d_mode_sliced";
    std::string_view sv(k_metal_navatala_cfd_spectral_conv1d_mode_sliced);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_cfd_spectral_conv2d_mode_sliced") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_cfd_spectral_conv2d_mode_sliced";
    std::string_view sv(k_metal_navatala_cfd_spectral_conv2d_mode_sliced);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_cfd_spectral_conv3d_mode_sliced") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_cfd_spectral_conv3d_mode_sliced";
    std::string_view sv(k_metal_navatala_cfd_spectral_conv3d_mode_sliced);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_cfd_spectral_conv4d_mode_sliced") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_cfd_spectral_conv4d_mode_sliced";
    std::string_view sv(k_metal_navatala_cfd_spectral_conv4d_mode_sliced);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_cfd_compute_face_area_pair_weights_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_cfd_compute_face_area_pair_weights_f32";
    std::string_view sv(k_metal_navatala_cfd_compute_face_area_pair_weights_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_cfd_restrict_by_agglomeration_gather_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_cfd_restrict_by_agglomeration_gather_f32";
    std::string_view sv(k_metal_navatala_cfd_restrict_by_agglomeration_gather_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_cfd_prolongate_by_agglomeration_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_cfd_prolongate_by_agglomeration_f32";
    std::string_view sv(k_metal_navatala_cfd_prolongate_by_agglomeration_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_cfd_diag_inv_apply_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_cfd_diag_inv_apply_f32";
    std::string_view sv(k_metal_navatala_cfd_diag_inv_apply_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_cfd_gamg_proc_pack_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_cfd_gamg_proc_pack_f32";
    std::string_view sv(k_metal_navatala_cfd_gamg_proc_pack_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_cfd_gamg_proc_add_gather_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_cfd_gamg_proc_add_gather_f32";
    std::string_view sv(k_metal_navatala_cfd_gamg_proc_add_gather_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  return false;
}

} // namespace NavatalaRegistry

