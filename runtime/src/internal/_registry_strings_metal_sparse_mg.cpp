// SPDX-License-Identifier: Apache-2.0
// Kernel registry partition for the `metal_sparse_mg` bucket. Compiles
// against the declarations in `navatala/_registry.hpp` and links
// against the trampoline in `_registry.cpp`. Partitioning keeps each
// translation unit well under the 50K-line threshold so a typical
// build can parallelise across the buckets.
#include "navatala/_registry.hpp"
#include <cstdint>
#include <string>
#include <string_view>

const char* k_metal_navatala_sparse_mg_restrict_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_sparse_mg_restrict_f32(device const uint* RrowPtr [[buffer(0)]], device const uint* RcolIdx [[buffer(1)]], device const float* Rvalues [[buffer(2)]], device const float* rFine [[buffer(3)]], device const uint* nCoarse [[buffer(4)]], device float* rCoarse [[buffer(5)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  int row = int(__gid.x);
  int N = ((int)(nCoarse[0]));
  if ((row < N)) {
    int rs = ((int)(RrowPtr[row]));
    int re = ((int)(RrowPtr[(row + 1)]));
    float acc = as_type<float>(0x00000000u);
    for (int j = 0; j < (int)((re - rs)); ++j) {
      int k = (rs + j);
      int col = ((int)(RcolIdx[k]));
      float rv = Rvalues[k];
      float rf = rFine[col];
      acc = (acc + (rv * rf));
    }
    rCoarse[row] = acc;
  }
}

)kernel";
const char* k_metal_navatala_sparse_mg_prolongate_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_sparse_mg_prolongate_f32(device const uint* ProwPtr [[buffer(0)]], device const uint* PcolIdx [[buffer(1)]], device const float* Pvalues [[buffer(2)]], device const float* eCoarse [[buffer(3)]], device const uint* nFine [[buffer(4)]], device float* xFine [[buffer(5)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  int row = int(__gid.x);
  int N = ((int)(nFine[0]));
  if ((row < N)) {
    int rs = ((int)(ProwPtr[row]));
    int re = ((int)(ProwPtr[(row + 1)]));
    float pe = as_type<float>(0x00000000u);
    for (int j = 0; j < (int)((re - rs)); ++j) {
      int k = (rs + j);
      int col = ((int)(PcolIdx[k]));
      float pv = Pvalues[k];
      float ec = eCoarse[col];
      pe = (pe + (pv * ec));
    }
    float xold = xFine[row];
    xFine[row] = (xold + pe);
  }
}

)kernel";
const char* k_metal_navatala_sparse_mg_convergence_absolute_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_sparse_mg_convergence_absolute_f32(device const float* residualNorm [[buffer(0)]], device const float* tolerance [[buffer(1)]], device uint* converged [[buffer(2)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  float norm = residualNorm[0];
  float tol = tolerance[0];
  uint flag = (((norm < tol)) ? (1u) : (0u));
  converged[0] = flag;
}

)kernel";
const char* k_metal_navatala_sparse_mg_convergence_relative_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_sparse_mg_convergence_relative_f32(device const float* residualNorm [[buffer(0)]], device const float* initialNorm [[buffer(1)]], device const float* tolerance [[buffer(2)]], device uint* converged [[buffer(3)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  float norm = residualNorm[0];
  float init = initialNorm[0];
  float tol = tolerance[0];
  float ratio = (norm / init);
  uint flag = (((ratio < tol)) ? (1u) : (0u));
  converged[0] = flag;
}

)kernel";
const char* k_metal_navatala_sparse_mg_compute_residual_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_sparse_mg_compute_residual_f32(device const float* b [[buffer(0)]], device const float* Ax [[buffer(1)]], device const uint* n [[buffer(2)]], device float* r [[buffer(3)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  int i = int(__gid.x);
  int N = ((int)(n[0]));
  if ((i < N)) {
    float bi = b[i];
    float axi = Ax[i];
    r[i] = (bi - axi);
  }
}

)kernel";
const char* k_metal_navatala_sparse_mg_weighted_restrict_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_sparse_mg_weighted_restrict_f32(device const uint* RrowPtr [[buffer(0)]], device const uint* RcolIdx [[buffer(1)]], device const float* Rvalues [[buffer(2)]], device const float* rFine [[buffer(3)]], device const float* weights [[buffer(4)]], device const uint* nCoarse [[buffer(5)]], device float* rCoarse [[buffer(6)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  int row = int(__gid.x);
  int N = ((int)(nCoarse[0]));
  if ((row < N)) {
    int rs = ((int)(RrowPtr[row]));
    int re = ((int)(RrowPtr[(row + 1)]));
    float acc = as_type<float>(0x00000000u);
    for (int j = 0; j < (int)((re - rs)); ++j) {
      int k = (rs + j);
      int col = ((int)(RcolIdx[k]));
      float rv = Rvalues[k];
      float rf = rFine[col];
      float w = weights[col];
      acc = (acc + (rv * (w * rf)));
    }
    rCoarse[row] = acc;
  }
}

)kernel";
const char* k_metal_navatala_sparse_mg_error_norm_partials_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_sparse_mg_error_norm_partials_f32(device const float* error [[buffer(0)]], device const uint* n [[buffer(1)]], device float* partials [[buffer(2)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  int gid = int(__gid.x);
  int lid = int(__tid.x);
  int N = ((int)(n[0]));
  float sq = (((gid < N)) ? ((error[gid] * error[gid])) : (as_type<float>(0x00000000u)));
  float warpSum = simd_sum(sq);
  threadgroup float sdata[32];
  int lane = int(__lane);
  if ((lane == 0)) {
    int warpIdx = (lid / 32);
    sdata[warpIdx] = warpSum;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if ((lid < 8)) {
    float val = sdata[lid];
    float finalSum = simd_sum(val);
    if ((lid == 0)) {
      int grpId = int(__tgid.x);
      partials[grpId] = finalSum;
    }
  }
}

)kernel";
const char* k_metal_navatala_sparse_mg_zero_vector_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_sparse_mg_zero_vector_f32(device const uint* n [[buffer(0)]], device float* v [[buffer(1)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  int i = int(__gid.x);
  int N = ((int)(n[0]));
  if ((i < N)) {
    v[i] = as_type<float>(0x00000000u);
  }
}

)kernel";
const char* k_metal_navatala_sparse_mg_prolongate_damped_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_sparse_mg_prolongate_damped_f32(device const uint* ProwPtr [[buffer(0)]], device const uint* PcolIdx [[buffer(1)]], device const float* Pvalues [[buffer(2)]], device const float* eCoarse [[buffer(3)]], device const float* omega [[buffer(4)]], device const uint* nFine [[buffer(5)]], device float* xFine [[buffer(6)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  int row = int(__gid.x);
  int N = ((int)(nFine[0]));
  if ((row < N)) {
    int rs = ((int)(ProwPtr[row]));
    int re = ((int)(ProwPtr[(row + 1)]));
    float pe = as_type<float>(0x00000000u);
    for (int j = 0; j < (int)((re - rs)); ++j) {
      int k = (rs + j);
      int col = ((int)(PcolIdx[k]));
      float pv = Pvalues[k];
      float ec = eCoarse[col];
      pe = (pe + (pv * ec));
    }
    float om = omega[0];
    float xold = xFine[row];
    xFine[row] = (xold + (om * pe));
  }
}

)kernel";

namespace NavatalaRegistry {

const KernelArgumentInfo kAbiArgs_metal_navatala_sparse_mg_restrict_f32[] = {
  { "RrowPtr", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "RcolIdx", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "Rvalues", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "rFine", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "nCoarse", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "rCoarse", 5, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_sparse_mg_restrict_f32 = {
  1,
  "navatala_sparse_mg_restrict_f32",
  "metal",
  "navatala_sparse_mg_restrict_f32",
  "kernel:metal:navatala_sparse_mg_restrict_f32",
  "abi-r1:metal:navatala_sparse_mg_restrict_f32",
  "abi-r1:metal:navatala_sparse_mg_restrict_f32",
  6,
  kAbiArgs_metal_navatala_sparse_mg_restrict_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_sparse_mg_prolongate_f32[] = {
  { "ProwPtr", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "PcolIdx", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "Pvalues", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "eCoarse", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "nFine", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "xFine", 5, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_sparse_mg_prolongate_f32 = {
  1,
  "navatala_sparse_mg_prolongate_f32",
  "metal",
  "navatala_sparse_mg_prolongate_f32",
  "kernel:metal:navatala_sparse_mg_prolongate_f32",
  "abi-r1:metal:navatala_sparse_mg_prolongate_f32",
  "abi-r1:metal:navatala_sparse_mg_prolongate_f32",
  6,
  kAbiArgs_metal_navatala_sparse_mg_prolongate_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_sparse_mg_convergence_absolute_f32[] = {
  { "residualNorm", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "tolerance", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "converged", 2, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_sparse_mg_convergence_absolute_f32 = {
  1,
  "navatala_sparse_mg_convergence_absolute_f32",
  "metal",
  "navatala_sparse_mg_convergence_absolute_f32",
  "kernel:metal:navatala_sparse_mg_convergence_absolute_f32",
  "abi-r1:metal:navatala_sparse_mg_convergence_absolute_f32",
  "abi-r1:metal:navatala_sparse_mg_convergence_absolute_f32",
  3,
  kAbiArgs_metal_navatala_sparse_mg_convergence_absolute_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_sparse_mg_convergence_relative_f32[] = {
  { "residualNorm", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "initialNorm", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "tolerance", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "converged", 3, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_sparse_mg_convergence_relative_f32 = {
  1,
  "navatala_sparse_mg_convergence_relative_f32",
  "metal",
  "navatala_sparse_mg_convergence_relative_f32",
  "kernel:metal:navatala_sparse_mg_convergence_relative_f32",
  "abi-r1:metal:navatala_sparse_mg_convergence_relative_f32",
  "abi-r1:metal:navatala_sparse_mg_convergence_relative_f32",
  4,
  kAbiArgs_metal_navatala_sparse_mg_convergence_relative_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_sparse_mg_compute_residual_f32[] = {
  { "b", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "Ax", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "r", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_sparse_mg_compute_residual_f32 = {
  1,
  "navatala_sparse_mg_compute_residual_f32",
  "metal",
  "navatala_sparse_mg_compute_residual_f32",
  "kernel:metal:navatala_sparse_mg_compute_residual_f32",
  "abi-r1:metal:navatala_sparse_mg_compute_residual_f32",
  "abi-r1:metal:navatala_sparse_mg_compute_residual_f32",
  4,
  kAbiArgs_metal_navatala_sparse_mg_compute_residual_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_sparse_mg_weighted_restrict_f32[] = {
  { "RrowPtr", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "RcolIdx", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "Rvalues", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "rFine", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "weights", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "nCoarse", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "rCoarse", 6, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_sparse_mg_weighted_restrict_f32 = {
  1,
  "navatala_sparse_mg_weighted_restrict_f32",
  "metal",
  "navatala_sparse_mg_weighted_restrict_f32",
  "kernel:metal:navatala_sparse_mg_weighted_restrict_f32",
  "abi-r1:metal:navatala_sparse_mg_weighted_restrict_f32",
  "abi-r1:metal:navatala_sparse_mg_weighted_restrict_f32",
  7,
  kAbiArgs_metal_navatala_sparse_mg_weighted_restrict_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_sparse_mg_error_norm_partials_f32[] = {
  { "error", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "partials", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_sparse_mg_error_norm_partials_f32 = {
  1,
  "navatala_sparse_mg_error_norm_partials_f32",
  "metal",
  "navatala_sparse_mg_error_norm_partials_f32",
  "kernel:metal:navatala_sparse_mg_error_norm_partials_f32",
  "abi-r1:metal:navatala_sparse_mg_error_norm_partials_f32",
  "abi-r1:metal:navatala_sparse_mg_error_norm_partials_f32",
  3,
  kAbiArgs_metal_navatala_sparse_mg_error_norm_partials_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_sparse_mg_zero_vector_f32[] = {
  { "n", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "v", 1, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_sparse_mg_zero_vector_f32 = {
  1,
  "navatala_sparse_mg_zero_vector_f32",
  "metal",
  "navatala_sparse_mg_zero_vector_f32",
  "kernel:metal:navatala_sparse_mg_zero_vector_f32",
  "abi-r1:metal:navatala_sparse_mg_zero_vector_f32",
  "abi-r1:metal:navatala_sparse_mg_zero_vector_f32",
  2,
  kAbiArgs_metal_navatala_sparse_mg_zero_vector_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_sparse_mg_prolongate_damped_f32[] = {
  { "ProwPtr", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "PcolIdx", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "Pvalues", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "eCoarse", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "omega", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "nFine", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "xFine", 6, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_sparse_mg_prolongate_damped_f32 = {
  1,
  "navatala_sparse_mg_prolongate_damped_f32",
  "metal",
  "navatala_sparse_mg_prolongate_damped_f32",
  "kernel:metal:navatala_sparse_mg_prolongate_damped_f32",
  "abi-r1:metal:navatala_sparse_mg_prolongate_damped_f32",
  "abi-r1:metal:navatala_sparse_mg_prolongate_damped_f32",
  7,
  kAbiArgs_metal_navatala_sparse_mg_prolongate_damped_f32
};

bool tryGetKernelAbiManifest_metal_sparse_mg(const std::string& backend, const std::string& kernelName, const KernelAbiManifestInfo*& out) {
  if (backend == "metal" && kernelName == "navatala_sparse_mg_restrict_f32") {
    out = &kAbiManifest_metal_navatala_sparse_mg_restrict_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_mg_prolongate_f32") {
    out = &kAbiManifest_metal_navatala_sparse_mg_prolongate_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_mg_convergence_absolute_f32") {
    out = &kAbiManifest_metal_navatala_sparse_mg_convergence_absolute_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_mg_convergence_relative_f32") {
    out = &kAbiManifest_metal_navatala_sparse_mg_convergence_relative_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_mg_compute_residual_f32") {
    out = &kAbiManifest_metal_navatala_sparse_mg_compute_residual_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_mg_weighted_restrict_f32") {
    out = &kAbiManifest_metal_navatala_sparse_mg_weighted_restrict_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_mg_error_norm_partials_f32") {
    out = &kAbiManifest_metal_navatala_sparse_mg_error_norm_partials_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_mg_zero_vector_f32") {
    out = &kAbiManifest_metal_navatala_sparse_mg_zero_vector_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_mg_prolongate_damped_f32") {
    out = &kAbiManifest_metal_navatala_sparse_mg_prolongate_damped_f32;
    return true;
  }
  out = nullptr;
  return false;
}

bool tryGetKernelSource_metal_sparse_mg(const std::string& backend, const std::string& kernelName, GpuRuntime::ProgramSource& out) {
  if (backend == "metal" && kernelName == "navatala_sparse_mg_restrict_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_sparse_mg_restrict_f32";
    std::string_view sv(k_metal_navatala_sparse_mg_restrict_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_mg_prolongate_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_sparse_mg_prolongate_f32";
    std::string_view sv(k_metal_navatala_sparse_mg_prolongate_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_mg_convergence_absolute_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_sparse_mg_convergence_absolute_f32";
    std::string_view sv(k_metal_navatala_sparse_mg_convergence_absolute_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_mg_convergence_relative_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_sparse_mg_convergence_relative_f32";
    std::string_view sv(k_metal_navatala_sparse_mg_convergence_relative_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_mg_compute_residual_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_sparse_mg_compute_residual_f32";
    std::string_view sv(k_metal_navatala_sparse_mg_compute_residual_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_mg_weighted_restrict_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_sparse_mg_weighted_restrict_f32";
    std::string_view sv(k_metal_navatala_sparse_mg_weighted_restrict_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_mg_error_norm_partials_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_sparse_mg_error_norm_partials_f32";
    std::string_view sv(k_metal_navatala_sparse_mg_error_norm_partials_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_mg_zero_vector_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_sparse_mg_zero_vector_f32";
    std::string_view sv(k_metal_navatala_sparse_mg_zero_vector_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_mg_prolongate_damped_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_sparse_mg_prolongate_damped_f32";
    std::string_view sv(k_metal_navatala_sparse_mg_prolongate_damped_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  return false;
}

} // namespace NavatalaRegistry

