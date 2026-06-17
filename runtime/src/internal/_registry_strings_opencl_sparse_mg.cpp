// SPDX-License-Identifier: Apache-2.0
// Kernel registry partition for the `opencl_sparse_mg` bucket. Compiles
// against the declarations in `navatala/_registry.hpp` and links
// against the trampoline in `_registry.cpp`. Partitioning keeps each
// translation unit well under the 50K-line threshold so a typical
// build can parallelise across the buckets.
#include "navatala/_registry.hpp"
#include <cstdint>
#include <string>
#include <string_view>

const char* k_opencl_navatala_sparse_mg_restrict_f32 = R"kernel(
__kernel void navatala_sparse_mg_restrict_f32(__global const uint* RrowPtr, __global const uint* RcolIdx, __global const float* Rvalues, __global const float* rFine, __global const uint* nCoarse, __global float* rCoarse) {
  int gid0 = (int)get_global_id(0);
  int row = (int)(get_global_id(0));
  int N = ((int)(nCoarse[0]));
  if ((row < N)) {
    int rs = ((int)(RrowPtr[row]));
    int re = ((int)(RrowPtr[(row + 1)]));
    float acc = as_float(0x00000000u);
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
const char* k_opencl_navatala_sparse_mg_restrict_f64 = R"kernel(
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
__kernel void navatala_sparse_mg_restrict_f64(__global const uint* RrowPtr, __global const uint* RcolIdx, __global const double* Rvalues, __global const double* rFine, __global const uint* nCoarse, __global double* rCoarse) {
  int gid0 = (int)get_global_id(0);
  int row = (int)(get_global_id(0));
  int N = ((int)(nCoarse[0]));
  if ((row < N)) {
    int rs = ((int)(RrowPtr[row]));
    int re = ((int)(RrowPtr[(row + 1)]));
    double acc = as_double(0x0000000000000000ul);
    for (int j = 0; j < (int)((re - rs)); ++j) {
      int k = (rs + j);
      int col = ((int)(RcolIdx[k]));
      double rv = Rvalues[k];
      double rf = rFine[col];
      acc = (acc + (rv * rf));
    }
    rCoarse[row] = acc;
  }
}

)kernel";
const char* k_opencl_navatala_sparse_mg_prolongate_f32 = R"kernel(
__kernel void navatala_sparse_mg_prolongate_f32(__global const uint* ProwPtr, __global const uint* PcolIdx, __global const float* Pvalues, __global const float* eCoarse, __global const uint* nFine, __global float* xFine) {
  int gid0 = (int)get_global_id(0);
  int row = (int)(get_global_id(0));
  int N = ((int)(nFine[0]));
  if ((row < N)) {
    int rs = ((int)(ProwPtr[row]));
    int re = ((int)(ProwPtr[(row + 1)]));
    float pe = as_float(0x00000000u);
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
const char* k_opencl_navatala_sparse_mg_prolongate_f64 = R"kernel(
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
__kernel void navatala_sparse_mg_prolongate_f64(__global const uint* ProwPtr, __global const uint* PcolIdx, __global const double* Pvalues, __global const double* eCoarse, __global const uint* nFine, __global double* xFine) {
  int gid0 = (int)get_global_id(0);
  int row = (int)(get_global_id(0));
  int N = ((int)(nFine[0]));
  if ((row < N)) {
    int rs = ((int)(ProwPtr[row]));
    int re = ((int)(ProwPtr[(row + 1)]));
    double pe = as_double(0x0000000000000000ul);
    for (int j = 0; j < (int)((re - rs)); ++j) {
      int k = (rs + j);
      int col = ((int)(PcolIdx[k]));
      double pv = Pvalues[k];
      double ec = eCoarse[col];
      pe = (pe + (pv * ec));
    }
    double xold = xFine[row];
    xFine[row] = (xold + pe);
  }
}

)kernel";
const char* k_opencl_navatala_sparse_mg_convergence_absolute_f32 = R"kernel(
__kernel void navatala_sparse_mg_convergence_absolute_f32(__global const float* residualNorm, __global const float* tolerance, __global uint* converged) {
  int gid0 = (int)get_global_id(0);
  float norm = residualNorm[0];
  float tol = tolerance[0];
  uint flag = (((norm < tol)) ? ((uint)(1u)) : ((uint)(0u)));
  converged[0] = flag;
}

)kernel";
const char* k_opencl_navatala_sparse_mg_convergence_relative_f32 = R"kernel(
__kernel void navatala_sparse_mg_convergence_relative_f32(__global const float* residualNorm, __global const float* initialNorm, __global const float* tolerance, __global uint* converged) {
  int gid0 = (int)get_global_id(0);
  float norm = residualNorm[0];
  float init = initialNorm[0];
  float tol = tolerance[0];
  float ratio = (norm / init);
  uint flag = (((ratio < tol)) ? ((uint)(1u)) : ((uint)(0u)));
  converged[0] = flag;
}

)kernel";
const char* k_opencl_navatala_sparse_mg_convergence_absolute_f64 = R"kernel(
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
__kernel void navatala_sparse_mg_convergence_absolute_f64(__global const double* residualNorm, __global const double* tolerance, __global uint* converged) {
  int gid0 = (int)get_global_id(0);
  double norm = residualNorm[0];
  double tol = tolerance[0];
  uint flag = (((norm < tol)) ? ((uint)(1u)) : ((uint)(0u)));
  converged[0] = flag;
}

)kernel";
const char* k_opencl_navatala_sparse_mg_convergence_relative_f64 = R"kernel(
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
__kernel void navatala_sparse_mg_convergence_relative_f64(__global const double* residualNorm, __global const double* initialNorm, __global const double* tolerance, __global uint* converged) {
  int gid0 = (int)get_global_id(0);
  double norm = residualNorm[0];
  double init = initialNorm[0];
  double tol = tolerance[0];
  double ratio = (norm / init);
  uint flag = (((ratio < tol)) ? ((uint)(1u)) : ((uint)(0u)));
  converged[0] = flag;
}

)kernel";
const char* k_opencl_navatala_sparse_mg_compute_residual_f32 = R"kernel(
__kernel void navatala_sparse_mg_compute_residual_f32(__global const float* b, __global const float* Ax, __global const uint* n, __global float* r) {
  int gid0 = (int)get_global_id(0);
  int i = (int)(get_global_id(0));
  int N = ((int)(n[0]));
  if ((i < N)) {
    float bi = b[i];
    float axi = Ax[i];
    r[i] = (bi - axi);
  }
}

)kernel";
const char* k_opencl_navatala_sparse_mg_compute_residual_f64 = R"kernel(
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
__kernel void navatala_sparse_mg_compute_residual_f64(__global const double* b, __global const double* Ax, __global const uint* n, __global double* r) {
  int gid0 = (int)get_global_id(0);
  int i = (int)(get_global_id(0));
  int N = ((int)(n[0]));
  if ((i < N)) {
    double bi = b[i];
    double axi = Ax[i];
    r[i] = (bi - axi);
  }
}

)kernel";
const char* k_opencl_navatala_sparse_mg_weighted_restrict_f32 = R"kernel(
__kernel void navatala_sparse_mg_weighted_restrict_f32(__global const uint* RrowPtr, __global const uint* RcolIdx, __global const float* Rvalues, __global const float* rFine, __global const float* weights, __global const uint* nCoarse, __global float* rCoarse) {
  int gid0 = (int)get_global_id(0);
  int row = (int)(get_global_id(0));
  int N = ((int)(nCoarse[0]));
  if ((row < N)) {
    int rs = ((int)(RrowPtr[row]));
    int re = ((int)(RrowPtr[(row + 1)]));
    float acc = as_float(0x00000000u);
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
const char* k_opencl_navatala_sparse_mg_weighted_restrict_f64 = R"kernel(
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
__kernel void navatala_sparse_mg_weighted_restrict_f64(__global const uint* RrowPtr, __global const uint* RcolIdx, __global const double* Rvalues, __global const double* rFine, __global const double* weights, __global const uint* nCoarse, __global double* rCoarse) {
  int gid0 = (int)get_global_id(0);
  int row = (int)(get_global_id(0));
  int N = ((int)(nCoarse[0]));
  if ((row < N)) {
    int rs = ((int)(RrowPtr[row]));
    int re = ((int)(RrowPtr[(row + 1)]));
    double acc = as_double(0x0000000000000000ul);
    for (int j = 0; j < (int)((re - rs)); ++j) {
      int k = (rs + j);
      int col = ((int)(RcolIdx[k]));
      double rv = Rvalues[k];
      double rf = rFine[col];
      double w = weights[col];
      acc = (acc + (rv * (w * rf)));
    }
    rCoarse[row] = acc;
  }
}

)kernel";
const char* k_opencl_navatala_sparse_mg_error_norm_partials_f32 = R"kernel(
#pragma OPENCL EXTENSION cl_khr_subgroups : enable
__kernel void navatala_sparse_mg_error_norm_partials_f32(__global const float* error, __global const uint* n, __global float* partials) {
  int gid0 = (int)get_global_id(0);
  int gid = (int)(get_global_id(0));
  int lid = (int)(get_local_id(0));
  int N = ((int)(n[0]));
  float sq = (((gid < N)) ? ((error[gid] * error[gid])) : (as_float(0x00000000u)));
  float warpSum = sub_group_reduce_add(sq);
  __local float sdata[32];
  int lane = (int)(get_sub_group_local_id());
  if ((lane == 0)) {
    int warpIdx = (lid / 32);
    sdata[warpIdx] = warpSum;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  if ((lid < 8)) {
    float val = sdata[lid];
    float finalSum = sub_group_reduce_add(val);
    if ((lid == 0)) {
      int grpId = (int)(get_group_id(0));
      partials[grpId] = finalSum;
    }
  }
}

)kernel";
const char* k_opencl_navatala_sparse_mg_error_norm_partials_f64 = R"kernel(
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#pragma OPENCL EXTENSION cl_khr_subgroups : enable
__kernel void navatala_sparse_mg_error_norm_partials_f64(__global const double* error, __global const uint* n, __global double* partials) {
  int gid0 = (int)get_global_id(0);
  int gid = (int)(get_global_id(0));
  int lid = (int)(get_local_id(0));
  int N = ((int)(n[0]));
  double sq = (((gid < N)) ? ((error[gid] * error[gid])) : (as_double(0x0000000000000000ul)));
  double warpSum = sub_group_reduce_add(sq);
  __local double sdata[32];
  int lane = (int)(get_sub_group_local_id());
  if ((lane == 0)) {
    int warpIdx = (lid / 32);
    sdata[warpIdx] = warpSum;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  if ((lid < 8)) {
    double val = sdata[lid];
    double finalSum = sub_group_reduce_add(val);
    if ((lid == 0)) {
      int grpId = (int)(get_group_id(0));
      partials[grpId] = finalSum;
    }
  }
}

)kernel";
const char* k_opencl_navatala_sparse_mg_zero_vector_f32 = R"kernel(
__kernel void navatala_sparse_mg_zero_vector_f32(__global const uint* n, __global float* v) {
  int gid0 = (int)get_global_id(0);
  int i = (int)(get_global_id(0));
  int N = ((int)(n[0]));
  if ((i < N)) {
    v[i] = as_float(0x00000000u);
  }
}

)kernel";
const char* k_opencl_navatala_sparse_mg_zero_vector_f64 = R"kernel(
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
__kernel void navatala_sparse_mg_zero_vector_f64(__global const uint* n, __global double* v) {
  int gid0 = (int)get_global_id(0);
  int i = (int)(get_global_id(0));
  int N = ((int)(n[0]));
  if ((i < N)) {
    v[i] = as_double(0x0000000000000000ul);
  }
}

)kernel";
const char* k_opencl_navatala_sparse_mg_prolongate_damped_f32 = R"kernel(
__kernel void navatala_sparse_mg_prolongate_damped_f32(__global const uint* ProwPtr, __global const uint* PcolIdx, __global const float* Pvalues, __global const float* eCoarse, __global const float* omega, __global const uint* nFine, __global float* xFine) {
  int gid0 = (int)get_global_id(0);
  int row = (int)(get_global_id(0));
  int N = ((int)(nFine[0]));
  if ((row < N)) {
    int rs = ((int)(ProwPtr[row]));
    int re = ((int)(ProwPtr[(row + 1)]));
    float pe = as_float(0x00000000u);
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
const char* k_opencl_navatala_sparse_mg_prolongate_damped_f64 = R"kernel(
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
__kernel void navatala_sparse_mg_prolongate_damped_f64(__global const uint* ProwPtr, __global const uint* PcolIdx, __global const double* Pvalues, __global const double* eCoarse, __global const double* omega, __global const uint* nFine, __global double* xFine) {
  int gid0 = (int)get_global_id(0);
  int row = (int)(get_global_id(0));
  int N = ((int)(nFine[0]));
  if ((row < N)) {
    int rs = ((int)(ProwPtr[row]));
    int re = ((int)(ProwPtr[(row + 1)]));
    double pe = as_double(0x0000000000000000ul);
    for (int j = 0; j < (int)((re - rs)); ++j) {
      int k = (rs + j);
      int col = ((int)(PcolIdx[k]));
      double pv = Pvalues[k];
      double ec = eCoarse[col];
      pe = (pe + (pv * ec));
    }
    double om = omega[0];
    double xold = xFine[row];
    xFine[row] = (xold + (om * pe));
  }
}

)kernel";

namespace NavatalaRegistry {

const KernelArgumentInfo kAbiArgs_opencl_navatala_sparse_mg_restrict_f32[] = {
  { "RrowPtr", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "RcolIdx", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "Rvalues", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "rFine", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "nCoarse", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "rCoarse", 5, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_sparse_mg_restrict_f32 = {
  1,
  "navatala_sparse_mg_restrict_f32",
  "opencl",
  "navatala_sparse_mg_restrict_f32",
  "kernel:opencl:navatala_sparse_mg_restrict_f32",
  "abi-r1:opencl:navatala_sparse_mg_restrict_f32",
  "abi-r1:opencl:navatala_sparse_mg_restrict_f32",
  6,
  kAbiArgs_opencl_navatala_sparse_mg_restrict_f32
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_sparse_mg_restrict_f64[] = {
  { "RrowPtr", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "RcolIdx", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "Rvalues", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "rFine", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "nCoarse", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "rCoarse", 5, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_sparse_mg_restrict_f64 = {
  1,
  "navatala_sparse_mg_restrict_f64",
  "opencl",
  "navatala_sparse_mg_restrict_f64",
  "kernel:opencl:navatala_sparse_mg_restrict_f64",
  "abi-r1:opencl:navatala_sparse_mg_restrict_f64",
  "abi-r1:opencl:navatala_sparse_mg_restrict_f64",
  6,
  kAbiArgs_opencl_navatala_sparse_mg_restrict_f64
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_sparse_mg_prolongate_f32[] = {
  { "ProwPtr", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "PcolIdx", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "Pvalues", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "eCoarse", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "nFine", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "xFine", 5, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_sparse_mg_prolongate_f32 = {
  1,
  "navatala_sparse_mg_prolongate_f32",
  "opencl",
  "navatala_sparse_mg_prolongate_f32",
  "kernel:opencl:navatala_sparse_mg_prolongate_f32",
  "abi-r1:opencl:navatala_sparse_mg_prolongate_f32",
  "abi-r1:opencl:navatala_sparse_mg_prolongate_f32",
  6,
  kAbiArgs_opencl_navatala_sparse_mg_prolongate_f32
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_sparse_mg_prolongate_f64[] = {
  { "ProwPtr", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "PcolIdx", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "Pvalues", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "eCoarse", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "nFine", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "xFine", 5, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_sparse_mg_prolongate_f64 = {
  1,
  "navatala_sparse_mg_prolongate_f64",
  "opencl",
  "navatala_sparse_mg_prolongate_f64",
  "kernel:opencl:navatala_sparse_mg_prolongate_f64",
  "abi-r1:opencl:navatala_sparse_mg_prolongate_f64",
  "abi-r1:opencl:navatala_sparse_mg_prolongate_f64",
  6,
  kAbiArgs_opencl_navatala_sparse_mg_prolongate_f64
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_sparse_mg_convergence_absolute_f32[] = {
  { "residualNorm", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "tolerance", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "converged", 2, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_sparse_mg_convergence_absolute_f32 = {
  1,
  "navatala_sparse_mg_convergence_absolute_f32",
  "opencl",
  "navatala_sparse_mg_convergence_absolute_f32",
  "kernel:opencl:navatala_sparse_mg_convergence_absolute_f32",
  "abi-r1:opencl:navatala_sparse_mg_convergence_absolute_f32",
  "abi-r1:opencl:navatala_sparse_mg_convergence_absolute_f32",
  3,
  kAbiArgs_opencl_navatala_sparse_mg_convergence_absolute_f32
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_sparse_mg_convergence_relative_f32[] = {
  { "residualNorm", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "initialNorm", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "tolerance", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "converged", 3, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_sparse_mg_convergence_relative_f32 = {
  1,
  "navatala_sparse_mg_convergence_relative_f32",
  "opencl",
  "navatala_sparse_mg_convergence_relative_f32",
  "kernel:opencl:navatala_sparse_mg_convergence_relative_f32",
  "abi-r1:opencl:navatala_sparse_mg_convergence_relative_f32",
  "abi-r1:opencl:navatala_sparse_mg_convergence_relative_f32",
  4,
  kAbiArgs_opencl_navatala_sparse_mg_convergence_relative_f32
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_sparse_mg_convergence_absolute_f64[] = {
  { "residualNorm", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 8, 8, 256, nullptr, 0, 0 },
  { "tolerance", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 8, 8, 256, nullptr, 0, 0 },
  { "converged", 2, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_sparse_mg_convergence_absolute_f64 = {
  1,
  "navatala_sparse_mg_convergence_absolute_f64",
  "opencl",
  "navatala_sparse_mg_convergence_absolute_f64",
  "kernel:opencl:navatala_sparse_mg_convergence_absolute_f64",
  "abi-r1:opencl:navatala_sparse_mg_convergence_absolute_f64",
  "abi-r1:opencl:navatala_sparse_mg_convergence_absolute_f64",
  3,
  kAbiArgs_opencl_navatala_sparse_mg_convergence_absolute_f64
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_sparse_mg_convergence_relative_f64[] = {
  { "residualNorm", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 8, 8, 256, nullptr, 0, 0 },
  { "initialNorm", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 8, 8, 256, nullptr, 0, 0 },
  { "tolerance", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 8, 8, 256, nullptr, 0, 0 },
  { "converged", 3, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_sparse_mg_convergence_relative_f64 = {
  1,
  "navatala_sparse_mg_convergence_relative_f64",
  "opencl",
  "navatala_sparse_mg_convergence_relative_f64",
  "kernel:opencl:navatala_sparse_mg_convergence_relative_f64",
  "abi-r1:opencl:navatala_sparse_mg_convergence_relative_f64",
  "abi-r1:opencl:navatala_sparse_mg_convergence_relative_f64",
  4,
  kAbiArgs_opencl_navatala_sparse_mg_convergence_relative_f64
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_sparse_mg_compute_residual_f32[] = {
  { "b", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "Ax", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "r", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_sparse_mg_compute_residual_f32 = {
  1,
  "navatala_sparse_mg_compute_residual_f32",
  "opencl",
  "navatala_sparse_mg_compute_residual_f32",
  "kernel:opencl:navatala_sparse_mg_compute_residual_f32",
  "abi-r1:opencl:navatala_sparse_mg_compute_residual_f32",
  "abi-r1:opencl:navatala_sparse_mg_compute_residual_f32",
  4,
  kAbiArgs_opencl_navatala_sparse_mg_compute_residual_f32
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_sparse_mg_compute_residual_f64[] = {
  { "b", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "Ax", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "r", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_sparse_mg_compute_residual_f64 = {
  1,
  "navatala_sparse_mg_compute_residual_f64",
  "opencl",
  "navatala_sparse_mg_compute_residual_f64",
  "kernel:opencl:navatala_sparse_mg_compute_residual_f64",
  "abi-r1:opencl:navatala_sparse_mg_compute_residual_f64",
  "abi-r1:opencl:navatala_sparse_mg_compute_residual_f64",
  4,
  kAbiArgs_opencl_navatala_sparse_mg_compute_residual_f64
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_sparse_mg_weighted_restrict_f32[] = {
  { "RrowPtr", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "RcolIdx", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "Rvalues", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "rFine", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "weights", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "nCoarse", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "rCoarse", 6, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_sparse_mg_weighted_restrict_f32 = {
  1,
  "navatala_sparse_mg_weighted_restrict_f32",
  "opencl",
  "navatala_sparse_mg_weighted_restrict_f32",
  "kernel:opencl:navatala_sparse_mg_weighted_restrict_f32",
  "abi-r1:opencl:navatala_sparse_mg_weighted_restrict_f32",
  "abi-r1:opencl:navatala_sparse_mg_weighted_restrict_f32",
  7,
  kAbiArgs_opencl_navatala_sparse_mg_weighted_restrict_f32
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_sparse_mg_weighted_restrict_f64[] = {
  { "RrowPtr", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "RcolIdx", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "Rvalues", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "rFine", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "weights", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "nCoarse", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "rCoarse", 6, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_sparse_mg_weighted_restrict_f64 = {
  1,
  "navatala_sparse_mg_weighted_restrict_f64",
  "opencl",
  "navatala_sparse_mg_weighted_restrict_f64",
  "kernel:opencl:navatala_sparse_mg_weighted_restrict_f64",
  "abi-r1:opencl:navatala_sparse_mg_weighted_restrict_f64",
  "abi-r1:opencl:navatala_sparse_mg_weighted_restrict_f64",
  7,
  kAbiArgs_opencl_navatala_sparse_mg_weighted_restrict_f64
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_sparse_mg_error_norm_partials_f32[] = {
  { "error", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "partials", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_sparse_mg_error_norm_partials_f32 = {
  1,
  "navatala_sparse_mg_error_norm_partials_f32",
  "opencl",
  "navatala_sparse_mg_error_norm_partials_f32",
  "kernel:opencl:navatala_sparse_mg_error_norm_partials_f32",
  "abi-r1:opencl:navatala_sparse_mg_error_norm_partials_f32",
  "abi-r1:opencl:navatala_sparse_mg_error_norm_partials_f32",
  3,
  kAbiArgs_opencl_navatala_sparse_mg_error_norm_partials_f32
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_sparse_mg_error_norm_partials_f64[] = {
  { "error", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "n", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "partials", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_sparse_mg_error_norm_partials_f64 = {
  1,
  "navatala_sparse_mg_error_norm_partials_f64",
  "opencl",
  "navatala_sparse_mg_error_norm_partials_f64",
  "kernel:opencl:navatala_sparse_mg_error_norm_partials_f64",
  "abi-r1:opencl:navatala_sparse_mg_error_norm_partials_f64",
  "abi-r1:opencl:navatala_sparse_mg_error_norm_partials_f64",
  3,
  kAbiArgs_opencl_navatala_sparse_mg_error_norm_partials_f64
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_sparse_mg_zero_vector_f32[] = {
  { "n", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "v", 1, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_sparse_mg_zero_vector_f32 = {
  1,
  "navatala_sparse_mg_zero_vector_f32",
  "opencl",
  "navatala_sparse_mg_zero_vector_f32",
  "kernel:opencl:navatala_sparse_mg_zero_vector_f32",
  "abi-r1:opencl:navatala_sparse_mg_zero_vector_f32",
  "abi-r1:opencl:navatala_sparse_mg_zero_vector_f32",
  2,
  kAbiArgs_opencl_navatala_sparse_mg_zero_vector_f32
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_sparse_mg_zero_vector_f64[] = {
  { "n", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "v", 1, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_sparse_mg_zero_vector_f64 = {
  1,
  "navatala_sparse_mg_zero_vector_f64",
  "opencl",
  "navatala_sparse_mg_zero_vector_f64",
  "kernel:opencl:navatala_sparse_mg_zero_vector_f64",
  "abi-r1:opencl:navatala_sparse_mg_zero_vector_f64",
  "abi-r1:opencl:navatala_sparse_mg_zero_vector_f64",
  2,
  kAbiArgs_opencl_navatala_sparse_mg_zero_vector_f64
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_sparse_mg_prolongate_damped_f32[] = {
  { "ProwPtr", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "PcolIdx", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "Pvalues", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "eCoarse", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "omega", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "nFine", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "xFine", 6, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_sparse_mg_prolongate_damped_f32 = {
  1,
  "navatala_sparse_mg_prolongate_damped_f32",
  "opencl",
  "navatala_sparse_mg_prolongate_damped_f32",
  "kernel:opencl:navatala_sparse_mg_prolongate_damped_f32",
  "abi-r1:opencl:navatala_sparse_mg_prolongate_damped_f32",
  "abi-r1:opencl:navatala_sparse_mg_prolongate_damped_f32",
  7,
  kAbiArgs_opencl_navatala_sparse_mg_prolongate_damped_f32
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_sparse_mg_prolongate_damped_f64[] = {
  { "ProwPtr", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "PcolIdx", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "Pvalues", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "eCoarse", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "omega", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 8, 8, 256, nullptr, 0, 0 },
  { "nFine", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "xFine", 6, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_sparse_mg_prolongate_damped_f64 = {
  1,
  "navatala_sparse_mg_prolongate_damped_f64",
  "opencl",
  "navatala_sparse_mg_prolongate_damped_f64",
  "kernel:opencl:navatala_sparse_mg_prolongate_damped_f64",
  "abi-r1:opencl:navatala_sparse_mg_prolongate_damped_f64",
  "abi-r1:opencl:navatala_sparse_mg_prolongate_damped_f64",
  7,
  kAbiArgs_opencl_navatala_sparse_mg_prolongate_damped_f64
};

bool tryGetKernelAbiManifest_opencl_sparse_mg(const std::string& backend, const std::string& kernelName, const KernelAbiManifestInfo*& out) {
  if (backend == "opencl" && kernelName == "navatala_sparse_mg_restrict_f32") {
    out = &kAbiManifest_opencl_navatala_sparse_mg_restrict_f32;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_sparse_mg_restrict_f64") {
    out = &kAbiManifest_opencl_navatala_sparse_mg_restrict_f64;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_sparse_mg_prolongate_f32") {
    out = &kAbiManifest_opencl_navatala_sparse_mg_prolongate_f32;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_sparse_mg_prolongate_f64") {
    out = &kAbiManifest_opencl_navatala_sparse_mg_prolongate_f64;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_sparse_mg_convergence_absolute_f32") {
    out = &kAbiManifest_opencl_navatala_sparse_mg_convergence_absolute_f32;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_sparse_mg_convergence_relative_f32") {
    out = &kAbiManifest_opencl_navatala_sparse_mg_convergence_relative_f32;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_sparse_mg_convergence_absolute_f64") {
    out = &kAbiManifest_opencl_navatala_sparse_mg_convergence_absolute_f64;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_sparse_mg_convergence_relative_f64") {
    out = &kAbiManifest_opencl_navatala_sparse_mg_convergence_relative_f64;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_sparse_mg_compute_residual_f32") {
    out = &kAbiManifest_opencl_navatala_sparse_mg_compute_residual_f32;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_sparse_mg_compute_residual_f64") {
    out = &kAbiManifest_opencl_navatala_sparse_mg_compute_residual_f64;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_sparse_mg_weighted_restrict_f32") {
    out = &kAbiManifest_opencl_navatala_sparse_mg_weighted_restrict_f32;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_sparse_mg_weighted_restrict_f64") {
    out = &kAbiManifest_opencl_navatala_sparse_mg_weighted_restrict_f64;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_sparse_mg_error_norm_partials_f32") {
    out = &kAbiManifest_opencl_navatala_sparse_mg_error_norm_partials_f32;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_sparse_mg_error_norm_partials_f64") {
    out = &kAbiManifest_opencl_navatala_sparse_mg_error_norm_partials_f64;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_sparse_mg_zero_vector_f32") {
    out = &kAbiManifest_opencl_navatala_sparse_mg_zero_vector_f32;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_sparse_mg_zero_vector_f64") {
    out = &kAbiManifest_opencl_navatala_sparse_mg_zero_vector_f64;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_sparse_mg_prolongate_damped_f32") {
    out = &kAbiManifest_opencl_navatala_sparse_mg_prolongate_damped_f32;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_sparse_mg_prolongate_damped_f64") {
    out = &kAbiManifest_opencl_navatala_sparse_mg_prolongate_damped_f64;
    return true;
  }
  out = nullptr;
  return false;
}

bool tryGetKernelSource_opencl_sparse_mg(const std::string& backend, const std::string& kernelName, GpuRuntime::ProgramSource& out) {
  if (backend == "opencl" && kernelName == "navatala_sparse_mg_restrict_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_sparse_mg_restrict_f32";
    std::string_view sv(k_opencl_navatala_sparse_mg_restrict_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_sparse_mg_restrict_f64") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_sparse_mg_restrict_f64";
    std::string_view sv(k_opencl_navatala_sparse_mg_restrict_f64);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_sparse_mg_prolongate_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_sparse_mg_prolongate_f32";
    std::string_view sv(k_opencl_navatala_sparse_mg_prolongate_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_sparse_mg_prolongate_f64") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_sparse_mg_prolongate_f64";
    std::string_view sv(k_opencl_navatala_sparse_mg_prolongate_f64);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_sparse_mg_convergence_absolute_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_sparse_mg_convergence_absolute_f32";
    std::string_view sv(k_opencl_navatala_sparse_mg_convergence_absolute_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_sparse_mg_convergence_relative_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_sparse_mg_convergence_relative_f32";
    std::string_view sv(k_opencl_navatala_sparse_mg_convergence_relative_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_sparse_mg_convergence_absolute_f64") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_sparse_mg_convergence_absolute_f64";
    std::string_view sv(k_opencl_navatala_sparse_mg_convergence_absolute_f64);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_sparse_mg_convergence_relative_f64") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_sparse_mg_convergence_relative_f64";
    std::string_view sv(k_opencl_navatala_sparse_mg_convergence_relative_f64);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_sparse_mg_compute_residual_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_sparse_mg_compute_residual_f32";
    std::string_view sv(k_opencl_navatala_sparse_mg_compute_residual_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_sparse_mg_compute_residual_f64") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_sparse_mg_compute_residual_f64";
    std::string_view sv(k_opencl_navatala_sparse_mg_compute_residual_f64);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_sparse_mg_weighted_restrict_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_sparse_mg_weighted_restrict_f32";
    std::string_view sv(k_opencl_navatala_sparse_mg_weighted_restrict_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_sparse_mg_weighted_restrict_f64") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_sparse_mg_weighted_restrict_f64";
    std::string_view sv(k_opencl_navatala_sparse_mg_weighted_restrict_f64);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_sparse_mg_error_norm_partials_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_sparse_mg_error_norm_partials_f32";
    std::string_view sv(k_opencl_navatala_sparse_mg_error_norm_partials_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_sparse_mg_error_norm_partials_f64") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_sparse_mg_error_norm_partials_f64";
    std::string_view sv(k_opencl_navatala_sparse_mg_error_norm_partials_f64);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_sparse_mg_zero_vector_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_sparse_mg_zero_vector_f32";
    std::string_view sv(k_opencl_navatala_sparse_mg_zero_vector_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_sparse_mg_zero_vector_f64") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_sparse_mg_zero_vector_f64";
    std::string_view sv(k_opencl_navatala_sparse_mg_zero_vector_f64);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_sparse_mg_prolongate_damped_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_sparse_mg_prolongate_damped_f32";
    std::string_view sv(k_opencl_navatala_sparse_mg_prolongate_damped_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_sparse_mg_prolongate_damped_f64") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_sparse_mg_prolongate_damped_f64";
    std::string_view sv(k_opencl_navatala_sparse_mg_prolongate_damped_f64);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  return false;
}

} // namespace NavatalaRegistry

