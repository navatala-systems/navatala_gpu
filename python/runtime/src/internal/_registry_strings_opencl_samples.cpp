// SPDX-License-Identifier: Apache-2.0
// Kernel registry partition for the `opencl_samples` bucket. Compiles
// against the declarations in `navatala/_registry.hpp` and links
// against the trampoline in `_registry.cpp`. Partitioning keeps each
// translation unit well under the 50K-line threshold so a typical
// build can parallelise across the buckets.
#include "navatala/_registry.hpp"
#include <cstdint>
#include <string>
#include <string_view>

const char* k_opencl_navatala_samples_float32_add = R"kernel(
__kernel void navatala_samples_float32_add(__global const float* a, __global const float* b, __global float* c) {
  int gid0 = (int)get_global_id(0);
  c[(int)(get_global_id(0))] = (a[(int)(get_global_id(0))] + b[(int)(get_global_id(0))]);
}

)kernel";
const char* k_opencl_navatala_samples_triangle_normals2 = R"kernel(
__kernel void navatala_samples_triangle_normals2(__global const float* vertices, __global const int* faces, __global float* normals) {
  int gid0 = (int)get_global_id(0);
  int i0 = faces[(((int)(get_global_id(0)))*3 + 0)];
  int i1 = faces[(((int)(get_global_id(0)))*3 + 1)];
  int i2 = faces[(((int)(get_global_id(0)))*3 + 2)];
  float p0x = vertices[((i0)*3 + 0)];
  float p0y = vertices[((i0)*3 + 1)];
  float p0z = vertices[((i0)*3 + 2)];
  float p1x = vertices[((i1)*3 + 0)];
  float p1y = vertices[((i1)*3 + 1)];
  float p1z = vertices[((i1)*3 + 2)];
  float p2x = vertices[((i2)*3 + 0)];
  float p2y = vertices[((i2)*3 + 1)];
  float p2z = vertices[((i2)*3 + 2)];
  float e1x = (p1x - p0x);
  float e1y = (p1y - p0y);
  float e1z = (p1z - p0z);
  float e2x = (p2x - p0x);
  float e2y = (p2y - p0y);
  float e2z = (p2z - p0z);
  float nx = ((e1y * e2z) - (e1z * e2y));
  float ny = ((e1z * e2x) - (e1x * e2z));
  float nz = ((e1x * e2y) - (e1y * e2x));
  normals[(((int)(get_global_id(0)))*3 + 0)] = nx;
  normals[(((int)(get_global_id(0)))*3 + 1)] = ny;
  normals[(((int)(get_global_id(0)))*3 + 2)] = nz;
}

)kernel";
const char* k_opencl_navatala_samples_axpy_fallback = R"kernel(
__kernel void navatala_samples_axpy_fallback(__global const float* x, __global const float* y, __global const float* alpha, __global float* _out) {
  int gid0 = (int)get_global_id(0);
  _out[(int)(get_global_id(0))] = (y[(int)(get_global_id(0))] + (x[(int)(get_global_id(0))] * alpha[0]));
}

)kernel";

namespace NavatalaRegistry {

const KernelArgumentInfo kAbiArgs_opencl_navatala_samples_float32_add[] = {
  { "a", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "b", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "c", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_samples_float32_add = {
  1,
  "navatala_samples_float32_add",
  "opencl",
  "navatala_samples_float32_add",
  "kernel:opencl:navatala_samples_float32_add",
  "abi-r1:opencl:navatala_samples_float32_add",
  "abi-r1:opencl:navatala_samples_float32_add",
  3,
  kAbiArgs_opencl_navatala_samples_float32_add
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_samples_triangle_normals2[] = {
  { "vertices", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "faces", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "normals", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_samples_triangle_normals2 = {
  1,
  "navatala_samples_triangle_normals2",
  "opencl",
  "navatala_samples_triangle_normals2",
  "kernel:opencl:navatala_samples_triangle_normals2",
  "abi-r1:opencl:navatala_samples_triangle_normals2",
  "abi-r1:opencl:navatala_samples_triangle_normals2",
  3,
  kAbiArgs_opencl_navatala_samples_triangle_normals2
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_samples_axpy_fallback[] = {
  { "x", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "y", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "alpha", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "out", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_samples_axpy_fallback = {
  1,
  "navatala_samples_axpy_fallback",
  "opencl",
  "navatala_samples_axpy_fallback",
  "kernel:opencl:navatala_samples_axpy_fallback",
  "abi-r1:opencl:navatala_samples_axpy_fallback",
  "abi-r1:opencl:navatala_samples_axpy_fallback",
  4,
  kAbiArgs_opencl_navatala_samples_axpy_fallback
};

bool tryGetKernelAbiManifest_opencl_samples(const std::string& backend, const std::string& kernelName, const KernelAbiManifestInfo*& out) {
  if (backend == "opencl" && kernelName == "navatala_samples_float32_add") {
    out = &kAbiManifest_opencl_navatala_samples_float32_add;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_samples_triangle_normals2") {
    out = &kAbiManifest_opencl_navatala_samples_triangle_normals2;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_samples_axpy_fallback") {
    out = &kAbiManifest_opencl_navatala_samples_axpy_fallback;
    return true;
  }
  out = nullptr;
  return false;
}

bool tryGetKernelSource_opencl_samples(const std::string& backend, const std::string& kernelName, GpuRuntime::ProgramSource& out) {
  if (backend == "opencl" && kernelName == "navatala_samples_float32_add") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_samples_float32_add";
    std::string_view sv(k_opencl_navatala_samples_float32_add);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_samples_triangle_normals2") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_samples_triangle_normals2";
    std::string_view sv(k_opencl_navatala_samples_triangle_normals2);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_samples_axpy_fallback") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_samples_axpy_fallback";
    std::string_view sv(k_opencl_navatala_samples_axpy_fallback);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  return false;
}

} // namespace NavatalaRegistry

