// SPDX-License-Identifier: Apache-2.0
// Kernel registry partition for the `metal_samples` bucket. Compiles
// against the declarations in `navatala/_registry.hpp` and links
// against the trampoline in `_registry.cpp`. Partitioning keeps each
// translation unit well under the 50K-line threshold so a typical
// build can parallelise across the buckets.
#include "navatala/_registry.hpp"
#include <cstdint>
#include <string>
#include <string_view>

const char* k_metal_navatala_samples_float32_add = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_samples_float32_add(device const float* a [[buffer(0)]], device const float* b [[buffer(1)]], device float* c [[buffer(2)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  c[int(__gid.x)] = (a[int(__gid.x)] + b[int(__gid.x)]);
}

)kernel";
const char* k_metal_navatala_samples_triangle_normals2 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_samples_triangle_normals2(device const float* vertices [[buffer(0)]], device const int* faces [[buffer(1)]], device float* normals [[buffer(2)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  int i0 = faces[((int(__gid.x))*3 + 0)];
  int i1 = faces[((int(__gid.x))*3 + 1)];
  int i2 = faces[((int(__gid.x))*3 + 2)];
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
  normals[((int(__gid.x))*3 + 0)] = nx;
  normals[((int(__gid.x))*3 + 1)] = ny;
  normals[((int(__gid.x))*3 + 2)] = nz;
}

)kernel";
const char* k_metal_navatala_samples_axpy_fallback = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_samples_axpy_fallback(device const float* x [[buffer(0)]], device const float* y [[buffer(1)]], device const float* alpha [[buffer(2)]], device float* _out [[buffer(3)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  _out[int(__gid.x)] = (y[int(__gid.x)] + (x[int(__gid.x)] * alpha[0]));
}

)kernel";

namespace NavatalaRegistry {

const KernelArgumentInfo kAbiArgs_metal_navatala_samples_float32_add[] = {
  { "a", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "b", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "c", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_samples_float32_add = {
  1,
  "navatala_samples_float32_add",
  "metal",
  "navatala_samples_float32_add",
  "kernel:metal:navatala_samples_float32_add",
  "abi-r1:metal:navatala_samples_float32_add",
  "abi-r1:metal:navatala_samples_float32_add",
  3,
  kAbiArgs_metal_navatala_samples_float32_add
};

const KernelArgumentInfo kAbiArgs_metal_navatala_samples_triangle_normals2[] = {
  { "vertices", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "faces", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "normals", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_samples_triangle_normals2 = {
  1,
  "navatala_samples_triangle_normals2",
  "metal",
  "navatala_samples_triangle_normals2",
  "kernel:metal:navatala_samples_triangle_normals2",
  "abi-r1:metal:navatala_samples_triangle_normals2",
  "abi-r1:metal:navatala_samples_triangle_normals2",
  3,
  kAbiArgs_metal_navatala_samples_triangle_normals2
};

const KernelArgumentInfo kAbiArgs_metal_navatala_samples_axpy_fallback[] = {
  { "x", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "y", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "alpha", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "out", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_samples_axpy_fallback = {
  1,
  "navatala_samples_axpy_fallback",
  "metal",
  "navatala_samples_axpy_fallback",
  "kernel:metal:navatala_samples_axpy_fallback",
  "abi-r1:metal:navatala_samples_axpy_fallback",
  "abi-r1:metal:navatala_samples_axpy_fallback",
  4,
  kAbiArgs_metal_navatala_samples_axpy_fallback
};

bool tryGetKernelAbiManifest_metal_samples(const std::string& backend, const std::string& kernelName, const KernelAbiManifestInfo*& out) {
  if (backend == "metal" && kernelName == "navatala_samples_float32_add") {
    out = &kAbiManifest_metal_navatala_samples_float32_add;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_samples_triangle_normals2") {
    out = &kAbiManifest_metal_navatala_samples_triangle_normals2;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_samples_axpy_fallback") {
    out = &kAbiManifest_metal_navatala_samples_axpy_fallback;
    return true;
  }
  out = nullptr;
  return false;
}

bool tryGetKernelSource_metal_samples(const std::string& backend, const std::string& kernelName, GpuRuntime::ProgramSource& out) {
  if (backend == "metal" && kernelName == "navatala_samples_float32_add") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_samples_float32_add";
    std::string_view sv(k_metal_navatala_samples_float32_add);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_samples_triangle_normals2") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_samples_triangle_normals2";
    std::string_view sv(k_metal_navatala_samples_triangle_normals2);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_samples_axpy_fallback") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_samples_axpy_fallback";
    std::string_view sv(k_metal_navatala_samples_axpy_fallback);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  return false;
}

} // namespace NavatalaRegistry

