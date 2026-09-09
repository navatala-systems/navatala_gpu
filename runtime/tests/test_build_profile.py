#!/usr/bin/env python3
"""Configure-contract controls; no toolkit, generation or runtime build required."""
import pathlib
import subprocess
import tempfile
import unittest

class BuildProfileTests(unittest.TestCase):
    def check_profile(self, changes=(), error=None):
        module = pathlib.Path(__file__).resolve().parents[1] / "cmake/GpuRuntimeProfile.cmake"
        settings = {
            "GPU_RUNTIME_PROFILE": "SEALED_EXACT",
            "GPU_RUNTIME_NAVATALA_FFI_STUB": "OFF",
            "GPU_RUNTIME_NAVATALA_WRAPPER": "ON",
            "GPU_RUNTIME_ENABLE_NVRTC": "OFF",
            "GPU_RUNTIME_REQUIRED_BACKENDS": "CUDA",
            "GPU_RUNTIME_USE_CUDA": "ON",
            "GPU_RUNTIME_HAVE_CUDA": "1",
        }
        settings.update(changes)
        script = "cmake_minimum_required(VERSION 3.18)\n"
        script += "\n".join(f'set({key} "{value}" CACHE STRING "" FORCE)'
                            for key, value in settings.items())
        script += f'\ninclude("{module.as_posix()}")\ngpu_runtime_validate_backend_profile()\n'
        script += 'message(STATUS "compiled=${GPU_RUNTIME_COMPILED_BACKENDS}")\n'
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "control.cmake"
            path.write_text(script)
            result = subprocess.run(["cmake", "-P", str(path)], capture_output=True, text=True)
        if error:
            self.assertNotEqual(result.returncode, 0)
            self.assertIn(error, result.stdout + result.stderr)
        else:
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("compiled=CUDA", result.stdout)

    def test_exact_backend(self):
        self.check_profile()

    def test_rejections(self):
        cases = [
            ({"GPU_RUNTIME_PROFILE": "UNKNOWN"}, "Unknown GPU_RUNTIME_PROFILE"),
            ({"GPU_RUNTIME_NAVATALA_FFI_STUB": "ON"}, "requires the real"),
            ({"GPU_RUNTIME_NAVATALA_WRAPPER": "OFF"}, "requires the real"),
            ({"GPU_RUNTIME_ENABLE_NVRTC": "ON"}, "NVRTC=OFF"),
            ({"GPU_RUNTIME_REQUIRED_BACKENDS": ""}, "explicit nonempty"),
            ({"GPU_RUNTIME_REQUIRED_BACKENDS": "AUTO"}, "Unknown required backend"),
            ({"GPU_RUNTIME_REQUIRED_BACKENDS": "CUDA;CUDA"}, "Duplicate"),
            ({"GPU_RUNTIME_HAVE_CUDA": "0"}, "differs from compiled"),
            ({"GPU_RUNTIME_USE_HIP": "ON"}, "Unrequested backend"),
            ({"GPU_RUNTIME_HAVE_HIP": "1"}, "differs from compiled"),
        ]
        for changes, error in cases:
            with self.subTest(changes=changes):
                self.check_profile(changes, error)

    def test_general_preserves_explicit_jit(self):
        self.check_profile({"GPU_RUNTIME_PROFILE": "GENERAL", "GPU_RUNTIME_ENABLE_NVRTC": "ON"})

if __name__ == "__main__":
    unittest.main()
