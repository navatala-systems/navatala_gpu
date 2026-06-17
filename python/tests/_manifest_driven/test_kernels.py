# SPDX-License-Identifier: Apache-2.0
import importlib

def test_kernels_exports_are_importable():
    mod = importlib.import_module("navatala_gpu.kernels")
