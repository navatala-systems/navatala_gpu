# SPDX-License-Identifier: Apache-2.0
project = "navatala_gpu"
copyright = "2026, Navatala"
author = "Navatala"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
]

templates_path = ["_templates"]
exclude_patterns = []
html_theme = "alabaster"

autodoc_member_order = "bysource"
autodoc_typehints = "description"
