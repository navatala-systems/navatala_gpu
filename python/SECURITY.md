<!-- SPDX-License-Identifier: Apache-2.0 -->

# Security Policy For navatala_gpu

## Supported Scope

This generated SDK exposes Python bindings for the public Navatala GPU C ABI.
Security fixes are supported for the generated Python package, generated C++ binding scaffold, and public runtime ABI surface shipped with the SDK.

The private formal specifications and generator implementation are not part of this generated SDK package.

## Reporting A Vulnerability

Do not disclose suspected vulnerabilities in a public issue before maintainers have had a chance to triage them.
Use the repository's private vulnerability reporting channel when enabled, or contact the maintainers through the project security contact configured for the public release.

Please include:

- affected package version or manifest id
- backend and driver/runtime version, if relevant
- minimal reproducer
- expected and observed behavior

## Generated Code Boundary

Generated files are marked with a manifest id. Security fixes that require generator changes should be made in the generator and regenerated, not patched only in generated output.
