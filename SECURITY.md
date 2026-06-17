# Security Policy

`navatala_gpu` is currently a developer-preview GPU runtime and kernel corpus.
Security reports are welcome even during the alpha period.

## Supported Versions

| Version | Supported |
| --- | --- |
| `0.1.x` developer preview | Security fixes are handled on a best-effort basis before the first stable release. |

## Reporting A Vulnerability

Please report suspected vulnerabilities privately by email to:

`security@navatalasystems.com`

Include, where possible:

- affected version, commit, or release artifact;
- operating system, GPU backend, driver/runtime versions, and Python version;
- a minimal reproducer, crash log, sanitizer output, or malformed input;
- whether the issue is remotely triggerable or requires local execution.

Do not open a public GitHub issue for a suspected vulnerability until we have
confirmed the impact and coordinated disclosure.

## Scope

In scope:

- memory-safety bugs in the public C++ runtime;
- Python binding crashes or reference/lifetime bugs;
- malformed kernel/manifest inputs that trigger unsafe behavior;
- build or packaging issues that could execute untrusted code unexpectedly.

Out of scope for private security reporting:

- general correctness bugs without a security impact;
- performance regressions;
- unsupported driver/runtime combinations already documented in the release notes.

## Disclosure Process

We aim to acknowledge reports within 5 business days. For confirmed issues, we
will coordinate a fix, publish an advisory or release note when appropriate, and
credit reporters unless they request otherwise.
