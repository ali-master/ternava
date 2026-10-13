# Security policy

`ternava` is in pre-alpha. The project is small enough that we don't yet
run a formal disclosure program, but we take security reports seriously.

## Reporting a vulnerability

If you find a security issue, **please do not open a public GitHub issue**.

Contact the [Ternava maintainer](https://github.com/ali-master) privately, or use
GitHub private vulnerability reporting when enabled, with:

- A description of the issue and its impact
- Steps to reproduce (or a proof-of-concept)
- Affected version, commit, or release

The maintainer aims to acknowledge reports and to coordinate a fix and disclosure timeline with you before publishing details.

## Scope

In-scope:

- The compiled WASM engine under `engine/`
- The `@usex/ternava` npm package under `packages/mini/`
- The training and packing pipeline under `training/`

Out-of-scope:

- The demo sites under `examples/` (deployed for illustration only)
- Vulnerabilities in third-party dependencies — please report those upstream

## Non-security bugs

For correctness, performance, or quality issues that aren't security-relevant,
please [open a GitHub issue](https://github.com/ali-master/ternava/issues).
