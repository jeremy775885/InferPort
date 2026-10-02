# Changelog

## 0.2.0 — Unreleased

- Accept codec-supported NumPy scalar backend outputs in typed contracts, preserving
  boolean/numeric separation, finite/range checks and the original backend payload.
- Add `Backend.describe()` / `Client.describe()` and portable typed input, output
  and reset-context contracts, with fixed/bounded arrays and directional compatibility checks.
- Require every backend to implement `describe() -> InferenceSpec`; remove untyped operation.
  Validate requests before backend state changes and outputs before transmission.
  Snapshot the contract on the backend worker per connection, before READY.
- Fetch and cache the contract during `Client.connect()` under the opening deadline;
  `Client.describe()` reads that cache and `infer()` validates received outputs.
  Invalid remote contracts/results close the connection with `ProtocolError`.
- Add the model-independent `robotics.joint-targets.v1` profile for RGB camera roles,
  ordered state/action channels, units, absolute targets and declared action horizons.
  Application diagnostics and seed options are declared by adapters, outside the base profile.
  No environment base class, action scheduling or conversions are added.
- Use the single unversioned `inferport` WebSocket subprotocol. Both endpoints must
  use this SDK, and existing backends must implement `describe()`; no legacy
  compatibility layer is provided. READY/reset remain empty acknowledgments.
- Package version 0.2.0 covers the contract API changes. The communication identifier
  remains `inferport`; the package version does not introduce a protocol suffix.

## 0.1.0 — 2026-09-26

- Add model-independent `Backend`, `Client`, and `serve` APIs for policy, value,
  reward, and other inference adapters.
- Exchange binary MessagePack messages and numeric NumPy arrays over WebSocket
  using the `inferport.v1` protocol.
- Define connection ownership, reset and cleanup behavior, bounded network waits,
  error responses, optional bearer authentication, and TLS support.
- Support Python 3.10+ and NumPy >=1.21.3,<3 with three runtime dependencies.
- Include examples, protocol documentation, compatibility tests, and benchmarks.

Real model and robot integrations remain in their owning repositories. See
[validation](docs/validation.md) for the tested scope and remaining gaps.
