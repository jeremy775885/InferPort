# Changelog

## 0.1.0

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
