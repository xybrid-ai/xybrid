/* Umbrella header for bindgen.
 *
 * mlx-c (pinned in vendor/mlx-apple/UPSTREAM_VERSIONS.txt) ships
 * `mlx/c/mlx.h` which transitively includes every public
 * sub-header (array, stream, device, ops, fast, fft, linalg, metal, memory,
 * random, transforms, io, compile, distributed, …). A single include here is
 * enough to produce the full Rust FFI surface.
 *
 * Do not add the MLX C++ headers (`mlx/mlx.h`, `mlx/array.h`, …) — they are
 * C++ only and would require bindgen's experimental C++ support. The mlx-c
 * shim exists precisely to avoid that.
 */
#include <mlx/c/mlx.h>
