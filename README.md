# RAIF

RAIF (Runtime for AI Inference) is a lightweight runtime designed to
accelerate AI model inference across different hardware targets. The
goal of this project is to provide a collection of optimized kernels
and a simple runtime that selects the best implementation based on the
underlying processor features (e.g. AVX2, SSE, etc.).

## Directory Structure

- `src/` - C++ implementation of the runtime and operators.
- `include/` - Public headers.
- `test/` - Unit tests and examples built with CMake.
- `examples/` - Sample applications (empty for now).
- `docs/` - Documentation (design notes, guides).
- `scripts/` - Helper scripts (build, tooling, etc.).

## Building

This project uses CMake:

```bash
mkdir build && cd build
cmake ..
make
ctest
```

`ctest` will build and run the sample test located in `test/`.

## Status

This repository currently contains minimal sample code to demonstrate
how the runtime might be organized. Real operator implementations and
additional hardware support will be added in the future.
