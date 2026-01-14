FEM3D in Rust for WASM
======================

A lightweight 3D finite element modal solver written in Rust, compiled to WASM. It uses the recommended stack of **nalgebra + nalgebra-sparse + lanczos** with a lumped mass + scaled operator for memory efficiency.

## What it does

- Builds a small 3D tetrahedral mesh (cube, base fixed) and assembles lumped-mass/stiffness matrices.
- Runs Lanczos to extract the lowest vibration modes.
- Exposes WASM bindings via `wasm-bindgen` and a browser demo.

## Prerequisites

- Rust toolchain
- `wasm-pack` for building the web demo (`cargo install wasm-pack` if you don't have it)

## Native build & test

```bash
cargo test
```

## Build WASM + demo (for GitHub Pages)

```bash
wasm-pack build --release --target web --out-dir docs/pkg
```

Then open `docs/index.html` locally (or serve `docs/`) to run the demo. To publish on GitHub Pages, point Pages to the `/docs` folder after running the build so that the generated `docs/pkg` assets are present.

## API surface (WASM)

- `compute_demo_modes()` → returns `{ frequencies_hz: number[] }` for the built-in cube demo.
- `compute_modes(node_positions, tets, fixed_nodes, young_modulus, poisson_ratio, density, modes)` → solve custom meshes (positions flattened `[x0,y0,z0,...]`, `tets` as `[i0,i1,i2,i3,...]`, `fixed_nodes` is a list of node indices to fully fix).

## Reference notes

See `reference/` for the background on the chosen stack and WASM considerations.
