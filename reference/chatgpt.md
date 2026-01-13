For “bar/beam” idiophones you typically end up with a **symmetric generalized eigenproblem**:

[
K,\phi = \omega^2, M,\phi \quad\Rightarrow\quad f = \frac{\omega}{2\pi}
]

So the library choice is mostly about:

* **matrix types + assembly** (dense or sparse),
* **(generalized) eigen solvers** for the lowest few modes,
* **WASM friendliness** (pure Rust, no BLAS/LAPACK dependencies, controllable allocations).

Below are the Rust crates that fit that well.

---

## The most practical stack for WASM FEM modal analysis

### 1) Core linear algebra (dense) — pick one

**Option A: `nalgebra`** (very ergonomic, great for small/medium problems)

* All the pure-Rust decompositions work on `wasm32-unknown-unknown`. ([nalgebra.rs][1])
* Has `SymmetricEigen` for symmetric eigen-decomposition (useful when your global matrices are small enough to keep dense). ([docs.rs][2])

**Option B: `faer`** (performance-focused, explicit workspace control)

* General-purpose linear algebra in pure Rust. ([docs.rs][3])
* If you care about minimizing allocations, `faer`’s ecosystem tends to expose **workspace sizing** and lets you reuse buffers.

### 2) Sparse matrices (assembly) — pick one

You’ll likely want sparse for FEM once you have more than a few hundred DOFs.

**Option A: `nalgebra-sparse`** (integrates naturally with nalgebra)

* CSR/CSC/COO formats, designed to work with `nalgebra`. ([docs.rs][4])

**Option B: `sprs`** (standalone sparse matrices, widely used)

* Pure Rust sparse matrix structures (CSR/CSC, triplet/COO, etc.). ([GitHub][5])

**Option C: `faer-sparse`** (if you go with faer)

* Sparse data structures + sparse factorizations/ordering utilities. ([docs.rs][6])

### 3) Eigen solver for “lowest N modes” (what you want for frequencies)

For modal frequencies you almost never want a full eigen decomposition; you want the **smallest few eigenpairs**.

**Best-fit for WASM sparse modal work: Lanczos**

* `lanczos` crate: Lanczos for Hermitian/symmetric matrices; supports dense and sparse (via `nalgebra_sparse`). ([docs.rs][7])

If you stay dense/small:

* `nalgebra::linalg::SymmetricEigen` is the simplest path. ([docs.rs][2])
* `faer-evd` exists for eigenvalue decompositions (including Hermitian), and exposes workspace requirements. ([docs.rs][8])

### 4) Sparse linear solves (optional but often needed)

If you do generalized eigenproblems in the common “operator form” (e.g., repeatedly applying (M^{-1}K) or shift-invert), you’ll need to solve linear systems.

* If you’re on `nalgebra-sparse`, check `nalgebra-sparse-linalg` (iterative solvers like CG are typical for SPD mass/stiffness usage). ([Crates][9])
* If you’re on `faer-sparse`, you have sparse Cholesky options in that ecosystem. ([docs.rs][6])

---

## What I’d choose (given “lightweight memory + WASM”)

### If your bars are 1D Euler–Bernoulli/Timoshenko (common for idiophones)

You can keep it **very small** and fast:

* **Start**: `nalgebra` (dense) + `SymmetricEigen` ([nalgebra.rs][1])
* If DOFs grow: switch to `nalgebra-sparse` + `lanczos` ([docs.rs][4])

This is usually enough because a bar model with, say, 50–300 nodes is still manageable, and you only need the first ~5–20 modes.

### If you expect bigger systems or want tighter control of allocations

* `faer` + `faer-sparse` + (dense fallback via `faer-evd` where appropriate) ([docs.rs][3])

---

## Small but important WASM tips (crate choices that help)

Not FEM-specific, but these keep the module “WASM-friendly”:

* `wasm-bindgen` (interface) + `serde` + `serde-wasm-bindgen` (pass arrays/structs cleanly)
* Avoid `*-linalg` crates that wrap system BLAS/LAPACK (they won’t work on `wasm32-unknown-unknown` without extra toolchain complexity).
* Prefer **COO → CSR/CSC** assembly to avoid repeated reallocations (both `sprs` and `nalgebra-sparse` support COO-style building). ([GitHub][5])

---

## A concrete “crate shortlist”

Pick one row:

**Simple + ergonomic**

* `nalgebra`, `nalgebra-sparse`, `lanczos`, `serde`, `wasm-bindgen` ([nalgebra.rs][1])

**Sparse-heavy + performance/workspace control**

* `faer`, `faer-sparse`, (optionally `faer-evd` for dense eigen), `serde`, `wasm-bindgen` ([docs.rs][3])

**Alternate sparse layer**

* `nalgebra` + `sprs` (if you prefer `sprs`’ API), but note: if you want the `lanczos` crate exactly as-is, it’s oriented around `nalgebra_sparse`. ([docs.rs][7])

---

If you tell me roughly the scale (e.g., “~100 elements per bar” vs “a few thousand”) and whether you’re doing **Euler–Bernoulli** or **Timoshenko**, I can recommend a tighter combo (dense vs sparse, and whether Lanczos is worth it) and a memory-minimal data layout for WASM.

[1]: https://www.nalgebra.rs/docs/user_guide/wasm_and_embedded_targets/?utm_source=chatgpt.com "WebAssembly and embedded targets - nalgebra"
[2]: https://docs.rs/nalgebra/latest/nalgebra/linalg/struct.SymmetricEigen.html?utm_source=chatgpt.com "SymmetricEigen in nalgebra::linalg - Rust"
[3]: https://docs.rs/faer/latest/faer/?utm_source=chatgpt.com "faer - Rust"
[4]: https://docs.rs/nalgebra-sparse?utm_source=chatgpt.com "nalgebra_sparse - Rust"
[5]: https://github.com/sparsemat/sprs?utm_source=chatgpt.com "sparsemat/sprs: sparse linear algebra library for rust"
[6]: https://docs.rs/faer-sparse?utm_source=chatgpt.com "faer_sparse - Rust"
[7]: https://docs.rs/lanczos?utm_source=chatgpt.com "lanczos - Rust"
[8]: https://docs.rs/faer-evd?utm_source=chatgpt.com "faer_evd - Rust"
[9]: https://crates.io/crates/nalgebra-sparse-linalg/0.1.7?utm_source=chatgpt.com "nalgebra-sparse-linalg - crates.io: Rust Package Registry"



3D (solid) FEM for vibration modes is a very different scale than 1D/beam FEM: DOFs explode, so for WASM you’ll almost certainly want **sparse + iterative eigen** (or even **matrix-free**) rather than dense factorizations.

### Recommended “WASM-friendly” Rust stack (pure Rust)

**Linear algebra + sparse storage**

* `nalgebra` (vectors/matrices; works well on `wasm32-unknown-unknown`) ([Nalgebra][1])
* `nalgebra-sparse` (CSR/CSC sparse matrices) ([Docs.rs][2])
* `nalgebra-sparse-linalg` (iterative solvers + preconditioners modules) ([Docs.rs][3])

**Eigenmodes (lowest frequencies)**

* `lanczos` for symmetric/Hermitian eigenpairs on dense *or sparse* matrices, via `nalgebra_sparse`, and it can directly ask for the *smallest* eigenvalues. ([Docs.rs][4])

**Mesh import (strongly recommended for 3D in WASM)**

* Generate meshes *offline* (Gmsh, etc.), then load them in WASM with `mshio` (pure Rust parser for Gmsh MSH v4.1). ([Docs.rs][5])

**Why this combo:** it stays in the nalgebra ecosystem, avoids external BLAS/LAPACK (which is the usual WASM pain), and gives you an iterative eigen solver you can drive with sparse or operator-based matvecs. Nalgebra explicitly notes pure-Rust decompositions work on wasm, while `nalgebra-lapack` won’t. ([Nalgebra][1])

---

### Alternative stack if you want tighter control + performance

* `faer` + `faer-sparse` for dense/sparse with a lot of attention to performance and sparse Cholesky support in the ecosystem. ([Docs.rs][6])
  You’d still need an eigen approach (Lanczos/LOBPCG-style). If you don’t find a crate that fits exactly, it’s common to implement the *outer* eigensolver yourself and rely on `faer` for the inner linear algebra.

---

## FEM-specific crates: useful, but watch maturity / WASM fit

* `fenris` is a real FEM library (assembly/quadrature/mesh tooling), built on `nalgebra` + `nalgebra_sparse`, **but it explicitly warns the API is unstable and production use is discouraged**. ([Docs.rs][7])
  *Still* very useful as reference code or for bootstrapping your own element routines.

* Avoid stacks that depend on external native libraries if you truly need `wasm32-unknown-unknown`. For example, the `russell` ecosystem aims for max performance via non-Rust deps like OpenBLAS/MKL/SuiteSparse/MUMPS. ([GitHub][8])

* 3D tetra meshing inside WASM is usually not worth it. Wrappers like `tritet` call Tetgen (C/C++), which generally isn’t what you want in-browser. ([Docs.rs][9])
  Mesh offline → ship mesh → parse with `mshio` is the usual practical route.

---

## One key design choice for WASM memory: assembled sparse vs matrix-free

If you assemble global `K` and `M` in CSR/CSC, memory will be dominated by the sparse storage.

If you go **matrix-free**, you store only:

* node coordinates,
* element connectivity,
* materials,
  and implement `y = Kx` and `y = Mx` by looping elements. Then an iterative eigensolver (Lanczos) only needs matvecs, which can be dramatically lighter in memory.

`lanczos` is compatible with this style if you wrap your operator to provide matvecs (and/or build a sparse matrix when small enough). ([Docs.rs][4])

---

### Practical recommendation

For a first working 3D modal solver in WASM:

**`nalgebra` + `nalgebra-sparse` + `nalgebra-sparse-linalg` + `lanczos` + `mshio`** ([Nalgebra][1])

…and strongly consider **matrix-free K/M application** if you expect anything beyond “toy” 3D meshes.

If you tell me roughly your target mesh size (e.g. ~20k DOFs vs ~500k DOFs) and element type (tet4 / tet10 / hex8), I’ll suggest the best solver strategy (plain Lanczos vs shift-invert, lumped vs consistent mass) that stays WASM-feasible.

[1]: https://www.nalgebra.rs/docs/user_guide/wasm_and_embedded_targets/ "WebAssembly and embedded targets | nalgebra"
[2]: https://docs.rs/nalgebra-sparse?utm_source=chatgpt.com "nalgebra_sparse - Rust"
[3]: https://docs.rs/nalgebra-sparse-linalg?utm_source=chatgpt.com "nalgebra_sparse_linalg - Rust"
[4]: https://docs.rs/lanczos "lanczos - Rust"
[5]: https://docs.rs/mshio "mshio - Rust"
[6]: https://docs.rs/faer/latest/faer/?utm_source=chatgpt.com "faer - Rust"
[7]: https://docs.rs/fenris "fenris - Rust"
[8]: https://github.com/cpmech/russell?utm_source=chatgpt.com "cpmech/russell: Rust Scientific Libary. ODE and DAE ..."
[9]: https://docs.rs/tritet "tritet - Rust"


At **5,000–10,000 DOF** you’re in a sweet spot where **assembled sparse** FEM is still very doable in WASM, *as long as you avoid dense eigensolvers and native BLAS/LAPACK*.

## Recommended Rust crates (WASM-friendly)

### Core math + sparse assembly (good default)

* **`nalgebra`** for vectors / small dense blocks (pure-Rust works on `wasm32-unknown-unknown`). ([Nalgebra][1])
* **`nalgebra-sparse`** for COO→CSR/CSC assembly and sparse matvecs. ([Docs.rs][2])
* **`lanczos`** to compute the *lowest* few eigenpairs (supports sparse via `nalgebra_sparse`, and has `Order::Smallest`). ([Docs.rs][3])

This combo is typically enough to get the first 10–50 modes of a 10k-DOF model.

### If you need shift-invert / direct sparse solves

`nalgebra-sparse` explicitly notes **limited/no sparse system solvers**. ([Docs.rs][2])
If you want shift-invert (often the fastest/most robust for smallest modes), use:

* **`faer-sparse`** (includes sparse Cholesky variants like LLT/LDLT/Bunch–Kaufman). ([Docs.rs][4])

You can still keep the rest of your code mostly the same conceptually; it’s just a different sparse backend.

## What to avoid for WASM

* Anything that relies on external BLAS/LAPACK at runtime. For example, `nalgebra-lapack` won’t compile for wasm because it binds to BLAS/LAPACK. ([Nalgebra][1])

## A practical approach for 3D modal frequencies at this DOF count

For linear elasticity you want:
[
K\phi = \lambda M\phi,\quad \lambda=\omega^2,\quad f=\frac{\sqrt{\lambda}}{2\pi}
]

To keep memory/complexity down in WASM, a common trick is:

* Use **lumped mass** (M \approx \mathrm{diag}(m)) (store as a single `Vec<f64>`), then solve a *standard* symmetric eigenproblem via scaling:
  [
  B = D^{-1/2} K D^{-1/2}
  ]
  and run Lanczos on matvecs:
  [
  y = D^{-1/2}\big(K(D^{-1/2}x)\big)
  ]
  This lets you store **only** sparse **K** + a **mass diagonal**, and still use `lanczos` (`Order::Smallest`). ([Docs.rs][3])

### One 3D-specific gotcha

If your bar is **free-free**, you’ll get **6 rigid-body modes ~0 Hz**. Lanczos with “smallest” will find those first; just skip them when reporting audible modes.

## Rough memory expectations (why sparse is fine here)

At 10k DOF, a typical 3D elasticity stiffness matrix might have on the order of **~0.5–1.5 million nonzeros** (depends on element type/connectivity). CSR storage (values + indices) usually lands in the **single-digit to low–tens of MB** per matrix in WASM. Keeping mass as a diagonal vector helps a lot.

---

### If you want one concrete “starter” choice

Use **`nalgebra` + `nalgebra-sparse` + `lanczos`**, with **lumped mass + scaled operator**. It’s the simplest route that stays WASM-native and fits 5k–10k DOF well. ([Nalgebra][1])

If you tell me your element type (tet4/tet10/hex8) and whether you’ll lump the mass or want consistent mass, I’ll point you to the best solver path (plain Lanczos vs shift-invert with `faer-sparse`) for that choice.

[1]: https://www.nalgebra.rs/docs/user_guide/wasm_and_embedded_targets/ "WebAssembly and embedded targets | nalgebra"
[2]: https://docs.rs/nalgebra-sparse "nalgebra_sparse - Rust"
[3]: https://docs.rs/lanczos "lanczos - Rust"
[4]: https://docs.rs/faer-sparse "faer_sparse - Rust"
