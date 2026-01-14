Advanced Computational Acoustic Analysis of Idiophone Bars: A Lightweight Finite Element Framework in Rust and WebAssembly
1. Introduction to Client-Side Computational Acoustics
The domain of musical instrument design, specifically the engineering of idiophones such as marimbas, xylophones, and vibraphones, has traditionally relied on the tacit knowledge of master craftsmen or, more recently, on heavy industrial finite element analysis (FEM) software packages running on high-performance workstations. These instruments produce sound through the vibration of the instrument body itself—most notably the "bar" or "key"—rather than through strings or air columns. The acoustic quality of these instruments is determined not merely by the fundamental pitch but by the precise harmonic alignment of overtones, a requirement that necessitates sub-millimeter geometric precision and a sophisticated understanding of elastodynamics.
The recent emergence of WebAssembly (WASM) as a compilation target has fundamentally altered the landscape of scientific computing, allowing high-fidelity numerical simulations to run near-natively within a web browser. This report outlines a comprehensive architectural and theoretical framework for implementing a lightweight, memory-efficient Finite Element Method module in the Rust programming language specifically for assessing idiophone bars. The objective is to transition from heavy, desktop-bound solvers to a nimble, accessible, yet rigorously accurate web-based tool capable of assisting luthiers and acoustic engineers in the fine-tuning of bar geometries.
To achieve this within the constrained memory environment of a browser (typically limited to 2GB or 4GB of addressable memory, with practical safety margins often lower), standard "brute force" engineering approaches—such as using dense tetrahedral meshes or storing full Krylov subspaces during eigenvalue iteration—must be abandoned. Instead, this framework prioritizes high-order hexahedral elements, algebraic mesh generation via Transfinite Interpolation (TFI), and memory-optimized iterative eigensolvers. This report details the physics of idiophone acoustics, the selection of appropriate continuum mechanics formulations, the evaluation of the current Rust scientific computing ecosystem, and the specific algorithmic implementations required to satisfy the dual constraints of high acoustic fidelity and low runtime memory usage.
2. Physics of Idiophone Acoustics and Frequency Analysis
To design a solver capable of assessing idiophone bars, one must first establish a rigorous definition of the physical phenomena being modeled. Unlike a string, which has a harmonic overtone series ($f, 2f, 3f, \dots$), a uniform rectangular bar exhibits inharmonic partials. The primary goal of idiophone tuning is to geometrically modify the bar—typically by carving an "undercut" arch on the underside—to force these inharmonic partials into a harmonic relationship.1
2.1 Classification of Vibration Modes
A robust FEM module must effectively capture and differentiate between three distinct classes of vibration modes: transverse (flexural), torsional, and longitudinal. The inability of simpler 1D beam models to accurately predict torsional and coupled modes necessitates the use of full 3D continuum mechanics.
2.1.1 Transverse (Flexural) Modes
The transverse modes involve bending in the plane of the bar's thickness (vertical plane) and are the primary source of sound radiation. The tuning of these modes defines the instrument's identity.
Fundamental Frequency ($f_1$): This corresponds to the pitch of the note (e.g., A4 = 440 Hz). The mode shape exhibits two nodes (points of zero displacement) located approximately 22.4% from each end of a uniform bar. However, as material is removed from the center of the bar (the undercut), the mass decreases and the stiffness decreases. Since the undercut is in a region of high bending moment for the fundamental, the stiffness reduction dominates, lowering the pitch.
Second Transverse Partial ($f_2$): For a uniform bar, the second bending mode occurs at approximately $2.76 \times f_1$, which is musically dissonant.
Marimba Tuning: The deep undercut of a marimba bar is designed to lower $f_1$ significantly while lowering $f_2$ less, or vice versa, to achieve a ratio of exactly 4:1 (two octaves). This harmonic alignment gives the marimba its dark, rich, and resonant timbre.2
Xylophone Tuning: The shallower undercut of a xylophone targets a ratio of 3:1 (an octave and a fifth). This produces the brighter, sharper sound characteristic of the instrument.3
Third Transverse Partial ($f_3$): In concert-grade instruments, the third partial is also tuned. For marimbas, the target is 10:1 (three octaves and a major third). For xylophones, it is 6:1. Achieving this requiring extremely precise shaping of the undercut flanks, often involving a "double arch" or cubic profile.4
The FEM module must be capable of calculating these frequencies with a precision of roughly 0.2% (approx. 3-5 cents) to be useful for high-level tuning.5
2.1.2 Torsional Modes
Torsional modes involve the twisting of the bar around its longitudinal axis. While these modes do not radiate sound efficiently due to acoustic short-circuiting (dipole cancellation), they act as energy sinks. If a torsional mode frequency coincides with a tuned transverse mode, it creates a "dead spot" where the energy from the mallet strike is rapidly transferred to the non-radiating twist, reducing sustain.1
Furthermore, untuned torsional modes can cause "beating" or roughness in the tone. 1D Timoshenko beam elements, even those with torsional degrees of freedom, often fail to predict these frequencies accurately because they cannot account for the complex warping of the non-prismatic cross-section created by the undercut.6 This failure of 1D approximations is a primary driver for the requirement of a 3D FEM formulation.
2.1.3 Lateral and Longitudinal Modes
Lateral Modes: These involve bending in the plane of the bar's width. Because the width is generally greater than the thickness, these modes occur at higher frequencies. However, in wide bars (bass notes), lateral modes can descend into the audible range and interfere with the spectrum.
Longitudinal Modes: These correspond to compression waves traveling the length of the bar. They are generally very high frequency ($f \propto \sqrt{E/\rho} / 2L$) and are rarely tuned, but they must be identified to ensure they do not coincide with a harmonic partial.
2.2 Material Behavior: Orthotropy in Tone Woods
The vast majority of professional idiophone bars are crafted from tropical hardwoods such as Honduras Rosewood (Dalbergia stevensonii) or Padauk (Pterocarpus soyauxii). These biological materials are not isotropic; their mechanical properties vary significantly relative to the grain direction.3
A lightweight FEM module must implement an Orthotropic Linear Elastic material model. The assumption of isotropy (single Young's Modulus $E$) is catastrophic for prediction accuracy, particularly for torsional modes which depend on shear moduli that are significantly softer than the longitudinal stiffness would suggest.
The constitutive relation relates the stress tensor $\boldsymbol{\sigma}$ to the strain tensor $\boldsymbol{\varepsilon}$ via the stiffness matrix $\mathbf{D}$ (often denoted $\mathbf{C}$ in physics contexts, but $\mathbf{D}$ in FEM to avoid confusion with Compliance). For an orthotropic material, this requires nine independent elastic constants:
Young's Moduli ($E_L, E_R, E_T$): Stiffness along the Longitudinal (grain), Radial (growth rings), and Tangential axes. For wood, $E_L \gg E_R \approx E_T$. Typically $E_L \approx 15-20$ GPa, while $E_R$ may be only 1-2 GPa.
Shear Moduli ($G_{LR}, G_{LT}, G_{RT}$): Stiffness resisting angular deformation. $G_{LT}$ is critical for the correct prediction of torsional frequencies.
Poisson's Ratios ($\nu_{LR}, \nu_{LT}, \nu_{RT}$): Ratios of transverse contraction to axial extension.
The solver must construct the material stiffness matrix $\mathbf{D}$ by inverting the Compliance Matrix $\mathbf{S}$. The compliance matrix is symmetric and defined as:

$$\mathbf{S} = \begin{bmatrix} \frac{1}{E_L} & -\frac{\nu_{TL}}{E_T} & -\frac{\nu_{RL}}{E_R} & 0 & 0 & 0 \\ -\frac{\nu_{LT}}{E_L} & \frac{1}{E_T} & -\frac{\nu_{RT}}{E_R} & 0 & 0 & 0 \\ -\frac{\nu_{LR}}{E_L} & -\frac{\nu_{TR}}{E_T} & \frac{1}{E_R} & 0 & 0 & 0 \\ 0 & 0 & 0 & \frac{1}{G_{RT}} & 0 & 0 \\ 0 & 0 & 0 & 0 & \frac{1}{G_{RL}} & 0 \\ 0 & 0 & 0 & 0 & 0 & \frac{1}{G_{TL}} \end{bmatrix}$$
Insight: In computational implementation, the inversion of $\mathbf{S}$ to obtain $\mathbf{D}$ should occur once per material definition, not per element. The resulting $6 \times 6$ stiffness matrix is then rotated into the global coordinate system if the grain direction does not align with the mesh axes (though for standard bars, they usually align).
2.3 Geometric Tuning Parameters and Mesh Implications
The geometric profile of the undercut is the primary variable in bar design.
Historical Profiles: Early acoustic models utilized simple parabolic arches ($z = ax^2 + b$).
Modern Profiles: To tune three partials simultaneously ($1:4:10$), modern manufacturers use complex shapes that may resemble cubic functions, double arches, or splines.4
The FEM module must therefore support arbitrary parametric definitions of the bottom surface. This requirement heavily influences the mesh generation strategy; a static library of pre-meshed bars is insufficient. The mesh must be generated procedurally at runtime to match the specific undercut curve defined by the user. This necessitates a meshing algorithm that is both fast and robust, pointing toward algebraic grid generation methods rather than unstructured Delaunay triangulation.
3. Mathematical Formulation of the Finite Element Method
To construct a solver from scratch or utilizing low-level libraries, one must define the weak form of the elastodynamic equations.
3.1 The Weak Form of Elastodynamics
The governing differential equation for the free vibration of a solid body $\Omega$ is:


$$\nabla \cdot \boldsymbol{\sigma} + \mathbf{f} = \rho \ddot{\mathbf{u}}$$

where $\mathbf{u}$ is displacement, $\rho$ is density, and body forces $\mathbf{f}$ are zero for free vibration.
Multiplying by an arbitrary test function $\mathbf{v}$ (virtual displacement) and integrating over the volume gives the weak form. Applying the divergence theorem and imposing boundary conditions leads to the standard discretized system:


$$\mathbf{M} \ddot{\mathbf{u}} + \mathbf{K} \mathbf{u} = \mathbf{0}$$
Assuming a harmonic solution $\mathbf{u}(t) = \boldsymbol{\phi} e^{i\omega t}$, we arrive at the Generalized Eigenvalue Problem:


$$(\mathbf{K} - \omega^2 \mathbf{M}) \boldsymbol{\phi} = \mathbf{0}$$

or


$$\mathbf{K} \boldsymbol{\phi} = \lambda \mathbf{M} \boldsymbol{\phi}$$

where $\lambda = \omega^2$ is the eigenvalue (squared circular frequency) and $\boldsymbol{\phi}$ is the eigenvector (mode shape).
3.2 Element Selection Strategies for Lightweight Accuracy
The choice of element type is the most critical decision for optimizing memory usage. We must balance the number of Degrees of Freedom (DOF) against the accuracy per DOF.
3.2.1 The Failure of Low-Order Elements (Tet4 and Hex8)
Linear Tetrahedra (Tet4): These elements exhibit "volumetric locking" (incompressibility issues) and "shear locking" (excessive stiffness in bending). To capture the bending of a marimba bar accurately, a Tet4 mesh would need hundreds of thousands of elements, creating a stiffness matrix $\mathbf{K}$ with millions of entries. This would likely exceed the 32-bit index limit and the memory constraints of WASM.9
Linear Hexahedra (Hex8): While better than Tet4, Hex8 elements famously suffer from shear locking. When a Hex8 element is subjected to a pure bending moment, the linear edges cannot curve, causing spurious shear strains to develop. This makes the element artificially stiff, causing the predicted frequencies to be significantly higher than reality.
Remedy: "Reduced Integration" can alleviate locking but introduces "hourglass modes" (zero-energy non-physical deformations) which require artificial stabilization parameters.10 This adds algorithmic complexity and potential inaccuracy if not tuned perfectly.
3.2.2 The Superiority of Quadratic Hexahedra (Hex20)
For the specific case of bending bars, the 20-node Quadratic Hexahedron (Hex20), often called the "Serendipity" element, is the optimal choice.11
Curved Edges: The quadratic shape functions allow the element edges to curve. This naturally models the bending behavior of the bar without parasitic shear strains, eliminating the need for reduced integration or stabilization.
Isoparametric Mapping: Hex20 elements can map to curved geometries (like the undercut of the bar) with high fidelity using fewer elements. A single Hex20 element can span the thickness of the bar, whereas multiple linear elements would be required.
DOF Efficiency: A standard marimba bar can be modeled with high precision using a coarse mesh of roughly $20 \times 4 \times 2$ Hex20 elements ($160$ elements total).
Node Count: $\approx 1,200$ nodes.
Total DOFs: $\approx 3,600$.
Matrix Size: A sparse matrix with 3,600 rows is trivial for modern hardware, consuming only a few megabytes of RAM. This is orders of magnitude smaller than an equivalent Tet mesh.
3.3 Mass Matrix Formulation: Consistent vs. Lumped
The mass matrix $\mathbf{M}$ represents the inertial properties of the system.


$$\mathbf{M}_e = \int_{\Omega_e} \rho \mathbf{N}^T \mathbf{N} \, d\Omega$$
Consistent Mass Matrix (CMM): Computed using the same shape functions $\mathbf{N}$ as the stiffness matrix. The resulting matrix is sparse but symmetric and non-diagonal. It accurately captures the rotational inertia coupled with translation, which is vital for the precise prediction of higher-order modes.13
Lumped Mass Matrix (LMM): Diagonalizes the mass by "lumping" it at the nodes. While LMMs are cheaper to store and invert (trivial inversion), they act as a low-pass filter and tend to underestimate natural frequencies. For high-order elements like Hex20, simple row-sum lumping can even produce negative masses, which is physically impossible and numerically unstable.11
Recommendation: For the high-precision requirements of instrument tuning, Consistent Mass Matrices must be used. The slight increase in storage (same sparsity pattern as $\mathbf{K}$) is justified by the gain in spectral accuracy.
4. Rust Ecosystem Analysis for Scientific Computing
The transition to a pure Rust implementation for WASM requires careful selection of libraries. The ecosystem is younger than C++ or Python, but specific crates have matured sufficiently to support this architecture.
4.1 Linear Algebra Backends
4.1.1 nalgebra and nalgebra-sparse
nalgebra is the general-purpose linear algebra library for Rust.
Pros: Excellent WASM support, wide adoption, good geometric types (Point3, Vector3).
Cons: The sparse matrix ecosystem within nalgebra is still developing. While nalgebra-sparse supports CSR/CSC formats and matrix-vector multiplication, it lacks robust direct solvers (like Cholesky or LU decomposition) for sparse matrices.15 It relies on iterative solvers (CG, BiCGStab), which can be slow or unstable for the shift-invert spectral transformations needed here.
4.1.2 faer (Recommended)
faer is a modern, high-performance linear algebra library written in pure Rust.
Pros: It implements Sparse Cholesky (LLT and LDLT) factorizations natively in Rust.16 This is the "killer feature" for this project. To solve the eigenvalue problem efficiently, we need to invert $(\mathbf{K} - \sigma \mathbf{M})$. faer allows us to compute the Cholesky factorization of this shifted matrix and solve linear systems efficiently.
Performance: Benchmarks indicate faer often outperforms nalgebra and rivals BLAS implementations due to aggressive SIMD usage and explicit parallelism.16
WASM: Being pure Rust, it compiles to WASM seamlessly.
4.2 Finite Element Assembly Libraries
4.2.1 fenris
fenris is a dedicated FEM library.18
Features: It provides high-level traits for assembly, quadrature rules, and integration with nalgebra. It supports Hex20 elements (referred to as Hex20) and arbitrary quadrature orders.
Risk: The documentation explicitly states the API is unstable and production usage is discouraged.19
Mitigation: For a robust module, one should not depend on fenris for the entire pipeline. Instead, use fenris for its quadrature rules and shape function definitions (which are mathematically static and unlikely to break), or "vendor" the specific element implementations needed into your own crate to ensure stability.
4.2.2 fem_2d
As the name implies, fem_2d 20 is restricted to two dimensions and is unsuitable for the 3D stress analysis required for torsional modes.
4.3 Eigensolvers
Finding eigenvalues of sparse matrices is non-trivial. Dense solvers ($O(N^3)$) are too slow.
4.3.1 lanczos Crate
The lanczos crate 21 implements the Hermitian Lanczos algorithm.
Function: It builds a Krylov subspace to approximate extremal eigenvalues.
Integration: It works with nalgebra-sparse.
Limitation: It typically finds the largest eigenvalues. To find frequencies (the smallest eigenvalues of $\mathbf{K}$), we must provide it with a linear operator that represents the inverse of the system, forcing it to find the largest eigenvalues of the inverse (which correspond to the smallest frequencies).
5. Architectural Implementation Strategy
5.1 Mesh Generation: Transfinite Interpolation (TFI)
To avoid the memory overhead of storing mesh files (VTK/OBJ) and the complexity of bundling an unstructured mesher (like TetGen) in WASM, this module should utilize Transfinite Interpolation (TFI).22
TFI allows the generation of a high-quality structured Hex20 mesh algebraically. The geometry of the bar is effectively a "deformed brick." We define the boundaries:
Top Surface ($w=1$): Flat ($z = H$).
Bottom Surface ($w=0$): Defined by the tuning curve $z = H - \text{undercut}(x)$.
Side Surfaces: Flat.
The TFI formula interpolates the internal nodes $(u, v, w)$ based on these boundaries.


$$\mathbf{x}(u,v,w) = (1-w)\mathbf{x}_{bottom}(u,v) + w \mathbf{x}_{top}(u,v)$$

For a bar where only the thickness varies along $x$:


$$z(u,v,w) = (1-w)(H - \text{depth}(u)) + wH$$

$$x(u,v,w) = u \cdot L$$

$$y(u,v,w) = v \cdot W$$
This generates the mesh on-the-fly using negligible memory. The mesh is defined entirely by the undercut parameters.
5.2 Sparse Matrix Assembly Pipeline
Sparsity Pattern Initialization: Since the mesh is structured (e.g., $N_x \times N_y \times N_z$ grid), the connectivity is deterministic. We can pre-calculate the exact sparsity pattern (CSR row pointers and column indices) before assembly. This prevents expensive dynamic resizing of vectors during the assembly loop, a crucial optimization for WASM.
Element Integration Loop:
Iterate over every element. For each element:
Get node coordinates from the TFI generator.
Compute the Jacobian $\mathbf{J}$ at each of the 27 Gauss points.
Compute the strain-displacement matrix $\mathbf{B}$.
Compute $\mathbf{K}_e = \sum \mathbf{B}^T \mathbf{D} \mathbf{B} \det(\mathbf{J}) w_i$.
Compute $\mathbf{M}_e = \sum \rho \mathbf{N}^T \mathbf{N} \det(\mathbf{J}) w_i$.
Add $\mathbf{K}_e$ and $\mathbf{M}_e$ into the global sparse matrices.
5.3 The Spectral Transformation Solver
We need to solve $\mathbf{K} \boldsymbol{\phi} = \lambda \mathbf{M} \boldsymbol{\phi}$ for the smallest $\lambda$.
Directly applying Lanczos to $\mathbf{K}$ finds the largest eigenvalues (highest frequencies), which are irrelevant. We must use the Shift-and-Invert strategy.
We define a shift $\sigma$ (e.g., $\sigma = -1.0$ to ensure the matrix is positive definite even if rigid body modes exist, or just near zero).
We define the operator $\mathbf{A} = (\mathbf{K} - \sigma \mathbf{M})^{-1} \mathbf{M}$.
The eigenvalues $\mu$ of $\mathbf{A}$ are related to the natural frequencies by $\mu = \frac{1}{\lambda - \sigma}$. The largest $\mu$ corresponds to the smallest $\lambda$.
The Algorithm:
Shift: Compute $\mathbf{K}_{\sigma} = \mathbf{K} - \sigma \mathbf{M}$.
Factorize: Compute the $LDL^T$ or Cholesky factorization of $\mathbf{K}_{\sigma}$ using faer. This is the most computationally expensive step but is done only once.
Lanczos Iteration:
Generate a vector $\mathbf{v}$.
Apply the operator: $\mathbf{y} = \mathbf{A}\mathbf{v}$. This involves:
Matrix-vector multiplication: $\mathbf{z} = \mathbf{M}\mathbf{v}$.
Linear Solve: Solve $\mathbf{K}_{\sigma} \mathbf{y} = \mathbf{z}$ using the cached factorization.
Orthogonalize and repeat.
5.4 Memory Optimization: The Two-Pass Lanczos
In a browser environment, memory is the bottleneck. The standard Lanczos algorithm stores the entire Krylov subspace basis (a set of vectors $\mathbf{V}_k = [\mathbf{v}_1, \dots, \mathbf{v}_m]$). If we request 50 modes for a large mesh, this basis can consume hundreds of megabytes.
The Two-Pass Lanczos 24 is a technique to drastically reduce this footprint.
Pass 1: Run the Lanczos iteration to generate the tridiagonal matrix $\mathbf{T}$, but discard the vectors $\mathbf{v}_i$ immediately after orthogonalization. We only need to keep the last two vectors in memory.
Compute the eigenvalues of the small matrix $\mathbf{T}$.
Pass 2: Once the eigenvalues are known, re-run the iteration from the same random seed. Regenerate the vectors $\mathbf{v}_i$ one by one and accumulate them to compute the eigenvectors (mode shapes) only for the specific modes of interest.
This technique trades computation time (doubling the matrix-vector products) for a massive reduction in memory, making it ideal for WASM deployment where CPU speed (via JIT) is often plentiful but memory is hard-capped.
6. Implementation Tables and Summaries
6.1 Comparison of Rust Linear Algebra Libraries for WASM
Library
Sparse Format
Direct Solvers (Cholesky/LU)
Pure Rust?
WASM Viability
Recommendation
nalgebra
CSR / CSC
No (Iterative only)
Yes
High
Use for small vectors/geometry types.
faer
CSR / CSC
Yes (LLT, LDLT, LU)
Yes
High
Primary solver engine.
sprs
CSR / CSC
Partial (limited)
Yes
Medium
Less performant than faer.
russell
CSR
Wraps C-libs (Mumps/Umfpack)
No
Low
Incompatible with WASM (requires C linking).

6.2 Recommended Element Formulation Data
Element Type
Shape Functions
Integration Rule
Locking Behavior
Memory Impact
Suitability
Hex8
Linear
$2 \times 2 \times 2$
Severe Shear Locking
Low
Poor (Inaccurate)
Hex8 (Reduced)
Linear
$1 \times 1 \times 1$
Hourglassing
Low
Poor (Unstable)
Tet10
Quadratic
4-point
Good
High (High Mesh Density)
Medium (High RAM usage)
Hex20
Quadratic
$3 \times 3 \times 3$
None
Optimal
Excellent

6.3 Wood Material Constants for FEM Input
Parameter
Symbol
Typical Value (Rosewood)
Description
Longitudinal Stiffness
$E_L$
18.5 GPa
Primary determinant of pitch ($f_1$).
Radial Stiffness
$E_R$
1.8 GPa
Influences cross-section warping.
Tangential Stiffness
$E_T$
1.2 GPa
Influences cross-section warping.
Shear Stiffness
$G_{LT}$
1.4 GPa
Critical for torsional modes.
Poisson Ratio
$\nu_{LT}$
0.35
Coupling between length and thickness.

7. Conclusions
The development of a WebAssembly-based FEM module for idiophone assessment is not only feasible but represents a significant advancement in the accessibility of acoustic engineering tools. By carefully selecting algorithms that align with the specific constraints of the browser environment—namely, memory scarcity and the inability to link against legacy Fortran/C libraries—a performant solution can be engineered in pure Rust.
The analysis dictates a clear departure from standard "black box" FEM approaches. The use of Transfinite Interpolation for mesh generation eliminates the file I/O bottleneck and ensures high-quality structured grids tailored to the specific "undercut" geometry of marimba and xylophone bars. The adoption of Hex20 (Quadratic Serendipity) elements provides the necessary continuum mechanics fidelity to capture torsional and coupled modes without the prohibitive memory cost of dense tetrahedral meshes.
Crucially, the linear algebra backbone must be built upon faer, which provides the sparse Cholesky factorization capability absent in other pure-Rust libraries. Coupling this with a Shift-Invert Two-Pass Lanczos solver creates a system that is robust, extremely lightweight, and capable of predicting eigenfrequencies with the sub-cent accuracy required for professional instrument tuning. This architecture delivers a desktop-class simulation experience within the lightweight footprint of a web application.
Works cited
Three-dimensional tuning of idiophone bar modes via finite element analysis - PubMed, accessed on January 13, 2026, https://pubmed.ncbi.nlm.nih.gov/34241415/
The Structure of the Marimba:There is craft to the design of the tone plates, too - Musical Instrument Guide - Yamaha Corporation, accessed on January 13, 2026, https://www.yamaha.com/en/musical_instrument_guide/marimba/mechanism/mechanism003.html
Basic physics of xylophone and marimba bars - ResearchGate, accessed on January 13, 2026, https://www.researchgate.net/publication/243492369_Basic_physics_of_xylophone_and_marimba_bars
Measurement-Based Comparison of Marimba Bar Modal Behaviour, accessed on January 13, 2026, https://pub.dega-akustik.de/ISMA2019/data/articles/000040.pdf
Tuning of idiophones using shape optimization on finite and boundary element 1D models, accessed on January 13, 2026, https://www.researchgate.net/publication/377884421_Tuning_of_idiophones_using_shape_optimization_on_finite_and_boundary_element_1D_models
GENERALIZED FINITE ELEMENT METHOD FOR VIBRATION ANALYSIS OF BARS - Blucher Proceedings, accessed on January 13, 2026, https://www.proceedings.blucher.com.br/article-details/generalized-finite-element-method-for-vibration-analysis-of-bars-9102
Developing the orthotropic linear-elastic model for wood applications using the FE method, accessed on January 13, 2026, https://pubs.rsc.org/en/content/articlehtml/2024/ma/d4ma00554f
Determining the Bar Shape - SuperMediocre, accessed on January 13, 2026, https://supermediocre.org/index.php/2016/01/10/computing-the-bar-shape/
automesh: Automatic mesh generation in Rust - Journal of Open Source Software, accessed on January 13, 2026, https://joss.theoj.org/papers/10.21105/joss.08768.pdf
Review of Solid Element Formulations in LS-DYNA, accessed on January 13, 2026, https://www.dynamore.de/de/download/papers/forum11/entwicklerforum-2011/erhart.pdf
PRECISE MID-NODE LUMPED MASS MATRICES FOR 3D 20-NODE HEXAHEDRAL AND 10-NODE TETRAHEDRAL FINITE ELEMENTS - 力学学报, accessed on January 13, 2026, https://lxxb.cstam.org.cn/en/article/cstr/32045.14.0459-1879-23-241
Extended Abstract+ Higher-order finite elements for lumped-mass explicit modeling of high- speed impacts, accessed on January 13, 2026, https://asmedigitalcollection.asme.org/hvis/proceedings-pdf/HVIS2019/883556/V001T02A009/6551025/v001t02a009-hvis2019-111.pdf
Analytical and numerical investigation of beam-spring systems with varying stiffness: a comparison of consistent and lumped mass matrices considerations - AIMS Press, accessed on January 13, 2026, https://www.aimspress.com/article/doi/10.3934/math.20241016?viewType=HTML
Finite Element Formulation - Computational Applied Mechanics, accessed on January 13, 2026, https://cam.uni-wuppertal.de/fileadmin/bauing/baumechanik/Dynamik_3/Vorlesungen/Lecture10-fe_formulation.pdf
nalgebra_sparse - Rust - Docs.rs, accessed on January 13, 2026, https://docs.rs/nalgebra-sparse
faer-sparse - crates.io: Rust Package Registry, accessed on January 13, 2026, https://crates.io/crates/faer-sparse
faer-cholesky — Rust math library // Lib.rs, accessed on January 13, 2026, https://lib.rs/crates/faer-cholesky
InteractiveComputerGraphics/fenris: A library for advanced finite element computations in Rust - GitHub, accessed on January 13, 2026, https://github.com/InteractiveComputerGraphics/fenris
fenris - Rust - Docs.rs, accessed on January 13, 2026, https://docs.rs/fenris/latest/fenris/
fem_2d - Rust - Docs.rs, accessed on January 13, 2026, https://docs.rs/fem_2d
lanczos - Rust - Docs.rs, accessed on January 13, 2026, https://docs.rs/lanczos
Transfinite interpolation - Wikipedia, accessed on January 13, 2026, https://en.wikipedia.org/wiki/Transfinite_interpolation
Unstructured and Semi-Structured Hexahedral Mesh Generation Methods - UPCommons, accessed on January 13, 2026, https://upcommons.upc.edu/bitstreams/b912c8c0-eeaf-4244-8301-89074613ae4b/download
A Cache Efficient, Low Memory, Lanczos Algorithm written in Rust - GitHub, accessed on January 13, 2026, https://github.com/lukefleed/two-pass-lanczos
