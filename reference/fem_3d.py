"""
3D Finite Element Analysis for Bar Vibration

Implements 3D solid element (8-node hexahedral) formulation for
more accurate frequency analysis of undercut bars.

This provides higher accuracy than the 2D Timoshenko beam model,
especially for complex undercut geometries and wide bars where
3D effects become significant.
"""

from typing import List, Tuple, Optional
import numpy as np
from scipy import linalg
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import eigsh
import math


def gauss_points_3d() -> Tuple[np.ndarray, np.ndarray]:
    """
    Return 2x2x2 Gauss quadrature points and weights for hexahedral element.

    Returns:
        Tuple of (points, weights) where points is (8, 3) and weights is (8,)
    """
    g = 1.0 / math.sqrt(3.0)
    points = np.array([
        [-g, -g, -g],
        [+g, -g, -g],
        [+g, +g, -g],
        [-g, +g, -g],
        [-g, -g, +g],
        [+g, -g, +g],
        [+g, +g, +g],
        [-g, +g, +g],
    ])
    weights = np.ones(8)
    return points, weights


def shape_functions_hex8(xi: float, eta: float, zeta: float) -> np.ndarray:
    """
    Compute shape functions for 8-node hexahedral element.

    Node numbering (natural coordinates):
        Node 0: (-1, -1, -1)
        Node 1: (+1, -1, -1)
        Node 2: (+1, +1, -1)
        Node 3: (-1, +1, -1)
        Node 4: (-1, -1, +1)
        Node 5: (+1, -1, +1)
        Node 6: (+1, +1, +1)
        Node 7: (-1, +1, +1)

    Args:
        xi, eta, zeta: Natural coordinates (-1 to +1)

    Returns:
        Array of 8 shape function values
    """
    return 0.125 * np.array([
        (1 - xi) * (1 - eta) * (1 - zeta),
        (1 + xi) * (1 - eta) * (1 - zeta),
        (1 + xi) * (1 + eta) * (1 - zeta),
        (1 - xi) * (1 + eta) * (1 - zeta),
        (1 - xi) * (1 - eta) * (1 + zeta),
        (1 + xi) * (1 - eta) * (1 + zeta),
        (1 + xi) * (1 + eta) * (1 + zeta),
        (1 - xi) * (1 + eta) * (1 + zeta),
    ])


def shape_function_derivatives_hex8(xi: float, eta: float, zeta: float) -> np.ndarray:
    """
    Compute shape function derivatives w.r.t. natural coordinates.

    Args:
        xi, eta, zeta: Natural coordinates

    Returns:
        (3, 8) array of derivatives [dN/dxi, dN/deta, dN/dzeta]
    """
    dN = np.zeros((3, 8))

    # dN/dxi
    dN[0, 0] = -0.125 * (1 - eta) * (1 - zeta)
    dN[0, 1] = +0.125 * (1 - eta) * (1 - zeta)
    dN[0, 2] = +0.125 * (1 + eta) * (1 - zeta)
    dN[0, 3] = -0.125 * (1 + eta) * (1 - zeta)
    dN[0, 4] = -0.125 * (1 - eta) * (1 + zeta)
    dN[0, 5] = +0.125 * (1 - eta) * (1 + zeta)
    dN[0, 6] = +0.125 * (1 + eta) * (1 + zeta)
    dN[0, 7] = -0.125 * (1 + eta) * (1 + zeta)

    # dN/deta
    dN[1, 0] = -0.125 * (1 - xi) * (1 - zeta)
    dN[1, 1] = -0.125 * (1 + xi) * (1 - zeta)
    dN[1, 2] = +0.125 * (1 + xi) * (1 - zeta)
    dN[1, 3] = +0.125 * (1 - xi) * (1 - zeta)
    dN[1, 4] = -0.125 * (1 - xi) * (1 + zeta)
    dN[1, 5] = -0.125 * (1 + xi) * (1 + zeta)
    dN[1, 6] = +0.125 * (1 + xi) * (1 + zeta)
    dN[1, 7] = +0.125 * (1 - xi) * (1 + zeta)

    # dN/dzeta
    dN[2, 0] = -0.125 * (1 - xi) * (1 - eta)
    dN[2, 1] = -0.125 * (1 + xi) * (1 - eta)
    dN[2, 2] = -0.125 * (1 + xi) * (1 + eta)
    dN[2, 3] = -0.125 * (1 - xi) * (1 + eta)
    dN[2, 4] = +0.125 * (1 - xi) * (1 - eta)
    dN[2, 5] = +0.125 * (1 + xi) * (1 - eta)
    dN[2, 6] = +0.125 * (1 + xi) * (1 + eta)
    dN[2, 7] = +0.125 * (1 - xi) * (1 + eta)

    return dN


def elasticity_matrix_3d(E: float, nu: float) -> np.ndarray:
    """
    Compute 3D elasticity matrix (6x6) for isotropic material.

    Args:
        E: Young's modulus (Pa)
        nu: Poisson's ratio

    Returns:
        6x6 elasticity matrix [sigma] = [D][epsilon]
        Stress/strain order: [xx, yy, zz, xy, yz, xz]
    """
    factor = E / ((1 + nu) * (1 - 2 * nu))
    D = np.zeros((6, 6))

    # Normal stress-strain
    D[0, 0] = D[1, 1] = D[2, 2] = factor * (1 - nu)
    D[0, 1] = D[1, 0] = factor * nu
    D[0, 2] = D[2, 0] = factor * nu
    D[1, 2] = D[2, 1] = factor * nu

    # Shear stress-strain
    D[3, 3] = D[4, 4] = D[5, 5] = factor * (1 - 2 * nu) / 2

    return D


def compute_hex8_matrices(
    node_coords: np.ndarray,
    E: float,
    nu: float,
    rho: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute element stiffness and mass matrices for 8-node hexahedral.

    Args:
        node_coords: (8, 3) array of node coordinates
        E: Young's modulus (Pa)
        nu: Poisson's ratio
        rho: Density (kg/m^3)

    Returns:
        Tuple of (Ke, Me) - 24x24 stiffness and mass matrices
    """
    D = elasticity_matrix_3d(E, nu)

    Ke = np.zeros((24, 24))
    Me = np.zeros((24, 24))

    gauss_pts, gauss_wts = gauss_points_3d()

    for gp in range(8):
        xi, eta, zeta = gauss_pts[gp]
        w = gauss_wts[gp]

        # Shape functions and derivatives
        N = shape_functions_hex8(xi, eta, zeta)
        dN_nat = shape_function_derivatives_hex8(xi, eta, zeta)

        # Jacobian matrix: J = dN/d(xi,eta,zeta) @ coords
        J = dN_nat @ node_coords  # 3x3
        detJ = np.linalg.det(J)

        if detJ <= 0:
            # Degenerate element, skip
            continue

        J_inv = np.linalg.inv(J)

        # Shape function derivatives w.r.t. physical coordinates
        dN_phys = J_inv @ dN_nat  # 3x8: [dN/dx, dN/dy, dN/dz]

        # Build strain-displacement matrix B (6x24)
        B = np.zeros((6, 24))
        for i in range(8):
            col = 3 * i
            dNdx, dNdy, dNdz = dN_phys[:, i]

            # epsilon_xx = du/dx
            B[0, col] = dNdx
            # epsilon_yy = dv/dy
            B[1, col + 1] = dNdy
            # epsilon_zz = dw/dz
            B[2, col + 2] = dNdz
            # gamma_xy = du/dy + dv/dx
            B[3, col] = dNdy
            B[3, col + 1] = dNdx
            # gamma_yz = dv/dz + dw/dy
            B[4, col + 1] = dNdz
            B[4, col + 2] = dNdy
            # gamma_xz = du/dz + dw/dx
            B[5, col] = dNdz
            B[5, col + 2] = dNdx

        # Stiffness contribution
        Ke += w * detJ * (B.T @ D @ B)

        # Mass matrix - build N matrix (3x24)
        N_mat = np.zeros((3, 24))
        for i in range(8):
            col = 3 * i
            N_mat[0, col] = N[i]
            N_mat[1, col + 1] = N[i]
            N_mat[2, col + 2] = N[i]

        Me += w * detJ * rho * (N_mat.T @ N_mat)

    return Ke, Me


def generate_bar_mesh_3d(
    length: float,
    width: float,
    element_heights: List[float],
    nx: int,
    ny: int = 2,
    nz: int = 2
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate 3D mesh for an undercut bar.

    The bar is oriented with:
    - X axis along length
    - Y axis along width
    - Z axis along height (thickness)

    The undercut is symmetric about the center (x = L/2).

    Args:
        length: Bar length (m)
        width: Bar width (m)
        element_heights: Height at each x-position (from 2D discretization)
        nx: Number of elements in x-direction (should match len(element_heights))
        ny: Number of elements in y-direction
        nz: Number of elements in z-direction

    Returns:
        Tuple of (nodes, elements, heights_per_element):
        - nodes: (num_nodes, 3) array of coordinates
        - elements: (num_elements, 8) array of node indices
        - heights_per_element: height value for each 3D element
    """
    # Ensure nx matches element_heights
    nx = len(element_heights)

    # Element sizes
    dx = length / nx
    dy = width / ny

    # Node grid dimensions
    nnx = nx + 1
    nny = ny + 1
    nnz = nz + 1

    nodes_list = []
    node_index = {}  # (ix, iy, iz) -> node index

    # Generate nodes - the z coordinates vary based on the undercut profile
    for ix in range(nnx):
        x = ix * dx

        # Determine height at this x position
        if ix == 0:
            h = element_heights[0]
        elif ix == nx:
            h = element_heights[-1]
        else:
            # Average of adjacent elements
            h = (element_heights[ix - 1] + element_heights[ix]) / 2

        dz = h / nz

        for iy in range(nny):
            y = iy * dy

            for iz in range(nnz):
                # z goes from 0 to h (undercut is from bottom)
                z = iz * dz

                node_idx = len(nodes_list)
                node_index[(ix, iy, iz)] = node_idx
                nodes_list.append([x, y, z])

    nodes = np.array(nodes_list)

    # Generate elements
    elements_list = []
    heights_list = []

    for ix in range(nx):
        h = element_heights[ix]

        for iy in range(ny):
            for iz in range(nz):
                # 8 nodes of hexahedron (following standard numbering)
                n0 = node_index[(ix, iy, iz)]
                n1 = node_index[(ix + 1, iy, iz)]
                n2 = node_index[(ix + 1, iy + 1, iz)]
                n3 = node_index[(ix, iy + 1, iz)]
                n4 = node_index[(ix, iy, iz + 1)]
                n5 = node_index[(ix + 1, iy, iz + 1)]
                n6 = node_index[(ix + 1, iy + 1, iz + 1)]
                n7 = node_index[(ix, iy + 1, iz + 1)]

                elements_list.append([n0, n1, n2, n3, n4, n5, n6, n7])
                heights_list.append(h)

    elements = np.array(elements_list)
    heights_per_element = np.array(heights_list)

    return nodes, elements, heights_per_element


def generate_bar_mesh_3d_adaptive(
    length: float,
    width: float,
    x_positions: List[float],
    element_heights: List[float],
    ny: int = 2,
    nz: int = 2
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate 3D mesh for an undercut bar with adaptive x-spacing.

    Unlike generate_bar_mesh_3d which uses uniform spacing, this function
    accepts pre-computed x-positions allowing for refined mesh at cut boundaries.

    Args:
        length: Bar length (m)
        width: Bar width (m)
        x_positions: X coordinates of element boundaries (length nx+1)
        element_heights: Height at each x-element (length nx)
        ny: Number of elements in y-direction
        nz: Number of elements in z-direction

    Returns:
        Tuple of (nodes, elements, heights_per_element):
        - nodes: (num_nodes, 3) array of coordinates
        - elements: (num_elements, 8) array of node indices
        - heights_per_element: height value for each 3D element
    """
    nx = len(element_heights)
    assert len(x_positions) == nx + 1, "x_positions must have length len(element_heights) + 1"

    dy = width / ny

    # Node grid dimensions
    nnx = nx + 1
    nny = ny + 1
    nnz = nz + 1

    nodes_list = []
    node_index = {}  # (ix, iy, iz) -> node index

    # Generate nodes - x coordinates from x_positions, z varies based on height
    for ix in range(nnx):
        x = x_positions[ix]

        # Determine height at this x position
        if ix == 0:
            h = element_heights[0]
        elif ix == nx:
            h = element_heights[-1]
        else:
            # Average of adjacent elements
            h = (element_heights[ix - 1] + element_heights[ix]) / 2

        dz = h / nz

        for iy in range(nny):
            y = iy * dy

            for iz in range(nnz):
                z = iz * dz

                node_idx = len(nodes_list)
                node_index[(ix, iy, iz)] = node_idx
                nodes_list.append([x, y, z])

    nodes = np.array(nodes_list)

    # Generate elements
    elements_list = []
    heights_list = []

    for ix in range(nx):
        h = element_heights[ix]

        for iy in range(ny):
            for iz in range(nz):
                # 8 nodes of hexahedron
                n0 = node_index[(ix, iy, iz)]
                n1 = node_index[(ix + 1, iy, iz)]
                n2 = node_index[(ix + 1, iy + 1, iz)]
                n3 = node_index[(ix, iy + 1, iz)]
                n4 = node_index[(ix, iy, iz + 1)]
                n5 = node_index[(ix + 1, iy, iz + 1)]
                n6 = node_index[(ix + 1, iy + 1, iz + 1)]
                n7 = node_index[(ix, iy + 1, iz + 1)]

                elements_list.append([n0, n1, n2, n3, n4, n5, n6, n7])
                heights_list.append(h)

    elements = np.array(elements_list)
    heights_per_element = np.array(heights_list)

    return nodes, elements, heights_per_element


def assemble_global_matrices_3d(
    nodes: np.ndarray,
    elements: np.ndarray,
    E: float,
    nu: float,
    rho: float,
    use_sparse: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Assemble global stiffness and mass matrices from 3D mesh.

    Args:
        nodes: (num_nodes, 3) array of node coordinates
        elements: (num_elements, 8) array of node indices
        E: Young's modulus (Pa)
        nu: Poisson's ratio
        rho: Density (kg/m^3)
        use_sparse: Whether to use sparse matrices (recommended for large meshes)

    Returns:
        Tuple of (K_global, M_global) matrices
    """
    num_nodes = len(nodes)
    num_dof = 3 * num_nodes
    num_elements = len(elements)

    if use_sparse:
        K_global = lil_matrix((num_dof, num_dof), dtype=np.float64)
        M_global = lil_matrix((num_dof, num_dof), dtype=np.float64)
    else:
        K_global = np.zeros((num_dof, num_dof), dtype=np.float64)
        M_global = np.zeros((num_dof, num_dof), dtype=np.float64)

    for e in range(num_elements):
        elem_nodes = elements[e]
        node_coords = nodes[elem_nodes]

        Ke, Me = compute_hex8_matrices(node_coords, E, nu, rho)

        # DOF mapping
        dof_map = []
        for n in elem_nodes:
            dof_map.extend([3 * n, 3 * n + 1, 3 * n + 2])

        for i in range(24):
            for j in range(24):
                gi, gj = dof_map[i], dof_map[j]
                K_global[gi, gj] += Ke[i, j]
                M_global[gi, gj] += Me[i, j]

    if use_sparse:
        K_global = csr_matrix(K_global)
        M_global = csr_matrix(M_global)

    return K_global, M_global


def solve_eigenvalue_3d(
    K: np.ndarray,
    M: np.ndarray,
    num_modes: int,
    use_sparse: bool = True
) -> List[float]:
    """
    Solve generalized eigenvalue problem for 3D FEM.

    For 3D analysis of a free-free bar, we look for transverse
    bending modes (primarily Z-direction motion).

    Args:
        K: Global stiffness matrix
        M: Global mass matrix
        num_modes: Number of modes to extract

    Returns:
        List of natural frequencies in Hz
    """
    if use_sparse:
        # Use sparse eigenvalue solver
        # Request more eigenvalues to filter rigid body modes
        num_request = min(num_modes + 10, K.shape[0] - 2)

        # Add small shift to avoid singularity issues
        sigma = 1.0  # Small shift

        try:
            eigenvalues, _ = eigsh(K, k=num_request, M=M, sigma=sigma, which='LM')
        except Exception:
            # Fallback: add regularization
            n = K.shape[0]
            M_reg = M + 1e-10 * csr_matrix(np.eye(n))
            eigenvalues, _ = eigsh(K, k=num_request, M=M_reg, sigma=sigma, which='LM')
    else:
        # Dense solver
        n = K.shape[0]

        # Regularize M
        M_reg = M.copy()
        for i in range(n):
            M_reg[i, i] += 1e-12 * max(abs(M[i, i]), 1e-20)

        try:
            L = linalg.cholesky(M_reg, lower=True)
            L_inv = linalg.solve_triangular(L, np.eye(n), lower=True)
            K_tilde = L_inv @ K @ L_inv.T
            K_tilde = (K_tilde + K_tilde.T) / 2
            eigenvalues, _ = linalg.eigh(K_tilde)
        except linalg.LinAlgError:
            for i in range(n):
                M_reg[i, i] += 1e-8
            L = linalg.cholesky(M_reg, lower=True)
            L_inv = linalg.solve_triangular(L, np.eye(n), lower=True)
            K_tilde = L_inv @ K @ L_inv.T
            K_tilde = (K_tilde + K_tilde.T) / 2
            eigenvalues, _ = linalg.eigh(K_tilde)

    # Sort and filter
    eigenvalues = np.sort(eigenvalues)

    # Filter rigid body modes (6 for 3D: 3 translations + 3 rotations)
    threshold = 100.0  # omega^2 threshold
    elastic_modes = [ev for ev in eigenvalues if ev > threshold]

    # Convert to frequencies
    frequencies = [math.sqrt(abs(ev)) / (2.0 * math.pi) for ev in elastic_modes[:num_modes]]

    return frequencies


def compute_frequencies_3d(
    element_heights: List[float],
    length: float,
    width: float,
    E: float,
    rho: float,
    nu: float,
    num_modes: int,
    ny: int = 2,
    nz: int = 2
) -> List[float]:
    """
    Compute natural frequencies using 3D FEM analysis.

    This is the main entry point for 3D frequency computation.

    Args:
        element_heights: Height of each element along bar length (m)
        length: Bar length (m)
        width: Bar width (m)
        E: Young's modulus (Pa)
        rho: Density (kg/m^3)
        nu: Poisson's ratio
        num_modes: Number of modes to extract
        ny: Number of elements in width direction
        nz: Number of elements in thickness direction

    Returns:
        List of natural frequencies in Hz
    """
    nx = len(element_heights)

    # Generate mesh
    nodes, elements, _ = generate_bar_mesh_3d(
        length, width, element_heights, nx, ny, nz
    )

    # Determine if we should use sparse matrices
    num_dof = 3 * len(nodes)
    use_sparse = num_dof > 1000

    # Assemble matrices
    K, M = assemble_global_matrices_3d(nodes, elements, E, nu, rho, use_sparse)

    # Solve eigenvalue problem
    frequencies = solve_eigenvalue_3d(K, M, num_modes, use_sparse)

    return frequencies


# =============================================================================
# Mode Classification (Soares' corner displacement method)
# =============================================================================

def find_corner_nodes(nodes: np.ndarray, tol: float = 1e-6) -> Tuple[int, int]:
    """
    Find the two corner nodes at x=0 end of bar.

    Assumes bar is oriented with length along x-axis,
    width along y-axis, height along z-axis.

    Args:
        nodes: (num_nodes, 3) array of node coordinates
        tol: Tolerance for coordinate comparison

    Returns:
        Tuple (s1_idx, s2_idx) of corner node indices at x=0 end
    """
    # Find nodes at x ≈ 0 (one end of bar)
    x_min = nodes[:, 0].min()
    end_mask = np.abs(nodes[:, 0] - x_min) < tol
    end_nodes = np.where(end_mask)[0]

    # Find corners at top surface (max z)
    z_vals = nodes[end_nodes, 2]
    z_max = z_vals.max()
    top_mask = np.abs(z_vals - z_max) < tol
    top_nodes = end_nodes[top_mask]
    top_y = nodes[top_nodes, 1]

    s1_idx = top_nodes[np.argmax(top_y)]  # max y
    s2_idx = top_nodes[np.argmin(top_y)]  # min y

    return int(s1_idx), int(s2_idx)


def classify_mode_soares(
    mode_shape: np.ndarray,
    nodes: np.ndarray,
    corner_indices: Tuple[int, int]
) -> str:
    """
    Classify mode using Soares' corner displacement method.

    Args:
        mode_shape: Eigenvector (n_nodes * 3,) with [ux1,uy1,uz1, ux2,uy2,uz2, ...]
        nodes: Node coordinates (n_nodes, 3)
        corner_indices: Tuple (s1_idx, s2_idx) of corner node indices at x=0 end

    Returns:
        mode_type: 'vertical_bending', 'torsional', 'lateral', or 'axial'
    """
    s1_idx, s2_idx = corner_indices

    # Extract 3D displacement at each corner
    psi_s1 = mode_shape[s1_idx*3 : s1_idx*3 + 3]  # [ψx, ψy, ψz]
    psi_s2 = mode_shape[s2_idx*3 : s2_idx*3 + 3]

    # Find direction of maximum displacement at s1
    abs_psi = np.abs(psi_s1)
    max_dir = np.argmax(abs_psi)  # 0=x, 1=y, 2=z

    if max_dir == 1:  # y-direction dominant
        return 'lateral'

    elif max_dir == 0:  # x-direction dominant
        return 'axial'

    else:  # z-direction dominant (max_dir == 2)
        # Disambiguate vertical bending vs torsional
        if np.sign(psi_s1[2]) == np.sign(psi_s2[2]):
            return 'vertical_bending'
        else:
            return 'torsional'


def classify_all_modes(
    frequencies: List[float],
    mode_shapes: np.ndarray,
    nodes: np.ndarray
) -> dict:
    """
    Classify all modes and organize by family.

    Args:
        frequencies: List of frequencies in Hz
        mode_shapes: (n_dof, n_modes) array of eigenvectors
        nodes: (n_nodes, 3) array of node coordinates

    Returns:
        Dict with frequencies organized by mode type:
        {
            'vertical_bending': [{'frequency': f, 'mode_index': i, 'mode_number': n}, ...],
            'torsional': [...],
            'lateral': [...],
            'axial': [...]
        }
    """
    corner_indices = find_corner_nodes(nodes)

    classified = {
        'vertical_bending': [],
        'torsional': [],
        'lateral': [],
        'axial': []
    }

    for i, freq in enumerate(frequencies):
        if i < mode_shapes.shape[1]:
            shape = mode_shapes[:, i]
            mode_type = classify_mode_soares(shape, nodes, corner_indices)
            classified[mode_type].append({
                'frequency': freq,
                'mode_index': i,
            })

    # Sort each family by frequency and assign mode numbers
    for family in classified:
        classified[family].sort(key=lambda m: m['frequency'])
        for j, mode in enumerate(classified[family]):
            mode['mode_number'] = j + 1  # V1, V2, V3... or T1, T2...

    return classified


def solve_eigenvalue_3d_with_vectors(
    K: np.ndarray,
    M: np.ndarray,
    num_modes: int,
    use_sparse: bool = True
) -> Tuple[List[float], np.ndarray]:
    """
    Solve generalized eigenvalue problem and return both frequencies and mode shapes.

    Args:
        K: Global stiffness matrix
        M: Global mass matrix
        num_modes: Number of modes to extract
        use_sparse: Whether matrices are sparse

    Returns:
        Tuple of (frequencies in Hz, mode_shapes array)
    """
    if use_sparse:
        num_request = min(num_modes + 12, K.shape[0] - 2)
        sigma = 1.0

        try:
            eigenvalues, eigenvectors = eigsh(K, k=num_request, M=M, sigma=sigma, which='LM')
        except Exception:
            n = K.shape[0]
            M_reg = M + 1e-10 * csr_matrix(np.eye(n))
            eigenvalues, eigenvectors = eigsh(K, k=num_request, M=M_reg, sigma=sigma, which='LM')
    else:
        n = K.shape[0]
        M_reg = M.copy()
        for i in range(n):
            M_reg[i, i] += 1e-12 * max(abs(M[i, i]), 1e-20)

        try:
            L = linalg.cholesky(M_reg, lower=True)
            L_inv = linalg.solve_triangular(L, np.eye(n), lower=True)
            K_tilde = L_inv @ K @ L_inv.T
            K_tilde = (K_tilde + K_tilde.T) / 2
            eigenvalues, eigenvectors_tilde = linalg.eigh(K_tilde)
            eigenvectors = L_inv.T @ eigenvectors_tilde
        except linalg.LinAlgError:
            for i in range(n):
                M_reg[i, i] += 1e-8
            L = linalg.cholesky(M_reg, lower=True)
            L_inv = linalg.solve_triangular(L, np.eye(n), lower=True)
            K_tilde = L_inv @ K @ L_inv.T
            K_tilde = (K_tilde + K_tilde.T) / 2
            eigenvalues, eigenvectors_tilde = linalg.eigh(K_tilde)
            eigenvectors = L_inv.T @ eigenvectors_tilde

    # Sort by eigenvalue
    sort_idx = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[sort_idx]
    eigenvectors = eigenvectors[:, sort_idx]

    # Filter rigid body modes
    threshold = 100.0
    elastic_mask = eigenvalues > threshold
    elastic_eigenvalues = eigenvalues[elastic_mask]
    elastic_eigenvectors = eigenvectors[:, elastic_mask]

    # Convert to frequencies
    frequencies = [math.sqrt(abs(ev)) / (2.0 * math.pi) for ev in elastic_eigenvalues[:num_modes]]
    mode_shapes = elastic_eigenvectors[:, :num_modes]

    return frequencies, mode_shapes


def compute_frequencies_3d_classified(
    element_heights: List[float],
    length: float,
    width: float,
    E: float,
    rho: float,
    nu: float,
    num_modes: int = 10,
    ny: int = 2,
    nz: int = 2
) -> Tuple[List[float], dict, np.ndarray]:
    """
    Compute natural frequencies using 3D FEM with mode classification.

    Returns frequencies, classified modes dict, and node coordinates.

    Args:
        element_heights: Height of each element along bar length (m)
        length: Bar length (m)
        width: Bar width (m)
        E: Young's modulus (Pa)
        rho: Density (kg/m^3)
        nu: Poisson's ratio
        num_modes: Number of modes to extract (request more for classification)
        ny: Number of elements in width direction
        nz: Number of elements in thickness direction

    Returns:
        Tuple of:
        - all_frequencies: List of all frequencies
        - classified: Dict with modes organized by type
        - nodes: Node coordinates for visualization
    """
    nx = len(element_heights)

    # Generate mesh
    nodes, elements, _ = generate_bar_mesh_3d(
        length, width, element_heights, nx, ny, nz
    )

    # Determine if we should use sparse matrices
    num_dof = 3 * len(nodes)
    use_sparse = num_dof > 1000

    # Assemble matrices
    K, M = assemble_global_matrices_3d(nodes, elements, E, nu, rho, use_sparse)

    # Solve eigenvalue problem with mode shapes
    frequencies, mode_shapes = solve_eigenvalue_3d_with_vectors(K, M, num_modes, use_sparse)

    # Classify modes
    classified = classify_all_modes(frequencies, mode_shapes, nodes)

    return frequencies, classified, nodes


def get_bending_frequencies_3d(
    element_heights: List[float],
    length: float,
    width: float,
    E: float,
    rho: float,
    nu: float,
    num_bending_modes: int = 3,
    ny: int = 2,
    nz: int = 2
) -> List[float]:
    """
    Compute only vertical bending frequencies from 3D FEM analysis.

    This filters out torsional, lateral, and axial modes to return
    only the vertical bending modes comparable to 2D beam analysis.

    Args:
        element_heights: Height of each element along bar length (m)
        length: Bar length (m)
        width: Bar width (m)
        E: Young's modulus (Pa)
        rho: Density (kg/m^3)
        nu: Poisson's ratio
        num_bending_modes: Number of bending modes to return
        ny: Number of elements in width direction
        nz: Number of elements in thickness direction

    Returns:
        List of vertical bending frequencies in Hz
    """
    # Request more modes to ensure we find enough bending modes
    num_request = num_bending_modes * 4 + 6

    _, classified, _ = compute_frequencies_3d_classified(
        element_heights, length, width, E, rho, nu,
        num_request, ny, nz
    )

    # Extract vertical bending frequencies
    bending_modes = classified['vertical_bending']
    bending_freqs = [m['frequency'] for m in bending_modes[:num_bending_modes]]

    return bending_freqs


def compute_frequencies_3d_adaptive(
    x_positions: List[float],
    element_heights: List[float],
    length: float,
    width: float,
    E: float,
    rho: float,
    nu: float,
    num_modes: int = 10,
    ny: int = 2,
    nz: int = 2
) -> Tuple[List[float], dict, np.ndarray, np.ndarray]:
    """
    Compute natural frequencies using 3D FEM with adaptive mesh and mode classification.

    This function uses pre-computed adaptive mesh positions for refined accuracy
    at cut boundaries while keeping coarser elements in uniform regions.

    Args:
        x_positions: X coordinates of element boundaries (from generate_adaptive_mesh_1d)
        element_heights: Height of each element along bar length (m)
        length: Bar length (m)
        width: Bar width (m)
        E: Young's modulus (Pa)
        rho: Density (kg/m^3)
        nu: Poisson's ratio
        num_modes: Number of modes to extract
        ny: Number of elements in width direction
        nz: Number of elements in thickness direction

    Returns:
        Tuple of:
        - all_frequencies: List of all frequencies
        - classified: Dict with modes organized by type
        - nodes: Node coordinates for visualization
        - elements: Element connectivity array
    """
    # Generate adaptive mesh
    nodes, elements, _ = generate_bar_mesh_3d_adaptive(
        length, width, x_positions, element_heights, ny, nz
    )

    # Determine if we should use sparse matrices
    num_dof = 3 * len(nodes)
    use_sparse = num_dof > 1000

    # Assemble matrices
    K, M = assemble_global_matrices_3d(nodes, elements, E, nu, rho, use_sparse)

    # Solve eigenvalue problem with mode shapes
    frequencies, mode_shapes = solve_eigenvalue_3d_with_vectors(K, M, num_modes, use_sparse)

    # Classify modes
    classified = classify_all_modes(frequencies, mode_shapes, nodes)

    return frequencies, classified, nodes, elements
