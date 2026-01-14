"""
Finite Element Assembly and Eigenvalue Solving

Implements global matrix assembly from element matrices and
generalized eigenvalue solving for natural frequencies.
"""

from typing import List, Tuple
import numpy as np
from scipy import linalg
import math

from .timoshenko import compute_element_stiffness, compute_element_mass


def assemble_global_matrices(
    element_heights: List[float],
    le: float,
    b: float,
    E: float,
    rho: float,
    nu: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Assemble global stiffness and mass matrices from element matrices.

    Args:
        element_heights: Height of each element (m)
        le: Element length (m)
        b: Bar width (m)
        E: Young's modulus (Pa)
        rho: Density (kg/m^3)
        nu: Poisson's ratio

    Returns:
        Tuple of (K_global, M_global) matrices
    """
    Ne = len(element_heights)
    num_dof = 2 * (Ne + 1)

    K_global = np.zeros((num_dof, num_dof), dtype=np.float64)
    M_global = np.zeros((num_dof, num_dof), dtype=np.float64)

    G = E / (2.0 * (1.0 + nu))  # Shear modulus

    for e in range(Ne):
        h = element_heights[e]
        A = b * h  # Cross-sectional area
        I = b * h**3 / 12.0  # Second moment of area

        Ke = compute_element_stiffness(le, E, I, G, A)
        Me = compute_element_mass(le, rho, A, I, E, G)

        # DOF mapping: element DOFs [0,1,2,3] -> global DOFs [2e, 2e+1, 2e+2, 2e+3]
        dof_map = [2*e, 2*e+1, 2*(e+1), 2*(e+1)+1]

        for i in range(4):
            for j in range(4):
                gi = dof_map[i]
                gj = dof_map[j]
                K_global[gi, gj] += Ke[i, j]
                M_global[gi, gj] += Me[i, j]

    return K_global, M_global


def solve_generalized_eigenvalue(
    K: np.ndarray,
    M: np.ndarray,
    num_modes: int
) -> List[float]:
    """
    Solve generalized eigenvalue problem K*phi = lambda*M*phi.
    Uses the standard form transformation via Cholesky decomposition.

    Args:
        K: Global stiffness matrix
        M: Global mass matrix
        num_modes: Number of modes to extract

    Returns:
        List of natural frequencies in Hz
    """
    n = K.shape[0]

    # Add small regularization to M for numerical stability
    M_reg = M.copy()
    for i in range(n):
        M_reg[i, i] += 1e-12 * max(abs(M[i, i]), 1e-20)

    try:
        # Compute Cholesky decomposition: M = L @ L.T
        L = linalg.cholesky(M_reg, lower=True)

        # Compute L^{-1}
        L_inv = linalg.solve_triangular(L, np.eye(n), lower=True)

        # K_tilde = L^{-1} @ K @ L^{-T}
        K_tilde = L_inv @ K @ L_inv.T

        # Symmetrize to remove numerical errors
        K_tilde = (K_tilde + K_tilde.T) / 2

        # Solve standard symmetric eigenvalue problem
        eigenvalues, _ = linalg.eigh(K_tilde)

    except linalg.LinAlgError:
        # Fallback: add more regularization
        for i in range(n):
            M_reg[i, i] += 1e-8

        L = linalg.cholesky(M_reg, lower=True)
        L_inv = linalg.solve_triangular(L, np.eye(n), lower=True)
        K_tilde = L_inv @ K @ L_inv.T
        K_tilde = (K_tilde + K_tilde.T) / 2
        eigenvalues, _ = linalg.eigh(K_tilde)

    # Sort eigenvalues
    eigenvalues = np.sort(eigenvalues)

    # Filter out rigid body modes (very small or negative eigenvalues)
    # For a free-free beam, there are 2 rigid body modes with ~zero eigenvalues
    threshold = 1.0  # omega^2 = 1 rad^2/s^2 -> f = 0.16 Hz
    elastic_modes = [ev for ev in eigenvalues if ev > threshold]

    # Convert eigenvalues to frequencies: f = sqrt(lambda) / (2*pi)
    frequencies = [math.sqrt(ev) / (2.0 * math.pi) for ev in elastic_modes[:num_modes]]

    return frequencies
