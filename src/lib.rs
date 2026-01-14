use std::collections::{HashMap, HashSet};

use lanczos::{Hermitian, Order};
use nalgebra::{DMatrix, DVector, Matrix4, Point3};
use nalgebra_sparse::{coo::CooMatrix, csr::CsrMatrix};
use serde::Serialize;
use wasm_bindgen::prelude::*;

const RIGID_MODE_TOL: f64 = 1e-8;

#[derive(Clone, Debug)]
pub struct Material {
    pub young_modulus: f64,
    pub poisson_ratio: f64,
    pub density: f64,
}

/// Linear tetrahedral mesh (TET4)
#[derive(Clone, Debug)]
pub struct Mesh {
    pub nodes: Vec<Point3<f64>>,
    pub elements: Vec<[usize; 4]>,
}

/// Quadratic tetrahedral mesh (TET10)
#[derive(Clone, Debug)]
pub struct Tet10Mesh {
    pub nodes: Vec<Point3<f64>>,
    pub elements: Vec<[usize; 10]>,
}

#[derive(Clone, Debug)]
pub struct AssembledModel {
    pub stiffness: CsrMatrix<f64>,
    pub mass_diag: Vec<f64>,
}

#[derive(Clone, Debug, Serialize)]
pub struct ModeSummary {
    pub frequencies_hz: Vec<f64>,
}

impl Mesh {
    /// Creates a rectangular box mesh with specified divisions and dimensions along each axis.
    pub fn regular_box(
        divisions: [usize; 3],
        size: [f64; 3],
    ) -> Self {
        assert!(divisions.iter().all(|&d| d > 0), "divisions must be at least 1");
        let steps = [
            size[0] / divisions[0] as f64,
            size[1] / divisions[1] as f64,
            size[2] / divisions[2] as f64,
        ];
        let nodes_per_axis = [divisions[0] + 1, divisions[1] + 1, divisions[2] + 1];

        let mut nodes = Vec::with_capacity(nodes_per_axis[0] * nodes_per_axis[1] * nodes_per_axis[2]);
        for k in 0..nodes_per_axis[2] {
            for j in 0..nodes_per_axis[1] {
                for i in 0..nodes_per_axis[0] {
                    nodes.push(Point3::new(
                        i as f64 * steps[0],
                        j as f64 * steps[1],
                        k as f64 * steps[2],
                    ));
                }
            }
        }

        let mut elements = Vec::with_capacity(divisions[0] * divisions[1] * divisions[2] * 5);
        let idx = |i: usize, j: usize, k: usize| -> usize {
            (k * nodes_per_axis[1] * nodes_per_axis[0]) + (j * nodes_per_axis[0]) + i
        };

        for k in 0..divisions[2] {
            for j in 0..divisions[1] {
                for i in 0..divisions[0] {
                    let n0 = idx(i, j, k);
                    let n1 = idx(i + 1, j, k);
                    let n2 = idx(i, j + 1, k);
                    let n3 = idx(i + 1, j + 1, k);
                    let n4 = idx(i, j, k + 1);
                    let n5 = idx(i + 1, j, k + 1);
                    let n6 = idx(i, j + 1, k + 1);
                    let n7 = idx(i + 1, j + 1, k + 1);

                    // Five-tetra subdivision of a hexahedron
                    elements.push([n0, n1, n3, n7]);
                    elements.push([n0, n3, n2, n7]);
                    elements.push([n0, n2, n6, n7]);
                    elements.push([n0, n6, n4, n7]);
                    elements.push([n0, n4, n5, n7]);
                }
            }
        }

        Self { nodes, elements }
    }

    pub fn regular_cube(divisions: usize, size: f64) -> Self {
        assert!(divisions > 0, "divisions must be at least 1");
        let step = size / divisions as f64;
        let nodes_per_axis = divisions + 1;

        let mut nodes = Vec::with_capacity(nodes_per_axis.pow(3));
        for k in 0..=divisions {
            for j in 0..=divisions {
                for i in 0..=divisions {
                    nodes.push(Point3::new(i as f64 * step, j as f64 * step, k as f64 * step));
                }
            }
        }

        let mut elements = Vec::with_capacity(divisions.pow(3) * 5);
        let idx = |i: usize, j: usize, k: usize| -> usize {
            (k * nodes_per_axis * nodes_per_axis) + (j * nodes_per_axis) + i
        };

        for k in 0..divisions {
            for j in 0..divisions {
                for i in 0..divisions {
                    let n0 = idx(i, j, k);
                    let n1 = idx(i + 1, j, k);
                    let n2 = idx(i, j + 1, k);
                    let n3 = idx(i + 1, j + 1, k);
                    let n4 = idx(i, j, k + 1);
                    let n5 = idx(i + 1, j, k + 1);
                    let n6 = idx(i, j + 1, k + 1);
                    let n7 = idx(i + 1, j + 1, k + 1);

                    // Five-tetra subdivision of a cube
                    elements.push([n0, n1, n3, n7]);
                    elements.push([n0, n3, n2, n7]);
                    elements.push([n0, n2, n6, n7]);
                    elements.push([n0, n6, n4, n7]);
                    elements.push([n0, n4, n5, n7]);
                }
            }
        }

        Self { nodes, elements }
    }
}

impl Tet10Mesh {
    /// Creates a rectangular box mesh with TET10 (quadratic) elements.
    /// Mid-edge nodes are added automatically.
    pub fn regular_box(divisions: [usize; 3], size: [f64; 3]) -> Self {
        // First create a linear tet mesh
        let linear_mesh = Mesh::regular_box(divisions, size);

        // Convert to TET10 by adding mid-edge nodes
        Self::from_linear_mesh(&linear_mesh)
    }

    /// Converts a linear TET4 mesh to a quadratic TET10 mesh by adding mid-edge nodes.
    pub fn from_linear_mesh(linear_mesh: &Mesh) -> Self {
        let mut nodes = linear_mesh.nodes.clone();
        let mut edge_midpoints: HashMap<(usize, usize), usize> = HashMap::new();

        // Helper to get or create mid-edge node
        let mut get_midpoint = |n1: usize, n2: usize, nodes: &mut Vec<Point3<f64>>| -> usize {
            let key = if n1 < n2 { (n1, n2) } else { (n2, n1) };
            if let Some(&idx) = edge_midpoints.get(&key) {
                return idx;
            }
            let p1 = nodes[n1];
            let p2 = nodes[n2];
            let mid = Point3::new(
                (p1.x + p2.x) / 2.0,
                (p1.y + p2.y) / 2.0,
                (p1.z + p2.z) / 2.0,
            );
            let idx = nodes.len();
            nodes.push(mid);
            edge_midpoints.insert(key, idx);
            idx
        };

        let mut elements = Vec::with_capacity(linear_mesh.elements.len());

        for &[n0, n1, n2, n3] in &linear_mesh.elements {
            // TET10 node numbering:
            // 0-3: corner nodes
            // 4: midpoint of edge 0-1
            // 5: midpoint of edge 1-2
            // 6: midpoint of edge 0-2
            // 7: midpoint of edge 0-3
            // 8: midpoint of edge 1-3
            // 9: midpoint of edge 2-3
            let n4 = get_midpoint(n0, n1, &mut nodes);
            let n5 = get_midpoint(n1, n2, &mut nodes);
            let n6 = get_midpoint(n0, n2, &mut nodes);
            let n7 = get_midpoint(n0, n3, &mut nodes);
            let n8 = get_midpoint(n1, n3, &mut nodes);
            let n9 = get_midpoint(n2, n3, &mut nodes);

            elements.push([n0, n1, n2, n3, n4, n5, n6, n7, n8, n9]);
        }

        Self { nodes, elements }
    }
}

pub fn assemble_model(mesh: &Mesh, material: &Material, fixed_nodes: &HashSet<usize>) -> AssembledModel {
    let total_dofs = mesh.nodes.len() * 3;
    let mut dof_map = vec![None; total_dofs];
    let mut free_dofs = 0usize;

    for (node_idx, _) in mesh.nodes.iter().enumerate() {
        let fixed = fixed_nodes.contains(&node_idx);
        for comp in 0..3 {
            let idx = node_idx * 3 + comp;
            if fixed {
                continue;
            }
            dof_map[idx] = Some(free_dofs);
            free_dofs += 1;
        }
    }

    let mut mass_diag = vec![0.0f64; free_dofs];
    let mut coo = CooMatrix::new(free_dofs, free_dofs);

    for &element in &mesh.elements {
        let element_nodes = [
            mesh.nodes[element[0]],
            mesh.nodes[element[1]],
            mesh.nodes[element[2]],
            mesh.nodes[element[3]],
        ];

        let (volume, ke) = element_stiffness(&element_nodes, material);
        if volume <= f64::EPSILON {
            continue;
        }

        let mass_per_node = material.density * volume / 4.0;
        let mass_per_dof = mass_per_node / 3.0;

        for (local_node, &global_node) in element.iter().enumerate() {
            for comp in 0..3 {
                let global_dof = global_node * 3 + comp;
                if let Some(row) = dof_map[global_dof] {
                    mass_diag[row] += mass_per_dof;
                }
            }

            for (local_other, &global_other) in element.iter().enumerate() {
                let block = ke.view((local_node * 3, local_other * 3), (3, 3));
                for r in 0..3 {
                    for c in 0..3 {
                        let global_r = global_node * 3 + r;
                        let global_c = global_other * 3 + c;
                        if let (Some(row), Some(col)) = (dof_map[global_r], dof_map[global_c]) {
                            let val = block[(r, c)];
                            if val.abs() > 0.0 {
                                coo.push(row, col, val);
                            }
                        }
                    }
                }
            }
        }
    }

    let stiffness = CsrMatrix::from(&coo);

    AssembledModel { stiffness, mass_diag }
}

fn element_stiffness(nodes: &[Point3<f64>; 4], material: &Material) -> (f64, DMatrix<f64>) {
    let m = Matrix4::new(
        1.0,
        nodes[0].x,
        nodes[0].y,
        nodes[0].z,
        1.0,
        nodes[1].x,
        nodes[1].y,
        nodes[1].z,
        1.0,
        nodes[2].x,
        nodes[2].y,
        nodes[2].z,
        1.0,
        nodes[3].x,
        nodes[3].y,
        nodes[3].z,
    );

    let det = m.determinant();
    let volume = det.abs() / 6.0;
    if volume <= f64::EPSILON {
        return (0.0, DMatrix::zeros(12, 12));
    }

    let inv = match m.try_inverse() {
        Some(inv) => inv,
        None => return (0.0, DMatrix::zeros(12, 12)),
    };

    let grads = [
        (inv[(1, 0)], inv[(2, 0)], inv[(3, 0)]),
        (inv[(1, 1)], inv[(2, 1)], inv[(3, 1)]),
        (inv[(1, 2)], inv[(2, 2)], inv[(3, 2)]),
        (inv[(1, 3)], inv[(2, 3)], inv[(3, 3)]),
    ];

    let mut b = DMatrix::<f64>::zeros(6, 12);
    for (i, (gx, gy, gz)) in grads.iter().enumerate() {
        let col = i * 3;
        b[(0, col)] = *gx;
        b[(1, col + 1)] = *gy;
        b[(2, col + 2)] = *gz;

        b[(3, col + 1)] = *gz;
        b[(3, col + 2)] = *gy;

        b[(4, col)] = *gz;
        b[(4, col + 2)] = *gx;

        b[(5, col)] = *gy;
        b[(5, col + 1)] = *gx;
    }

    let d = constitutive_matrix(material);
    let ke = b.transpose() * d * b * volume;
    (volume, ke)
}

fn constitutive_matrix(material: &Material) -> DMatrix<f64> {
    let e = material.young_modulus;
    let nu = material.poisson_ratio;

    let lambda = e * nu / ((1.0 + nu) * (1.0 - 2.0 * nu));
    let mu = e / (2.0 * (1.0 + nu));

    DMatrix::from_row_slice(
        6,
        6,
        &[
            lambda + 2.0 * mu,
            lambda,
            lambda,
            0.0,
            0.0,
            0.0,
            lambda,
            lambda + 2.0 * mu,
            lambda,
            0.0,
            0.0,
            0.0,
            lambda,
            lambda,
            lambda + 2.0 * mu,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            mu,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            mu,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            mu,
        ],
    )
}

/// TET10 shape functions in natural coordinates (xi, eta, zeta)
/// L4 = 1 - xi - eta - zeta (fourth barycentric coordinate)
fn tet10_shape_functions(xi: f64, eta: f64, zeta: f64) -> [f64; 10] {
    let l4 = 1.0 - xi - eta - zeta;
    [
        l4 * (2.0 * l4 - 1.0),      // N0: corner 0
        xi * (2.0 * xi - 1.0),       // N1: corner 1
        eta * (2.0 * eta - 1.0),     // N2: corner 2
        zeta * (2.0 * zeta - 1.0),   // N3: corner 3
        4.0 * l4 * xi,               // N4: mid-edge 0-1
        4.0 * xi * eta,              // N5: mid-edge 1-2
        4.0 * l4 * eta,              // N6: mid-edge 0-2
        4.0 * l4 * zeta,             // N7: mid-edge 0-3
        4.0 * xi * zeta,             // N8: mid-edge 1-3
        4.0 * eta * zeta,            // N9: mid-edge 2-3
    ]
}

/// Derivatives of TET10 shape functions with respect to natural coordinates
/// Returns [dN/dxi, dN/deta, dN/dzeta] for each of the 10 nodes
fn tet10_shape_derivatives(xi: f64, eta: f64, zeta: f64) -> [[f64; 3]; 10] {
    let l4 = 1.0 - xi - eta - zeta;
    [
        // dN0/d(xi, eta, zeta) - corner 0
        [-(4.0 * l4 - 1.0), -(4.0 * l4 - 1.0), -(4.0 * l4 - 1.0)],
        // dN1/d(xi, eta, zeta) - corner 1
        [4.0 * xi - 1.0, 0.0, 0.0],
        // dN2/d(xi, eta, zeta) - corner 2
        [0.0, 4.0 * eta - 1.0, 0.0],
        // dN3/d(xi, eta, zeta) - corner 3
        [0.0, 0.0, 4.0 * zeta - 1.0],
        // dN4/d(xi, eta, zeta) - mid-edge 0-1
        [4.0 * (l4 - xi), -4.0 * xi, -4.0 * xi],
        // dN5/d(xi, eta, zeta) - mid-edge 1-2
        [4.0 * eta, 4.0 * xi, 0.0],
        // dN6/d(xi, eta, zeta) - mid-edge 0-2
        [-4.0 * eta, 4.0 * (l4 - eta), -4.0 * eta],
        // dN7/d(xi, eta, zeta) - mid-edge 0-3
        [-4.0 * zeta, -4.0 * zeta, 4.0 * (l4 - zeta)],
        // dN8/d(xi, eta, zeta) - mid-edge 1-3
        [4.0 * zeta, 0.0, 4.0 * xi],
        // dN9/d(xi, eta, zeta) - mid-edge 2-3
        [0.0, 4.0 * zeta, 4.0 * eta],
    ]
}

/// 4-point Gauss quadrature for tetrahedra
/// Returns (xi, eta, zeta, weight) for each integration point
fn tet_gauss_points_4() -> [(f64, f64, f64, f64); 4] {
    let a = 0.5854101966249685;  // (5 + 3*sqrt(5)) / 20
    let b = 0.1381966011250105;  // (5 - sqrt(5)) / 20
    let w = 1.0 / 24.0;          // weight for each point (volume of reference tet = 1/6)
    [
        (b, b, b, w),
        (a, b, b, w),
        (b, a, b, w),
        (b, b, a, w),
    ]
}

/// Compute TET10 element stiffness matrix using numerical integration
fn tet10_element_stiffness(nodes: &[Point3<f64>; 10], material: &Material) -> (f64, DMatrix<f64>) {
    let d_mat = constitutive_matrix(material);
    let gauss_pts = tet_gauss_points_4();

    let mut ke = DMatrix::<f64>::zeros(30, 30);
    let mut total_volume = 0.0;

    for (xi, eta, zeta, weight) in gauss_pts {
        let dn_dnat = tet10_shape_derivatives(xi, eta, zeta);

        // Build Jacobian matrix J = [dx/dxi, dx/deta, dx/dzeta; ...]
        let mut jacobian = DMatrix::<f64>::zeros(3, 3);
        for i in 0..10 {
            jacobian[(0, 0)] += dn_dnat[i][0] * nodes[i].x;
            jacobian[(0, 1)] += dn_dnat[i][1] * nodes[i].x;
            jacobian[(0, 2)] += dn_dnat[i][2] * nodes[i].x;
            jacobian[(1, 0)] += dn_dnat[i][0] * nodes[i].y;
            jacobian[(1, 1)] += dn_dnat[i][1] * nodes[i].y;
            jacobian[(1, 2)] += dn_dnat[i][2] * nodes[i].y;
            jacobian[(2, 0)] += dn_dnat[i][0] * nodes[i].z;
            jacobian[(2, 1)] += dn_dnat[i][1] * nodes[i].z;
            jacobian[(2, 2)] += dn_dnat[i][2] * nodes[i].z;
        }

        let det_j = jacobian[(0, 0)] * (jacobian[(1, 1)] * jacobian[(2, 2)] - jacobian[(1, 2)] * jacobian[(2, 1)])
                  - jacobian[(0, 1)] * (jacobian[(1, 0)] * jacobian[(2, 2)] - jacobian[(1, 2)] * jacobian[(2, 0)])
                  + jacobian[(0, 2)] * (jacobian[(1, 0)] * jacobian[(2, 1)] - jacobian[(1, 1)] * jacobian[(2, 0)]);

        if det_j.abs() <= f64::EPSILON {
            continue;
        }

        // Inverse Jacobian
        let inv_det = 1.0 / det_j;
        let mut inv_j = DMatrix::<f64>::zeros(3, 3);
        inv_j[(0, 0)] = inv_det * (jacobian[(1, 1)] * jacobian[(2, 2)] - jacobian[(1, 2)] * jacobian[(2, 1)]);
        inv_j[(0, 1)] = inv_det * (jacobian[(0, 2)] * jacobian[(2, 1)] - jacobian[(0, 1)] * jacobian[(2, 2)]);
        inv_j[(0, 2)] = inv_det * (jacobian[(0, 1)] * jacobian[(1, 2)] - jacobian[(0, 2)] * jacobian[(1, 1)]);
        inv_j[(1, 0)] = inv_det * (jacobian[(1, 2)] * jacobian[(2, 0)] - jacobian[(1, 0)] * jacobian[(2, 2)]);
        inv_j[(1, 1)] = inv_det * (jacobian[(0, 0)] * jacobian[(2, 2)] - jacobian[(0, 2)] * jacobian[(2, 0)]);
        inv_j[(1, 2)] = inv_det * (jacobian[(0, 2)] * jacobian[(1, 0)] - jacobian[(0, 0)] * jacobian[(1, 2)]);
        inv_j[(2, 0)] = inv_det * (jacobian[(1, 0)] * jacobian[(2, 1)] - jacobian[(1, 1)] * jacobian[(2, 0)]);
        inv_j[(2, 1)] = inv_det * (jacobian[(0, 1)] * jacobian[(2, 0)] - jacobian[(0, 0)] * jacobian[(2, 1)]);
        inv_j[(2, 2)] = inv_det * (jacobian[(0, 0)] * jacobian[(1, 1)] - jacobian[(0, 1)] * jacobian[(1, 0)]);

        // Transform shape function derivatives to physical coordinates
        // dN/dx_j = Σ_k (dN/dξ_k) * (dξ_k/dx_j) = Σ_k dn_dnat[k] * inv_j[k,j]
        let mut dn_dx = [[0.0f64; 3]; 10];
        for i in 0..10 {
            for j in 0..3 {
                dn_dx[i][j] = inv_j[(0, j)] * dn_dnat[i][0]
                            + inv_j[(1, j)] * dn_dnat[i][1]
                            + inv_j[(2, j)] * dn_dnat[i][2];
            }
        }

        // Build B matrix (6x30)
        let mut b = DMatrix::<f64>::zeros(6, 30);
        for i in 0..10 {
            let col = i * 3;
            let (gx, gy, gz) = (dn_dx[i][0], dn_dx[i][1], dn_dx[i][2]);

            b[(0, col)] = gx;
            b[(1, col + 1)] = gy;
            b[(2, col + 2)] = gz;
            b[(3, col + 1)] = gz;
            b[(3, col + 2)] = gy;
            b[(4, col)] = gz;
            b[(4, col + 2)] = gx;
            b[(5, col)] = gy;
            b[(5, col + 1)] = gx;
        }

        // Integration: K += B^T * D * B * det(J) * weight
        let btd = b.transpose() * &d_mat;
        let btdb = &btd * &b;
        ke += btdb * (det_j.abs() * weight);
        total_volume += det_j.abs() * weight;
    }

    (total_volume, ke)
}

/// Assemble global stiffness and mass matrices for a TET10 mesh
pub fn assemble_tet10_model(mesh: &Tet10Mesh, material: &Material, fixed_nodes: &HashSet<usize>) -> AssembledModel {
    let total_dofs = mesh.nodes.len() * 3;
    let mut dof_map = vec![None; total_dofs];
    let mut free_dofs = 0usize;

    for (node_idx, _) in mesh.nodes.iter().enumerate() {
        let fixed = fixed_nodes.contains(&node_idx);
        for comp in 0..3 {
            let idx = node_idx * 3 + comp;
            if fixed {
                continue;
            }
            dof_map[idx] = Some(free_dofs);
            free_dofs += 1;
        }
    }

    let mut mass_diag = vec![0.0f64; free_dofs];
    let mut coo = CooMatrix::new(free_dofs, free_dofs);

    for element in &mesh.elements {
        let element_nodes: [Point3<f64>; 10] = [
            mesh.nodes[element[0]],
            mesh.nodes[element[1]],
            mesh.nodes[element[2]],
            mesh.nodes[element[3]],
            mesh.nodes[element[4]],
            mesh.nodes[element[5]],
            mesh.nodes[element[6]],
            mesh.nodes[element[7]],
            mesh.nodes[element[8]],
            mesh.nodes[element[9]],
        ];

        let (volume, ke) = tet10_element_stiffness(&element_nodes, material);
        if volume <= f64::EPSILON {
            continue;
        }

        // Lumped mass: distribute element mass equally to all DOFs
        // Total element mass = density * volume
        // 10 nodes * 3 DOFs = 30 DOFs per element
        let mass_per_dof = material.density * volume / 30.0;

        for (local_node, &global_node) in element.iter().enumerate() {
            for comp in 0..3 {
                let global_dof = global_node * 3 + comp;
                if let Some(row) = dof_map[global_dof] {
                    mass_diag[row] += mass_per_dof;
                }
            }

            for (local_other, &global_other) in element.iter().enumerate() {
                let block = ke.view((local_node * 3, local_other * 3), (3, 3));
                for r in 0..3 {
                    for c in 0..3 {
                        let global_r = global_node * 3 + r;
                        let global_c = global_other * 3 + c;
                        if let (Some(row), Some(col)) = (dof_map[global_r], dof_map[global_c]) {
                            let val = block[(r, c)];
                            if val.abs() > 0.0 {
                                coo.push(row, col, val);
                            }
                        }
                    }
                }
            }
        }
    }

    let stiffness = CsrMatrix::from(&coo);
    AssembledModel { stiffness, mass_diag }
}

pub fn solve_modes(model: &AssembledModel, modes: usize) -> ModeSummary {
    let op = MassScaledOperator::new(&model.stiffness, &model.mass_diag);
    let dim = op.ncols().max(1);
    let iterations = dim.min(dim.max(modes * 2 + 4));
    let eigen = op.eigsh(iterations, Order::Smallest);

    let mut frequencies_hz = Vec::new();
    for lambda in eigen.eigenvalues.iter().copied() {
        if lambda <= RIGID_MODE_TOL {
            continue;
        }
        if frequencies_hz.len() >= modes {
            break;
        }
        let freq = lambda.sqrt() / (2.0 * std::f64::consts::PI);
        frequencies_hz.push(freq);
    }

    ModeSummary { frequencies_hz }
}

pub fn demo_modes() -> ModeSummary {
    let mesh = Mesh::regular_cube(1, 1.0);
    let mut fixed = HashSet::new();
    for (idx, node) in mesh.nodes.iter().enumerate() {
        if node.z <= f64::EPSILON {
            fixed.insert(idx);
        }
    }

    let material = Material {
        young_modulus: 210e9,
        poisson_ratio: 0.3,
        density: 7800.0,
    };

    let model = assemble_model(&mesh, &material, &fixed);
    solve_modes(&model, 6)
}

#[wasm_bindgen]
pub fn compute_demo_modes() -> Result<JsValue, JsValue> {
    let summary = demo_modes();
    serde_wasm_bindgen::to_value(&summary).map_err(to_js_error)
}

#[wasm_bindgen]
pub fn compute_modes(
    node_positions: Vec<f64>,
    tets: Vec<u32>,
    fixed_nodes: Vec<u32>,
    young_modulus: f64,
    poisson_ratio: f64,
    density: f64,
    modes: usize,
) -> Result<JsValue, JsValue> {
    if node_positions.len() % 3 != 0 {
        return Err(JsValue::from_str("node_positions length must be a multiple of 3"));
    }
    if tets.len() % 4 != 0 {
        return Err(JsValue::from_str("tets length must be a multiple of 4"));
    }

    let nodes = node_positions
        .chunks_exact(3)
        .map(|c| Point3::new(c[0], c[1], c[2]))
        .collect();

    let elements = tets
        .chunks_exact(4)
        .map(|c| [c[0] as usize, c[1] as usize, c[2] as usize, c[3] as usize])
        .collect();

    let fixed: HashSet<usize> = fixed_nodes.into_iter().map(|v| v as usize).collect();
    let mesh = Mesh { nodes, elements };
    let material = Material {
        young_modulus,
        poisson_ratio,
        density,
    };

    let model = assemble_model(&mesh, &material, &fixed);
    if model.stiffness.ncols() == 0 {
        return Err(JsValue::from_str("no free degrees of freedom after applying constraints"));
    }

    let summary = solve_modes(&model, modes.max(1));
    serde_wasm_bindgen::to_value(&summary).map_err(to_js_error)
}

fn to_js_error(err: impl ToString) -> JsValue {
    JsValue::from_str(&err.to_string())
}

struct MassScaledOperator<'a> {
    stiffness: &'a CsrMatrix<f64>,
    d_inv_sqrt: Vec<f64>,
}

impl<'a> MassScaledOperator<'a> {
    fn new(stiffness: &'a CsrMatrix<f64>, mass_diag: &[f64]) -> Self {
        let d_inv_sqrt = mass_diag
            .iter()
            .map(|m| if *m > 0.0 { 1.0 / m.sqrt() } else { 0.0 })
            .collect();

        Self {
            stiffness,
            d_inv_sqrt,
        }
    }
}

impl<'a> Hermitian<f64> for MassScaledOperator<'a> {
    fn nrows(&self) -> usize {
        self.stiffness.nrows()
    }

    fn ncols(&self) -> usize {
        self.stiffness.ncols()
    }

    fn vector_product(&self, v: nalgebra::DVectorView<f64>) -> DVector<f64> {
        let mut scaled = DVector::<f64>::zeros(v.len());
        for (i, val) in v.iter().enumerate() {
            scaled[i] = val * self.d_inv_sqrt[i];
        }

        let mut out = self.stiffness * scaled;
        for i in 0..out.len() {
            out[i] *= self.d_inv_sqrt[i];
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn demo_returns_positive_modes() {
        let result = demo_modes();
        assert!(!result.frequencies_hz.is_empty());
        assert!(result.frequencies_hz[0].is_finite());
        assert!(result.frequencies_hz.iter().all(|f| *f > 0.0));
    }

    #[test]
    fn sapele_bar_450x32x24_free_free_modes() {
        // Sapele wood material properties (from wood-database.com)
        let material = Material {
            young_modulus: 12.35e9,  // 12.35 GPa along grain
            poisson_ratio: 0.35,
            density: 665.0,          // kg/m³
        };

        // Bar dimensions: 450mm x 32mm x 24mm (converted to meters)
        // X = length (450mm), Y = width (32mm), Z = thickness (24mm)
        let size = [0.450, 0.032, 0.024];

        // Mesh divisions - linear tets need fine meshes for bending accuracy
        // Note: Linear tetrahedra exhibit shear locking in bending, producing
        // frequencies higher than analytical beam theory.
        let divisions = [30, 4, 4];

        let mesh = Mesh::regular_box(divisions, size);

        // Free-free boundary conditions (no fixed nodes)
        // Real xylophone bars are supported at nodal points (~22.4% from each end)
        let fixed = HashSet::new();

        let model = assemble_model(&mesh, &material, &fixed);
        let result = solve_modes(&model, 6);

        // Free-free bar should have 6 rigid body modes filtered out,
        // then flexible modes starting with bending modes
        assert!(!result.frequencies_hz.is_empty(), "Should have flexible modes");
        assert!(result.frequencies_hz.iter().all(|f| f.is_finite()), "All frequencies should be finite");
        assert!(result.frequencies_hz.iter().all(|f| *f > 0.0), "All frequencies should be positive");

        // Analytical first bending frequency (Euler-Bernoulli beam theory):
        // f1 = (4.73^2 / 2π) * sqrt(E*I / (ρ*A*L^4)) ≈ 525 Hz with correct properties
        // Linear tets will be stiffer, giving higher frequencies.
        let first_flexible_freq = result.frequencies_hz.iter()
            .find(|&&f| f > 1.0)
            .copied()
            .unwrap_or(0.0);

        // With this mesh, expect first mode around 1000-1200 Hz due to tet stiffness
        assert!(first_flexible_freq > 500.0, "First flexible mode should be above 500 Hz");
        assert!(first_flexible_freq < 2000.0, "First flexible mode should be below 2000 Hz");

        // Print frequencies for debugging/verification
        println!("Sapele bar 450x32x24mm free-free mode frequencies (TET4):");
        println!("(Note: Linear tets overestimate stiffness; analytical f1 ≈ 525 Hz)");
        for (i, freq) in result.frequencies_hz.iter().enumerate() {
            println!("  Mode {}: {:.2} Hz", i + 1, freq);
        }
    }

    #[test]
    fn sapele_bar_450x32x24_tet10_modes() {
        // Sapele wood material properties (from wood-database.com)
        let material = Material {
            young_modulus: 12.35e9,  // 12.35 GPa along grain
            poisson_ratio: 0.35,
            density: 665.0,          // kg/m³
        };

        // Bar dimensions: 450mm x 32mm x 24mm (converted to meters)
        let size = [0.450, 0.032, 0.024];

        // TET10 needs fewer elements than TET4 for similar accuracy
        let divisions = [18, 2, 2];

        let mesh = Tet10Mesh::regular_box(divisions, size);

        // Free-free boundary conditions (no fixed nodes)
        let fixed = HashSet::new();

        let model = assemble_tet10_model(&mesh, &material, &fixed);
        let result = solve_modes(&model, 6);

        // Free-free bar should have 6 rigid body modes filtered out,
        // then flexible modes starting with bending modes
        assert!(!result.frequencies_hz.is_empty(), "Should have flexible modes");
        assert!(result.frequencies_hz.iter().all(|f| f.is_finite()), "All frequencies should be finite");
        assert!(result.frequencies_hz.iter().all(|f| *f > 0.0), "All frequencies should be positive");

        // Analytical first bending frequency (Euler-Bernoulli beam theory):
        // f1 = (4.73^2 / 2π) * sqrt(E*I / (ρ*A*L^4)) ≈ 525 Hz with correct properties
        // TET10 should be much closer to analytical than TET4
        let first_flexible_freq = result.frequencies_hz.iter()
            .find(|&&f| f > 1.0)
            .copied()
            .unwrap_or(0.0);

        // TET10 gives better accuracy than TET4
        assert!(first_flexible_freq > 500.0, "First flexible mode should be above 500 Hz");
        assert!(first_flexible_freq < 1500.0, "First flexible mode should be below 1500 Hz");

        // Print frequencies for verification
        println!("Sapele bar 450x32x24mm free-free mode frequencies (TET10):");
        println!("(Analytical f1 ≈ 525 Hz)");
        for (i, freq) in result.frequencies_hz.iter().enumerate() {
            println!("  Mode {}: {:.2} Hz", i + 1, freq);
        }
    }

    #[test]
    fn sapele_bar_450x32x24_opposing_corners_constrained() {
        // Sapele wood material properties (from Soares 2021 / wood-database.com)
        let material = Material {
            young_modulus: 12.35e9,  // 12.35 GPa along grain
            poisson_ratio: 0.35,
            density: 665.0,          // kg/m³
        };

        // Bar dimensions: 450mm x 32mm x 24mm (converted to meters)
        let size = [0.450, 0.032, 0.024];

        // TET10 mesh
        let divisions = [18, 2, 2];
        let mesh = Tet10Mesh::regular_box(divisions, size);

        // Constrain opposing corners as described in Soares 2021
        // Corner 1: (0, 0, 0)
        // Opposing corner: (0.45, 0.032, 0.024)
        let mut fixed = HashSet::new();
        let tol = 1e-9;

        for (idx, node) in mesh.nodes.iter().enumerate() {
            // Check for corner at origin (0, 0, 0)
            if node.x.abs() < tol && node.y.abs() < tol && node.z.abs() < tol {
                fixed.insert(idx);
                println!("Fixed node {} at origin: ({:.4}, {:.4}, {:.4})", idx, node.x, node.y, node.z);
            }
            // Check for opposing corner at (0.45, 0.032, 0.024)
            if (node.x - size[0]).abs() < tol
                && (node.y - size[1]).abs() < tol
                && (node.z - size[2]).abs() < tol {
                fixed.insert(idx);
                println!("Fixed node {} at far corner: ({:.4}, {:.4}, {:.4})", idx, node.x, node.y, node.z);
            }
        }

        assert_eq!(fixed.len(), 2, "Should have exactly 2 fixed nodes (opposing corners)");

        let model = assemble_tet10_model(&mesh, &material, &fixed);
        let result = solve_modes(&model, 6);

        assert!(!result.frequencies_hz.is_empty(), "Should have flexible modes");
        assert!(result.frequencies_hz.iter().all(|f| f.is_finite()), "All frequencies should be finite");

        // With opposing corners fixed, there's one remaining rotational DOF about the diagonal
        // axis, giving one near-zero frequency. The first bending mode follows.
        let first_flexible_freq = result.frequencies_hz.iter()
            .find(|&&f| f > 10.0)  // Skip near-rigid modes
            .copied()
            .unwrap_or(0.0);

        println!("Sapele bar 450x32x24mm with opposing corners constrained (TET10):");
        println!("(Boundary condition: opposing corners fixed - first bending ~384 Hz)");
        for (i, freq) in result.frequencies_hz.iter().enumerate() {
            println!("  Mode {}: {:.2} Hz", i + 1, freq);
        }

        // First bending mode should be close to expected ~400 Hz
        assert!(first_flexible_freq > 300.0, "First bending mode should be above 300 Hz");
        assert!(first_flexible_freq < 500.0, "First bending mode should be below 500 Hz");
    }

    #[test]
    fn sapele_bar_551x32x24_opposing_corners_constrained() {
        // Sapele wood material properties
        let material = Material {
            young_modulus: 12.35e9,  // 12.35 GPa along grain
            poisson_ratio: 0.35,
            density: 665.0,          // kg/m³
        };

        // Bar dimensions: 551mm x 32mm x 24mm (converted to meters)
        // Expected first bending frequency: ~350 Hz
        let size = [0.551, 0.032, 0.024];

        // TET10 mesh - scale divisions proportionally to length
        let divisions = [22, 2, 2];
        let mesh = Tet10Mesh::regular_box(divisions, size);

        // Constrain opposing corners
        let mut fixed = HashSet::new();
        let tol = 1e-9;

        for (idx, node) in mesh.nodes.iter().enumerate() {
            if node.x.abs() < tol && node.y.abs() < tol && node.z.abs() < tol {
                fixed.insert(idx);
            }
            if (node.x - size[0]).abs() < tol
                && (node.y - size[1]).abs() < tol
                && (node.z - size[2]).abs() < tol {
                fixed.insert(idx);
            }
        }

        assert_eq!(fixed.len(), 2, "Should have exactly 2 fixed nodes (opposing corners)");

        let model = assemble_tet10_model(&mesh, &material, &fixed);
        let result = solve_modes(&model, 6);

        let first_flexible_freq = result.frequencies_hz.iter()
            .find(|&&f| f > 10.0)
            .copied()
            .unwrap_or(0.0);

        println!("Sapele bar 551x32x24mm with opposing corners constrained (TET10):");
        println!("(Expected first bending: ~350 Hz)");
        for (i, freq) in result.frequencies_hz.iter().enumerate() {
            println!("  Mode {}: {:.2} Hz", i + 1, freq);
        }

        // First bending mode should be close to expected ~350 Hz
        assert!(first_flexible_freq > 250.0, "First bending mode should be above 250 Hz");
        assert!(first_flexible_freq < 450.0, "First bending mode should be below 450 Hz");
    }
}
