use std::collections::HashSet;

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

#[derive(Clone, Debug)]
pub struct Mesh {
    pub nodes: Vec<Point3<f64>>,
    pub elements: Vec<[usize; 4]>,
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

        b[(3, col)] = *gy;
        b[(3, col + 1)] = *gx;

        b[(4, col + 1)] = *gz;
        b[(4, col + 2)] = *gy;

        b[(5, col)] = *gz;
        b[(5, col + 2)] = *gx;
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
        fn element_stiffness_matches_constitutive_order() {
            let material = Material {
                young_modulus: 1.0,
                poisson_ratio: 0.25,
                density: 1.0,
            };

            let nodes = [
                Point3::new(0.0, 0.0, 0.0),
                Point3::new(1.0, 0.0, 0.0),
                Point3::new(0.0, 1.0, 0.0),
                Point3::new(0.0, 0.0, 1.0),
            ];

            let grads = [
                (-1.0, -1.0, -1.0),
                (1.0, 0.0, 0.0),
                (0.0, 1.0, 0.0),
                (0.0, 0.0, 1.0),
            ];

            // Strain ordering matches [exx, eyy, ezz, gxy, gyz, gxz].
            let mut expected_b = DMatrix::<f64>::zeros(6, 12);
            for (i, (gx, gy, gz)) in grads.iter().copied().enumerate() {
                let col = i * 3;
                expected_b[(0, col)] = gx;
                expected_b[(1, col + 1)] = gy;
                expected_b[(2, col + 2)] = gz;
                expected_b[(3, col)] = gy;
                expected_b[(3, col + 1)] = gx;
                expected_b[(4, col + 1)] = gz;
                expected_b[(4, col + 2)] = gy;
                expected_b[(5, col)] = gz;
                expected_b[(5, col + 2)] = gx;
            }

            let expected_volume = 1.0 / 6.0;
            let expected_ke = expected_b.transpose() * constitutive_matrix(&material) * expected_b * expected_volume;

            let (volume, ke) = element_stiffness(&nodes, &material);
            assert!((volume - expected_volume).abs() < 1e-12);

            let max_err = ke
                .iter()
                .zip(expected_ke.iter())
                .fold(0.0f64, |acc, (a, b)| acc.max((a - b).abs()));
            assert!(max_err < 1e-10, "max_err was {}", max_err);
        }
    }
