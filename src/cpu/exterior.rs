use super::{CpuDomain, CpuState, CpuTensor, CpuTopology};
use crate::model::Domain;
use crate::numerics::{TensorFaceState, check_transmissive};
use crate::solver::{
    Axis, BoundaryFace, BoundaryForcing, EngineError, ForcingInterval, SolverConfig,
    TransmissiveDiagnostics,
};
use numr::ops::{BinaryOps, ConditionalOps, IndexingOps};
use numr::runtime::cpu::{CpuClient, CpuDevice};
use std::collections::BTreeMap;

#[derive(Debug)]
pub(crate) struct CpuExterior {
    left: GhostSide,
    right: GhostSide,
    faces: Vec<BoundaryFace>,
    cell_ids: CpuTensor,
    signs: CpuTensor,
    x_mask: CpuTensor,
}

#[derive(Debug)]
struct GhostSide {
    count: usize,
    ids: CpuTensor,
    h: CpuTensor,
    normal: CpuTensor,
    tangent: CpuTensor,
}

#[derive(Default)]
struct GhostValues {
    ids: Vec<i64>,
    h: Vec<f64>,
    normal: Vec<f64>,
    tangent: Vec<f64>,
}

impl GhostValues {
    fn cache(self, device: &CpuDevice) -> Result<GhostSide, EngineError> {
        let count = self.ids.len();
        Ok(GhostSide {
            count,
            ids: CpuTensor::from_slice(&self.ids, &[count], device)?,
            h: tensor(&self.h, device)?,
            normal: tensor(&self.normal, device)?,
            tangent: tensor(&self.tangent, device)?,
        })
    }
}

impl GhostSide {
    fn replace(
        &self,
        client: &CpuClient,
        state: TensorFaceState,
    ) -> Result<TensorFaceState, EngineError> {
        if self.count == 0 {
            return Ok(state);
        }
        Ok(TensorFaceState {
            h: client.index_put(&state.h, 0, &self.ids, &self.h)?,
            q_normal: client.index_put(&state.q_normal, 0, &self.ids, &self.normal)?,
            q_tangent: client.index_put(&state.q_tangent, 0, &self.ids, &self.tangent)?,
            bed: state.bed,
        })
    }
}

impl CpuExterior {
    pub(crate) fn new(
        domain: &Domain,
        topology: &CpuTopology,
        forcing: &ForcingInterval,
        device: &CpuDevice,
    ) -> Result<Self, EngineError> {
        let mut left = GhostValues::default();
        let mut right = GhostValues::default();
        let mut faces = Vec::new();
        let mut cells = Vec::new();
        let mut signs = Vec::new();
        let mut x_mask = Vec::new();
        let selected: Vec<_> = forcing
            .boundaries
            .iter()
            .filter(|value| {
                matches!(
                    value,
                    BoundaryForcing::Exterior(_) | BoundaryForcing::Transmissive(_)
                )
            })
            .collect();
        let supports: BTreeMap<_, _> = if selected.is_empty() {
            BTreeMap::new()
        } else {
            topology
                .faces
                .iter()
                .enumerate()
                .map(|(i, support)| (support.face, (i, support)))
                .collect()
        };
        for value in selected {
            let face = value.face();
            let &(index, support) = supports.get(&face).ok_or_else(|| {
                EngineError::rejected(
                    "CPU exterior",
                    format!("face={face:?}: require cached exterior support"),
                )
            })?;
            let sign = if support.left.is_some() { 1.0 } else { -1.0 };
            match value {
                BoundaryForcing::Exterior(exterior) => {
                    let [h, normal, tangent] = exterior.conserved(domain)?;
                    let values = if support.left.is_none() {
                        &mut left
                    } else {
                        &mut right
                    };
                    values.ids.push(index_i64(index)?);
                    values.h.push(h);
                    values.normal.push(sign * normal);
                    values.tangent.push(tangent);
                }
                BoundaryForcing::Transmissive(_) => {
                    let cell = support.left.or(support.right).ok_or_else(|| {
                        EngineError::rejected(
                            "CPU exterior",
                            format!("face={face:?}: require active support"),
                        )
                    })?;
                    faces.push(face);
                    cells.push(index_i64(cell)?);
                    signs.push(sign);
                    x_mask.push(if face.axis == Axis::X { 1.0 } else { 0.0 });
                }
                _ => {}
            }
        }
        Ok(Self {
            left: left.cache(device)?,
            right: right.cache(device)?,
            faces,
            cell_ids: CpuTensor::from_slice(&cells, &[cells.len()], device)?,
            signs: tensor(&signs, device)?,
            x_mask: tensor(&x_mask, device)?,
        })
    }

    pub(crate) fn prepare(
        &self,
        cpu: &CpuDomain,
        state: &CpuState,
        left: TensorFaceState,
        right: TensorFaceState,
        config: &SolverConfig,
    ) -> Result<(TensorFaceState, TensorFaceState, Option<TensorFaceState>), EngineError> {
        let client = &cpu.client;
        let witness = if self.faces.is_empty() {
            None
        } else {
            let gather = |field| client.index_select(field, 0, &self.cell_ids);
            let qx = gather(&state.qx)?;
            let qy = gather(&state.qy)?;
            let selected = TensorFaceState {
                h: gather(&state.h)?,
                q_normal: client.mul(&client.where_cond(&self.x_mask, &qx, &qy)?, &self.signs)?,
                q_tangent: client.where_cond(&self.x_mask, &qy, &qx)?,
                bed: gather(&cpu.bed)?,
            };
            check_transmissive(client, &selected, &self.faces, config)?;
            Some(selected)
        };
        Ok((
            self.left.replace(client, left)?,
            self.right.replace(client, right)?,
            witness,
        ))
    }

    pub(crate) fn results(
        &self,
        witness: Option<&TensorFaceState>,
    ) -> Result<Vec<TransmissiveDiagnostics>, EngineError> {
        let Some(state) = witness else {
            return Ok(Vec::new());
        };
        let h = state.h.try_to_vec::<f64>()?;
        let q = state.q_normal.try_to_vec::<f64>()?;
        let tangent = state.q_tangent.try_to_vec::<f64>()?;
        if [h.len(), q.len(), tangent.len()]
            .iter()
            .any(|length| *length != self.faces.len())
        {
            return Err(EngineError::rejected(
                "CPU transmissive output",
                "tensor projection differs from cached support: inspect runtime output shapes",
            ));
        }
        Ok(self
            .faces
            .iter()
            .enumerate()
            .map(|(i, face)| TransmissiveDiagnostics {
                face: *face,
                depth: h[i],
                outward_discharge_per_width: q[i],
                tangential_discharge_per_width: tangent[i],
            })
            .collect())
    }
}

fn tensor(values: &[f64], device: &CpuDevice) -> Result<CpuTensor, EngineError> {
    Ok(CpuTensor::from_slice(values, &[values.len()], device)?)
}

fn index_i64(index: usize) -> Result<i64, EngineError> {
    i64::try_from(index).map_err(|_| {
        EngineError::rejected(
            "CPU exterior",
            format!("index={index}: reduce grid dimensions below i64 index range"),
        )
    })
}
