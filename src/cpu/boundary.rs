use super::{CpuDomain, CpuState, CpuTensor, CpuTopology};
use crate::model::Domain;
use crate::numerics::{
    TensorBoundaryRequest, TensorBoundaryTrace, TensorFaceState, characteristic_trace,
};
use crate::solver::{
    Axis, BoundaryFace, BoundaryForcing, BoundaryTraceDiagnostics, EngineError, ForcingInterval,
    SolverConfig,
};
use numr::ops::{BinaryOps, ConditionalOps, IndexingOps, ScatterReduceOp};
use numr::runtime::cpu::CpuDevice;
use std::collections::BTreeMap;

#[derive(Debug)]
pub(crate) struct CpuBoundaries {
    pub faces: Vec<BoundaryFace>,
    pub face_ids: CpuTensor,
    cell_ids: CpuTensor,
    signs: CpuTensor,
    x_mask: CpuTensor,
    widths: CpuTensor,
    request: TensorBoundaryRequest,
}

pub(crate) struct CpuBoundaryTrace {
    pub trace: TensorBoundaryTrace,
}

impl CpuBoundaries {
    pub(crate) fn new(
        domain: &Domain,
        topology: &CpuTopology,
        forcing: &ForcingInterval,
        device: &CpuDevice,
    ) -> Result<Self, EngineError> {
        let mut faces = Vec::new();
        let mut face_ids = Vec::new();
        let mut cell_ids = Vec::new();
        let mut signs = Vec::new();
        let mut x_mask = Vec::new();
        let mut widths = Vec::new();
        let mut stage = Vec::new();
        let mut discharge = Vec::new();
        let mut kinds = Vec::new();
        let mut tangent = Vec::new();
        let mut has_tangent = Vec::new();
        let (dx, dy) = domain.grid().spacing();
        let supports: BTreeMap<_, _> = if forcing
            .boundaries
            .iter()
            .any(|value| !matches!(value, BoundaryForcing::Flux(_)))
        {
            topology
                .faces
                .iter()
                .enumerate()
                .map(|(index, support)| (support.face, (index, support)))
                .collect()
        } else {
            BTreeMap::new()
        };
        for value in &forcing.boundaries {
            if matches!(value, BoundaryForcing::Flux(_)) {
                continue;
            }
            let face = value.face();
            let &(index, support) = supports.get(&face).ok_or_else(|| {
                EngineError::rejected(
                    "CPU boundary",
                    format!("face={face:?}: require cached exterior support"),
                )
            })?;
            let cell = support.left.or(support.right).ok_or_else(|| {
                EngineError::rejected(
                    "CPU boundary",
                    format!("face={face:?}: require active support"),
                )
            })?;
            let width = if face.axis == Axis::X { dy } else { dx };
            let (depth, rate, kind, incoming) = match value {
                BoundaryForcing::Stage(stage) => (
                    stage.elevation - domain.bed()[cell],
                    0.0,
                    0.0,
                    stage.inflow_tangential_velocity,
                ),
                BoundaryForcing::Discharge(discharge) => (
                    1.0,
                    discharge.outward_rate / width,
                    1.0,
                    discharge.inflow_tangential_velocity,
                ),
                BoundaryForcing::Flux(_) => continue,
            };
            if !rate.is_finite()
                || (matches!(value, BoundaryForcing::Discharge(item) if item.outward_rate != 0.0)
                    && rate == 0.0)
            {
                return Err(EngineError::rejected(
                    "CPU boundary",
                    format!("face={face:?}, width={width}: rescale the prescribed discharge"),
                ));
            }
            faces.push(face);
            face_ids.push(i64::try_from(index).map_err(|_| {
                EngineError::rejected("CPU boundary", "face index exceeds i64: reduce grid size")
            })?);
            cell_ids.push(i64::try_from(cell).map_err(|_| {
                EngineError::rejected("CPU boundary", "cell index exceeds i64: reduce grid size")
            })?);
            signs.push(if support.left.is_some() { 1.0 } else { -1.0 });
            x_mask.push(if face.axis == Axis::X { 1.0 } else { 0.0 });
            widths.push(width);
            stage.push(depth);
            discharge.push(rate);
            kinds.push(kind);
            tangent.push(incoming.unwrap_or(0.0));
            has_tangent.push(if incoming.is_some() { 1.0 } else { 0.0 });
        }
        let tensor = |values: &[f64]| {
            CpuTensor::from_slice(values, &[values.len()], device).map_err(EngineError::from)
        };
        Ok(Self {
            faces,
            face_ids: CpuTensor::from_slice(&face_ids, &[face_ids.len()], device)?,
            cell_ids: CpuTensor::from_slice(&cell_ids, &[cell_ids.len()], device)?,
            signs: tensor(&signs)?,
            x_mask: tensor(&x_mask)?,
            widths: tensor(&widths)?,
            request: TensorBoundaryRequest {
                stage_depth: tensor(&stage)?,
                outward_discharge: tensor(&discharge)?,
                discharge_mask: tensor(&kinds)?,
                inflow_tangent: tensor(&tangent)?,
                has_inflow_tangent: tensor(&has_tangent)?,
            },
        })
    }

    pub(crate) fn calculate(
        &self,
        cpu: &CpuDomain,
        state: &CpuState,
        config: &SolverConfig,
    ) -> Result<Option<CpuBoundaryTrace>, EngineError> {
        if self.faces.is_empty() {
            return Ok(None);
        }
        let client = &cpu.client;
        let gather = |value| client.index_select(value, 0, &self.cell_ids);
        let qx = gather(&state.qx)?;
        let qy = gather(&state.qy)?;
        let interior = TensorFaceState {
            h: gather(&state.h)?,
            q_normal: client.mul(&client.where_cond(&self.x_mask, &qx, &qy)?, &self.signs)?,
            q_tangent: client.where_cond(&self.x_mask, &qy, &qx)?,
            bed: gather(&cpu.bed)?,
        };
        Ok(Some(CpuBoundaryTrace {
            trace: characteristic_trace(
                client,
                &interior,
                &self.request,
                &self.faces,
                config.gravity,
                config,
            )?,
        }))
    }

    pub(crate) fn override_flux(
        &self,
        cpu: &CpuDomain,
        result: &CpuBoundaryTrace,
        zero: &CpuTensor,
    ) -> Result<([CpuTensor; 3], CpuTensor), EngineError> {
        let client = &cpu.client;
        let scatter = |values: &CpuTensor| {
            client.scatter_reduce(zero, 0, &self.face_ids, values, ScatterReduceOp::Sum, true)
        };
        let flux = &result.trace.flux;
        let mass = client.mul(&flux[0], &self.signs)?;
        let tangent = client.mul(&flux[2], &self.signs)?;
        Ok((
            [
                scatter(&mass)?,
                scatter(&client.where_cond(&self.x_mask, &flux[1], &tangent)?)?,
                scatter(&client.where_cond(&self.x_mask, &tangent, &flux[1])?)?,
            ],
            scatter(&result.trace.speed)?,
        ))
    }

    pub(crate) fn results(
        &self,
        cpu: &CpuDomain,
        result: &CpuBoundaryTrace,
    ) -> Result<Vec<BoundaryTraceDiagnostics>, EngineError> {
        let client = &cpu.client;
        let trace = &result.trace;
        let depth = trace.h.try_to_vec::<f64>()?;
        let normal = client.div(&trace.q_normal, &trace.h)?.try_to_vec::<f64>()?;
        let tangent = trace.tangential_velocity.try_to_vec::<f64>()?;
        let rate = client
            .mul(&trace.q_normal, &self.widths)?
            .try_to_vec::<f64>()?;
        let discharge = trace.discharge_residual.try_to_vec::<f64>()?;
        let characteristic = trace.characteristic_residual.try_to_vec::<f64>()?;
        let speed = trace.speed.try_to_vec::<f64>()?;
        let count = self.faces.len();
        for (field, length) in [
            ("depth", depth.len()),
            ("normal velocity", normal.len()),
            ("tangential velocity", tangent.len()),
            ("water rate", rate.len()),
            ("discharge residual", discharge.len()),
            ("characteristic residual", characteristic.len()),
            ("signal speed", speed.len()),
        ] {
            if length != count {
                return Err(EngineError::rejected(
                    "CPU boundary output",
                    format!(
                        "field={field}, length={length}: require {count} values for cached physical faces"
                    ),
                ));
            }
        }
        Ok(self
            .faces
            .iter()
            .enumerate()
            .map(|(i, face)| BoundaryTraceDiagnostics {
                face: *face,
                depth: depth[i],
                normal_velocity: normal[i],
                tangential_velocity: tangent[i],
                outward_water_rate: rate[i],
                discharge_residual_per_width: discharge[i],
                characteristic_velocity_residual: characteristic[i],
                signal_speed: speed[i],
            })
            .collect())
    }
}
