use super::boundary_trace::check_transmissive;
use super::{TensorDomain, TensorState, TensorTopology};
use crate::contract::{
    Axis, BoundaryFace, BoundaryForcing, EngineError, ForcingInterval, SolverConfig,
};
use crate::model::Domain;
use crate::numerics::TensorFaceState;
use crate::solver::TransmissiveDiagnostics;
use numr::dtype::DType;
use numr::ops::{BinaryOps, ConditionalOps, IndexingOps};
use numr::runtime::Runtime;
use numr::tensor::Tensor;
use std::collections::BTreeMap;

type PreparedExterior<State> = (State, State, Option<State>);

#[derive(Debug)]
pub(crate) struct TensorExterior<R: Runtime<DType = DType>> {
    left: GhostSide<R>,
    right: GhostSide<R>,
    faces: Vec<BoundaryFace>,
    cell_ids: Tensor<R>,
    signs: Tensor<R>,
    x_mask: Tensor<R>,
}

#[derive(Debug)]
struct GhostSide<R: Runtime<DType = DType>> {
    count: usize,
    ids: Tensor<R>,
    h: Tensor<R>,
    normal: Tensor<R>,
    tangent: Tensor<R>,
}

#[derive(Default)]
struct GhostValues {
    ids: Vec<i64>,
    h: Vec<f64>,
    normal: Vec<f64>,
    tangent: Vec<f64>,
}

impl GhostValues {
    fn cache<R: Runtime<DType = DType>>(
        self,
        device: &R::Device,
    ) -> Result<GhostSide<R>, EngineError> {
        let count = self.ids.len();
        Ok(GhostSide {
            count,
            ids: Tensor::<R>::from_slice(&self.ids, &[count], device)?,
            h: tensor(&self.h, device)?,
            normal: tensor(&self.normal, device)?,
            tangent: tensor(&self.tangent, device)?,
        })
    }
}

impl<R: Runtime<DType = DType>> GhostSide<R>
where
    R::Client: IndexingOps<R>,
{
    fn replace(
        &self,
        client: &R::Client,
        state: TensorFaceState<R>,
    ) -> Result<TensorFaceState<R>, EngineError> {
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

impl<R: Runtime<DType = DType>> TensorExterior<R> {
    pub(crate) fn new(
        domain: &Domain,
        topology: &TensorTopology<R>,
        forcing: &ForcingInterval,
        device: &R::Device,
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
                    "tensor exterior",
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
                            "tensor exterior",
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
            cell_ids: Tensor::<R>::from_slice(&cells, &[cells.len()], device)?,
            signs: tensor(&signs, device)?,
            x_mask: tensor(&x_mask, device)?,
        })
    }
}

impl<R: Runtime<DType = DType>> TensorExterior<R>
where
    R::Client: BinaryOps<R>
        + ConditionalOps<R>
        + IndexingOps<R>
        + numr::ops::CompareOps<R>
        + numr::ops::ReduceOps<R>
        + numr::ops::ScalarOps<R>
        + numr::ops::UnaryOps<R>,
{
    pub(crate) fn prepare(
        &self,
        context: &TensorDomain<R>,
        state: &TensorState<R>,
        left: TensorFaceState<R>,
        right: TensorFaceState<R>,
        config: &SolverConfig,
    ) -> Result<PreparedExterior<TensorFaceState<R>>, EngineError> {
        let client = &context.client;
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
                bed: gather(&context.bed)?,
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
        witness: Option<&TensorFaceState<R>>,
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
                "tensor transmissive output",
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

fn tensor<R: Runtime<DType = DType>>(
    values: &[f64],
    device: &R::Device,
) -> Result<Tensor<R>, EngineError> {
    Ok(Tensor::<R>::from_slice(values, &[values.len()], device)?)
}

fn index_i64(index: usize) -> Result<i64, EngineError> {
    i64::try_from(index).map_err(|_| {
        EngineError::rejected(
            "tensor exterior",
            format!("index={index}: reduce grid dimensions below i64 index range"),
        )
    })
}
