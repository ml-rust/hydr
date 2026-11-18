//! Coordinator port faces: ghost-side states before the HLL flux, and per-port flux receipts.
//!
//! A port face has one active surface cell. Its other side holds a ghost state from a reach
//! end. Lanes run in port order, then along each port's face line.

use super::TensorDomain;
use crate::contract::EngineError;
use crate::numerics::TensorFaceState;
use crate::numerics::tensor_types::TensorFaceFlux;
use numr::dtype::DType;
use numr::ops::{
    BinaryOps, CompareOps, ConditionalOps, IndexingOps, ReduceOps, ScalarOps, ScatterReduceOp,
    UnaryOps,
};
use numr::runtime::Runtime;
use numr::tensor::Tensor;

/// One port face as the device topology needs it.
#[derive(Clone, Copy, Debug)]
pub(crate) struct PortFaceLane {
    /// Port lane of this face.
    pub port: usize,
    /// Index into the transport face list.
    pub face_index: usize,
    /// The ghost is the left side of the face, so water entering the surface moves along `+axis`.
    pub ghost_left: bool,
    /// Sign that maps reach-positive discharge onto the face's positive axis.
    pub reach_sign: f64,
    /// Ghost bed elevation in m.
    pub bed: f64,
    /// Face width in m.
    pub width: f64,
    /// The face normal is the x axis.
    pub x_axis: bool,
}

/// Face lanes whose ghost lies on one side of their faces.
#[derive(Debug)]
struct GhostLanes<R: Runtime<DType = DType>> {
    count: usize,
    face_ids: Tensor<R>,
    lanes: Tensor<R>,
}

impl<R: Runtime<DType = DType>> GhostLanes<R>
where
    R::Client: IndexingOps<R>,
{
    fn new(values: &[(i64, i64)], device: &R::Device) -> Result<Self, EngineError> {
        let faces: Vec<i64> = values.iter().map(|value| value.0).collect();
        let lanes: Vec<i64> = values.iter().map(|value| value.1).collect();
        Ok(Self {
            count: values.len(),
            face_ids: Tensor::<R>::from_slice(&faces, &[faces.len()], device)?,
            lanes: Tensor::<R>::from_slice(&lanes, &[lanes.len()], device)?,
        })
    }

    /// Writes the ghost lanes of `ghost` into the matching faces of `state`, bed included.
    fn replace(
        &self,
        client: &R::Client,
        state: TensorFaceState<R>,
        ghost: &TensorFaceState<R>,
    ) -> Result<TensorFaceState<R>, EngineError> {
        if self.count == 0 {
            return Ok(state);
        }
        let put = |target: &Tensor<R>, source: &Tensor<R>| -> Result<Tensor<R>, EngineError> {
            let values = client.index_select(source, 0, &self.lanes)?;
            Ok(client.index_put(target, 0, &self.face_ids, &values)?)
        };
        Ok(TensorFaceState {
            h: put(&state.h, &ghost.h)?,
            q_normal: put(&state.q_normal, &ghost.q_normal)?,
            q_tangent: put(&state.q_tangent, &ghost.q_tangent)?,
            bed: put(&state.bed, &ghost.bed)?,
        })
    }
}

/// Device topology of every port face, in lane order.
#[derive(Debug)]
pub(crate) struct PortFaces<R: Runtime<DType = DType>> {
    face_ids: Tensor<R>,
    port_of_face: Tensor<R>,
    left: GhostLanes<R>,
    right: GhostLanes<R>,
    /// `1.0` where the ghost is the left side, else `0.0`.
    ghost_left: Tensor<R>,
    /// `+1` where water entering the surface moves along `+axis`, else `-1`.
    into: Tensor<R>,
    reach_sign: Tensor<R>,
    bed: Tensor<R>,
    widths: Tensor<R>,
    x_mask: Tensor<R>,
    port_zero: Tensor<R>,
}

impl<R: Runtime<DType = DType>> PortFaces<R>
where
    R::Client: IndexingOps<R>,
{
    pub(crate) fn new(
        lanes: &[PortFaceLane],
        ports: usize,
        device: &R::Device,
    ) -> Result<Self, EngineError> {
        let index = |value: usize| {
            i64::try_from(value).map_err(|_| {
                EngineError::rejected(
                    "surface ports",
                    format!("index={value} exceeds I64 indexing: reduce grid dimensions"),
                )
            })
        };
        let mut face_ids = Vec::with_capacity(lanes.len());
        let mut port_of_face = Vec::with_capacity(lanes.len());
        let mut left = Vec::new();
        let mut right = Vec::new();
        for (lane, value) in lanes.iter().enumerate() {
            let face = index(value.face_index)?;
            face_ids.push(face);
            port_of_face.push(index(value.port)?);
            if value.ghost_left {
                left.push((face, index(lane)?));
            } else {
                right.push((face, index(lane)?));
            }
        }
        let tensor = |values: Vec<f64>| -> Result<Tensor<R>, EngineError> {
            Ok(Tensor::<R>::from_slice(&values, &[values.len()], device)?)
        };
        let select = |f: fn(&PortFaceLane) -> f64| lanes.iter().map(f).collect::<Vec<_>>();
        Ok(Self {
            face_ids: Tensor::<R>::from_slice(&face_ids, &[face_ids.len()], device)?,
            port_of_face: Tensor::<R>::from_slice(&port_of_face, &[port_of_face.len()], device)?,
            left: GhostLanes::new(&left, device)?,
            right: GhostLanes::new(&right, device)?,
            ghost_left: tensor(select(|lane| if lane.ghost_left { 1.0 } else { 0.0 }))?,
            into: tensor(select(|lane| if lane.ghost_left { 1.0 } else { -1.0 }))?,
            reach_sign: tensor(select(|lane| lane.reach_sign))?,
            bed: tensor(select(|lane| lane.bed))?,
            widths: tensor(select(|lane| lane.width))?,
            x_mask: tensor(select(|lane| if lane.x_axis { 1.0 } else { 0.0 }))?,
            port_zero: tensor(vec![0.0; ports])?,
        })
    }
}

impl<R: Runtime<DType = DType>> PortFaces<R>
where
    R::Client: BinaryOps<R> + IndexingOps<R> + ScalarOps<R>,
{
    /// Expands per-port ghost depth and reach-positive discharge per width onto port faces.
    ///
    /// The ghost carries depth `h`, normal discharge per width `q · s` along the face's
    /// positive axis, zero tangential discharge, and the ghost bed.
    pub(crate) fn ghost_states(
        &self,
        client: &R::Client,
        depth: &Tensor<R>,
        discharge: &Tensor<R>,
    ) -> Result<TensorFaceState<R>, EngineError> {
        let h = client.index_select(depth, 0, &self.port_of_face)?;
        let q = client.index_select(discharge, 0, &self.port_of_face)?;
        Ok(TensorFaceState {
            q_normal: client.mul(&q, &self.reach_sign)?,
            q_tangent: client.mul_scalar(&h, 0.0)?,
            h,
            bed: self.bed.clone(),
        })
    }

    /// Replaces the missing side of every port face with its ghost state.
    pub(crate) fn replace(
        &self,
        client: &R::Client,
        left: TensorFaceState<R>,
        right: TensorFaceState<R>,
        ghost: &TensorFaceState<R>,
    ) -> Result<(TensorFaceState<R>, TensorFaceState<R>), EngineError> {
        Ok((
            self.left.replace(client, left, ghost)?,
            self.right.replace(client, right, ghost)?,
        ))
    }
}

/// Per-port-face and per-port flux receipts of one prepared surface interval.
///
/// Face tensors have one lane per port face, in port order, then along each face line.
/// Port tensors have one lane per port. "Into the surface" is the direction from the ghost
/// side into the active cell, which is also the outward normal of the reach end.
#[doc(hidden)]
#[derive(Clone, Debug)]
pub struct PortFluxes<R: Runtime<DType = DType>> {
    /// Face width in m.
    pub face_width_m: Tensor<R>,
    /// Water flux per width into the surface, in m²/s.
    pub face_water_into_surface_m2_per_s: Tensor<R>,
    /// Shared normal momentum flux per width, in m³/s². It has no orientation sign.
    pub face_normal_momentum_m3_per_s2: Tensor<R>,
    /// Shared flux per width of tangential momentum into the surface, in m³/s².
    ///
    /// The tangential axis is y for x faces and x for y faces.
    pub face_tangential_momentum_into_surface_m3_per_s2: Tensor<R>,
    /// Hydrostatic pressure correction on the ghost side, per width, in m³/s².
    pub face_ghost_pressure_m3_per_s2: Tensor<R>,
    /// Hydrostatic pressure correction on the surface side, per width, in m³/s².
    pub face_surface_pressure_m3_per_s2: Tensor<R>,
    /// Face signal speed `max(|lower|, |upper|)`, in m/s.
    pub face_signal_speed_m_per_s: Tensor<R>,
    /// Lower HLL wave bound along the direction into the surface, in m/s.
    pub face_lower_wave_into_surface_m_per_s: Tensor<R>,
    /// Upper HLL wave bound along the direction into the surface, in m/s.
    pub face_upper_wave_into_surface_m_per_s: Tensor<R>,
    /// Post-source depth of the active surface cell, in m.
    pub face_surface_depth_m: Tensor<R>,
    /// Post-source normal discharge per width of the active cell into the surface, in m²/s.
    pub face_surface_discharge_into_surface_m2_per_s: Tensor<R>,
    /// Water rate leaving the reach and entering the surface, per port, in m³/s.
    pub reach_outward_water_rate_m3_per_s: Tensor<R>,
    /// Normal momentum flux along the reach's outward normal, ghost pressure included, per port, in m⁴/s².
    pub reach_outward_momentum_rate_m4_per_s2: Tensor<R>,
    /// Momentum rate entering the surface cells in grid x and y, per port, in m⁴/s².
    pub surface_momentum_rate_m4_per_s2: [Tensor<R>; 2],
    /// Momentum leaving the reach minus momentum entering the surface, grid x and y, per port, in m⁴/s².
    ///
    /// It holds the bed-step reaction and the tangential momentum that leaves the surface.
    pub junction_reaction_rate_m4_per_s2: [Tensor<R>; 2],
    /// Largest face signal speed per port, in m/s.
    pub port_signal_speed_m_per_s: Tensor<R>,
    /// Summed water rate entering the surface through port faces, in m³/s.
    pub inflow_rate_m3_per_s: f64,
    /// Summed water rate leaving the surface through port faces, in m³/s.
    pub outflow_rate_m3_per_s: f64,
}

/// Transport face tensors that the port receipts read, each over every transport face.
pub(crate) struct TransportFaces<'a, R: Runtime<DType = DType>> {
    pub left: &'a TensorFaceState<R>,
    pub right: &'a TensorFaceState<R>,
    pub result: &'a TensorFaceFlux<R>,
    /// Grid-frame shared flux: water, x momentum, y momentum.
    pub flux: &'a [Tensor<R>; 3],
    /// Face-frame shared normal and tangential momentum flux.
    pub normal: &'a Tensor<R>,
    pub tangent: &'a Tensor<R>,
    pub pressures: [&'a Tensor<R>; 2],
    pub speed: &'a Tensor<R>,
}

impl<R: Runtime<DType = DType>> PortFaces<R>
where
    R::Client: BinaryOps<R>
        + CompareOps<R>
        + ConditionalOps<R>
        + IndexingOps<R>
        + ReduceOps<R>
        + ScalarOps<R>
        + UnaryOps<R>,
{
    /// Gathers the port lanes of the transport faces into per-face and per-port receipts.
    pub(crate) fn fluxes(
        &self,
        context: &TensorDomain<R>,
        faces: TransportFaces<'_, R>,
    ) -> Result<PortFluxes<R>, EngineError> {
        let client = &context.client;
        let gather = |values: &Tensor<R>| client.index_select(values, 0, &self.face_ids);
        let ghost_left = &self.ghost_left;
        // The surface side is the right side where the ghost is the left side.
        let by_side = |left: &Tensor<R>, right: &Tensor<R>| -> Result<_, EngineError> {
            let left = gather(left)?;
            let right = gather(right)?;
            Ok((
                client.where_cond(ghost_left, &left, &right)?,
                client.where_cond(ghost_left, &right, &left)?,
            ))
        };
        let into = &self.into;
        let widths = &self.widths;
        let water = client.mul(&gather(&faces.flux[0])?, into)?;
        let normal = gather(faces.normal)?;
        let tangent = client.mul(&gather(faces.tangent)?, into)?;
        let (ghost_pressure, surface_pressure) = by_side(faces.pressures[0], faces.pressures[1])?;
        let speed = gather(faces.speed)?;
        let lower = gather(&faces.result.lower)?;
        let upper = gather(&faces.result.upper)?;
        let lower_into = client.where_cond(ghost_left, &lower, &client.neg(&upper)?)?;
        let upper_into = client.where_cond(ghost_left, &upper, &client.neg(&lower)?)?;
        let (_, surface_depth) = by_side(&faces.left.h, &faces.right.h)?;
        let (_, surface_discharge) = by_side(&faces.left.q_normal, &faces.right.q_normal)?;
        let surface_discharge = client.mul(&surface_discharge, into)?;
        let scatter = |values: &Tensor<R>, op| {
            client.scatter_reduce(&self.port_zero, 0, &self.port_of_face, values, op, true)
        };
        let sum = |values: &Tensor<R>| scatter(values, ScatterReduceOp::Sum);
        let rate = client.mul(&water, widths)?;
        let zero = client.mul_scalar(&rate, 0.0)?;
        let inflow_rate = client
            .sum(&client.maximum(&rate, &zero)?, &[0], false)?
            .item::<f64>()?;
        let outflow_rate = client
            .sum(&client.maximum(&client.neg(&rate)?, &zero)?, &[0], false)?
            .item::<f64>()?;
        let reach_momentum = client.mul(&client.add(&normal, &ghost_pressure)?, widths)?;
        let y_mask = client.rsub_scalar(&self.x_mask, 1.0)?;
        let mut surface = Vec::with_capacity(2);
        let mut reaction = Vec::with_capacity(2);
        for (component, direction) in [(1, &self.x_mask), (2, &y_mask)] {
            // The surface cell gains the shared flux plus its own pressure, as transport adds them.
            let gained = client.mul(
                &client.mul(
                    &client.add(
                        &gather(&faces.flux[component])?,
                        &client.mul(&surface_pressure, direction)?,
                    )?,
                    widths,
                )?,
                into,
            )?;
            let leaving = client.mul(&client.mul(&reach_momentum, direction)?, into)?;
            reaction.push(sum(&client.sub(&leaving, &gained)?)?);
            surface.push(sum(&gained)?);
        }
        let [surface_x, surface_y]: [Tensor<R>; 2] = surface.try_into().map_err(|_| {
            EngineError::rejected("port fluxes", "missing surface momentum component")
        })?;
        let [reaction_x, reaction_y]: [Tensor<R>; 2] = reaction.try_into().map_err(|_| {
            EngineError::rejected("port fluxes", "missing junction reaction component")
        })?;
        Ok(PortFluxes {
            face_width_m: widths.clone(),
            reach_outward_water_rate_m3_per_s: sum(&rate)?,
            reach_outward_momentum_rate_m4_per_s2: sum(&reach_momentum)?,
            surface_momentum_rate_m4_per_s2: [surface_x, surface_y],
            junction_reaction_rate_m4_per_s2: [reaction_x, reaction_y],
            port_signal_speed_m_per_s: scatter(&speed, ScatterReduceOp::Max)?,
            face_water_into_surface_m2_per_s: water,
            face_normal_momentum_m3_per_s2: normal,
            face_tangential_momentum_into_surface_m3_per_s2: tangent,
            face_ghost_pressure_m3_per_s2: ghost_pressure,
            face_surface_pressure_m3_per_s2: surface_pressure,
            face_signal_speed_m_per_s: speed,
            face_lower_wave_into_surface_m_per_s: lower_into,
            face_upper_wave_into_surface_m_per_s: upper_into,
            face_surface_depth_m: surface_depth,
            face_surface_discharge_into_surface_m2_per_s: surface_discharge,
            inflow_rate_m3_per_s: inflow_rate,
            outflow_rate_m3_per_s: outflow_rate,
        })
    }
}
