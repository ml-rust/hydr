use super::{TensorDomain, TensorSources, TensorState};
use crate::numerics::{TensorFaceState, hydrostatic_hll_tensor};
use crate::solver::{EngineError, FaceTransfer, SolverConfig};
use numr::dtype::DType;
use numr::ops::{
    BinaryOps, CompareOps, ConditionalOps, IndexingOps, ReduceOps, ScalarOps, ScatterReduceOp,
    UnaryOps,
};
use numr::runtime::Runtime;
use numr::tensor::Tensor;

#[derive(Clone)]
pub(crate) struct TensorTransport<R: Runtime<DType = DType>> {
    pub derivative: TensorState<R>,
    pub outgoing: Tensor<R>,
    pub flux: [Tensor<R>; 3],
    pub pressures: [Tensor<R>; 2],
    pub signal_speed: Tensor<R>,
    pub transmissive_state: Option<TensorFaceState<R>>,
    pub speeds: [f64; 2],
    pub boundary_in_rate: f64,
    pub boundary_out_rate: f64,
    pub boundary_trace: Option<super::boundary::TensorBoundaryReceipt<R>>,
}

pub(crate) fn calculate<R: Runtime<DType = DType>>(
    context: &TensorDomain<R>,
    state: &TensorState<R>,
    source: &TensorSources<R>,
    config: &SolverConfig,
    area: f64,
) -> Result<TensorTransport<R>, EngineError>
where
    R::Client: BinaryOps<R>
        + CompareOps<R>
        + ConditionalOps<R>
        + IndexingOps<R>
        + ReduceOps<R>
        + ScalarOps<R>
        + UnaryOps<R>
        + solvr::optimize::roots::BatchedBisectionAlgorithms<R>,
{
    let client = &context.client;
    let topology = &context.topology;
    let gather = |field: &Tensor<R>, ids: &Tensor<R>| client.index_select(field, 0, ids);
    let side = |ids: &Tensor<R>, sign: &Tensor<R>| -> Result<TensorFaceState<R>, EngineError> {
        let qx = gather(&state.qx, ids)?;
        let qy = gather(&state.qy, ids)?;
        let normal = client.where_cond(&topology.x_mask, &qx, &qy)?;
        let tangent = client.where_cond(&topology.x_mask, &qy, &qx)?;
        let sign = client.where_cond(
            &source.open_mask,
            &client.add_scalar(&client.mul_scalar(sign, 0.0)?, 1.0)?,
            sign,
        )?;
        Ok(TensorFaceState {
            h: gather(&state.h, ids)?,
            q_normal: client.mul(&normal, &sign)?,
            q_tangent: tangent,
            bed: gather(&context.bed, ids)?,
        })
    };
    let left = side(&topology.left_ids, &topology.left_normal_sign)?;
    let right = side(&topology.right_ids, &topology.right_normal_sign)?;
    let (left, right, transmissive_state) = source
        .exterior
        .prepare(context, state, left, right, config)?;
    let zero_face = client.mul_scalar(&topology.widths, 0.0)?;
    let one_face = client.add_scalar(&zero_face, 1.0)?;
    let characteristic_speed = |state: &TensorFaceState<R>| -> Result<Tensor<R>, EngineError> {
        let depth = client.where_cond(&client.gt(&state.h, &zero_face)?, &state.h, &one_face)?;
        Ok(client.add(
            &client.abs(&client.div(&state.q_normal, &depth)?)?,
            &client.mul_scalar(&client.sqrt(&state.h)?, config.gravity.sqrt())?,
        )?)
    };
    let known_speed = client.maximum(
        &characteristic_speed(&left)?,
        &characteristic_speed(&right)?,
    )?;
    let invalid_port = client.mul(
        &source.port_mask,
        &client.gt(&known_speed, &source.port_speed)?,
    )?;
    if client.max(&invalid_port, &[0], false)?.item::<f64>()? != 0.0 {
        let flags = invalid_port.try_to_vec::<f64>()?;
        let index = flags
            .iter()
            .position(|value| *value != 0.0)
            .ok_or_else(|| {
                EngineError::rejected(
                    "port bound",
                    "inconsistent reduction: inspect numr comparisons",
                )
            })?;
        return Err(EngineError::rejected(
            "port bound",
            format!(
                "face={:?}: supplied speed excludes an interior characteristic, supply a conservative compatible boundary provider",
                topology.faces[index].face
            ),
        ));
    }
    let result =
        match hydrostatic_hll_tensor(client, left.clone(), right.clone(), config.gravity, config) {
            Ok(result) => result,
            Err(error) => return Err(face_error(context, (&left, &right), config, error)?),
        };
    let direct_mask = client.maximum(&source.port_mask, &source.trace_mask)?;
    let pressure_left = client.where_cond(&direct_mask, &zero_face, &result.left_pressure)?;
    let pressure_right = client.where_cond(&direct_mask, &zero_face, &result.right_pressure)?;
    let normal_flux = client.where_cond(&topology.x_mask, &result.shared[1], &result.shared[2])?;
    let tangent_flux = client.where_cond(&topology.x_mask, &result.shared[2], &result.shared[1])?;
    let mut flux = [
        client.where_cond(&source.port_mask, &source.port_flux[0], &result.shared[0])?,
        client.where_cond(&source.port_mask, &source.port_flux[1], &normal_flux)?,
        client.where_cond(&source.port_mask, &source.port_flux[2], &tangent_flux)?,
    ];
    let mut speed = client.where_cond(&source.port_mask, &source.port_speed, &result.speed)?;
    let boundary_trace = source.characteristic.calculate(context, state, config)?;
    if let Some(trace) = &boundary_trace {
        let (physical_flux, physical_speed) = source
            .characteristic
            .override_flux(context, trace, &zero_face)?;
        for component in 0..3 {
            flux[component] = client.where_cond(
                &source.trace_mask,
                &physical_flux[component],
                &flux[component],
            )?;
        }
        speed = client.where_cond(&source.trace_mask, &physical_speed, &speed)?;
    }
    let y_mask = client.rsub_scalar(&topology.x_mask, 1.0)?;
    let speeds = [
        client
            .max(&client.mul(&speed, &topology.x_mask)?, &[0], false)?
            .item::<f64>()?,
        client
            .max(&client.mul(&speed, &y_mask)?, &[0], false)?
            .item::<f64>()?,
    ];
    let scatter = |ids: &Tensor<R>, values: &Tensor<R>| {
        client.scatter_reduce(&context.zero, 0, ids, values, ScatterReduceOp::Sum, true)
    };
    let rate = client.mul(&flux[0], &topology.widths)?;
    let exterior_rate = client.mul(&rate, &source.outward_sign)?;
    let boundary_out_rate = client
        .sum(&client.maximum(&exterior_rate, &zero_face)?, &[0], false)?
        .item::<f64>()?;
    let boundary_in_rate = client
        .sum(
            &client.maximum(&client.neg(&exterior_rate)?, &zero_face)?,
            &[0],
            false,
        )?
        .item::<f64>()?;
    let outgoing_left = client.mul(
        &client.maximum(&rate, &zero_face)?,
        &client.mul_scalar(&topology.left_sign, -1.0)?,
    )?;
    let outgoing_right = client.mul(
        &client.maximum(&client.mul_scalar(&rate, -1.0)?, &zero_face)?,
        &topology.right_sign,
    )?;
    let outgoing = client.add(
        &scatter(&topology.left_ids, &outgoing_left)?,
        &scatter(&topology.right_ids, &outgoing_right)?,
    )?;
    let mut derivatives = Vec::with_capacity(3);
    for (component, shared) in flux.iter().enumerate() {
        let direction = match component {
            1 => &topology.x_mask,
            2 => &y_mask,
            _ => &zero_face,
        };
        let left = client.mul(
            &client.add(shared, &client.mul(&pressure_left, direction)?)?,
            &topology.widths,
        )?;
        let left = client.mul(&client.div_scalar(&left, area)?, &topology.left_sign)?;
        let right = client.mul(
            &client.add(shared, &client.mul(&pressure_right, direction)?)?,
            &topology.widths,
        )?;
        let right = client.mul(&client.div_scalar(&right, area)?, &topology.right_sign)?;
        derivatives.push(client.add(
            &scatter(&topology.left_ids, &left)?,
            &scatter(&topology.right_ids, &right)?,
        )?);
    }
    let mut derivative = derivatives.into_iter();
    let h = derivative.next().ok_or_else(|| {
        EngineError::rejected(
            "transport",
            "missing mass derivative: inspect component assembly",
        )
    })?;
    let qx = derivative.next().ok_or_else(|| {
        EngineError::rejected(
            "transport",
            "missing x derivative: inspect component assembly",
        )
    })?;
    let qy = derivative.next().ok_or_else(|| {
        EngineError::rejected(
            "transport",
            "missing y derivative: inspect component assembly",
        )
    })?;
    Ok(TensorTransport {
        derivative: TensorState { h, qx, qy },
        outgoing,
        flux,
        pressures: [pressure_left, pressure_right],
        speeds,
        signal_speed: speed,
        transmissive_state,
        boundary_in_rate,
        boundary_out_rate,
        boundary_trace,
    })
}

fn face_slice<R: Runtime<DType = DType>>(
    state: &TensorFaceState<R>,
    start: usize,
    count: usize,
) -> Result<TensorFaceState<R>, EngineError> {
    Ok(TensorFaceState {
        h: state.h.narrow(0, start, count)?,
        q_normal: state.q_normal.narrow(0, start, count)?,
        q_tangent: state.q_tangent.narrow(0, start, count)?,
        bed: state.bed.narrow(0, start, count)?,
    })
}

/// Isolate numerical errors with the same tensor kernel, only after a rejected batch.
fn face_error<R: Runtime<DType = DType>>(
    context: &TensorDomain<R>,
    states: (&TensorFaceState<R>, &TensorFaceState<R>),
    config: &SolverConfig,
    error: EngineError,
) -> Result<EngineError, EngineError>
where
    R::Client: BinaryOps<R>
        + CompareOps<R>
        + ConditionalOps<R>
        + IndexingOps<R>
        + ReduceOps<R>
        + ScalarOps<R>
        + UnaryOps<R>
        + solvr::optimize::roots::BatchedBisectionAlgorithms<R>,
{
    let mut start = 0;
    let mut count = context.topology.faces.len();
    while count > 1 {
        let half = count / 2;
        if hydrostatic_hll_tensor(
            &context.client,
            face_slice(states.0, start, half)?,
            face_slice(states.1, start, half)?,
            config.gravity,
            config,
        )
        .is_err()
        {
            count = half;
        } else if hydrostatic_hll_tensor(
            &context.client,
            face_slice(states.0, start + half, count - half)?,
            face_slice(states.1, start + half, count - half)?,
            config.gravity,
            config,
        )
        .is_err()
        {
            start += half;
            count -= half;
        } else {
            return Ok(EngineError::rejected(
                "face batch",
                format!(
                    "{error}: error does not reproduce on partitioned faces, inspect the numr runtime"
                ),
            ));
        }
    }
    let values = |state: &TensorFaceState<R>| -> Result<[f64; 4], EngineError> {
        Ok([
            state.h.narrow(0, start, 1)?.item::<f64>()?,
            state.q_normal.narrow(0, start, 1)?.item::<f64>()?,
            state.q_tangent.narrow(0, start, 1)?.item::<f64>()?,
            state.bed.narrow(0, start, 1)?.item::<f64>()?,
        ])
    };
    Ok(EngineError::rejected(
        "face flux",
        format!(
            "face={:?}, left [h,qn,qt,bed]={:?}, right={:?}: {error}",
            context.topology.faces[start].face,
            values(states.0)?,
            values(states.1)?
        ),
    ))
}

impl<R: Runtime<DType = DType>> TensorTransport<R>
where
    R::Client: BinaryOps<R> + ScalarOps<R>,
{
    /// Projects accepted tensor fluxes into owned output records. No calculation uses this projection.
    pub(crate) fn results(
        &self,
        context: &TensorDomain<R>,
        dt: f64,
    ) -> Result<Vec<FaceTransfer>, EngineError> {
        let mass = self.flux[0].try_to_vec::<f64>()?;
        let qx = self.flux[1].try_to_vec::<f64>()?;
        let qy = self.flux[2].try_to_vec::<f64>()?;
        let pl = self.pressures[0].try_to_vec::<f64>()?;
        let pr = self.pressures[1].try_to_vec::<f64>()?;
        let speed = self.signal_speed.try_to_vec::<f64>()?;
        if speed.len() != context.topology.faces.len() {
            return Err(EngineError::rejected(
                "tensor face output",
                "signal-speed projection differs from face support: inspect runtime shapes",
            ));
        }
        let volume = context
            .client
            .mul_scalar(
                &context
                    .client
                    .mul(&self.flux[0], &context.topology.widths)?,
                dt,
            )?
            .try_to_vec::<f64>()?;
        Ok(context
            .topology
            .faces
            .iter()
            .enumerate()
            .map(|(index, support)| FaceTransfer {
                face: support.face,
                left: support.left,
                right: support.right,
                flux: [mass[index], qx[index], qy[index]],
                pressure_corrections: [pl[index], pr[index]],
                signal_speed: speed[index],
                volume: volume[index],
            })
            .collect())
    }
}
