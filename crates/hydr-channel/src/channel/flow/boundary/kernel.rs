use super::{
    DirectBoundaryFrame, EndpointInterior, NativeBoundaryTraces,
    algebra::{Bounds, Interior},
    depth::{Depth, DepthRequest},
    port::PortLane,
};
use crate::channel::flow::{
    ChannelError, ChannelTensorBoundary, context::Context, geometry::TensorGeometry,
};
use numr::{
    dtype::DType,
    ops::{
        BinaryOps, CompareOps, ConditionalOps, ReduceOps, ScalarOps, ShapeOps, TwoSumOps, UnaryOps,
    },
    runtime::{Runtime, SquareRootContract},
    tensor::Tensor,
};
use solvr::optimize::roots::BatchedBisectionAlgorithms;
use std::sync::Arc;

pub(super) struct Lane<R: Runtime<DType = DType>> {
    pub trace: NativeBoundaryTraces<R>,
    pub longitudinal: Tensor<R>,
    pub water_bound: Tensor<R>,
    pub momentum_bound: Tensor<R>,
}
/// Solved scalar lanes in endpoint order and their packed receipts.
type ScalarLanes<R> = (Vec<Lane<R>>, Arc<NativeBoundaryTraces<R>>);

struct Pending<R: Runtime<DType = DType>> {
    endpoint: usize,
    interior: Interior<R>,
    coefficient: Tensor<R>,
    cap: Tensor<R>,
    depth: DepthRequest<R>,
    target: Option<(Tensor<R>, Tensor<R>)>,
    reservoir: Option<super::reservoir::Reservoir<R>>,
}
impl<R: Runtime<DType = DType>> Context<R>
where
    R::Client: BinaryOps<R>
        + CompareOps<R>
        + ConditionalOps<R>
        + ReduceOps<R>
        + ScalarOps<R>
        + ShapeOps<R>
        + UnaryOps<R>
        + TwoSumOps<R>
        + BatchedBisectionAlgorithms<R>,
{
    /// Rates at the direct end faces, or `None` when both faces solve a face Riemann problem.
    pub(in crate::channel::flow) fn direct_boundary_frame(
        &self,
        interior: &EndpointInterior<R>,
        left: &ChannelTensorBoundary<R>,
        right: &ChannelTensorBoundary<R>,
    ) -> Result<Option<DirectBoundaryFrame<R>>, ChannelError> {
        let ports = self.port_lanes(left, right)?;
        let scalar = self.scalar_lanes(interior, left, right)?;
        if ports.is_empty() && scalar.is_none() {
            return Ok(None);
        }
        let c = &self.client;
        let (lanes, traces) = match scalar {
            Some((lanes, traces)) => (lanes, Some(traces)),
            None => (Vec::new(), None),
        };
        let mut endpoints = Vec::with_capacity(2);
        let mut entries = Vec::with_capacity(2);
        for endpoint in 0..2 {
            if let Some(lane) = lanes.iter().find(|l| l.trace.endpoints == [endpoint]) {
                entries.push([
                    &lane.longitudinal,
                    &lane.trace.momentum_flux_m4_per_s2,
                    &lane.trace.signal_speed_bound_m_per_s,
                    &lane.water_bound,
                    &lane.momentum_bound,
                    &lane.trace.water_flux_width_m3_per_s,
                    &lane.trace.momentum_flux_width_m4_per_s2,
                ]);
            } else if let Some(port) = ports.iter().find(|p| p.endpoint == endpoint) {
                entries.push([
                    &port.longitudinal,
                    &port.momentum,
                    &port.speed,
                    &port.zero,
                    &port.zero,
                    &port.zero,
                    &port.zero,
                ]);
            } else {
                continue;
            }
            endpoints.push(endpoint);
        }
        let pack = |field: usize| -> Result<Tensor<R>, ChannelError> {
            let refs: Vec<_> = entries.iter().map(|entry| entry[field]).collect();
            Ok(c.cat(&refs, 0)?)
        };
        Ok(Some(DirectBoundaryFrame {
            endpoints,
            longitudinal_water: pack(0)?,
            momentum_flux_m4_per_s2: pack(1)?,
            signal_speed_bound_m_per_s: pack(2)?,
            water_underflow_bound: pack(3)?,
            momentum_underflow_bound: pack(4)?,
            water_flux_width_m3_per_s: pack(5)?,
            momentum_flux_width_m4_per_s2: pack(6)?,
            root_iterations: traces
                .as_ref()
                .and_then(|t| t.root_iterations.iter().copied().max())
                .unwrap_or(0),
            traces,
            ports: ports.into_iter().map(PortLane::front).collect(),
        }))
    }

    /// Solved scalar trace lanes and their packed receipts, or `None` without a scalar trace.
    fn scalar_lanes(
        &self,
        interior: &EndpointInterior<R>,
        left: &ChannelTensorBoundary<R>,
        right: &ChannelTensorBoundary<R>,
    ) -> Result<Option<ScalarLanes<R>>, ChannelError> {
        let mut pending = Vec::with_capacity(2);
        for (endpoint, request) in [left, right].into_iter().enumerate() {
            if !super::assembly::is_scalar(request) {
                continue;
            }
            if R::square_root_contract(DType::F64) != SquareRootContract::Ieee754RoundToNearest {
                return Err(ChannelError::input(
                    "scalar channel boundaries require correctly rounded F64 square roots: select an admitted runtime",
                ));
            }
            pending.push(self.boundary_prepare(interior, endpoint, request)?);
        }
        if pending.is_empty() {
            return Ok(None);
        }
        let roots: Vec<_> = pending
            .iter()
            .filter_map(|lane| match &lane.depth {
                DepthRequest::Discharge(root) => Some(root.as_ref()),
                DepthRequest::Known(_) | DepthRequest::Reservoir(_) => None,
            })
            .collect();
        let mut solved = self.boundary_discharge_solve(&roots)?.into_iter();
        let reservoir_roots: Vec<_> = pending
            .iter()
            .filter_map(|lane| match &lane.depth {
                DepthRequest::Reservoir(root) => Some(root.as_ref()),
                _ => None,
            })
            .collect();
        let mut reservoir_solved = self.boundary_reservoir_solve(&reservoir_roots)?.into_iter();
        let mut lanes = Vec::with_capacity(pending.len());
        for lane in pending {
            let depth = match lane.depth {
                DepthRequest::Known(depth) => *depth,
                DepthRequest::Reservoir(_) => reservoir_solved.next().ok_or_else(|| {
                    ChannelError::numerical(
                        "scalar reservoir root mapping is incomplete: inspect packed endpoint metadata",
                    )
                })?,
                DepthRequest::Discharge(_) => solved.next().ok_or_else(|| {
                    ChannelError::numerical(
                        "scalar discharge root mapping is incomplete: inspect packed root endpoint metadata",
                    )
                })?,
            };
            let zero = self.constant(&depth.value, 0.0)?;
            self.boundary_require(
                &self.client.gt(&depth.bounds.lo, &zero)?,
                "trace depth is dry",
            )?;
            self.boundary_require(
                &self.client.le(&depth.bounds.hi, &lane.cap)?,
                "trace depth exceeds the geometric cap",
            )?;
            let bounds = depth.bounds.clone();
            let mut evaluated = self.boundary_evaluate(
                lane.endpoint,
                &lane.interior,
                &lane.coefficient,
                depth,
                lane.target,
            )?;
            if let Some(reservoir) = lane.reservoir {
                self.boundary_reservoir_receipt(&mut evaluated, &reservoir, &bounds)?;
            }
            lanes.push(evaluated);
        }
        let c = &self.client;
        macro_rules! pack {
            ($field:ident) => {{
                let refs: Vec<_> = lanes.iter().map(|l| &l.trace.$field).collect();
                c.cat(&refs, 0)?
            }};
        }
        let reservoirs: Vec<_> = lanes
            .iter()
            .filter_map(|lane| lane.trace.reservoir.as_ref())
            .collect();
        let reservoir = if reservoirs.is_empty() {
            None
        } else {
            macro_rules! reservoir_pack {
                ($field:ident) => {{
                    let refs: Vec<_> = reservoirs.iter().map(|r| &r.$field).collect();
                    c.cat(&refs, 0)?
                }};
            }
            Some(super::types::NativeReservoirTraces {
                depth_lower_m: reservoir_pack!(depth_lower_m),
                depth_upper_m: reservoir_pack!(depth_upper_m),
                head_compatibility_abs_bound_m: reservoir_pack!(head_compatibility_abs_bound_m),
                dissipation_head_lower_m: reservoir_pack!(dissipation_head_lower_m),
                dissipation_head_upper_m: reservoir_pack!(dissipation_head_upper_m),
                endpoints: reservoirs
                    .iter()
                    .flat_map(|r| r.endpoints.iter().copied())
                    .collect(),
            })
        };
        let traces = NativeBoundaryTraces {
            depth_m: pack!(depth_m),
            area_m2: pack!(area_m2),
            outward_velocity_m_per_s: pack!(outward_velocity_m_per_s),
            outward_discharge_m3_per_s: pack!(outward_discharge_m3_per_s),
            momentum_flux_m4_per_s2: pack!(momentum_flux_m4_per_s2),
            signal_speed_bound_m_per_s: pack!(signal_speed_bound_m_per_s),
            velocity_compatibility_abs_bound_m_per_s: pack!(
                velocity_compatibility_abs_bound_m_per_s
            ),
            water_compatibility_abs_bound_m3_per_s: pack!(water_compatibility_abs_bound_m3_per_s),
            momentum_compatibility_abs_bound_m4_per_s2: pack!(
                momentum_compatibility_abs_bound_m4_per_s2
            ),
            water_flux_width_m3_per_s: pack!(water_flux_width_m3_per_s),
            momentum_flux_width_m4_per_s2: pack!(momentum_flux_width_m4_per_s2),
            cap_compatibility_approximation: pack!(cap_compatibility_approximation),
            reservoir,
            endpoints: lanes
                .iter()
                .flat_map(|l| l.trace.endpoints.iter().copied())
                .collect(),
            root_iterations: lanes
                .iter()
                .flat_map(|l| l.trace.root_iterations.iter().copied())
                .collect(),
        };
        Ok(Some((lanes, Arc::new(traces))))
    }
    fn boundary_prepare(
        &self,
        interior: &EndpointInterior<R>,
        endpoint: usize,
        request: &ChannelTensorBoundary<R>,
    ) -> Result<Pending<R>, ChannelError> {
        let c = &self.client;
        let index = if endpoint == 0 { 0 } else { self.count() - 1 };
        let normal = if endpoint == 0 { -1.0 } else { 1.0 };
        let hi = interior.depth.narrow(0, endpoint, 1)?;
        let ui = c.mul_scalar(&interior.velocity.narrow(0, endpoint, 1)?, normal)?;
        let zero = self.constant(&hi, 0.0)?;
        self.boundary_require(&c.gt(&hi, &zero)?, "interior depth is dry")?;
        self.finite(&hi, "scalar boundary interior depth")?;
        self.finite(&ui, "scalar boundary interior velocity")?;

        let coefficient = match &self.geometry {
            TensorGeometry::Surveyed(_) => {
                return Err(ChannelError::input(
                    "surveyed scalar boundary is unsupported: supply Wall or complete Exterior depth and velocity",
                ));
            }
            TensorGeometry::Rectangular(g) => g.width.narrow(0, index, 1)?,
            TensorGeometry::Triangular(g) => g.coefficient.clone(),
        };
        let area = interior.area.narrow(0, endpoint, 1)?;
        let discharge = c.mul_scalar(&interior.discharge.narrow(0, endpoint, 1)?, normal)?;
        self.finite(&area, "scalar boundary interior area")?;
        self.finite(&discharge, "scalar boundary interior discharge")?;
        self.boundary_require(&c.gt(&area, &zero)?, "interior area is dry")?;
        let coefficient_bounds = if self.boundary_triangle() {
            Bounds {
                lo: super::super::triangular::envelope::lower::<R>(c, &coefficient)?,
                hi: super::super::triangular::envelope::upper::<R>(c, &coefficient)?,
            }
        } else {
            Bounds::exact(&coefficient)
        };
        let depth_bounds = if self.boundary_triangle() {
            self.bdiv(
                &self.bsqrt(&Bounds::exact(&area))?,
                &self.bsqrt(&coefficient_bounds)?,
            )?
        } else {
            self.bdiv(&Bounds::exact(&area), &coefficient_bounds)?
        };
        let velocity_bounds = self.bdiv(&Bounds::exact(&discharge), &Bounds::exact(&area))?;
        let state = Interior {
            depth: hi,
            velocity: ui,
            depth_bounds,
            velocity_bounds,
            area,
            discharge,
        };
        self.boundary_subcritical(&state.depth_bounds, &state.velocity_bounds)?;
        let cap = self.cap.narrow(0, index, 1)?;
        let mut reservoir = None;
        let (depth, target) = match request {
            ChannelTensorBoundary::Stage { elevation_m } => {
                let stage = elevation_m;
                let bed = self.bed.narrow(0, index, 1)?;
                let (depth, low) = c.two_sum(stage, &c.neg(&bed)?)?;
                let bounds = Bounds {
                    lo: c.where_cond(
                        &c.ge(&low, &zero)?,
                        &depth,
                        &super::super::triangular::envelope::lower::<R>(c, &depth)?,
                    )?,
                    hi: c.where_cond(
                        &c.le(&low, &zero)?,
                        &depth,
                        &super::super::triangular::envelope::upper::<R>(c, &depth)?,
                    )?,
                };
                (
                    DepthRequest::Known(Box::new(Depth {
                        value: depth,
                        bounds,
                        approximate: false,
                        iterations: 0,
                    })),
                    None,
                )
            }
            ChannelTensorBoundary::QuiescentReservoir {
                elevation_m,
                entrance_loss_coefficient,
            } => {
                let bed = self.bed.narrow(0, index, 1)?;
                let (depth, input) = self.boundary_reservoir_prepare(
                    &state,
                    &cap,
                    elevation_m,
                    &bed,
                    entrance_loss_coefficient,
                )?;
                reservoir = Some(input);
                (depth, None)
            }
            ChannelTensorBoundary::Discharge { discharge_m3_per_s } => {
                let longitudinal = discharge_m3_per_s.clone();
                let outward = c.mul_scalar(&longitudinal, normal)?;
                (
                    self.boundary_discharge_prepare(&state, &coefficient, &cap, &outward)?,
                    Some((outward, longitudinal)),
                )
            }
            _ => {
                return Err(ChannelError::input(
                    "scalar boundary trace requires Stage or Discharge",
                ));
            }
        };
        Ok(Pending {
            endpoint,
            interior: state,
            coefficient,
            cap,
            depth,
            target,
            reservoir,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::super::fixture::{context, interior};
    use super::*;

    type TestResult = Result<(), Box<dyn std::error::Error>>;

    #[test]
    fn exact_rest_retains_original_depth_and_section_enclosure() -> TestResult {
        for triangle in [false, true] {
            for bed in [0.0, 16.0] {
                let ctx = context(triangle, bed)?;
                let state = interior(&ctx, 0.0, 0.0)?;
                for loss in [0.0, 1.0] {
                    let (request, _) = ctx.boundary_reservoir_prepare(
                        &state,
                        &ctx.tensor(&[4.0])?,
                        &ctx.tensor(&[bed + 1.0])?,
                        &ctx.tensor(&[bed])?,
                        &ctx.tensor(&[loss])?,
                    )?;
                    let DepthRequest::Known(depth) = request else {
                        return Err("exact resting depth enters numerical solving".into());
                    };
                    assert_eq!(depth.value.item::<f64>()?.to_bits(), 1.0_f64.to_bits());
                    assert_eq!(
                        depth.bounds.lo.item::<f64>()?,
                        state.depth_bounds.lo.item::<f64>()?
                    );
                    assert_eq!(
                        depth.bounds.hi.item::<f64>()?,
                        state.depth_bounds.hi.item::<f64>()?
                    );
                    assert_eq!(depth.iterations, 0);
                    assert!(!depth.approximate);
                    for endpoint in [0, 1] {
                        let inputs = EndpointInterior {
                            depth: ctx.tensor(&[1.0, 1.0])?,
                            area: ctx.tensor(&[1.0, 1.0])?,
                            velocity: ctx.tensor(&[0.0, 0.0])?,
                            discharge: ctx.tensor(&[0.0, 0.0])?,
                        };
                        let policy = ChannelTensorBoundary::QuiescentReservoir {
                            elevation_m: ctx.tensor(&[bed + 1.0])?,
                            entrance_loss_coefficient: ctx.tensor(&[loss])?,
                        };
                        let wall = ChannelTensorBoundary::Wall;
                        let (left, right) = if endpoint == 0 {
                            (&policy, &wall)
                        } else {
                            (&wall, &policy)
                        };
                        let frame = ctx
                            .direct_boundary_frame(&inputs, left, right)?
                            .ok_or("missing resting reservoir frame")?;
                        let traces = frame
                            .traces
                            .as_deref()
                            .ok_or("missing resting reservoir trace")?;
                        assert_eq!(traces.depth_m.item::<f64>()?.to_bits(), 1.0_f64.to_bits());
                        assert_eq!(traces.outward_discharge_m3_per_s.item::<f64>()?, 0.0);
                        assert_eq!(frame.longitudinal_water.item::<f64>()?, 0.0);
                        let momentum = if triangle { 1.0 / 3.0 } else { 0.5 };
                        assert_eq!(traces.momentum_flux_m4_per_s2.item::<f64>()?, momentum);
                        assert_eq!(frame.momentum_flux_m4_per_s2.item::<f64>()?, momentum);
                        assert_eq!(traces.root_iterations, vec![0]);
                        assert_eq!(traces.endpoints, vec![endpoint]);
                        assert_eq!(frame.endpoints, vec![endpoint]);
                        assert!(traces.reservoir.is_some());
                        assert!(frame.ports.is_empty());
                    }
                }
            }
        }
        Ok(())
    }
}
