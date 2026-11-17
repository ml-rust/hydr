use super::{
    EndpointInterior, NativeBoundaryTraces, ScalarBoundaryFrame,
    algebra::{Bounds, Interior},
    depth::{Depth, DepthRequest},
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
    pub(in crate::channel::flow) fn scalar_boundary_frame(
        &self,
        interior: &EndpointInterior<R>,
        left: &ChannelTensorBoundary<R>,
        right: &ChannelTensorBoundary<R>,
    ) -> Result<Option<ScalarBoundaryFrame<R>>, ChannelError> {
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
        let longitudinal: Vec<_> = lanes.iter().map(|l| &l.longitudinal).collect();
        let water: Vec<_> = lanes.iter().map(|l| &l.water_bound).collect();
        let momentum: Vec<_> = lanes.iter().map(|l| &l.momentum_bound).collect();
        Ok(Some(ScalarBoundaryFrame {
            longitudinal_water: c.cat(&longitudinal, 0)?,
            water_underflow_bound: c.cat(&water, 0)?,
            momentum_underflow_bound: c.cat(&momentum, 0)?,
            traces: Arc::new(traces),
        }))
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
