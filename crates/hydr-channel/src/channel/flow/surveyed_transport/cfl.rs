//! Surveyed CFL speeds and dry-front states from the certified point Riemann solver.

use super::error::riemann_error;
use crate::channel::flow::{
    ChannelError, ChannelTensorBoundary,
    admission::Fields,
    boundary::{DirectBoundaryFrame, assembly::numerical_range},
    cfl_fronts::{DryFronts, SideFronts, has_dry_contact, side_fronts},
    cfl_speeds::CflSpeeds,
    conserved_update::{exact_over, expanded},
    context::Context,
    geometry::SurveyedGeometry,
    speed_memo::SpeedEntry,
    surveyed_points::PrismaticPointRiemannResult,
};
use numr::{
    dtype::DType,
    ops::{
        BinaryOps, CompareOps, MidpointOps, NextRepresentableOps, ScalarOps, TensorOps, TwoSumOps,
    },
    runtime::{Runtime, RuntimeClient},
    tensor::Tensor,
};
use solvr::{
    integrate::CertifiedSimpsonAlgorithms,
    optimize::{
        CertifiedMappedBisectionAlgorithms,
        roots::{
            BatchedBisectionAlgorithms, CertifiedBisectionAlgorithms,
            CertifiedUpperBracketAlgorithms,
        },
    },
};
use std::sync::Arc;

impl<R: Runtime<DType = DType>> Context<R>
where
    R::Client: TensorOps<R>
        + CompareOps<R>
        + NextRepresentableOps<R>
        + MidpointOps<R>
        + ScalarOps<R>
        + TwoSumOps<R>
        + RuntimeClient<R>
        + CertifiedSimpsonAlgorithms<R>
        + CertifiedBisectionAlgorithms<R>
        + CertifiedMappedBisectionAlgorithms<R>
        + CertifiedUpperBracketAlgorithms<R>
        + BatchedBisectionAlgorithms<R>,
{
    pub(in crate::channel::flow) fn surveyed_speed(
        &self,
        geometry: &SurveyedGeometry<R>,
        fields: &Fields<R>,
        left: &ChannelTensorBoundary<R>,
        right: &ChannelTensorBoundary<R>,
    ) -> Result<SpeedEntry<R>, ChannelError> {
        let frame = self
            .surveyed_direct_frame(geometry, fields, left, right, false)?
            .map(Arc::new);
        let (_, len) = numerical_range(self.count(), left, right);
        let result = if len > 0 {
            Some(Arc::new(
                self.surveyed_point_riemann(geometry, fields, left, right)?,
            ))
        } else {
            None
        };
        Ok(SpeedEntry {
            speeds: self.surveyed_cfl_speeds(
                geometry,
                fields,
                left,
                right,
                frame.as_deref(),
                result.as_deref(),
            )?,
            riemann: result,
            frame,
        })
    }

    /// Face speeds and dry fronts from the point Riemann `result` and the direct end faces.
    ///
    /// Riemann faces split their complete wave interval. A direct face holds one unsigned bound
    /// for both directions.
    pub(super) fn surveyed_cfl_speeds(
        &self,
        geometry: &SurveyedGeometry<R>,
        fields: &Fields<R>,
        left: &ChannelTensorBoundary<R>,
        right: &ChannelTensorBoundary<R>,
        frame: Option<&DirectBoundaryFrame<R>>,
        result: Option<&PrismaticPointRiemannResult<R>>,
    ) -> Result<CflSpeeds<R>, ChannelError> {
        let numerical = result
            .map(|result| self.surveyed_face_speeds(result.wave_speed_m_per_s()))
            .transpose()?;
        Ok(CflSpeeds {
            outer: self.assemble_face_speeds(frame, numerical.as_ref())?,
            internal: None,
            fronts: self.surveyed_fronts(geometry, fields, left, right, frame)?,
        })
    }

    /// Dry-front states from one certified potential per state.
    ///
    /// The bounds enclose `u + φ(A)` from above and `u - φ(A)` from below. Each direct end face
    /// takes its port exterior or trace state in place of the end cell copy. Returns `None`
    /// without evaluating the bounds when no face is wet on exactly one side.
    pub(in crate::channel::flow) fn surveyed_fronts(
        &self,
        geometry: &SurveyedGeometry<R>,
        fields: &Fields<R>,
        left: &ChannelTensorBoundary<R>,
        right: &ChannelTensorBoundary<R>,
        frame: Option<&DirectBoundaryFrame<R>>,
    ) -> Result<Option<DryFronts<R>>, ChannelError> {
        let c = &self.client;
        let (area, discharge) = self.surveyed_states(geometry, fields, left, right)?;
        let states = area.high.numel();
        let faces = states - 1;
        let zero = self.constant(&area.high, 0.0)?;
        let wet = c.maximum(&c.ne(&area.high, &zero)?, &c.ne(&area.low, &zero)?)?;
        let contact = self.direct_wet(&wet, frame)?;
        if !has_dry_contact(
            c,
            &contact.narrow(0, 0, faces)?,
            &contact.narrow(0, 1, faces)?,
        )? {
            return Ok(None);
        }
        let (area, discharge) = expanded(c, "surveyed dry-front states", |a| {
            Ok((
                exact_over(a, &area.high, &area.low, &zero)?,
                exact_over(a, &discharge.high, &discharge.low, &zero)?,
            ))
        })?;
        let owners = Tensor::<R>::zeros(&[states], DType::I64, c.device())?;
        let (lower, upper) = geometry
            .section
            .dry_front_bounds_tensor(&owners, &area, &discharge, &geometry.riemann)
            .map_err(riemann_error)?;
        let fronts = side_fronts(c, &lower, &upper, &wet)?;
        let side = |start: usize| {
            Ok::<_, ChannelError>(SideFronts {
                rightward: fronts.rightward.narrow(0, start, faces)?,
                leftward: fronts.leftward.narrow(0, start, faces)?,
                wet: fronts.wet.narrow(0, start, faces)?,
            })
        };
        let trace = |depth: &Tensor<R>, area: &Tensor<R>, velocity: &Tensor<R>| {
            self.surveyed_trace_fronts(geometry, depth, area, velocity)
        };
        self.dry_fronts(side(0)?, side(1)?, frame, &trace)
    }

    /// State wetness with each direct end replaced by the wetness its front splice uses.
    ///
    /// A port is wet when it holds an exterior invariant. A scalar trace is wet at positive depth.
    fn direct_wet(
        &self,
        wet: &Tensor<R>,
        frame: Option<&DirectBoundaryFrame<R>>,
    ) -> Result<Tensor<R>, ChannelError> {
        let Some(frame) = frame else {
            return Ok(wet.clone());
        };
        let c = &self.client;
        let mut values = wet.try_to_vec::<f64>()?;
        let last = values.len() - 1;
        let index = |endpoint: usize| if endpoint == 0 { 0 } else { last };
        if let Some(traces) = frame.traces.as_deref() {
            let depths = traces.depth_m.try_to_vec::<f64>()?;
            for (&endpoint, depth) in traces.endpoints.iter().zip(depths) {
                values[index(endpoint)] = if depth > 0.0 { 1.0 } else { 0.0 };
            }
        }
        for port in &frame.ports {
            values[index(port.endpoint)] = if port.invariant.is_some() { 1.0 } else { 0.0 };
        }
        Ok(Tensor::from_slice(&values, &[values.len()], c.device())?)
    }

    /// Signed dry-front bounds `(u - φ, u + φ)` of trace states from area and velocity.
    ///
    /// The trace discharge is the exact product `A u`. The trace depth is not read.
    pub(super) fn surveyed_trace_fronts(
        &self,
        geometry: &SurveyedGeometry<R>,
        _depth: &Tensor<R>,
        area: &Tensor<R>,
        velocity: &Tensor<R>,
    ) -> Result<(Tensor<R>, Tensor<R>), ChannelError> {
        let c = &self.client;
        let (area, discharge) = expanded(c, "surveyed trace dry-front states", |a| {
            let zero = Tensor::zeros(area.shape(), DType::F64, c.device())?;
            let area = exact_over(a, area, &zero, &zero)?;
            let velocity = a.expansion_point(velocity)?;
            let discharge = a.expansion_mul(&area, &velocity)?;
            Ok((area, discharge))
        })?;
        let owners = Tensor::<R>::zeros(&[area.high().numel()], DType::I64, c.device())?;
        geometry
            .section
            .dry_front_bounds_tensor(&owners, &area, &discharge, &geometry.riemann)
            .map_err(riemann_error)
    }
}

#[cfg(test)]
mod tests {
    use crate::channel::flow::{
        ChannelTensorBoundary, ChannelTensorFluxPort,
        cfl_fronts::bound_dry_faces,
        geometry::TensorGeometry,
        surveyed_update::tests::{TestResult, cpu, moving_reach},
    };
    use numr::runtime::cpu::CpuRuntime;

    /// Trace states on the width-2 rectangle have `φ = 2 sqrt(g h)`.
    #[test]
    fn surveyed_trace_fronts_enclose_the_rectangular_invariant() -> TestResult {
        let reach = moving_reach::<CpuRuntime>(cpu(), &[20.0, 10.0], &[0.0; 2], &[0.0; 2])?;
        let context = &reach.context;
        let TensorGeometry::Surveyed(geometry) = &context.geometry else {
            return Err("surveyed geometry required".into());
        };
        let gravity = context.definition.config.gravity;
        let (depths, velocities) = ([2.0, 1.0, 0.0], [0.5, -0.25, 0.0]);
        let (lower, upper) = context.surveyed_trace_fronts(
            geometry,
            &context.tensor(&depths)?,
            &context.tensor(&depths.map(|h| 2.0 * h))?,
            &context.tensor(&velocities)?,
        )?;
        let (lower, upper) = (lower.try_to_vec::<f64>()?, upper.try_to_vec::<f64>()?);
        for lane in 0..2 {
            let phi = 2.0 * (gravity * depths[lane]).sqrt();
            let (exact_lower, exact_upper) = (velocities[lane] - phi, velocities[lane] + phi);
            assert!(lower[lane] <= exact_lower && lower[lane] >= exact_lower - 1e-9 * phi);
            assert!(upper[lane] >= exact_upper && upper[lane] <= exact_upper + 1e-9 * phi);
        }
        assert_eq!([lower[2], upper[2]], [0.0, 0.0]);
        Ok(())
    }

    /// A wet port exterior beside an all-dry reach forms a front only through its splice.
    #[test]
    fn a_wet_port_exterior_fronts_an_all_dry_reach() -> TestResult {
        let reach = moving_reach::<CpuRuntime>(cpu(), &[0.0, 0.0], &[0.0; 2], &[0.0; 2])?;
        let context = &reach.context;
        let port = ChannelTensorBoundary::FluxPort(Box::new(ChannelTensorFluxPort::new(
            context.tensor(&[-0.01])?,
            context.tensor(&[0.0])?,
            context.tensor(&[1.0])?,
            Some(context.tensor(&[7.0])?),
            "dry-front fixture",
        )?));
        let wall = ChannelTensorBoundary::Wall;
        let fields = context.fields(&reach.accepted.volume, &reach.accepted.momentum)?;
        assert!(
            context
                .candidate_speed(&fields, &wall, &wall)?
                .fronts
                .is_none()
        );
        let flux = context.flux(&fields, &port, &wall)?;
        let fronts = flux
            .dry_fronts
            .as_ref()
            .ok_or("the port front is missing")?;
        assert_eq!(fronts.left.wet.try_to_vec::<f64>()?, [1.0, 0.0, 0.0]);
        assert_eq!(fronts.left.rightward.try_to_vec::<f64>()?, [7.0, 0.0, 0.0]);
        let bounded = bound_dry_faces(&context.client, &flux.speed, fronts)?;
        assert_eq!(bounded.rightward.try_to_vec::<f64>()?[0], 7.0);
        Ok(())
    }

    /// Two wet cells of depths 2 m and 1 m at rest beside two dry cells, between walls.
    #[test]
    fn surveyed_fronts_enclose_the_rectangular_invariant() -> TestResult {
        let volumes = [20.0, 10.0, 0.0, 0.0];
        let reach = moving_reach::<CpuRuntime>(cpu(), &volumes, &[0.0; 4], &[0.0; 4])?;
        let context = &reach.context;
        let wall = ChannelTensorBoundary::Wall;
        let fields = context.fields(&reach.accepted.volume, &reach.accepted.momentum)?;
        let speed = context.candidate_speed(&fields, &wall, &wall)?;
        let fronts = speed
            .fronts
            .as_ref()
            .ok_or("a wet/dry face must yield fronts")?;
        let gravity = context.definition.config.gravity;
        // A rectangle has φ(h) = 2 sqrt(g h). At rest u + φ = φ.
        let deep = 2.0 * (gravity * 2.0).sqrt();
        let shallow = 2.0 * gravity.sqrt();
        let rightward = fronts.left.rightward.try_to_vec::<f64>()?;
        assert_eq!(
            fronts.left.wet.try_to_vec::<f64>()?,
            [1.0, 1.0, 1.0, 0.0, 0.0]
        );
        for (value, exact) in rightward.iter().zip([deep, deep, shallow]) {
            assert!(
                *value >= exact && *value <= exact * (1.0 + 1e-9),
                "{value} vs {exact}"
            );
        }
        assert_eq!(&rightward[3..], [0.0, 0.0]);
        // The right states are the cells, then the wall mirror of dry cell 3.
        let leftward = fronts.right.leftward.try_to_vec::<f64>()?;
        for (value, exact) in leftward.iter().zip([deep, shallow]) {
            assert!(
                *value >= exact && *value <= exact * (1.0 + 1e-9),
                "{value} vs {exact}"
            );
        }
        assert_eq!(&leftward[2..], [0.0, 0.0, 0.0]);
        let bounded = bound_dry_faces(&context.client, &speed.outer, fronts)?;
        let into_dry = bounded.rightward.try_to_vec::<f64>()?[2];
        assert!(
            into_dry >= deep,
            "{into_dry} must bound the deep cell front {deep}"
        );
        let local = speed.outer.rightward.try_to_vec::<f64>()?[2];
        assert!(
            local < deep,
            "the local face speed {local} misses the deep cell front"
        );
        Ok(())
    }
}
