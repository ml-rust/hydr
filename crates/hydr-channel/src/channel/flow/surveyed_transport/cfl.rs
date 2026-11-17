//! Surveyed CFL speeds and dry-front states from the certified point Riemann solver.

use super::error::riemann_error;
use crate::channel::flow::{
    ChannelError, ChannelTensorBoundary,
    admission::Fields,
    cfl_fronts::{DryFronts, SideFronts, has_dry_contact, side_fronts},
    cfl_speeds::CflSpeeds,
    conserved_update::{exact_over, expanded},
    context::Context,
    geometry::SurveyedGeometry,
    speed_memo::SpeedEntry,
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
        roots::{CertifiedBisectionAlgorithms, CertifiedUpperBracketAlgorithms},
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
        + CertifiedUpperBracketAlgorithms<R>,
{
    pub(in crate::channel::flow) fn surveyed_speed(
        &self,
        geometry: &SurveyedGeometry<R>,
        fields: &Fields<R>,
        left: &ChannelTensorBoundary<R>,
        right: &ChannelTensorBoundary<R>,
    ) -> Result<SpeedEntry<R>, ChannelError> {
        let result = self.surveyed_point_riemann(geometry, fields, left, right)?;
        Ok(SpeedEntry {
            speeds: CflSpeeds {
                outer: self.surveyed_face_speeds(result.wave_speed_m_per_s())?,
                internal: None,
                fronts: self.surveyed_fronts(geometry, fields, left, right)?,
            },
            riemann: Some(Arc::new(result)),
        })
    }

    /// Dry-front states from one certified potential per state.
    ///
    /// The bounds enclose `u + φ(A)` from above and `u - φ(A)` from below. Returns `None`
    /// without evaluating them when no face is wet on exactly one side.
    pub(in crate::channel::flow) fn surveyed_fronts(
        &self,
        geometry: &SurveyedGeometry<R>,
        fields: &Fields<R>,
        left: &ChannelTensorBoundary<R>,
        right: &ChannelTensorBoundary<R>,
    ) -> Result<Option<DryFronts<R>>, ChannelError> {
        let c = &self.client;
        let (area, discharge) = self.surveyed_states(geometry, fields, left, right)?;
        let states = area.high.numel();
        let faces = states - 1;
        let zero = self.constant(&area.high, 0.0)?;
        let wet = c.maximum(&c.ne(&area.high, &zero)?, &c.ne(&area.low, &zero)?)?;
        if !has_dry_contact(c, &wet.narrow(0, 0, faces)?, &wet.narrow(0, 1, faces)?)? {
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
        Ok(Some(DryFronts {
            left: side(0)?,
            right: side(1)?,
        }))
    }
}

#[cfg(test)]
mod tests {
    use crate::channel::flow::{
        ChannelTensorBoundary,
        cfl_fronts::bound_dry_faces,
        surveyed_update::tests::{TestResult, cpu, moving_reach},
    };
    use numr::runtime::cpu::CpuRuntime;

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
