//! Face-side dry-front states for the channel CFL rule, with scalar boundary traces.

use super::{
    ChannelError,
    boundary::{NativeBoundaryTraces, ScalarBoundaryFrame},
    cfl_fronts::{DryFronts, SideFronts, has_dry_contact, side_fronts},
    context::Context,
};
use crate::numerics::{TensorFaceState, dry_front_bounds};
use numr::dtype::DType;
use numr::ops::{BinaryOps, CompareOps, ConditionalOps, ReduceOps, ScalarOps, ShapeOps, UnaryOps};
use numr::runtime::Runtime;
use numr::tensor::Tensor;

/// Signed dry-front bounds `(u - φ, u + φ)` of trace states from depth, area and x velocity.
pub(super) type TraceFronts<'a, R> =
    dyn Fn(&Tensor<R>, &Tensor<R>, &Tensor<R>) -> Result<(Tensor<R>, Tensor<R>), ChannelError> + 'a;

impl<R: Runtime<DType = DType>> Context<R>
where
    R::Client: BinaryOps<R>
        + CompareOps<R>
        + ConditionalOps<R>
        + ReduceOps<R>
        + ScalarOps<R>
        + ShapeOps<R>
        + UnaryOps<R>,
{
    /// Rectangular fronts from the reconstructed face states, with `φ = 2 sqrt(g h)`.
    pub fn rectangular_fronts(
        &self,
        left: &TensorFaceState<R>,
        right: &TensorFaceState<R>,
        frame: Option<&ScalarBoundaryFrame<R>>,
    ) -> Result<Option<DryFronts<R>>, ChannelError> {
        let c = &self.client;
        let gravity = self.definition.config.gravity;
        let side = |state: &TensorFaceState<R>| -> Result<SideFronts<R>, ChannelError> {
            let wet = c.gt(&state.h, &self.constant(&state.h, 0.0)?)?;
            let depth = c.where_cond(&wet, &state.h, &self.constant(&state.h, 1.0)?)?;
            let velocity = c.div(&state.q_normal, &depth)?;
            self.finite(&velocity, "rectangular dry-front velocity")?;
            let (lower, upper) = dry_front_bounds(c, &state.h, &velocity, gravity)?;
            side_fronts(c, &lower, &upper, &wet)
        };
        let trace = |depth: &Tensor<R>,
                     _area: &Tensor<R>,
                     velocity: &Tensor<R>|
         -> Result<(Tensor<R>, Tensor<R>), ChannelError> {
            Ok(dry_front_bounds(c, depth, velocity, gravity)?)
        };
        self.dry_fronts(side(left)?, side(right)?, frame, &trace)
    }

    /// Fronts with each scalar trace in place of its exterior copy.
    ///
    /// Returns `None` when no face has water on exactly one side.
    pub fn dry_fronts(
        &self,
        left: SideFronts<R>,
        right: SideFronts<R>,
        frame: Option<&ScalarBoundaryFrame<R>>,
        trace: &TraceFronts<'_, R>,
    ) -> Result<Option<DryFronts<R>>, ChannelError> {
        let (left, right) = match frame {
            Some(frame) => self.splice_traces(left, right, &frame.traces, trace)?,
            None => (left, right),
        };
        if !has_dry_contact(&self.client, &left.wet, &right.wet)? {
            return Ok(None);
        }
        Ok(Some(DryFronts { left, right }))
    }

    /// Replace the outer state at each scalar endpoint face with its boundary trace.
    ///
    /// The trace velocity points outward. Its compatibility defect widens both front bounds.
    fn splice_traces(
        &self,
        left: SideFronts<R>,
        right: SideFronts<R>,
        traces: &NativeBoundaryTraces<R>,
        trace: &TraceFronts<'_, R>,
    ) -> Result<(SideFronts<R>, SideFronts<R>), ChannelError> {
        let Some(last) = traces.endpoints.len().checked_sub(1) else {
            return Ok((left, right));
        };
        let c = &self.client;
        let signs: Vec<f64> = traces
            .endpoints
            .iter()
            .map(|&endpoint| if endpoint == 0 { -1.0 } else { 1.0 })
            .collect();
        let velocity = c.mul(&self.tensor(&signs)?, &traces.outward_velocity_m_per_s)?;
        let defect = &traces.velocity_compatibility_abs_bound_m_per_s;
        let (_, upper) = trace(&traces.depth_m, &traces.area_m2, &c.add(&velocity, defect)?)?;
        let (lower, _) = trace(&traces.depth_m, &traces.area_m2, &c.sub(&velocity, defect)?)?;
        let wet = c.gt(&traces.depth_m, &self.constant(&traces.depth_m, 0.0)?)?;
        let fronts = side_fronts(c, &lower, &upper, &wet)?;
        let left = if traces.endpoints.first() == Some(&0) {
            self.splice_side(left, 0, &fronts, 0)?
        } else {
            left
        };
        let right = if traces.endpoints.last() == Some(&1) {
            let face = right.wet.numel().saturating_sub(1);
            self.splice_side(right, face, &fronts, last)?
        } else {
            right
        };
        Ok((left, right))
    }

    fn splice_side(
        &self,
        side: SideFronts<R>,
        face: usize,
        source: &SideFronts<R>,
        entry: usize,
    ) -> Result<SideFronts<R>, ChannelError> {
        let n = side.wet.numel();
        let splice = |target: &Tensor<R>, value: &Tensor<R>| -> Result<Tensor<R>, ChannelError> {
            let value = value.narrow(0, entry, 1)?;
            let mut pieces = Vec::with_capacity(3);
            if face > 0 {
                pieces.push(target.narrow(0, 0, face)?);
            }
            pieces.push(value);
            if face + 1 < n {
                pieces.push(target.narrow(0, face + 1, n - face - 1)?);
            }
            let refs: Vec<_> = pieces.iter().collect();
            Ok(self.client.cat(&refs, 0)?)
        };
        Ok(SideFronts {
            rightward: splice(&side.rightward, &source.rightward)?,
            leftward: splice(&side.leftward, &source.leftward)?,
            wet: splice(&side.wet, &source.wet)?,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::super::{ChannelReach, ChannelTensorBoundary, cfl_fronts::bound_dry_faces};
    use crate::channel::{
        ChannelConfig, ChannelGeometry, ChannelInitialState, ChannelReachDefinition,
        ChannelTriangularReconstruction, RectangularChannelCell, TriangularChannelCell,
    };
    use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime, ParallelismConfig};

    type TestResult = Result<(), Box<dyn std::error::Error>>;

    fn reach(
        geometry: ChannelGeometry,
        config: ChannelConfig,
        volumes: &[f64],
    ) -> Result<ChannelReach<CpuRuntime>, Box<dyn std::error::Error>> {
        let client = CpuClient::new(CpuDevice::new())
            .with_parallelism(ParallelismConfig::new(Some(1), None));
        Ok(ChannelReach::new(
            client,
            ChannelReachDefinition {
                vertical_datum: "synthetic datum".into(),
                provenance: "dry-front CFL fixture".into(),
                geometry,
                lateral_bindings: vec![],
                config,
            },
            ChannelInitialState {
                time_s: 0.0,
                volume_m3: volumes.to_vec(),
                longitudinal_momentum_m4_per_s: vec![0.0; volumes.len()],
            },
        )?)
    }

    /// Two wet cells at rest beside a dry cell, between walls, with `φ` of each wet cell.
    ///
    /// `exact` compares every state front with `φ`. Reconstructed face states skip that check.
    fn check(reach: &ChannelReach<CpuRuntime>, phi: [f64; 2], exact: bool) -> TestResult {
        let context = &reach.context;
        let wall = ChannelTensorBoundary::Wall;
        let fields = context.fields(&reach.accepted.volume, &reach.accepted.momentum)?;
        let speed = context.candidate_speed(&fields, &wall, &wall)?;
        let fronts = speed
            .fronts
            .as_ref()
            .ok_or("a wet/dry face must yield fronts")?;
        let rightward = fronts.left.rightward.try_to_vec::<f64>()?;
        // The left wall mirrors cell 0. At rest `u + φ = φ`.
        if exact {
            for (value, phi) in rightward.iter().zip([phi[0], phi[0], phi[1]]) {
                assert!(
                    *value >= phi && *value <= phi * (1.0 + 1e-12),
                    "{value} vs {phi}"
                );
            }
        }
        assert_eq!(rightward[3], 0.0);
        let bounded = bound_dry_faces(&context.client, &speed.outer, fronts)?;
        let into_dry = bounded.rightward.try_to_vec::<f64>()?[2];
        assert!(
            into_dry >= phi[0],
            "{into_dry} must bound the deep cell front {}",
            phi[0]
        );
        Ok(())
    }

    #[test]
    fn rectangular_fronts_use_twice_the_sound_speed() -> TestResult {
        let cell = |id: &str| RectangularChannelCell {
            id: id.into(),
            length_m: 1.0,
            width_m: 1.0,
            bed_elevation_m: 0.0,
            maximum_depth_m: 10.0,
            manning_n: 0.0,
        };
        let geometry = ChannelGeometry::Rectangular {
            cells: vec![cell("a"), cell("b"), cell("c")],
        };
        // Depths 2 m and 1 m with g = 1: φ = 2 sqrt(h).
        let config = ChannelConfig {
            gravity: 1.0,
            ..ChannelConfig::default()
        };
        let reach = reach(geometry, config, &[2.0, 1.0, 0.0])?;
        check(&reach, [2.0 * 2f64.sqrt(), 2.0], true)
    }

    #[test]
    fn triangular_fronts_use_four_times_the_sound_speed() -> TestResult {
        let cell = |id: &str| TriangularChannelCell {
            id: id.into(),
            length_m: 1.0,
            maximum_depth_m: 10.0,
            manning_n: 0.0,
        };
        let geometry = || ChannelGeometry::PrismaticTriangular {
            left_bank_slope_h_per_v: 1.0,
            right_bank_slope_h_per_v: 1.0,
            bed_elevation_m: 0.0,
            cells: vec![cell("a"), cell("b"), cell("c")],
        };
        // A = h² gives depths 2 m and 1 m. With g = 8, c = 2 sqrt(h) and φ = 4c.
        let phi = [8.0 * 2f64.sqrt(), 8.0];
        for (reconstruction, exact) in [
            (ChannelTriangularReconstruction::FirstOrderReference, true),
            (ChannelTriangularReconstruction::LimitedLinear, false),
        ] {
            let config = ChannelConfig {
                gravity: 8.0,
                triangular_reconstruction: reconstruction,
                ..ChannelConfig::default()
            };
            check(&reach(geometry(), config, &[4.0, 1.0, 0.0])?, phi, exact)?;
        }
        Ok(())
    }
}
