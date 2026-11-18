use super::values;
use crate::channel::flow::output::ChannelFrictionGeometryDiagnostics;
use crate::channel::flow::{ChannelBoundaryTrace, ChannelRiemannDiagnostics};

pub(super) fn riemann(
    a: Option<&ChannelRiemannDiagnostics>,
    b: Option<&ChannelRiemannDiagnostics>,
) -> bool {
    match (a, b) {
        (None, None) => true,
        (Some(a), Some(b)) => {
            a.root_iterations == b.root_iterations
                && a.surveyed_work == b.surveyed_work
                && values(&a.water_flux_width_m3_per_s, &b.water_flux_width_m3_per_s)
                && values(
                    &a.momentum_flux_width_m4_per_s2,
                    &b.momentum_flux_width_m4_per_s2,
                )
                && values(
                    &[
                        a.max_water_flux_width_m3_per_s,
                        a.max_momentum_flux_width_m4_per_s2,
                    ],
                    &[
                        b.max_water_flux_width_m3_per_s,
                        b.max_momentum_flux_width_m4_per_s2,
                    ],
                )
        }
        _ => false,
    }
}

pub(super) fn friction_geometry(
    a: Option<&ChannelFrictionGeometryDiagnostics>,
    b: Option<&ChannelFrictionGeometryDiagnostics>,
) -> bool {
    match (a, b) {
        (None, None) => true,
        (Some(a), Some(b)) => {
            values(
                &a.momentum_abs_bound_m4_per_s,
                &b.momentum_abs_bound_m4_per_s,
            ) && a.max_momentum_abs_bound_m4_per_s.to_bits()
                == b.max_momentum_abs_bound_m4_per_s.to_bits()
        }
        _ => false,
    }
}

pub(super) fn boundary_trace(
    a: Option<&ChannelBoundaryTrace>,
    b: Option<&ChannelBoundaryTrace>,
) -> bool {
    match (a, b) {
        (None, None) => true,
        (Some(a), Some(b)) => a.same_bits(b),
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::channel::flow::snapshot::fixture::trace;
    use crate::channel::{ChannelError, ChannelReservoirTrace};

    #[test]
    fn riemann_diagnostics_retain_every_projected_bit() -> Result<(), ChannelError> {
        let original = ChannelRiemannDiagnostics {
            surveyed_work: None,
            water_flux_width_m3_per_s: vec![0.0, 1e-12],
            momentum_flux_width_m4_per_s2: vec![2e-12, 0.0],
            max_water_flux_width_m3_per_s: 1e-12,
            max_momentum_flux_width_m4_per_s2: 2e-12,
            root_iterations: 7,
        };
        assert!(riemann(None, None));
        assert!(!riemann(Some(&original), None));
        assert!(!riemann(None, Some(&original)));
        assert!(riemann(Some(&original), Some(&original)));
        for index in 0..5 {
            let mut changed = original.clone();
            match index {
                0 => changed.water_flux_width_m3_per_s[0] = -0.0,
                1 => changed.momentum_flux_width_m4_per_s2[1] = -0.0,
                2 => changed.max_water_flux_width_m3_per_s = 0.0,
                3 => changed.max_momentum_flux_width_m4_per_s2 = 0.0,
                _ => changed.root_iterations += 1,
            }
            assert!(!riemann(Some(&original), Some(&changed)));
        }
        let mut surveyed = original.clone();
        surveyed.surveyed_work = Some(crate::channel::PrismaticRiemannWork::default());
        assert!(!riemann(Some(&original), Some(&surveyed)));
        let work = surveyed.surveyed_work.as_mut().ok_or_else(|| {
            ChannelError::input("test surveyed work absent: retain the original diagnostics")
        })?;
        work.maximum_root_iterations = 7;
        for index in 0..12 {
            let mut changed = surveyed.clone();
            let work = changed.surveyed_work.as_mut().ok_or_else(|| {
                ChannelError::input("test surveyed work absent: retain the original diagnostics")
            })?;
            match index {
                0 => work.root_solves += 1,
                1 => work.root_iterations += 1,
                2 => work.maximum_root_iterations += 1,
                3 => work.residual_points += 1,
                4 => work.simpson_value_points += 1,
                5 => work.simpson_derivative_points += 1,
                6 => work.refinements += 1,
                7 => work.maximum_simpson_panels += 1,
                8 => work.bracket_expansion_rounds += 1,
                9 => work.maximum_bracket_expansion_rounds += 1,
                10 => work.bracket_solves += 1,
                _ => changed.surveyed_work = None,
            }
            assert!(!riemann(Some(&surveyed), Some(&changed)));
        }
        Ok(())
    }

    #[test]
    fn friction_geometry_diagnostics_retain_every_projected_bit() {
        let original = ChannelFrictionGeometryDiagnostics {
            momentum_abs_bound_m4_per_s: vec![0.0, 1e-12],
            max_momentum_abs_bound_m4_per_s: 1e-12,
        };
        assert!(friction_geometry(None, None));
        assert!(!friction_geometry(Some(&original), None));
        assert!(!friction_geometry(None, Some(&original)));
        assert!(friction_geometry(Some(&original), Some(&original)));
        let mut changed = original.clone();
        changed.momentum_abs_bound_m4_per_s[0] = -0.0;
        assert!(!friction_geometry(Some(&original), Some(&changed)));
        let mut changed = original.clone();
        changed.max_momentum_abs_bound_m4_per_s = 0.0;
        assert!(!friction_geometry(Some(&original), Some(&changed)));
    }

    #[test]
    fn boundary_receipts_retain_every_field_and_optional_entry() {
        let original = trace();
        assert!(boundary_trace(None, None));
        assert!(boundary_trace(Some(&original), Some(&original)));
        assert!(!boundary_trace(None, Some(&original)));
        assert!(!boundary_trace(Some(&original), None));
        for index in 0..13 {
            let mut changed = original.clone();
            match index {
                0 => changed.depth_m = -0.0,
                1 => changed.area_m2 = -0.0,
                2 => changed.outward_velocity_m_per_s = -0.0,
                3 => changed.outward_discharge_m3_per_s = -0.0,
                4 => changed.momentum_flux_m4_per_s2 = -0.0,
                5 => changed.signal_speed_bound_m_per_s = -0.0,
                6 => changed.velocity_compatibility_abs_bound_m_per_s = -0.0,
                7 => changed.water_compatibility_abs_bound_m3_per_s = -0.0,
                8 => changed.momentum_compatibility_abs_bound_m4_per_s2 = -0.0,
                9 => changed.water_flux_width_m3_per_s = -0.0,
                10 => changed.momentum_flux_width_m4_per_s2 = -0.0,
                11 => changed.cap_compatibility_approximation = true,
                _ => changed.root_iterations += 1,
            }
            assert!(!boundary_trace(Some(&original), Some(&changed)));
        }
    }

    #[test]
    fn reservoir_receipts_retain_each_bound_bit_and_policy_presence() -> Result<(), ChannelError> {
        let mut original = trace();
        original.reservoir = Some(ChannelReservoirTrace {
            depth_lower_m: 0.0,
            depth_upper_m: 0.0,
            head_compatibility_abs_bound_m: 0.0,
            dissipation_head_lower_m: 0.0,
            dissipation_head_upper_m: 0.0,
        });
        assert!(original.same_bits(&original));
        assert!(!original.same_bits(&trace()));
        for index in 0..5 {
            let mut changed = original.clone();
            let receipt = changed.reservoir.as_mut().ok_or_else(|| {
                ChannelError::input(
                    "test reservoir receipt absent: retain the original certificate",
                )
            })?;
            match index {
                0 => receipt.depth_lower_m = -0.0,
                1 => receipt.depth_upper_m = -0.0,
                2 => receipt.head_compatibility_abs_bound_m = -0.0,
                3 => receipt.dissipation_head_lower_m = -0.0,
                _ => receipt.dissipation_head_upper_m = -0.0,
            }
            assert!(!original.same_bits(&changed));
        }
        Ok(())
    }
}
