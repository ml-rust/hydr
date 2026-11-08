//! Hydrostatic reconstruction and conservative normal Riemann wave bounds.

use std::fmt;

/// Cell state expressed in the face's common normal and tangent directions.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct FaceState {
    /// Nonnegative water depth in metres. Zero depth requires zero discharge.
    pub h: f64,
    /// Discharge per width along the common normal, in square metres per second.
    pub q_normal: f64,
    /// Discharge per width along the common tangent, in square metres per second.
    pub q_tangent: f64,
    /// Bed elevation in metres, using the same datum on both sides.
    pub bed: f64,
}

/// Depth-bracket tolerances and separate expansion/bisection iteration limits.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct WaveConfig {
    pub absolute_depth: f64,
    pub relative_depth: f64,
    pub iterations: usize,
}

impl Default for WaveConfig {
    fn default() -> Self {
        Self {
            absolute_depth: 1e-12,
            relative_depth: 1e-12,
            iterations: 128,
        }
    }
}

/// Shared mass/normal/tangent flux and side-specific normal pressure additions.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct FaceFlux {
    pub shared: [f64; 3],
    pub left_pressure: f64,
    pub right_pressure: f64,
    /// Maximum absolute normal wave bound, in metres per second.
    pub speed: f64,
}

/// Invalid input or unrepresentable face calculation, retaining original states.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct FluxError {
    pub reason: &'static str,
    pub left: FaceState,
    pub right: FaceState,
}

impl fmt::Display for FluxError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "face flux error: {}; left={:?}, right={:?}; inspect face inputs and wave limits",
            self.reason, self.left, self.right
        )
    }
}

impl std::error::Error for FluxError {}

type Calculation<T> = Result<T, &'static str>;

/// Calculate a flux from finite cell states and positive finite gravity.
///
/// The caller assigns face identity to errors and rotates momentum for y-faces.
/// Positive films remain wet. Reflected states implement walls at the caller.
pub(crate) fn hydrostatic_hll(
    left: FaceState,
    right: FaceState,
    gravity: f64,
    config: WaveConfig,
) -> Result<FaceFlux, FluxError> {
    calculate(left, right, gravity, config).map_err(|reason| FluxError {
        reason,
        left,
        right,
    })
}

fn calculate(
    left: FaceState,
    right: FaceState,
    gravity: f64,
    config: WaveConfig,
) -> Calculation<FaceFlux> {
    check_state(left)?;
    check_state(right)?;
    if !gravity.is_finite() || gravity <= 0.0 {
        return Err("gravity must be finite and positive");
    }
    if !config.absolute_depth.is_finite()
        || config.absolute_depth <= 0.0
        || !config.relative_depth.is_finite()
        || config.relative_depth < 0.0
        || config.iterations == 0
    {
        return Err("invalid root-bracket tolerances or iteration limit");
    }
    let bed = left.bed.max(right.bed);
    let l = reconstruct(left, bed)?;
    let r = reconstruct(right, bed)?;
    let left_pressure = correction(left.h, l.h, gravity)?;
    let right_pressure = correction(right.h, r.h, gravity)?;
    if l.h == 0.0 && r.h == 0.0 {
        return Ok(FaceFlux {
            shared: [0.0; 3],
            left_pressure,
            right_pressure,
            speed: 0.0,
        });
    }
    let (sl, sr) = wave_bounds(l, r, gravity, config)?;
    if sl > sr {
        return Err("normal wave bounds are reversed");
    }
    let fl = physical_flux(l, gravity)?;
    let fr = physical_flux(r, gravity)?;
    let shared = if sl >= 0.0 {
        fl
    } else if sr <= 0.0 {
        fr
    } else {
        let span = finite(sr - sl, "HLL wave span overflow")?;
        if span <= 0.0 {
            return Err("HLL wave span is nonpositive");
        }
        let wl = sr / span;
        let wr = -sl / span;
        let diffusion = finite(wl * sl, "HLL diffusion overflow")?;
        let ul = [l.h, l.q_normal, l.q_tangent];
        let ur = [r.h, r.q_normal, r.q_tangent];
        let mut result = [0.0; 3];
        for component in 0..3 {
            result[component] = finite(
                wl * fl[component]
                    + wr * fr[component]
                    + diffusion * (ur[component] - ul[component]),
                "HLL flux overflow",
            )?;
        }
        result
    };
    Ok(FaceFlux {
        shared,
        left_pressure,
        right_pressure,
        speed: sl.abs().max(sr.abs()),
    })
}

fn check_state(state: FaceState) -> Calculation<()> {
    if ![state.h, state.q_normal, state.q_tangent, state.bed]
        .into_iter()
        .all(f64::is_finite)
        || state.h < 0.0
        || (state.h == 0.0 && (state.q_normal != 0.0 || state.q_tangent != 0.0))
    {
        return Err("state requires finite values, nonnegative depth, and zero dry discharge");
    }
    Ok(())
}

fn reconstruct(state: FaceState, bed: f64) -> Calculation<FaceState> {
    // Subtract the bed step first to preserve films at large bed elevations.
    let step = finite(bed - state.bed, "bed step overflow")?;
    let h = (state.h - step).max(0.0);
    let ratio = if state.h > 0.0 { h / state.h } else { 0.0 };
    Ok(FaceState {
        h,
        q_normal: state.q_normal * ratio,
        q_tangent: state.q_tangent * ratio,
        bed,
    })
}

fn correction(h: f64, reconstructed: f64, gravity: f64) -> Calculation<f64> {
    if h == reconstructed {
        return Ok(0.0);
    }
    finite(
        (0.5 * gravity * (h - reconstructed)) * (h + reconstructed),
        "hydrostatic pressure correction overflow",
    )
}

fn physical_flux(state: FaceState, gravity: f64) -> Calculation<[f64; 3]> {
    if state.h == 0.0 {
        return Ok([0.0; 3]);
    }
    let u = finite(state.q_normal / state.h, "normal velocity overflow")?;
    Ok([
        state.q_normal,
        finite(
            state.q_normal * u + (0.5 * gravity * state.h) * state.h,
            "normal physical flux overflow",
        )?,
        finite(state.q_tangent * u, "tangent physical flux overflow")?,
    ])
}

fn wave_bounds(
    left: FaceState,
    right: FaceState,
    gravity: f64,
    config: WaveConfig,
) -> Calculation<(f64, f64)> {
    let velocity = |s: FaceState| {
        if s.h == 0.0 {
            Ok(0.0)
        } else {
            finite(s.q_normal / s.h, "normal velocity overflow")
        }
    };
    let ul = velocity(left)?;
    let ur = velocity(right)?;
    let cl = finite(gravity * left.h, "left celerity overflow")?.sqrt();
    let cr = finite(gravity * right.h, "right celerity overflow")?.sqrt();
    let (sl, sr) = if right.h == 0.0 {
        (ul - cl, ul + 2.0 * cl)
    } else if left.h == 0.0 {
        (ur - 2.0 * cr, ur + cr)
    } else {
        let jump = finite(ur - ul, "normal velocity jump overflow")?;
        let vacuum = finite(2.0 * (cl + cr), "vacuum threshold overflow")?;
        if jump >= vacuum {
            (ul - cl, ur + cr)
        } else {
            let hi = root_upper(left.h, right.h, cl, cr, jump, gravity, config)?;
            (
                ul - cl * shock_multiplier(hi, left.h)?,
                ur + cr * shock_multiplier(hi, right.h)?,
            )
        }
    };
    Ok((
        finite(sl, "left wave bound overflow")?,
        finite(sr, "right wave bound overflow")?,
    ))
}

fn shock_multiplier(upper: f64, h: f64) -> Calculation<f64> {
    if upper <= h {
        return Ok(1.0);
    }
    let ratio = finite(upper / h, "shock depth ratio overflow")?;
    Ok(finite(0.5 * ratio * (1.0 + ratio), "shock multiplier overflow")?.sqrt())
}

fn root_upper(
    hl: f64,
    hr: f64,
    cl: f64,
    cr: f64,
    jump: f64,
    gravity: f64,
    config: WaveConfig,
) -> Calculation<f64> {
    let residual = |h| {
        finite(
            wave_curve(h, hl, cl, gravity)? + wave_curve(h, hr, cr, gravity)? + jump,
            "root residual overflow",
        )
    };
    let mut lo = 0.0;
    let mut hi = hl.max(hr);
    let mut bracketed = residual(hi)? >= 0.0;
    if !bracketed {
        // For H >= max(hl, hr), each shock curve exceeds
        // (H - max(hl, hr)) * sqrt(g / (2h)).
        let coefficient_scale = gravity.sqrt() * std::f64::consts::FRAC_1_SQRT_2;
        let coefficient = finite(
            coefficient_scale / hl.sqrt() + coefficient_scale / hr.sqrt(),
            "compression bracket coefficient overflow",
        )?;
        if coefficient <= 0.0 {
            return Err("compression bracket coefficient is nonpositive");
        }
        hi = finite(
            hi + (-jump).max(0.0) / coefficient,
            "compression bracket depth overflow",
        )?;
        bracketed = residual(hi)? >= 0.0;
    }
    for _ in 0..config.iterations {
        if bracketed {
            break;
        }
        hi = finite(hi * 2.0, "root bracket expansion overflow")?;
        bracketed = residual(hi)? >= 0.0;
    }
    if !bracketed {
        return Err("root bracket expansion iteration limit reached");
    }
    for _ in 0..config.iterations {
        let tolerance = finite(
            config.absolute_depth + config.relative_depth * hi,
            "root tolerance overflow",
        )?;
        if hi - lo <= tolerance {
            return Ok(hi);
        }
        let middle = lo + 0.5 * (hi - lo);
        if middle == lo || middle == hi {
            return Err("root bracket cannot reach requested tolerance");
        }
        if residual(middle)? >= 0.0 {
            hi = middle;
        } else {
            lo = middle;
        }
    }
    let tolerance = finite(
        config.absolute_depth + config.relative_depth * hi,
        "root tolerance overflow",
    )?;
    if hi - lo <= tolerance {
        Ok(hi)
    } else {
        Err("root bisection iteration limit reached")
    }
}

fn wave_curve(h: f64, reference: f64, c: f64, gravity: f64) -> Calculation<f64> {
    if h <= reference {
        finite(
            2.0 * (finite(gravity * h, "root celerity overflow")?.sqrt() - c),
            "rarefaction curve overflow",
        )
    } else {
        // Factor out reference before taking square roots to avoid 1/reference.
        finite(
            ((h - reference) / reference.sqrt())
                * gravity.sqrt()
                * (0.5 * (1.0 + reference / h)).sqrt(),
            "shock curve overflow",
        )
    }
}

fn finite(value: f64, reason: &'static str) -> Calculation<f64> {
    if value.is_finite() {
        Ok(value)
    } else {
        Err(reason)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn state(h: f64, u: f64, v: f64, bed: f64) -> FaceState {
        FaceState {
            h,
            q_normal: h * u,
            q_tangent: h * v,
            bed,
        }
    }

    #[test]
    fn uniform_flow_transports_normal_and_tangent_momentum() -> Result<(), FluxError> {
        let s = state(2.0, 3.0, -4.0, 0.0);
        let flux = hydrostatic_hll(s, s, 9.81, WaveConfig::default())?;
        for (actual, expected) in flux.shared.into_iter().zip([6.0, 37.62, -24.0]) {
            assert!((actual - expected).abs() < 1e-12);
        }
        Ok(())
    }

    #[test]
    fn stepped_lake_retains_each_cells_pressure() -> Result<(), FluxError> {
        let flux = hydrostatic_hll(
            state(2.0, 0.0, 0.0, 0.0),
            state(1.0, 0.0, 0.0, 1.0),
            9.81,
            WaveConfig::default(),
        )?;
        assert_eq!(flux.shared[0], 0.0);
        assert!((flux.shared[1] + flux.left_pressure - 19.62).abs() < 1e-12);
        assert!((flux.shared[1] + flux.right_pressure - 4.905).abs() < 1e-12);
        Ok(())
    }

    #[test]
    fn dry_front_uses_double_celerity_bound() -> Result<(), FluxError> {
        let wet = state(1.0, 0.0, 0.0, 0.0);
        let dry = state(0.0, 0.0, 0.0, 0.0);
        let forward = hydrostatic_hll(wet, dry, 9.81, WaveConfig::default())?;
        let reverse = hydrostatic_hll(dry, wet, 9.81, WaveConfig::default())?;
        assert_eq!(forward.speed, 2.0 * 9.81_f64.sqrt());
        assert_eq!(forward.shared[0], -reverse.shared[0]);
        Ok(())
    }

    #[test]
    fn separating_wet_states_use_vacuum_heads() -> Result<(), FluxError> {
        let flux = hydrostatic_hll(
            state(1.0, -8.0, 0.0, 0.0),
            state(1.0, 8.0, 0.0, 0.0),
            9.81,
            WaveConfig::default(),
        )?;
        assert_eq!(flux.speed, 8.0 + 9.81_f64.sqrt());
        assert_eq!(flux.shared[0], 0.0);
        Ok(())
    }

    #[test]
    fn root_retains_nonnegative_residual() -> Calculation<()> {
        let c = 9.81_f64.sqrt();
        let upper = root_upper(1.0, 1.0, c, c, -20.0, 9.81, WaveConfig::default())?;
        assert!(2.0 * wave_curve(upper, 1.0, c, 9.81)? - 20.0 >= 0.0);
        assert!(upper > 1.0);
        Ok(())
    }

    #[test]
    fn iteration_exhaustion_retains_original_states() {
        let left = state(1.0, 0.0, 0.0, 0.0);
        let right = state(0.25, 0.0, 0.0, 0.0);
        let result = hydrostatic_hll(
            left,
            right,
            9.81,
            WaveConfig {
                iterations: 1,
                ..WaveConfig::default()
            },
        );
        assert!(
            matches!(result, Err(FluxError { reason: "root bisection iteration limit reached", left: l, right: r }) if l == left && r == right)
        );
    }

    #[test]
    fn positive_films_survive_large_bed_datum() -> Result<(), FluxError> {
        let film = state(1e-15, 1.0, 0.0, 1e12);
        let flux = hydrostatic_hll(film, film, 9.81, WaveConfig::default())?;
        assert_eq!(flux.shared[0], film.q_normal);
        assert!(flux.speed >= 1.0);
        Ok(())
    }

    #[test]
    fn compressed_positive_films_retain_a_finite_bracket() -> Result<(), FluxError> {
        let left = FaceState {
            h: 5.7660025090732485e-86,
            q_normal: 1.0669783925783614e-85,
            q_tangent: 0.0,
            bed: 0.0,
        };
        let right = FaceState {
            h: 4.440970925099901e-89,
            q_normal: 8.183764274505537e-89,
            q_tangent: 0.0,
            bed: 0.0,
        };
        let flux = hydrostatic_hll(left, right, 9.81, WaveConfig::default())?;
        assert!(flux.speed.is_finite());
        assert!(flux.shared.into_iter().all(f64::is_finite));
        Ok(())
    }

    #[test]
    fn shock_curve_supports_subnormal_reference_depth() -> Calculation<()> {
        let h: f64 = 1e-320;
        let expected = 9.81_f64.sqrt() * h.sqrt() * 0.75_f64.sqrt();
        let actual = wave_curve(2.0 * h, h, (9.81 * h).sqrt(), 9.81)?;
        assert!((actual / expected - 1.0).abs() < 1e-12);
        Ok(())
    }

    #[test]
    fn overflow_and_dry_discharge_return_errors() {
        let ordinary = state(1.0, 0.0, 0.0, 0.0);
        let huge = state(f64::MAX, 0.0, 0.0, 0.0);
        assert!(hydrostatic_hll(huge, ordinary, 9.81, WaveConfig::default()).is_err());
        let dry = FaceState {
            h: 0.0,
            q_normal: 1.0,
            q_tangent: 0.0,
            bed: 0.0,
        };
        assert!(hydrostatic_hll(dry, ordinary, 9.81, WaveConfig::default()).is_err());
    }
}
