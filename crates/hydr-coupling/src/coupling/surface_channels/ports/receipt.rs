//! Port receipts of one accepted interval: regimes, pairing, and lifetime port counters.
//!
//! Signed port volumes are positive from the surface into the channel, as link volumes are.
//! The reach side counts `rate · dt` of its own port rate. The surface side counts the sum of
//! its per-face volumes `f_k w_k dt`.

use super::super::SurfaceChannelError;
use super::super::accounting::within;
use crate::numerics::outward::{upward_add, upward_mul};

/// Base of the pairing bound factor `2n + 2` of an `n`-face port, in units of `f64::EPSILON`.
const PAIR_BASE_ULPS: f64 = 2.0;

/// Flow regime of one port over one interval.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in super::super) enum PortRegime {
    /// Both waves straddle the port. The HLL flux mixes both states.
    Subcritical,
    /// Every wave runs into the surface. The flux is the reach state's own flux.
    SupercriticalToSurface,
    /// Every wave runs into the reach. The flux is the surface state's own flux.
    SupercriticalToReach,
    /// A rarefaction fan holds its sonic point at the port.
    Transcritical,
    /// One side holds no water above the higher bed.
    WetDryFront,
    /// Neither side holds water above the higher bed. The port moves nothing.
    Dry,
}

/// Host flux receipt of one port for one attempt, from the surface stage.
#[derive(Clone, Debug, PartialEq)]
pub(in super::super) struct PortFlux {
    /// Water rate into the surface of each face, `f_k w_k`, in face order, in m³/s.
    pub face_rate_m3_per_s: Vec<f64>,
    /// Water rate leaving the reach, in m³/s.
    pub outward_water_rate_m3_per_s: f64,
    /// Normal momentum rate along the reach's outward normal, pressure included, in m⁴/s².
    pub outward_momentum_rate_m4_per_s2: f64,
    /// Momentum rate entering the surface cells in grid x and y, in m⁴/s².
    pub surface_momentum_rate_m4_per_s2: [f64; 2],
    /// Momentum leaving the reach minus momentum entering the surface, grid x and y, in m⁴/s².
    pub junction_reaction_rate_m4_per_s2: [f64; 2],
    /// Signal speed bound the reach receives, in m/s.
    pub signal_speed_m_per_s: f64,
    /// Into-reach invariant `u + 2c` of the surface water, `None` when every face cell is dry.
    pub exterior_invariant_m_per_s: Option<f64>,
    pub regime: PortRegime,
}

/// Receipt of one port over one accepted interval.
#[derive(Clone, Debug, PartialEq)]
pub(in super::super) struct PortTransfer {
    pub port_id: String,
    /// Water the reach takes from the surface through the port, in m³. Negative leaves it.
    pub signed_volume_m3: f64,
    /// Normal momentum leaving the reach along its outward normal, in m⁴/s.
    pub reach_momentum_impulse_m4_per_s: f64,
    /// Momentum entering the surface cells in grid x and y, in m⁴/s.
    pub surface_momentum_impulse_m4_per_s: [f64; 2],
    /// Momentum leaving the reach minus momentum entering the surface, grid x and y, in m⁴/s.
    pub junction_reaction_m4_per_s: [f64; 2],
    pub signal_speed_m_per_s: f64,
    pub regime: PortRegime,
    /// The signed volume changes sign against the previous interval.
    pub reversed: bool,
    /// `surface port volume + reach outward volume`, in m³.
    pub paired_gap_m3: f64,
    /// Bound of `|paired_gap_m3|`, in m³.
    pub paired_bound_m3: f64,
}

/// Port receipts and water totals of one accepted interval.
#[derive(Clone, Debug, Default, PartialEq)]
pub(in super::super) struct PortInterval {
    pub transfers: Vec<PortTransfer>,
    /// Port water the surface counts in its exchange totals: into the surface, out of it, in m³.
    pub surface_m3: [f64; 2],
    /// Port water the reaches count in their boundary totals: into the reaches, out of them, in m³.
    pub reach_m3: [f64; 2],
    /// Sum of every port's pairing bound, in m³.
    pub paired_bound_m3: f64,
}

impl PortInterval {
    /// Receipts of `fluxes` over `dt`. Rejects a port whose two sides disagree past its bound.
    ///
    /// The surface volume sums `f_k w_k dt` over the `n` faces. The reach volume is its own
    /// `rate · dt`, where the rate sums the same `f_k w_k` in any order. Each sum rounds by at
    /// most `(n - 1) ε Σ|f_k w_k|`, and each product by `ε / 2` of its magnitude. The bound is
    /// `(2n + 2) ε Σ|f_k w_k| dt`, rounded up, plus `n + 1` subnormal quanta.
    ///
    /// `surface_rate` holds the surface's summed port rates into it and out of it.
    /// `previous` holds the receipts of the previous interval, if any.
    pub fn new(
        ids: &[String],
        fluxes: &[PortFlux],
        surface_rate: [f64; 2],
        dt: f64,
        previous: Option<&[PortTransfer]>,
    ) -> Result<Self, SurfaceChannelError> {
        let numerical = |error: crate::numerics::outward::OutwardError| {
            SurfaceChannelError::Numerical(format!("port pairing bound: {error}"))
        };
        let mut interval = Self {
            transfers: Vec::with_capacity(fluxes.len()),
            surface_m3: [surface_rate[0] * dt, surface_rate[1] * dt],
            reach_m3: [0.0, 0.0],
            paired_bound_m3: 0.0,
        };
        for (index, (id, flux)) in ids.iter().zip(fluxes).enumerate() {
            let outward = flux.outward_water_rate_m3_per_s * dt;
            let mut into_surface = 0.0;
            let mut magnitude = 0.0;
            for &rate in &flux.face_rate_m3_per_s {
                into_surface += rate * dt;
                magnitude += rate.abs();
            }
            let faces = flux.face_rate_m3_per_s.len() as f64;
            let gap = outward - into_surface;
            let scale = upward_mul(
                upward_mul((2.0 * faces + PAIR_BASE_ULPS) * f64::EPSILON, magnitude)
                    .map_err(numerical)?,
                dt,
            )
            .map_err(numerical)?;
            let bound = upward_add(scale, (faces + 1.0) * f64::from_bits(1)).map_err(numerical)?;
            if !within(gap, bound) {
                return Err(SurfaceChannelError::Numerical(format!(
                    "port={id:?} pairing: surface volume={}, reach outward volume={outward}, bound={bound}: the two sides disagree; shorten the common interval",
                    -into_surface
                )));
            }
            // A port that moves nothing reports `+0.0`.
            let signed = 0.0 - outward;
            let reversed = previous
                .and_then(|transfers| transfers.get(index))
                .is_some_and(|before| {
                    let prior = before.signed_volume_m3;
                    (prior > 0.0 && signed < 0.0) || (prior < 0.0 && signed > 0.0)
                });
            interval.reach_m3[0] += signed.max(0.0);
            interval.reach_m3[1] += outward.max(0.0);
            interval.paired_bound_m3 =
                upward_add(interval.paired_bound_m3, bound).map_err(numerical)?;
            let [sx, sy] = flux.surface_momentum_rate_m4_per_s2;
            let [rx, ry] = flux.junction_reaction_rate_m4_per_s2;
            interval.transfers.push(PortTransfer {
                port_id: id.clone(),
                signed_volume_m3: signed,
                reach_momentum_impulse_m4_per_s: flux.outward_momentum_rate_m4_per_s2 * dt,
                surface_momentum_impulse_m4_per_s: [sx * dt, sy * dt],
                junction_reaction_m4_per_s: [rx * dt, ry * dt],
                signal_speed_m_per_s: flux.signal_speed_m_per_s,
                regime: flux.regime,
                reversed,
                paired_gap_m3: gap,
                paired_bound_m3: bound,
            });
        }
        Ok(interval)
    }
}

/// Lifetime port volumes, host resident, in port identity order.
#[derive(Clone, Debug, Default, PartialEq)]
pub(in super::super) struct PortCounters {
    /// Water each reach took from the surface, in m³.
    pub to_channel_m3: Vec<f64>,
    /// Water each reach gave to the surface, in m³.
    pub to_surface_m3: Vec<f64>,
    /// Upward-rounded sum of every addition error of both counters, in m³.
    pub roundoff_bound_m3: Vec<f64>,
    /// Port water the surface counted into it and out of it, in m³.
    pub surface_m3: [f64; 2],
    /// Upward-rounded sum of every interval's pairing bound, in m³.
    pub paired_bound_m3: f64,
}

/// Lifetime directional volumes of one port.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(
    not(test),
    expect(
        dead_code,
        reason = "checkpoint data and port ledger projections read it"
    )
)]
pub(in super::super) struct PortLedger {
    pub port_id: String,
    pub surface_to_channel_m3: f64,
    pub channel_to_surface_m3: f64,
    pub roundoff_bound_m3: f64,
}

/// `a + b` with its exact rounding error.
fn two_sum(a: f64, b: f64) -> (f64, f64) {
    let sum = a + b;
    let b_part = sum - a;
    let a_part = sum - b_part;
    (sum, (a - a_part) + (b - b_part))
}

impl PortCounters {
    /// Zero counters for `ports` ports.
    pub fn zero(ports: usize) -> Self {
        Self {
            to_channel_m3: vec![0.0; ports],
            to_surface_m3: vec![0.0; ports],
            roundoff_bound_m3: vec![0.0; ports],
            surface_m3: [0.0; 2],
            paired_bound_m3: 0.0,
        }
    }

    /// True without ports.
    pub fn is_empty(&self) -> bool {
        self.to_channel_m3.is_empty()
    }

    /// Lifetime totals over every port: into the reaches, out of them, and the rounding bound.
    pub fn reach_totals(&self) -> (f64, f64, f64) {
        (
            self.to_channel_m3.iter().sum(),
            self.to_surface_m3.iter().sum(),
            self.roundoff_bound_m3.iter().sum(),
        )
    }

    /// Adds one interval's port volumes with error-free addition bounds.
    pub fn advance(&self, interval: &PortInterval) -> Result<Self, SurfaceChannelError> {
        let numerical = |error: crate::numerics::outward::OutwardError| {
            SurfaceChannelError::Numerical(format!("port counters: {error}"))
        };
        let mut next = self.clone();
        for (index, transfer) in interval.transfers.iter().enumerate() {
            let (Some(to_channel), Some(to_surface), Some(bound)) = (
                next.to_channel_m3.get_mut(index),
                next.to_surface_m3.get_mut(index),
                next.roundoff_bound_m3.get_mut(index),
            ) else {
                return Err(SurfaceChannelError::Numerical(
                    "port counters lack a port of the interval".into(),
                ));
            };
            let signed = transfer.signed_volume_m3;
            let (forward, forward_low) = two_sum(*to_channel, signed.max(0.0));
            let (backward, backward_low) = two_sum(*to_surface, (-signed).max(0.0));
            *to_channel = forward;
            *to_surface = backward;
            if forward_low != 0.0 || backward_low != 0.0 {
                let low = upward_add(forward_low.abs(), backward_low.abs()).map_err(numerical)?;
                *bound = upward_add(*bound, low).map_err(numerical)?;
            }
        }
        next.surface_m3 = [
            self.surface_m3[0] + interval.surface_m3[0],
            self.surface_m3[1] + interval.surface_m3[1],
        ];
        next.paired_bound_m3 =
            upward_add(self.paired_bound_m3, interval.paired_bound_m3).map_err(numerical)?;
        Ok(next)
    }

    /// Host ledgers with ports in `ids` order.
    #[cfg_attr(
        not(test),
        expect(
            dead_code,
            reason = "checkpoint data and port ledger projections read it"
        )
    )]
    pub fn ledgers(&self, ids: &[String]) -> Vec<PortLedger> {
        ids.iter()
            .zip(&self.to_channel_m3)
            .zip(&self.to_surface_m3)
            .zip(&self.roundoff_bound_m3)
            .map(|(((id, &to_channel), &to_surface), &bound)| PortLedger {
                port_id: id.clone(),
                surface_to_channel_m3: to_channel,
                channel_to_surface_m3: to_surface,
                roundoff_bound_m3: bound,
            })
            .collect()
    }
}

/// One side of a port face in the frame pointing into the surface.
#[derive(Clone, Copy, Debug)]
pub(in super::super) struct FaceSide {
    /// Depth above the higher bed of the face, in m.
    pub depth_m: f64,
    /// Velocity into the surface, in m/s.
    pub velocity_m_per_s: f64,
}

/// Regime of one port face from its reconstructed sides and HLL wave bounds into the surface.
///
/// `ghost` is the reach side and `surface` the surface side.
pub(in super::super) fn face_regime(
    ghost: FaceSide,
    surface: FaceSide,
    [lower, upper]: [f64; 2],
    gravity: f64,
) -> PortRegime {
    match (ghost.depth_m > 0.0, surface.depth_m > 0.0) {
        (false, false) => return PortRegime::Dry,
        (true, false) | (false, true) => return PortRegime::WetDryFront,
        (true, true) => {}
    }
    if lower >= 0.0 {
        return PortRegime::SupercriticalToSurface;
    }
    if upper <= 0.0 {
        return PortRegime::SupercriticalToReach;
    }
    let celerity = |side: FaceSide| (gravity * side.depth_m).sqrt();
    let (cg, cs) = (celerity(ghost), celerity(surface));
    let (ug, us) = (ghost.velocity_m_per_s, surface.velocity_m_per_s);
    let toward_reach = ug - cg < 0.0 && 0.0 < us - cs;
    let toward_surface = ug + cg < 0.0 && 0.0 < us + cs;
    if toward_reach || toward_surface {
        PortRegime::Transcritical
    } else {
        PortRegime::Subcritical
    }
}

/// Regime of a port from the regimes of its faces.
///
/// Every dry face makes the port dry. Otherwise a wet/dry front, then a sonic point, wins.
/// A port is supercritical only when every wet face runs the same way.
pub(in super::super) fn port_regime(faces: &[PortRegime]) -> PortRegime {
    let wet: Vec<PortRegime> = faces
        .iter()
        .copied()
        .filter(|regime| *regime != PortRegime::Dry)
        .collect();
    let Some(&first) = wet.first() else {
        return PortRegime::Dry;
    };
    for regime in [PortRegime::WetDryFront, PortRegime::Transcritical] {
        if wet.contains(&regime) {
            return regime;
        }
    }
    if wet.iter().all(|regime| *regime == first) {
        first
    } else {
        PortRegime::Subcritical
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn side(depth_m: f64, velocity_m_per_s: f64) -> FaceSide {
        FaceSide {
            depth_m,
            velocity_m_per_s,
        }
    }

    #[test]
    fn each_face_regime_follows_its_sides_and_bounds() {
        let g = 9.81;
        let wet = side(1.0, 0.0);
        assert_eq!(
            face_regime(side(0.0, 0.0), side(0.0, 0.0), [0.0; 2], g),
            PortRegime::Dry
        );
        assert_eq!(
            face_regime(wet, side(0.0, 0.0), [-1.0, 6.0], g),
            PortRegime::WetDryFront
        );
        assert_eq!(
            face_regime(side(0.0, 0.0), wet, [-6.0, 1.0], g),
            PortRegime::WetDryFront
        );
        assert_eq!(
            face_regime(side(1.0, 7.0), side(1.0, 7.0), [3.8, 10.2], g),
            PortRegime::SupercriticalToSurface
        );
        assert_eq!(
            face_regime(side(1.0, -7.0), side(1.0, -7.0), [-10.2, -3.8], g),
            PortRegime::SupercriticalToReach
        );
        // A subcritical reach end beside a supercritical surface cell: the 1-fan is sonic.
        assert_eq!(
            face_regime(side(1.0, 1.0), side(0.25, 2.0), [-2.2, 4.0], g),
            PortRegime::Transcritical
        );
        assert_eq!(
            face_regime(wet, wet, [-3.2, 3.2], g),
            PortRegime::Subcritical
        );
    }

    #[test]
    fn port_regimes_combine_their_faces() {
        use PortRegime::*;
        assert_eq!(port_regime(&[Dry, Dry]), Dry);
        assert_eq!(port_regime(&[Dry, Subcritical]), Subcritical);
        assert_eq!(port_regime(&[Subcritical, WetDryFront]), WetDryFront);
        assert_eq!(port_regime(&[Transcritical, Subcritical]), Transcritical);
        assert_eq!(
            port_regime(&[SupercriticalToSurface, SupercriticalToSurface]),
            SupercriticalToSurface
        );
        assert_eq!(
            port_regime(&[SupercriticalToSurface, SupercriticalToReach]),
            Subcritical
        );
    }

    fn flux(face_rates: Vec<f64>, outward: f64) -> PortFlux {
        PortFlux {
            face_rate_m3_per_s: face_rates,
            outward_water_rate_m3_per_s: outward,
            outward_momentum_rate_m4_per_s2: 2.0,
            surface_momentum_rate_m4_per_s2: [1.0, 0.5],
            junction_reaction_rate_m4_per_s2: [0.25, -0.5],
            signal_speed_m_per_s: 4.0,
            exterior_invariant_m_per_s: Some(1.0),
            regime: PortRegime::Subcritical,
        }
    }

    #[test]
    fn pairing_bounds_the_face_sum_against_the_reach_rate() -> Result<(), SurfaceChannelError> {
        let ids = vec!["p".to_string()];
        let rates = vec![0.1, 0.2, 0.3];
        // The reach rate is the face sum in another order.
        let outward = (0.3 + 0.2) + 0.1;
        let interval =
            PortInterval::new(&ids, &[flux(rates.clone(), outward)], [0.6, 0.0], 0.5, None)?;
        let transfer = &interval.transfers[0];
        assert!(transfer.paired_gap_m3.abs() <= transfer.paired_bound_m3);
        assert_eq!(transfer.signed_volume_m3, -(outward * 0.5));
        assert_eq!(transfer.reach_momentum_impulse_m4_per_s, 1.0);
        assert_eq!(transfer.surface_momentum_impulse_m4_per_s, [0.5, 0.25]);
        assert_eq!(transfer.junction_reaction_m4_per_s, [0.125, -0.25]);
        assert!(!transfer.reversed);
        assert_eq!(interval.reach_m3, [0.0, outward * 0.5]);
        assert_eq!(interval.surface_m3, [0.3, 0.0]);
        // A gap far past the rounding bound rejects.
        let wrong = PortInterval::new(&ids, &[flux(rates, 0.7)], [0.6, 0.0], 0.5, None);
        assert!(
            matches!(wrong, Err(SurfaceChannelError::Numerical(message)) if message.contains("pairing"))
        );
        Ok(())
    }

    #[test]
    fn reversal_and_counters_follow_the_signed_volume() -> Result<(), SurfaceChannelError> {
        let ids = vec!["p".to_string()];
        let out = PortInterval::new(&ids, &[flux(vec![1.0], 1.0)], [1.0, 0.0], 1.0, None)?;
        let back = PortInterval::new(
            &ids,
            &[flux(vec![-0.5], -0.5)],
            [0.0, 0.5],
            1.0,
            Some(&out.transfers),
        )?;
        assert!(back.transfers[0].reversed);
        let again = PortInterval::new(
            &ids,
            &[flux(vec![-0.25], -0.25)],
            [0.0, 0.25],
            1.0,
            Some(&back.transfers),
        )?;
        assert!(!again.transfers[0].reversed);
        // A zero volume never reverses.
        let still = PortInterval::new(
            &ids,
            &[flux(vec![0.0], 0.0)],
            [0.0; 2],
            1.0,
            Some(&again.transfers),
        )?;
        assert!(!still.transfers[0].reversed);
        let counters = PortCounters::zero(1).advance(&out)?.advance(&back)?;
        assert_eq!(counters.to_surface_m3, [1.0]);
        assert_eq!(counters.to_channel_m3, [0.5]);
        assert_eq!(counters.roundoff_bound_m3, [0.0]);
        assert_eq!(counters.surface_m3, [1.0, 0.5]);
        assert_eq!(counters.reach_totals(), (0.5, 1.0, 0.0));
        let ledgers = counters.ledgers(&ids);
        assert_eq!(ledgers[0].channel_to_surface_m3, 1.0);
        Ok(())
    }
}
