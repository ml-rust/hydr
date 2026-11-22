//! Profile depths of the slope bounds, and friction over the conveyance peak enclosure of arc
//! rows for the falling bounds.
//!
//! An arc barrel's construction certificate proves that `Sf` falls with depth up to `s_N` and
//! rises from `s_F`, and encloses `ln k` over `[s_N, s_F]`. It does not prove that `Sf` has one
//! minimum inside that enclosure. So over a falling profile that reaches the enclosure, `Sf`
//! is at most the larger of its ends and `K exp(-2 ln k_low)`, and `S - Sf` is at least the
//! smaller of its ends and `S - K exp(-2 ln k_low)`.

use super::integrand::ProfileParams;
use super::jet;
use super::slope::SlopeOperands;
use super::trig::fill;
use crate::structures::numerics::culvert::{Arithmetic, BarrelClient, choose};
use crate::structures::tensor::culvert::TensorBarrelContext;
use numr::dtype::DType;
use numr::ops::{BinaryOps, CompareOps, ConditionalOps, UnaryOps};
use numr::runtime::Runtime;
use numr::tensor::Tensor;
use solvr::common::arithmetic::{ArithmeticResult, Interval};

/// Depth enclosures of the free profile: inlet, start and the peak bound.
pub(super) struct Depths<R: Runtime<DType = DType>> {
    pub(super) end: Interval<R>,
    pub(super) start: Interval<R>,
    pub(super) low: Interval<R>,
    /// 1 where `Sf` falls with depth over the whole profile.
    pub(super) monotone: Tensor<R>,
    /// Depth enclosure of the conveyance peak, `D` on box rows.
    pub(super) peak: Interval<R>,
}

/// Depth enclosures of the free-inlet rows `free` of a march.
pub(super) fn depths<R>(
    a: &mut Arithmetic<'_, R>,
    operands: &SlopeOperands<'_, R>,
    free: &Tensor<R>,
) -> ArithmeticResult<Depths<R>>
where
    R: Runtime<DType = DType>,
    R::Client: BarrelClient<R>,
{
    let SlopeOperands {
        barrel,
        march,
        tailwater,
        ..
    } = *operands;
    let c = a.client();
    let classes = &march.classes;
    let rise = &barrel.rise;
    let half = &barrel.half_rise;
    let [inlet_low, inlet_high] = &march.headwater.inlet;
    let end = a.bounds(
        &c.where_cond(free, inlet_low, half)?,
        &c.where_cond(free, inlet_high, half)?,
    )?;
    let [yc_low, yc_high] = &classes.critical;
    let start = a.bounds(
        &c.where_cond(&classes.falling, rise, &c.maximum(tailwater, yc_low)?)?,
        &c.where_cond(&classes.falling, rise, &c.maximum(tailwater, yc_high)?)?,
    )?;
    // A box has no conveyance peak below the crown. Circle and arc rows hold its certified
    // enclosure, `[s_N, s_F]` on arc rows.
    let fraction = a.bounds(&barrel.conveyance_peak[0], &barrel.conveyance_peak[1])?;
    let d = a.point(rise)?;
    let circle_peak = a.mul(&fraction, &d)?;
    let peak = choose(a, &barrel.circular, &circle_peak, &d)?;
    let peak = match &operands.outlet.arcs {
        None => peak,
        Some(arcs) => choose(a, &arcs.arced, &circle_peak, &peak)?,
    };
    let deepest = c.maximum(start.upper(), end.upper())?;
    let monotone = c.maximum(&barrel.boxed, &c.le(&deepest, peak.lower())?)?;
    let low = a.bounds(
        &c.maximum(end.lower(), peak.lower())?,
        &c.maximum(end.upper(), peak.upper())?,
    )?;
    Ok(Depths {
        end,
        start,
        low,
        monotone,
        peak,
    })
}

/// Upper bounds over the peak enclosure on the rows that reach it.
pub(super) struct Zone<R: Runtime<DType = DType>> {
    /// 1 on arc rows whose profile depths may reach the peak enclosure.
    pub(super) touches: Tensor<R>,
    /// Upper bound of `|sigma_e| / (S - Sf)` over the enclosure.
    pub(super) ratio: Tensor<R>,
    /// Upper bound of `Sf` over the enclosure.
    pub(super) friction: Tensor<R>,
    /// 1 where `S - Sf` is certified positive over the enclosure, or the row does not reach it.
    pub(super) known: Tensor<R>,
}

/// Bounds over the peak depth enclosure `peak` for falling profiles between the depth
/// enclosures `ends`, with `|sigma_e|` enclosed by `gap`.
pub(super) fn falling<R>(
    a: &mut Arithmetic<'_, R>,
    barrel: &TensorBarrelContext<R>,
    params: &ProfileParams<R>,
    arced: &Tensor<R>,
    [end, start]: [&Interval<R>; 2],
    peak: &Interval<R>,
    gap: &Interval<R>,
) -> ArithmeticResult<Zone<R>>
where
    R: Runtime<DType = DType>,
    R::Client: BarrelClient<R>,
{
    let c = a.client();
    let like = &params.rise;
    let shallow = c.minimum(end.lower(), start.lower())?;
    let deep = c.maximum(end.upper(), start.upper())?;
    let reach = c.mul(&c.le(&shallow, peak.upper())?, &c.ge(&deep, peak.lower())?)?;
    let touches = c.mul(arced, &reach)?;
    let log_peak = a.bounds(&barrel.log_peak[0], &barrel.log_peak[1])?;
    let two = a.point(&fill(like, 2.0)?)?;
    let twice = a.mul(&two, &log_peak)?;
    let log_friction = a.bounds(&params.log_friction[0], &params.log_friction[1])?;
    let exponent = a.sub(&log_friction, &twice)?;
    let friction = jet::exp_interval(a, &exponent)?;
    let slope = a.point(&params.slope)?;
    let room = a.sub(&slope, &friction)?;
    let positive = c.gt(room.lower(), &fill(like, 0.0)?)?;
    let one = fill(like, 1.0)?;
    let safe = a.bounds(
        &c.where_cond(&positive, room.lower(), &one)?,
        &c.where_cond(&positive, room.upper(), &one)?,
    )?;
    let ratio = a.div_positive_interval(gap, &safe)?;
    let known = c.maximum(&c.sub(&one, &touches)?, &positive)?;
    Ok(Zone {
        touches,
        ratio: c.abs(ratio.upper())?,
        friction: friction.upper().clone(),
        known,
    })
}

#[cfg(test)]
mod tests {
    use crate::structures::numerics::culvert::RowNames;
    use crate::structures::numerics::culvert::outlet::{
        Accuracy, SlopeOperands, head_slopes, march, prepare,
    };
    use crate::structures::tensor::culvert::TensorBarrelContext;
    use crate::structures::{CulvertBarrel, CulvertOutletRow, StructureError};
    use numr::dtype::DType;
    use numr::runtime::Runtime;
    use numr::runtime::cpu::{CpuDevice, CpuRuntime};
    use numr::tensor::Tensor;
    use solvr::common::arithmetic::with_interval_arithmetic;

    type Cpu = CpuRuntime;

    /// Relative step of the centred differences.
    const STEP: f64 = 1e-5;

    /// Relative slack of each comparison: enclosure widths over the step.
    const SLACK: f64 = 1e-3;

    /// Headwater bounds per lane, with the slope bounds when `slopes` is set.
    struct Run {
        low: Vec<f64>,
        high: Vec<f64>,
        slopes: Option<[Vec<f64>; 3]>,
    }

    fn run(
        rows: &[CulvertOutletRow],
        flows: &[f64],
        tails: &[f64],
        slopes: bool,
    ) -> Result<Run, StructureError> {
        let device = CpuDevice::new();
        let client = Cpu::default_client(&device);
        let barrels: Vec<CulvertBarrel> = rows.iter().map(|row| row.barrel).collect();
        let barrel = TensorBarrelContext::<Cpu>::new(client.clone(), &barrels)?;
        let unit = barrel.unit()?;
        let outlet = prepare(&barrel, rows)?;
        let q = Tensor::<Cpu>::from_slice(flows, &[flows.len()], &device)?;
        let tw = Tensor::<Cpu>::from_slice(tails, &[tails.len()], &device)?;
        let names = RowNames::new(&barrels);
        let found = march(names, &barrel, &unit, &outlet, &q, &tw, Accuracy::Fine)?;
        let [low, high] = &found.headwater.headwater;
        let slopes = if slopes {
            let operands = SlopeOperands {
                barrel: &barrel,
                outlet: &outlet,
                march: &found,
                discharge: &q,
                tailwater: &tw,
            };
            let bounds = with_interval_arithmetic(&client, DType::F64, "arc slopes", |a| {
                let bounds = head_slopes(a, &operands)?;
                Ok([
                    bounds.gain.try_to_vec::<f64>()?,
                    bounds.tail.try_to_vec::<f64>()?,
                    bounds.known.try_to_vec::<f64>()?,
                ])
            })?;
            Some(bounds)
        } else {
            None
        };
        Ok(Run {
            low: low.try_to_vec::<f64>()?,
            high: high.try_to_vec::<f64>()?,
            slopes,
        })
    }

    /// Frozen criteria: on arc lanes of every shape, mild and steep, with free and drowned
    /// outlets, each known lower bound of `Q dHW/dQ` lies at or below the largest centred
    /// difference the headwater enclosures allow, and each known upper bound of `|dHW/dTW|` at
    /// or above the smallest, both within `SLACK`. At least half the lanes are known.
    #[test]
    fn arc_slope_bounds_hold_against_centred_differences() -> Result<(), StructureError> {
        let shapes = [
            CulvertBarrel::horizontal_ellipse(1.524, 0.9652, 0.419_475_92)?,
            CulvertBarrel::vertical_ellipse(0.9652, 1.524, 0.419_475_92)?,
            CulvertBarrel::pipe_arch(1.8542, 1.397, 0.4572, 1.938_528)?,
            CulvertBarrel::arch(3.048, 1.6002)?,
        ];
        // Slope, discharge, tailwater over the rise.
        let cases = [
            (0.001, 0.5, 0.0),
            (0.001, 0.5, 0.5),
            (0.001, 1.5, 0.9),
            (0.001, 1.5, 0.3),
            (0.03, 1.0, 0.0),
            (0.03, 1.0, 1.5),
            (0.0, 0.8, 0.2),
            (-0.002, 0.8, 0.6),
        ];
        let mut rows = Vec::new();
        let mut flows = Vec::new();
        let mut tails = Vec::new();
        for barrel in shapes {
            for (slope, flow, tail) in cases {
                rows.push(CulvertOutletRow {
                    barrel,
                    length_m: 40.0,
                    manning_n: 0.013,
                    entrance_loss: 0.5,
                    slope,
                    gravity_m_per_s2: 9.81,
                });
                flows.push(flow);
                tails.push(tail * barrel.rise_m());
            }
        }
        let shift = |values: &[f64], by: &dyn Fn(usize, f64) -> f64| -> Vec<f64> {
            values.iter().enumerate().map(|(i, &v)| by(i, v)).collect()
        };
        let tail_step = |i: usize| STEP * rows[i].barrel.rise_m();
        let centre = run(&rows, &flows, &tails, true)?;
        let more_q = run(
            &rows,
            &shift(&flows, &|_, q| q * (1.0 + STEP)),
            &tails,
            false,
        )?;
        let less_q = run(
            &rows,
            &shift(&flows, &|_, q| q * (1.0 - STEP)),
            &tails,
            false,
        )?;
        let more_t = run(
            &rows,
            &flows,
            &shift(&tails, &|i, t| t + tail_step(i)),
            false,
        )?;
        let less_t = run(
            &rows,
            &flows,
            &shift(&tails, &|i, t| (t - tail_step(i)).max(0.0)),
            false,
        )?;
        let [gain, tail, known] = centre
            .slopes
            .ok_or_else(|| StructureError::definition("no slope bounds"))?;
        let mut held = 0;
        for (i, &flag) in known.iter().enumerate() {
            if flag != 1.0 || !centre.low[i].is_finite() {
                continue;
            }
            held += 1;
            let q = flows[i];
            let largest = q * (more_q.high[i] - less_q.low[i]) / (2.0 * STEP * q);
            assert!(
                gain[i] <= largest + SLACK * largest.abs(),
                "lane {i}: gain {:e} above the centred difference {largest:e}",
                gain[i]
            );
            let span = (more_t.high[i] - more_t.low[i]).max(0.0);
            let down = tails[i] - (tails[i] - tail_step(i)).max(0.0);
            let width = tail_step(i) + down;
            let rise = (more_t.low[i] - less_t.high[i]) / width;
            let fall = (less_t.low[i] - more_t.high[i]) / width;
            let smallest = rise.max(fall).max(0.0);
            assert!(
                tail[i] >= smallest * (1.0 - SLACK) - span / width,
                "lane {i}: tail {:e} below the centred difference {smallest:e}",
                tail[i]
            );
        }
        assert!(
            held * 2 >= rows.len(),
            "{held} of {} lanes known",
            rows.len()
        );
        Ok(())
    }
}
