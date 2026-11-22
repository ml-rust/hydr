//! The piece cover, its refinement rounds and the per-row reductions.

use super::ArcCertificates;
use super::slopes::{PieceSlopes, over_pieces};
use crate::structures::numerics::culvert::constants::{CIRCLE_CRITICAL_TOP, CRITICAL_FLOOR};
use crate::structures::numerics::culvert::{Arithmetic, BarrelClient, section};
use crate::structures::tensor::culvert::{ArcRows, TensorBarrelContext};
use numr::dtype::DType;
use numr::ops::{BinaryOps, CompareOps, ConditionalOps, IndexingOps, ReduceOps};
use numr::runtime::Runtime;
use numr::tensor::Tensor;
use solvr::common::arithmetic::ArithmeticResult;

/// Pieces of the first cover per row.
const COVER_PIECES: usize = 512;

/// Pieces of each refinement round per row.
const ROUND_PIECES: usize = 32;

/// Refinement rounds of a context's certificates.
pub(crate) const REFINEMENT_ROUNDS: usize = 16;

/// Widest certified peak enclosure `s_F - s_N`, `2^-20`.
const PEAK_WIDTH: f64 = 1.0 / 1_048_576.0;

/// Pieces of every row: the row of each piece and the fractions of its ends in its span.
struct Grid<R: Runtime<DType = DType>> {
    owner: Tensor<R>,
    from: Tensor<R>,
    to: Tensor<R>,
    rows: usize,
    pieces: usize,
    /// `unit` gathered to one row per piece.
    unit: TensorBarrelContext<R>,
}

impl<R: Runtime<DType = DType>> Grid<R> {
    /// `fractions` holds the piece ends from 0 to 1, each exact.
    fn new(unit: &TensorBarrelContext<R>, fractions: &[f64]) -> numr::error::Result<Self>
    where
        R::Client: BarrelClient<R>,
    {
        let rows = unit.width();
        let pieces = fractions.len() - 1;
        let mut owner = Vec::with_capacity(rows * pieces);
        let mut from = Vec::with_capacity(rows * pieces);
        let mut to = Vec::with_capacity(rows * pieces);
        for row in 0..rows {
            for pair in fractions.windows(2) {
                let [low, high] = pair else { continue };
                owner.push(row as i64);
                from.push(*low);
                to.push(*high);
            }
        }
        let shape = [rows * pieces];
        let device = &unit.device;
        let owner = Tensor::<R>::from_slice(&owner, &shape, device)?;
        Ok(Self {
            from: Tensor::<R>::from_slice(&from, &shape, device)?,
            to: Tensor::<R>::from_slice(&to, &shape, device)?,
            rows,
            pieces,
            unit: unit.select(&owner)?,
            owner,
        })
    }

    /// Piece ends `[lower, upper]` over `[low, high]` per row, ordered and inside it.
    fn ends(
        &self,
        c: &R::Client,
        low: &Tensor<R>,
        high: &Tensor<R>,
    ) -> numr::error::Result<[Tensor<R>; 2]>
    where
        R::Client: BarrelClient<R>,
    {
        let start = c.index_select(low, 0, &self.owner)?;
        let end = c.index_select(high, 0, &self.owner)?;
        let span = c.sub(&end, &start)?;
        // Rounding is monotone, so the ends rise with the fractions.
        let at = |fraction: &Tensor<R>| -> numr::error::Result<Tensor<R>> {
            let reached = c.minimum(&c.add(&start, &c.mul(&span, fraction)?)?, &end)?;
            let last = c.eq(fraction, &self.unit.one)?;
            c.where_cond(&last, &end, &reached)
        };
        Ok([at(&self.from)?, at(&self.to)?])
    }

    /// Least value of each row's pieces.
    fn least(&self, c: &R::Client, value: &Tensor<R>) -> numr::error::Result<Tensor<R>>
    where
        R::Client: BarrelClient<R>,
    {
        c.min(&value.reshape(&[self.rows, self.pieces])?, &[1], false)
    }

    /// Greatest value of each row's pieces.
    fn most(&self, c: &R::Client, value: &Tensor<R>) -> numr::error::Result<Tensor<R>>
    where
        R::Client: BarrelClient<R>,
    {
        c.max(&value.reshape(&[self.rows, self.pieces])?, &[1], false)
    }

    /// Row values spread to the pieces.
    fn spread(&self, c: &R::Client, value: &Tensor<R>) -> numr::error::Result<Tensor<R>>
    where
        R::Client: BarrelClient<R>,
    {
        c.index_select(value, 0, &self.owner)
    }
}

/// The rising run, the peak gap `[low, high]` and the failure flags of every row.
struct Run<R: Runtime<DType = DType>> {
    low: Tensor<R>,
    high: Tensor<R>,
    slope: Tensor<R>,
    failed: Tensor<R>,
    /// 1 where a negative piece lies inside the gap.
    inner: Tensor<R>,
}

impl<R: Runtime<DType = DType>> Run<R> {
    /// This run on rows where `active` is 1 and `kept` on the others.
    fn unless(self, c: &R::Client, active: &Tensor<R>, kept: Self) -> numr::error::Result<Self>
    where
        R::Client: BarrelClient<R>,
    {
        let pick = |new: &Tensor<R>, old: &Tensor<R>| c.where_cond(active, new, old);
        Ok(Self {
            low: pick(&self.low, &kept.low)?,
            high: pick(&self.high, &kept.high)?,
            slope: pick(&self.slope, &kept.slope)?,
            failed: pick(&self.failed, &kept.failed)?,
            inner: pick(&self.inner, &kept.inner)?,
        })
    }
}

/// Certifies the `[k]` unit-rise arc rows `unit` with `rounds` refinement rounds.
pub(crate) fn certify<R>(
    a: &mut Arithmetic<'_, R>,
    unit: &TensorBarrelContext<R>,
    rows: &ArcRows<R>,
    rounds: usize,
) -> ArithmeticResult<ArcCertificates<R>>
where
    R: Runtime<DType = DType>,
    R::Client: BarrelClient<R>,
{
    let c = a.client();
    let floor = unit.constant(CRITICAL_FLOOR)?;
    let top = unit.constant(CIRCLE_CRITICAL_TOP)?;
    let start = c.maximum(&rows.widest_depth[0], &floor)?;

    // Pieces shrink as (1 - u)^2 toward the top, as equal normal angles do at the crown.
    let quadratic: Vec<f64> = (0..=COVER_PIECES)
        .map(|i| {
            let rest = (COVER_PIECES - i) as f64 / COVER_PIECES as f64;
            1.0 - rest * rest
        })
        .collect();
    let cover = Grid::new(unit, &quadratic)?;
    let [lower, upper] = cover.ends(c, &start, &top)?;
    let slopes = slopes_on(a, &cover, &lower, &upper)?;
    let minus_inf = cover.unit.constant(f64::NEG_INFINITY)?;
    let critical = c.where_cond(&slopes.known, &slopes.critical_lower, &minus_inf)?;
    let critical = c.minimum(&cover.least(c, &critical)?, &unit.constant(2.0)?)?;
    let unbounded = c.le(&critical, &unit.zero)?;
    let critical_slope = c.where_cond(&unbounded, &unit.one, &critical)?;

    let mut run = Run {
        low: start,
        high: top.clone(),
        slope: unit.constant(3.0)?,
        failed: unbounded,
        inner: unit.zero.clone(),
    };
    run = advance(c, &cover, &run, [&lower, &upper], &slopes)?;
    let linear: Vec<f64> = (0..=ROUND_PIECES)
        .map(|i| i as f64 / ROUND_PIECES as f64)
        .collect();
    let round = Grid::new(unit, &linear)?;
    let width = unit.constant(PEAK_WIDTH)?;
    for _ in 0..rounds {
        let [lower, upper] = round.ends(c, &run.low, &run.high)?;
        let slopes = slopes_on(a, &round, &lower, &upper)?;
        // A row whose gap is already narrow keeps it, so its slope floor stays clear of zero.
        let active = c.gt(&c.sub(&run.high, &run.low)?, &width)?;
        let next = advance(c, &round, &run, [&lower, &upper], &slopes)?;
        run = next.unless(c, &active, run)?;
    }

    let open = c.ge(&run.high, &top)?;
    let wide = c.gt(&c.sub(&run.high, &run.low)?, &unit.constant(PEAK_WIDTH)?)?;
    let failed = c.maximum(
        &c.maximum(&run.failed, &run.inner)?,
        &c.maximum(&open, &wide)?,
    )?;
    let geometry = section::over_depth(a, unit, &run.low, &run.high)?.geometry;
    let ln_area = a.log_positive(&geometry.area)?;
    let ln_perimeter = a.log_positive(&geometry.perimeter)?;
    let five = a.point(&unit.constant(5.0)?)?;
    let two = a.point(&unit.constant(2.0)?)?;
    let area_term = a.mul(&five, &ln_area)?;
    let perimeter_term = a.mul(&two, &ln_perimeter)?;
    let log_cube = a.sub(&area_term, &perimeter_term)?;
    let log_peak = a.div_positive(&log_cube, &unit.constant(3.0)?)?;
    Ok(ArcCertificates {
        critical_slope,
        normal_top: run.low.clone(),
        normal_slope: run.slope,
        peak: [run.low, run.high],
        log_peak: [log_peak.lower().clone(), log_peak.upper().clone()],
        failed,
    })
}

/// Slopes over the pieces of `grid`.
fn slopes_on<R>(
    a: &mut Arithmetic<'_, R>,
    grid: &Grid<R>,
    lower: &Tensor<R>,
    upper: &Tensor<R>,
) -> ArithmeticResult<PieceSlopes<R>>
where
    R: Runtime<DType = DType>,
    R::Client: BarrelClient<R>,
{
    let Some(rows) = &grid.unit.arcs else {
        return Err(numr::error::Error::InvalidArgument {
            arg: "unit",
            reason: "arc certificates need arc rows".into(),
        }
        .into());
    };
    over_pieces(a, &grid.unit, rows, lower, upper)
}

/// Moves the run past the positive pieces and the gap top before the trailing negative run.
fn advance<R>(
    c: &R::Client,
    grid: &Grid<R>,
    run: &Run<R>,
    [lower, upper]: [&Tensor<R>; 2],
    slopes: &PieceSlopes<R>,
) -> numr::error::Result<Run<R>>
where
    R: Runtime<DType = DType>,
    R::Client: BarrelClient<R>,
{
    let zero = &grid.unit.zero;
    let positive = c.mul(&slopes.known, &c.gt(&slopes.normal_lower, zero)?)?;
    let negative = c.mul(&slopes.known, &c.lt(&slopes.normal_upper, zero)?)?;
    let gap_high = grid.spread(c, &run.high)?;
    let gap_low = grid.spread(c, &run.low)?;
    let first_open = grid.least(c, &c.where_cond(&positive, &gap_high, lower)?)?;
    let last_open = grid.most(c, &c.where_cond(&negative, &gap_low, upper)?)?;
    let last_open = c.maximum(&last_open, &first_open)?;

    let reach = grid.spread(c, &first_open)?;
    let rising = c.mul(&positive, &c.le(upper, &reach)?)?;
    let three = grid.unit.constant(3.0)?;
    let slope = grid.least(c, &c.where_cond(&rising, &slopes.normal_lower, &three)?)?;
    let slope = c.minimum(&run.slope, &slope)?;

    let inf = grid.unit.constant(f64::INFINITY)?;
    let minus_inf = grid.unit.constant(f64::NEG_INFINITY)?;
    let first_negative = grid.least(c, &c.where_cond(&negative, lower, &inf)?)?;
    let last_positive = grid.most(c, &c.where_cond(&positive, upper, &minus_inf)?)?;
    let reversed = c.lt(&first_negative, &last_positive)?;
    let tail = grid.spread(c, &last_open)?;
    let inside = c.mul(&negative, &c.lt(lower, &tail)?)?;
    let inner = grid.most(c, &inside)?;
    Ok(Run {
        low: first_open,
        high: last_open,
        slope,
        failed: c.maximum(&run.failed, &reversed)?,
        inner,
    })
}

#[cfg(test)]
mod tests {
    use super::super::certificate_rows::{CERTIFIED, Certified, DEPTHS_SHA256, Shape};
    use crate::structures::tensor::culvert::TensorBarrelContext;
    use crate::structures::{CulvertBarrel, StructureError};
    use numr::runtime::Runtime;
    use numr::runtime::cpu::{CpuDevice, CpuRuntime};

    fn barrel(entry: &Certified) -> Result<CulvertBarrel, StructureError> {
        let inputs: Vec<f64> = entry
            .barrel
            .inputs
            .iter()
            .map(|b| f64::from_bits(*b))
            .collect();
        match (entry.barrel.shape, inputs.as_slice()) {
            (Shape::HorizontalEllipse, &[b, d, r]) => CulvertBarrel::horizontal_ellipse(b, d, r),
            (Shape::VerticalEllipse, &[b, d, r]) => CulvertBarrel::vertical_ellipse(b, d, r),
            (Shape::PipeArch, &[b, d, c, r]) => CulvertBarrel::pipe_arch(b, d, c, r),
            (Shape::Arch, &[b, d]) => CulvertBarrel::arch(b, d),
            _ => Err(StructureError::definition(format!(
                "fixture barrel {}: wrong input count",
                entry.barrel.name
            ))),
        }
    }

    #[test]
    fn certificates_hold_the_independent_minima_and_peaks() -> Result<(), StructureError> {
        assert_eq!(
            DEPTHS_SHA256,
            "52858a2e2daa57c74f5cff3193e81bf6e1132fd547c6dae8a11cb7c7754e36d4"
        );
        let barrels = CERTIFIED
            .iter()
            .map(barrel)
            .collect::<Result<Vec<_>, _>>()?;
        let device = CpuDevice::new();
        let context =
            TensorBarrelContext::<CpuRuntime>::new(CpuRuntime::default_client(&device), &barrels)?;
        let read = |t: &numr::tensor::Tensor<CpuRuntime>| t.try_to_vec::<f64>();
        let critical = read(&context.critical_slope)?;
        let top = read(&context.normal_top)?;
        let normal = read(&context.normal_slope)?;
        let [peak_lower, peak_upper] = [
            read(&context.conveyance_peak[0])?,
            read(&context.conveyance_peak[1])?,
        ];
        let [log_lower, log_upper] = [read(&context.log_peak[0])?, read(&context.log_peak[1])?];
        let mut failures = Vec::new();
        for (row, entry) in CERTIFIED.iter().enumerate() {
            let name = entry.barrel.name;
            let [kc, s_n, kn] = [critical[row], top[row], normal[row]];
            let [down, up] = entry.peak.map(f64::from_bits);
            let [log_down, log_up] = entry.log_peak.map(f64::from_bits);
            if !(0.0 < kc && kc <= f64::from_bits(entry.critical_minimum)) {
                failures.push(format!("{name}: kappa_c={kc:e}"));
            }
            // The true minimum over [2^-30, s_j] bounds the minimum over [2^-30, s_N] above.
            let below = entry
                .rising
                .iter()
                .filter(|(s, _)| f64::from_bits(*s) <= s_n)
                .map(|(_, m)| f64::from_bits(*m))
                .next_back();
            if !(0.0 < kn && below.is_some_and(|m| kn <= m)) {
                failures.push(format!("{name}: kappa_n={kn:e} against {below:?}"));
            }
            if !(peak_lower[row] <= down && up <= peak_upper[row] && s_n < down) {
                failures.push(format!(
                    "{name}: peak [{:e}, {:e}] misses [{down:e}, {up:e}]",
                    peak_lower[row], peak_upper[row]
                ));
            }
            if !(log_lower[row] <= log_down && log_up <= log_upper[row]) {
                failures.push(format!(
                    "{name}: log peak [{:e}, {:e}] misses [{log_down:e}, {log_up:e}]",
                    log_lower[row], log_upper[row]
                ));
            }
        }
        assert!(failures.is_empty(), "{}", failures.join("\n"));
        Ok(())
    }
}
