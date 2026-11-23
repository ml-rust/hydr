//! Compact `[m]` barrel rows: shape masks, dimensions, series coefficients, arc layouts,
//! depth certificates, full-state bounds and each row's frame.

use super::arcs::ArcRows;
use super::floor::FloorRows;
use super::frame::{BarrelFrame, framed_barrel};
use super::series::SeriesTensors;
use crate::structures::numerics::culvert::arcs::ArcInputs;
use crate::structures::numerics::culvert::certify::REFINEMENT_ROUNDS;
use crate::structures::numerics::culvert::{BarrelClient, constants, section};
use crate::structures::{CulvertBarrel, CulvertBarrelShape, StructureError};
use numr::dtype::DType;
use numr::ops::{BinaryOps, ConditionalOps, IndexingOps};
use numr::runtime::{Runtime, RuntimeClient, SquareRootContract};
use numr::tensor::Tensor;
use solvr::common::arithmetic::with_interval_arithmetic;

/// Lower and upper endpoint tensors of one stored enclosure.
pub(crate) type Bounds<R> = [Tensor<R>; 2];

/// One row per barrel. Circle, box and arc rows mix and the shape masks select per row.
///
/// Every field lives on the device, so [`TensorBarrelContext::select`] gathers rows without a
/// host read. The owner of the context keeps the host barrel list.
pub(crate) struct TensorBarrelContext<R: Runtime<DType = DType>> {
    pub(crate) client: R::Client,
    pub(crate) device: R::Device,
    pub(crate) zero: Tensor<R>,
    pub(crate) one: Tensor<R>,
    /// 1 on circle rows, 0 on other rows.
    pub(crate) circular: Tensor<R>,
    /// 1 on box rows, 0 on other rows.
    pub(crate) boxed: Tensor<R>,
    /// Rise `D`, the diameter of a circle.
    pub(crate) rise: Tensor<R>,
    /// Span `B`, `D` for a circle and the floor chord for a flat-floor arch.
    pub(crate) span: Tensor<R>,
    /// `D / 2`, exact.
    pub(crate) half_rise: Tensor<R>,
    /// `D / 16`, exact. Circle depths below it take the shallow series.
    pub(crate) shallow_top: Tensor<R>,
    /// `D / 32`, exact. Harmless shallow operand on other lanes.
    pub(crate) shallow_operand: Tensor<R>,
    pub(crate) series: SeriesTensors<R>,
    /// Upper end of the critical-depth domain in `s = y/D`: `1 - 2^-53` circle and arc, `1` box.
    /// Floored circle and arc rows hold `1 - 2^-52` in `s = y/D'`.
    pub(crate) critical_top: Tensor<R>,
    /// Lower bound of `d r_c / ds` on the critical domain: `5.4` circle, `3` box, the
    /// certified `kappa_c` arc and floored circle.
    pub(crate) critical_slope: Tensor<R>,
    /// Upper end of the normal-depth domain in `s = y/D`: `s_N = 0.93818` circle, `1` box, the
    /// certified rising-limb top arc and floored circle.
    pub(crate) normal_top: Tensor<R>,
    /// Lower bound of `d r_n / ds` on the normal domain: `5.42e-5` circle, `3` box, the
    /// certified `kappa_n` arc and floored circle.
    pub(crate) normal_slope: Tensor<R>,
    /// Enclosure in `s` of the conveyance peak: the frozen circle bounds, `[1, 1]` for a box,
    /// whose conveyance rises to the crown, and the certified `[s_N, s_F]` arc and floored circle.
    pub(crate) conveyance_peak: Bounds<R>,
    /// Enclosure of `ln k_max`, the log peak conveyance `a^(5/3) / p^(2/3)` in unit geometry:
    /// the frozen circle bounds and the certified arc bounds. Zero on box rows, which never read it.
    /// Floored circle and arc rows hold `(5 ln A - 2 ln P_n) / 3` in their frame.
    pub(crate) log_peak: Bounds<R>,
    /// `[s_j, kappa_j]` per rising-limb knot: `kappa_j` bounds `d r_n / ds` below on
    /// `[2^-30, s_j]` of an arc row or a floored circle row. Other rows hold `[1/2, 1]`, which no
    /// query reads. `None` when no row is an arc barrel or a floored circle.
    pub(crate) limb: Option<Vec<Bounds<R>>>,
    pub(crate) full_area: Bounds<R>,
    pub(crate) full_perimeter: Bounds<R>,
    pub(crate) full_radius: Bounds<R>,
    /// Arc layouts in metres, with a harmless arc barrel on circle and box rows. `None` when no
    /// row is an arc barrel, so circle and box queries skip the arc branch. Each is its frame
    /// layout times exact powers of two.
    pub(crate) arcs: Option<ArcRows<R>>,
    /// The arc layouts in units of the rise, read by the depth residuals and certificates.
    pub(crate) unit_arcs: Option<ArcRows<R>>,
    /// Each row's frame, where point geometry runs. `None` when every row is its own frame.
    pub(crate) frame: Option<Box<BarrelFrame<R>>>,
    /// Open sections above each declared floor. `None` when no row has a floor. The geometry
    /// fields above describe the barrel without its floor. The depth domains, slope floors and
    /// peak bounds of a floored row describe its open section.
    pub(crate) floor: Option<Box<FloorRows<R>>>,
}

impl<R: Runtime<DType = DType>> TensorBarrelContext<R> {
    /// Checks the runtime, uploads the rows and encloses the full-state constants.
    pub(crate) fn new(client: R::Client, barrels: &[CulvertBarrel]) -> Result<Self, StructureError>
    where
        R::Client: BarrelClient<R>,
    {
        if barrels.is_empty() {
            return Err(StructureError::definition(
                "barrels=[]: supply at least one culvert barrel",
            ));
        }
        if let Some(error) = crate::tensor::probe_f64::<R, _>(&client)? {
            return Err(StructureError::definition(error.message("culvert barrel")));
        }
        if let Some(error) = crate::tensor::probe_correctly_rounded::<R>() {
            return Err(StructureError::definition(error.message("culvert barrel")));
        }
        let sqrt = R::square_root_contract(DType::F64);
        if sqrt != SquareRootContract::Ieee754RoundToNearest {
            return Err(StructureError::definition(format!(
                "unsupported {} square root: contract={sqrt:?}. Select a runtime with F64 square roots rounded to nearest before creating the culvert barrel context",
                R::name()
            )));
        }
        Self::build(client, barrels, REFINEMENT_ROUNDS)
    }

    /// A context of unit-rise rows with this context's width and client: unit-diameter circles
    /// on circle and box rows, and each arc barrel's layout in units of its rise.
    ///
    /// The critical and normal residuals read geometry in `s = y/D` from it, so tiny rises never
    /// underflow the area.
    pub(crate) fn unit(&self) -> Result<Self, StructureError>
    where
        R::Client: BarrelClient<R>,
    {
        let circle = CulvertBarrel::circular(1.0)?;
        let mut unit = Self::build(self.client.clone(), &vec![circle; self.width()], 0)?;
        let Some(arcs) = &self.unit_arcs else {
            return Ok(unit);
        };
        let c = &self.client;
        let arced = &arcs.arced;
        let merge = |[lower, upper]: &Bounds<R>,
                     [arc_lower, arc_upper]: &Bounds<R>|
         -> numr::error::Result<Bounds<R>> {
            Ok([
                c.where_cond(arced, arc_lower, lower)?,
                c.where_cond(arced, arc_upper, upper)?,
            ])
        };
        unit.circular = c.sub(&unit.one, arced)?;
        unit.full_area = merge(&unit.full_area, &arcs.full_area)?;
        unit.full_perimeter = merge(&unit.full_perimeter, &arcs.full_perimeter)?;
        unit.full_radius = merge(&unit.full_radius, &arcs.full_radius)?;
        unit.arcs = Some(arcs.select(&|value| Ok(value.clone()))?);
        Ok(unit)
    }

    /// Gathers the rows at the I64 indices `rows`, in their order. Indices may repeat.
    pub(crate) fn select(&self, rows: &Tensor<R>) -> numr::error::Result<Self>
    where
        R::Client: BarrelClient<R>,
    {
        let c = &self.client;
        let take = |value: &Tensor<R>| c.index_select(value, 0, rows);
        let pair = |[lower, upper]: &Bounds<R>| -> numr::error::Result<Bounds<R>> {
            Ok([take(lower)?, take(upper)?])
        };
        let table = |values: &[Bounds<R>]| -> numr::error::Result<Vec<Bounds<R>>> {
            values.iter().map(pair).collect()
        };
        Ok(Self {
            client: c.clone(),
            device: self.device.clone(),
            zero: take(&self.zero)?,
            one: take(&self.one)?,
            circular: take(&self.circular)?,
            boxed: take(&self.boxed)?,
            rise: take(&self.rise)?,
            span: take(&self.span)?,
            half_rise: take(&self.half_rise)?,
            shallow_top: take(&self.shallow_top)?,
            shallow_operand: take(&self.shallow_operand)?,
            series: SeriesTensors {
                asin: table(&self.series.asin)?,
                area: table(&self.series.area)?,
                moment: table(&self.series.moment)?,
            },
            critical_top: take(&self.critical_top)?,
            critical_slope: take(&self.critical_slope)?,
            normal_top: take(&self.normal_top)?,
            normal_slope: take(&self.normal_slope)?,
            conveyance_peak: pair(&self.conveyance_peak)?,
            log_peak: pair(&self.log_peak)?,
            limb: self.limb.as_deref().map(table).transpose()?,
            full_area: pair(&self.full_area)?,
            full_perimeter: pair(&self.full_perimeter)?,
            full_radius: pair(&self.full_radius)?,
            arcs: self
                .arcs
                .as_ref()
                .map(|arcs| arcs.select(&take))
                .transpose()?,
            unit_arcs: self
                .unit_arcs
                .as_ref()
                .map(|arcs| arcs.select(&take))
                .transpose()?,
            frame: self
                .frame
                .as_ref()
                .map(|frame| frame.select(rows).map(Box::new))
                .transpose()?,
            floor: self
                .floor
                .as_ref()
                .map(|floor| floor.select(&take).map(Box::new))
                .transpose()?,
        })
    }

    /// Uploads the rows of an admitted runtime, encloses the full-state constants and
    /// certifies the arc rows with `rounds` refinement rounds.
    ///
    /// Rows whose rise lies outside `[1, 2)` also get their frame, where point geometry runs.
    pub(super) fn build(
        client: R::Client,
        barrels: &[CulvertBarrel],
        rounds: usize,
    ) -> Result<Self, StructureError>
    where
        R::Client: BarrelClient<R>,
    {
        let framed = barrels
            .iter()
            .enumerate()
            .map(|(row, b)| framed_barrel(row, b))
            .collect::<Result<Vec<_>, _>>()?;
        let frame = if framed.as_slice() == barrels {
            None
        } else {
            Some(Box::new(BarrelFrame::new(
                client.clone(),
                barrels,
                &framed,
            )?))
        };
        Self::rows(client, barrels, frame)?
            .certify_arcs(barrels, rounds)?
            .certify_floors(barrels, rounds)
    }

    /// Uploads the rows and encloses the full-state constants, without depth certificates.
    ///
    /// With a frame, the arc layouts are the frame layouts times exact powers of two, so no
    /// metre product underflows or overflows on the way. Without one, the rows are their own
    /// frame.
    pub(crate) fn rows(
        client: R::Client,
        barrels: &[CulvertBarrel],
        frame: Option<Box<BarrelFrame<R>>>,
    ) -> Result<Self, StructureError>
    where
        R::Client: BarrelClient<R>,
    {
        let device = client.device().clone();
        let m = barrels.len();
        let column = |f: &dyn Fn(&CulvertBarrel) -> f64| -> numr::error::Result<Tensor<R>> {
            let values: Vec<f64> = barrels.iter().map(f).collect();
            Tensor::<R>::from_slice(&values, &[m], &device)
        };
        let circle = |b: &CulvertBarrel| b.shape() == CulvertBarrelShape::Circular;
        let flag = |value: bool| f64::from(u8::from(value));
        let circular = column(&|b| flag(circle(b)))?;
        let boxed = column(&|b| flag(b.shape() == CulvertBarrelShape::Box))?;
        let arc_rows = column(&|b| flag(b.shape().arced()))?;
        let rise = column(&|b| b.rise_m())?;
        let span = column(&|b| b.span_m())?;
        let half_rise = column(&|b| b.rise_m() * 0.5)?;
        let shallow_top = column(&|b| b.rise_m() * 0.0625)?;
        let shallow_operand = column(&|b| b.rise_m() * 0.03125)?;
        // Circle value, box value. Arc rows hold NaN until their certificates fill them.
        let certified = |[on_circle, on_box]: [f64; 2]| {
            column(&|b| match b.shape() {
                CulvertBarrelShape::Circular => on_circle,
                CulvertBarrelShape::Box => on_box,
                _ => f64::NAN,
            })
        };
        let critical_top = certified([constants::CIRCLE_CRITICAL_TOP, 1.0])?;
        let critical_slope = certified([
            constants::CIRCLE_CRITICAL_SLOPE,
            constants::BOX_CRITICAL_SLOPE,
        ])?;
        let normal_top = certified([constants::RISING_LIMB_TOP, 1.0])?;
        let normal_slope =
            certified([constants::CIRCLE_NORMAL_SLOPE, constants::BOX_NORMAL_SLOPE])?;
        let [peak_lower, peak_upper] = constants::CONVEYANCE_PEAK.map(f64::from_bits);
        let [log_lower, log_upper] = constants::LOG_PEAK_CONVEYANCE.map(f64::from_bits);
        let conveyance_peak = [certified([peak_lower, 1.0])?, certified([peak_upper, 1.0])?];
        let log_peak = [certified([log_lower, 0.0])?, certified([log_upper, 0.0])?];
        let series = SeriesTensors::constant(m, &device)?;
        let any_arc = barrels.iter().any(|b| b.shape().arced());
        let inputs = match &frame {
            None if any_arc => Some(ArcInputs::<R>::new(barrels, &device)?),
            _ => None,
        };
        let framed_arcs = frame
            .as_ref()
            .and_then(|frame| frame.context.arcs.as_ref().map(|rows| (frame, rows)));
        let (closed, arcs) = with_interval_arithmetic(
            &client,
            DType::F64,
            "culvert barrel full-state constants",
            |a| {
                let closed = section::full_constants(a, &circular, &span, &rise)?;
                let arcs = match (&framed_arcs, &inputs) {
                    (Some((frame, rows)), _) => Some(rows.scaled(a, frame)?),
                    (None, Some(inputs)) => {
                        Some(ArcRows::build(a, inputs, &arc_rows, &span, &rise)?)
                    }
                    (None, None) => None,
                };
                let unit = match (&arcs, &framed_arcs) {
                    (Some(rows), None) => Some(rows.unit(a, &rise)?),
                    _ => None,
                };
                Ok((closed, (arcs, unit)))
            },
        )?;
        let (arcs, unit_arcs) = match (arcs, framed_arcs) {
            ((Some(rows), _), Some((frame, _))) => {
                let unit = frame
                    .context
                    .unit_arcs
                    .as_ref()
                    .map(|unit| unit.select(&|value| Ok(value.clone())))
                    .transpose()?;
                (Some(rows), unit)
            }
            ((rows, unit), None) => (rows, unit),
            ((None, _), Some(_)) => (None, None),
        };
        let merge = |closed: Bounds<R>,
                     arc: fn(&ArcRows<R>) -> &Bounds<R>|
         -> numr::error::Result<Bounds<R>> {
            let Some(arcs) = &arcs else {
                return Ok(closed);
            };
            let [lower, upper] = &closed;
            let [arc_lower, arc_upper] = arc(arcs);
            Ok([
                client.where_cond(&arc_rows, arc_lower, lower)?,
                client.where_cond(&arc_rows, arc_upper, upper)?,
            ])
        };
        let [area, perimeter, radius] = closed;
        let full_area = merge(area, |arcs| &arcs.full_area)?;
        let full_perimeter = merge(perimeter, |arcs| &arcs.full_perimeter)?;
        let full_radius = merge(radius, |arcs| &arcs.full_radius)?;
        let mut context = Self {
            zero: Tensor::<R>::zeros(&[m], DType::F64, &device)?,
            one: Tensor::<R>::ones(&[m], DType::F64, &device)?,
            client,
            device,
            circular,
            boxed,
            rise,
            span,
            half_rise,
            shallow_top,
            shallow_operand,
            series,
            critical_top,
            critical_slope,
            normal_top,
            normal_slope,
            conveyance_peak,
            log_peak,
            limb: None,
            full_area,
            full_perimeter,
            full_radius,
            arcs,
            unit_arcs,
            frame,
            floor: None,
        };
        context.floor = FloorRows::new(&context, barrels)?.map(Box::new);
        Ok(context)
    }

    /// Rise of the depth domain per row: the open rise `D'` above a floor, `D` elsewhere. Exact.
    pub(crate) fn open_rise(&self) -> &Tensor<R> {
        self.floor.as_deref().map_or(&self.rise, |floor| &floor.top)
    }

    /// Barrel count `m`.
    pub(crate) fn width(&self) -> usize {
        self.rise.shape()[0]
    }

    /// `[m]` tensor holding `value` in every row.
    pub(crate) fn constant(&self, value: f64) -> numr::error::Result<Tensor<R>> {
        Tensor::<R>::full_scalar(&[self.width()], DType::F64, value, &self.device)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::structures::StructureInputErrorKind;
    use crate::structures::numerics::culvert::constants::CIRCLE_CRITICAL_TOP;
    use numr::runtime::cpu::{CpuDevice, CpuRuntime};

    #[test]
    fn empty_barrel_list_is_rejected_with_action() {
        let device = CpuDevice::new();
        let client = CpuRuntime::default_client(&device);
        let error = TensorBarrelContext::<CpuRuntime>::new(client, &[]).err();
        assert!(matches!(
            error,
            Some(StructureError::InvalidInput { kind: StructureInputErrorKind::Definition, ref context, .. })
                if context.contains("supply at least one culvert barrel")
        ));
    }

    #[test]
    fn arc_rows_join_the_context_with_certificates() -> Result<(), StructureError> {
        let device = CpuDevice::new();
        let client = CpuRuntime::default_client(&device);
        let barrels = [
            CulvertBarrel::circular(0.3)?,
            CulvertBarrel::pipe_arch(1.8542, 1.397, 0.4572, 1.938_528)?,
            CulvertBarrel::rectangular_box(2.0, 1.0)?,
            CulvertBarrel::arch(3.048, 1.6002)?,
        ];
        let context = TensorBarrelContext::<CpuRuntime>::new(client.clone(), &barrels)?;
        assert_eq!(context.boxed.try_to_vec::<f64>()?, [0.0, 0.0, 1.0, 0.0]);
        let arcs = context
            .arcs
            .as_ref()
            .ok_or_else(|| StructureError::definition("context lost its arc rows"))?;
        assert_eq!(arcs.arced.try_to_vec::<f64>()?, [0.0, 1.0, 0.0, 1.0]);
        assert_eq!(
            arcs.floor_width.try_to_vec::<f64>()?,
            [0.0, 0.0, 0.0, 3.048]
        );
        let slopes = context.critical_slope.try_to_vec::<f64>()?;
        assert!(slopes[1] > 0.0 && slopes[3] > 0.0 && slopes[0] == 5.4 && slopes[2] == 3.0);
        let tops = context.critical_top.try_to_vec::<f64>()?;
        assert_eq!(
            tops,
            [
                CIRCLE_CRITICAL_TOP,
                CIRCLE_CRITICAL_TOP,
                1.0,
                CIRCLE_CRITICAL_TOP
            ]
        );
        let unit = context.unit()?;
        assert_eq!(unit.circular.try_to_vec::<f64>()?, [1.0, 0.0, 1.0, 0.0]);
        assert!(unit.unit_arcs.is_none() && unit.arcs.is_some());
        let rows = Tensor::<CpuRuntime>::from_slice(&[3i64, 2], &[2], &device)?;
        let picked = context.select(&rows)?;
        let picked_arcs = picked
            .arcs
            .as_ref()
            .ok_or_else(|| StructureError::definition("selection lost its arc rows"))?;
        assert_eq!(picked_arcs.floor_width.try_to_vec::<f64>()?, [3.048, 0.0]);
        let circles = TensorBarrelContext::<CpuRuntime>::new(client, &barrels[..1])?;
        assert!(circles.arcs.is_none());
        Ok(())
    }

    #[test]
    fn rows_hold_exact_fractions_of_the_rise() -> Result<(), StructureError> {
        let device = CpuDevice::new();
        let client = CpuRuntime::default_client(&device);
        let barrels = vec![
            CulvertBarrel::circular(0.3)?,
            CulvertBarrel::rectangular_box(2.0, 1.0)?,
        ];
        let context = TensorBarrelContext::<CpuRuntime>::new(client, &barrels)?;
        assert_eq!(context.circular.try_to_vec::<f64>()?, [1.0, 0.0]);
        assert_eq!(context.span.try_to_vec::<f64>()?, [0.3, 2.0]);
        assert_eq!(
            context.shallow_top.try_to_vec::<f64>()?,
            [0.3 / 16.0, 0.0625]
        );
        assert_eq!(context.series.asin.len(), constants::SERIES_TERMS + 1);
        let rows = Tensor::<CpuRuntime>::from_slice(&[1i64, 0, 1], &[3], &device)?;
        let picked = context.select(&rows)?;
        assert_eq!(picked.width(), 3);
        assert_eq!(picked.rise.try_to_vec::<f64>()?, [1.0, 0.3, 1.0]);
        assert_eq!(picked.critical_slope.try_to_vec::<f64>()?, [3.0, 5.4, 3.0]);
        assert_eq!(picked.normal_top.try_to_vec::<f64>()?, [1.0, 0.93818, 1.0]);
        assert_eq!(
            picked.normal_slope.try_to_vec::<f64>()?,
            [3.0, 5.42e-5, 3.0]
        );
        assert_eq!(picked.full_perimeter[1].try_to_vec::<f64>()?[2], 6.0);
        assert_eq!(picked.series.moment[3][0].shape(), [3]);
        // Box area B D = 2 and perimeter 2 (B + D) = 6 are exact singletons.
        for (bounds, expected) in [(&context.full_area, 2.0), (&context.full_perimeter, 6.0)] {
            for endpoint in bounds {
                assert_eq!(endpoint.try_to_vec::<f64>()?[1], expected);
            }
        }
        Ok(())
    }
}
