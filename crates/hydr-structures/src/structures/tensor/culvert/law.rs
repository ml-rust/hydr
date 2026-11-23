//! Compact culvert law rows: one lane per culvert and two oriented rows per lane.
//!
//! Lane `j` is the `j`-th culvert element. Row `j` holds the A-end inlet and row `m + j` the
//! B-end inlet. A culvert without one end repeats its other end in that row, so admission
//! never sees an undeclared inlet. The flap keeps those rows unused.

use super::{Bounds, TensorBarrelContext, TensorInletContext, TensorOutletContext};
use crate::structures::blockage::{CulvertGeometrySummary, effective};
use crate::structures::numerics::culvert::{inlet::anchors, outlet};
use crate::structures::tensor::{
    TensorStructureContext, TensorStructureInput, TensorStructureResult,
};
use crate::structures::{
    CulvertBarrel, CulvertBarrelShape, CulvertClient, CulvertSide, DirectionPolicy, StructureError,
    StructureLaw,
};
use numr::dtype::DType;
use numr::ops::{BinaryOps, ConditionalOps};
use numr::runtime::Runtime;
use numr::tensor::Tensor;

/// Culvert evaluation, built once with the client operations the culvert numerics need.
///
/// Evaluation through this trait needs no client bound beyond the scalar laws'.
pub(crate) trait CulvertLaw<R: Runtime<DType = DType>>: Send + Sync {
    /// Writes the culvert lanes of `result`.
    fn evaluate(
        &self,
        context: &TensorStructureContext<R>,
        input: &TensorStructureInput<R>,
        result: TensorStructureResult<R>,
    ) -> Result<TensorStructureResult<R>, StructureError>;

    /// Precomputed full-area and floor-width enclosures of each culvert element.
    fn geometry(&self) -> &[CulvertGeometrySummary];
}

/// Culvert operands per element, `[n]` wide and zero off culvert lanes.
#[derive(Clone, Debug)]
pub(crate) struct TensorCulvertResult<R: Runtime<DType = DType>> {
    /// Donor stage above the inlet invert.
    pub headwater: Tensor<R>,
    /// Receiver stage above the outlet invert, zero for a free outfall.
    pub tailwater: Tensor<R>,
    /// Lower bound of the per-barrel discharge.
    pub lower: Tensor<R>,
    /// Upper bound of the per-barrel discharge.
    pub upper: Tensor<R>,
    /// Outlet roots of this evaluation, for the search of a later one.
    pub hint: CulvertHint<R>,
}

/// Outlet roots of one culvert evaluation, each `[m]` over the culvert lanes.
///
/// A later evaluation of the same laws at nearby heads searches its outlet root near these
/// first. The hint only sets where the search starts. Every enclosure stays certified by the
/// residual signs at its ends, so a stale hint costs work and never accuracy.
///
/// A hinted evaluation returns its own roots over the hint's, lane by lane.
#[doc(hidden)]
#[derive(Clone, Debug)]
pub struct CulvertHint<R: Runtime<DType = DType>> {
    /// 1 where the outlet governs or ties with a searched root.
    pub(crate) held: Tensor<R>,
    /// Oriented row of each lane, as F64.
    pub(crate) rows: Tensor<R>,
    /// Donor stage above the inlet invert, in metres.
    pub(crate) headwater: Tensor<R>,
    /// Receiver stage above the outlet invert, in metres.
    pub(crate) tailwater: Tensor<R>,
    /// Outlet root enclosure per barrel, in m³/s.
    pub(crate) root: Bounds<R>,
    /// Bounds on `|dQ/dHW| / Q` and `|dQ/dTW| / Q` of the outlet root, per metre.
    pub(crate) slope: Bounds<R>,
}

impl<R: Runtime<DType = DType>> CulvertHint<R> {
    /// This hint on its held lanes, and `older` on the others.
    pub(crate) fn over(self, older: &Self, c: &R::Client) -> numr::error::Result<Self>
    where
        R::Client: BinaryOps<R> + ConditionalOps<R>,
    {
        let pick = |new: &Tensor<R>, old: &Tensor<R>| c.where_cond(&self.held, new, old);
        Ok(Self {
            rows: pick(&self.rows, &older.rows)?,
            headwater: pick(&self.headwater, &older.headwater)?,
            tailwater: pick(&self.tailwater, &older.tailwater)?,
            root: [
                pick(&self.root[0], &older.root[0])?,
                pick(&self.root[1], &older.root[1])?,
            ],
            slope: [
                pick(&self.slope[0], &older.slope[0])?,
                pick(&self.slope[1], &older.slope[1])?,
            ],
            held: c.maximum(&self.held, &older.held)?,
        })
    }
}

/// Culvert rows of one structure batch.
pub(crate) struct TensorCulvertLaw<R: Runtime<DType = DType>> {
    /// Element index of each lane.
    pub(crate) elements: Vec<usize>,
    /// Element index of each lane, I64 `[m]`.
    pub(crate) lanes: Tensor<R>,
    /// Barrel of each row, `2m` long, for error names.
    pub(crate) barrels: Vec<CulvertBarrel>,
    /// `2m` barrel rows.
    pub(crate) barrel: TensorBarrelContext<R>,
    /// The A-end rows `0..m`, one per lane. Both rows of a lane share its barrel.
    pub(crate) lane: TensorBarrelContext<R>,
    /// `2m` unit-circle rows.
    pub(crate) unit: TensorBarrelContext<R>,
    /// `2m` inlet rows.
    pub(crate) inlet: TensorInletContext<R>,
    /// `2m` outlet rows.
    pub(crate) outlet: TensorOutletContext<R>,
    /// Bed elevations at A and B per lane (`fl(z + h)` above sediment, invert otherwise).
    pub(crate) invert: [Tensor<R>; 2],
    /// 1 where flow may leave A, and 1 where flow may leave B, per lane.
    pub(crate) allow: [Tensor<R>; 2],
    /// Effective open barrel count `N - k` per lane.
    pub(crate) count: Tensor<R>,
    /// Lane index `j` per lane, as F64.
    pub(crate) forward: Tensor<R>,
    /// Lane count `m`, the offset of the B-end rows.
    pub(crate) width: usize,
    /// Full-area and floor-width enclosures per culvert element, for geometry reports.
    pub(crate) geometry: Vec<CulvertGeometrySummary>,
}

impl<R: Runtime<DType = DType>> TensorCulvertLaw<R> {
    /// Builds the rows of every culvert element and admits each inlet and outlet row.
    ///
    /// Returns `None` when the batch holds no culvert. A rejected row names its element.
    pub(crate) fn new(context: &TensorStructureContext<R>) -> Result<Option<Self>, StructureError>
    where
        R::Client: CulvertClient<R>,
    {
        let mut elements = Vec::new();
        let mut forward_rows = Vec::new();
        let mut reverse_rows = Vec::new();
        let mut invert = [Vec::new(), Vec::new()];
        let mut allow = [Vec::new(), Vec::new()];
        let mut count = Vec::new();
        for (index, element) in context.elements.iter().enumerate() {
            let StructureLaw::Culvert(parameters) = &element.law else {
                continue;
            };
            let blockage = effective::culvert_blockage(&element.condition);
            let a = parameters
                .effective_rows(CulvertSide::A, blockage)
                .map_err(|e| e.at(index))?;
            let b = parameters
                .effective_rows(CulvertSide::B, blockage)
                .map_err(|e| e.at(index))?;
            let (Some(forward), Some(reverse)) = (a.or(b), b.or(a)) else {
                return Err(StructureError::definition(
                    "culvert declares no end: supply the end of at least one flow direction",
                )
                .at(index));
            };
            let bed_a = parameters
                .bed_elevation(CulvertSide::A, blockage)
                .map_err(|e| e.at(index))?;
            let bed_b = parameters
                .bed_elevation(CulvertSide::B, blockage)
                .map_err(|e| e.at(index))?;
            let open_barrels = parameters
                .barrel_count()
                .saturating_sub(blockage.map_or(0, |b| b.closed_barrels));
            elements.push(index);
            forward_rows.push(forward);
            reverse_rows.push(reverse);
            invert[0].push(bed_a);
            invert[1].push(bed_b);
            allow[0].push(flag(element.direction != DirectionPolicy::BtoAOnly));
            allow[1].push(flag(element.direction != DirectionPolicy::AtoBOnly));
            count.push(f64::from(open_barrels));
        }
        if elements.is_empty() {
            return Ok(None);
        }
        let width = elements.len();
        let rows: Vec<_> = forward_rows.into_iter().chain(reverse_rows).collect();
        let at = |error: StructureError| match error.element() {
            Some(row) => error.at(elements[row % width]),
            None => error,
        };
        let barrels: Vec<CulvertBarrel> = rows.iter().map(|(inlet, _)| inlet.barrel).collect();
        let inlet_rows: Vec<_> = rows.iter().map(|(inlet, _)| *inlet).collect();
        let outlet_rows: Vec<_> = rows.iter().map(|(_, outlet)| *outlet).collect();
        let barrel = TensorBarrelContext::new(context.client.clone(), &barrels).map_err(at)?;
        let unit = barrel.unit()?;
        let inlet = anchors::prepare(&barrels, &barrel, &unit, &inlet_rows).map_err(at)?;
        let outlet = outlet::prepare(&barrel, &outlet_rows).map_err(at)?;
        let device = &context.device;
        let column = |values: &[f64]| Tensor::<R>::from_slice(values, &[values.len()], device);
        let lanes: Vec<i64> = elements
            .iter()
            .map(|&index| i64::try_from(index))
            .collect::<Result<_, _>>()
            .map_err(|_| StructureError::definition("culvert element index exceeds I64"))?;
        let forward: Vec<f64> = (0..width).map(|lane| lane as f64).collect();
        let front: Vec<i64> = (0..lanes.len())
            .map(|lane| i64::try_from(lane).unwrap_or(i64::MAX))
            .collect();
        let lane = barrel.select(&Tensor::<R>::from_slice(&front, &[width], device)?)?;
        let geometry = summarize_geometry(&elements, &barrels[..width], &lane)?;
        Ok(Some(Self {
            lane,
            lanes: Tensor::<R>::from_slice(&lanes, &[width], device)?,
            elements,
            barrels,
            barrel,
            unit,
            inlet,
            outlet,
            invert: [column(&invert[0])?, column(&invert[1])?],
            allow: [column(&allow[0])?, column(&allow[1])?],
            count: column(&count)?,
            forward: column(&forward)?,
            width,
            geometry,
        }))
    }
}

fn summarize_geometry<R: Runtime<DType = DType>>(
    elements: &[usize],
    barrels: &[CulvertBarrel],
    lane: &TensorBarrelContext<R>,
) -> Result<Vec<CulvertGeometrySummary>, StructureError>
where
    R::Client: CulvertClient<R>,
{
    let base_low = lane.full_area[0].try_to_vec::<f64>()?;
    let base_high = lane.full_area[1].try_to_vec::<f64>()?;
    let open = lane.open_full_area()?;
    let eff_low = open[0].try_to_vec::<f64>()?;
    let eff_high = open[1].try_to_vec::<f64>()?;
    let floor_info = match lane.floor.as_deref() {
        Some(floor) => Some((
            floor.floored.try_to_vec::<f64>()?,
            floor.bed_width[0].try_to_vec::<f64>()?,
            floor.bed_width[1].try_to_vec::<f64>()?,
        )),
        None => None,
    };
    let mut out = Vec::with_capacity(elements.len());
    for (j, (&element, barrel)) in elements.iter().zip(barrels).enumerate() {
        let base_floor = if matches!(
            barrel.shape(),
            CulvertBarrelShape::Box | CulvertBarrelShape::Arch
        ) {
            let span = barrel.span_m();
            (span, span)
        } else {
            (0.0, 0.0)
        };
        let eff_floor = match &floor_info {
            Some((floored, low, high)) if floored[j] != 0.0 => (low[j], high[j]),
            _ => base_floor,
        };
        out.push(CulvertGeometrySummary {
            element,
            base_full_area_m2: (base_low[j], base_high[j]),
            effective_full_area_m2: (eff_low[j], eff_high[j]),
            base_floor_width_m: base_floor,
            effective_floor_width_m: eff_floor,
        });
    }
    Ok(out)
}

fn flag(value: bool) -> f64 {
    f64::from(u8::from(value))
}
