//! Arc barrels in the march coordinate: the normal angle `psi`, band by band, in units of `D`.
//!
//! Band `k` runs from `psi_k` to `psi_{k+1}`, and the top band to `pi`. On band `k`, with arc
//! centre `(x_k, y_k)` and radius `r_k`:
//!
//! | Quantity | Form |
//! | --- | --- |
//! | `t` | `2 (x_k + r_k sin psi)` |
//! | `a` | `c_k - 2 x_k r_k cos psi + r_k^2 (psi - sin psi cos psi)` |
//! | `p` | `p_k + 2 r_k (psi - psi_k)` |
//! | `s` | `y_k - r_k cos psi` |
//! | `ds/dpsi` | `r_k sin psi` |
//!
//! `c_k = a_k + 2 x_k r_k cos psi_k - r_k^2 (psi_k - sin psi_k cos psi_k)` folds the start area
//! `a_k` into one constant per band. A circle of diameter `D` is one band with `x = 0`,
//! `r = 1/2` and `psi = phi`, the circle half angle.
//!
//! `T` and the normal are continuous at a junction, but `ds/dpsi` jumps with the radius. So a
//! derivative jet holds only over coordinates inside one band. A band certainly describes the
//! closed range from the upper bound of its start angle to the lower bound of its end angle.
//! Between the two bounds of a junction angle, a sliver a few F64 wide, either band may hold.
//! A value there takes the hull of both bands. A flat-floor arch repeats its one arc in every
//! band, so its top band holds from the floor normal to the crown.

mod jets;
mod maps;
mod outline;

pub(crate) use jets::{Trig, shape_jets};
pub(crate) use maps::{angle_of, fraction_of};
pub(crate) use outline::{Outline, hull};

use super::trig::fill;
use crate::structures::numerics::culvert::constants::PI;
use crate::structures::numerics::culvert::{Arithmetic, BarrelClient};
use crate::structures::tensor::culvert::{ArcRows, Bounds};
use numr::dtype::DType;
use numr::ops::{BinaryOps, CompareOps, ConditionalOps, IndexingOps, ScalarOps};
use numr::runtime::Runtime;
use numr::tensor::Tensor;
use solvr::common::arithmetic::ArithmeticResult;

/// One band of each row in units of the rise.
#[derive(Clone)]
pub(crate) struct MarchBand<R: Runtime<DType = DType>> {
    pub(crate) centre_x: Bounds<R>,
    pub(crate) centre_y: Bounds<R>,
    pub(crate) radius: Bounds<R>,
    pub(crate) start_angle: Bounds<R>,
    pub(crate) end_angle: Bounds<R>,
    pub(crate) start_depth: Bounds<R>,
    pub(crate) end_depth: Bounds<R>,
    pub(crate) start_perimeter: Bounds<R>,
    /// `c_k`, the area constant of the band.
    pub(crate) area_offset: Bounds<R>,
}

impl<R: Runtime<DType = DType>> MarchBand<R> {
    fn map(
        &self,
        take: &dyn Fn(&Tensor<R>) -> numr::error::Result<Tensor<R>>,
    ) -> numr::error::Result<Self> {
        let pair = |[lower, upper]: &Bounds<R>| -> numr::error::Result<Bounds<R>> {
            Ok([take(lower)?, take(upper)?])
        };
        Ok(Self {
            centre_x: pair(&self.centre_x)?,
            centre_y: pair(&self.centre_y)?,
            radius: pair(&self.radius)?,
            start_angle: pair(&self.start_angle)?,
            end_angle: pair(&self.end_angle)?,
            start_depth: pair(&self.start_depth)?,
            end_depth: pair(&self.end_depth)?,
            start_perimeter: pair(&self.start_perimeter)?,
            area_offset: pair(&self.area_offset)?,
        })
    }

    /// Takes `on` where `mask` is 1 and `off` where it is 0, field by field.
    fn choose(c: &R::Client, mask: &Tensor<R>, on: &Self, off: &Self) -> numr::error::Result<Self>
    where
        R::Client: BarrelClient<R>,
    {
        let pair = |x: &Bounds<R>, y: &Bounds<R>| -> numr::error::Result<Bounds<R>> {
            Ok([
                c.where_cond(mask, &x[0], &y[0])?,
                c.where_cond(mask, &x[1], &y[1])?,
            ])
        };
        Ok(Self {
            centre_x: pair(&on.centre_x, &off.centre_x)?,
            centre_y: pair(&on.centre_y, &off.centre_y)?,
            radius: pair(&on.radius, &off.radius)?,
            start_angle: pair(&on.start_angle, &off.start_angle)?,
            end_angle: pair(&on.end_angle, &off.end_angle)?,
            start_depth: pair(&on.start_depth, &off.start_depth)?,
            end_depth: pair(&on.end_depth, &off.end_depth)?,
            start_perimeter: pair(&on.start_perimeter, &off.start_perimeter)?,
            area_offset: pair(&on.area_offset, &off.area_offset)?,
        })
    }
}

/// Arc rows of a march in units of the rise, with a harmless arc layout on other rows.
#[derive(Clone)]
pub(crate) struct ArcShape<R: Runtime<DType = DType>> {
    /// 1 on arc rows.
    pub(crate) arced: Tensor<R>,
    /// 1 on flat-floor arch rows.
    pub(crate) floored: Tensor<R>,
    /// The three bands from the invert, or the one band that holds each lane.
    pub(crate) bands: Vec<MarchBand<R>>,
    /// Lower bound of the least band radius.
    pub(crate) least_radius: Tensor<R>,
    /// A coordinate inside every row's domain: the middle of the arc on arch rows, 1/2 on others.
    pub(crate) harmless: Tensor<R>,
}

impl<R: Runtime<DType = DType>> ArcShape<R> {
    /// The march bands of the unit-rise arc layouts `rows`.
    pub(crate) fn new(a: &mut Arithmetic<'_, R>, rows: &ArcRows<R>) -> ArithmeticResult<Self>
    where
        R::Client: BarrelClient<R>,
    {
        let c = a.client();
        let like = &rows.arced;
        let pi = [
            fill(like, f64::from_bits(PI[0]))?,
            fill(like, f64::from_bits(PI[1]))?,
        ];
        let crown = [fill(like, 1.0)?, fill(like, 1.0)?];
        let two = a.point(&fill(like, 2.0)?)?;
        let mut bands = Vec::with_capacity(rows.bands.len());
        for (k, band) in rows.bands.iter().enumerate() {
            let (end_angle, end_depth) = match rows.bands.get(k + 1) {
                Some(next) => (next.start_angle.clone(), next.start_depth.clone()),
                None => (pi.clone(), crown.clone()),
            };
            let x = a.bounds(&band.centre_x[0], &band.centre_x[1])?;
            let r = a.bounds(&band.radius[0], &band.radius[1])?;
            let angle = a.bounds(&band.start_angle[0], &band.start_angle[1])?;
            let sine = a.bounds(&band.start_sin[0], &band.start_sin[1])?;
            let cosine = a.bounds(&band.start_cos[0], &band.start_cos[1])?;
            let start = a.bounds(&band.start_area[0], &band.start_area[1])?;
            let lever = a.mul(&two, &x)?;
            let lever = a.mul(&lever, &r)?;
            let lift = a.mul(&lever, &cosine)?;
            let product = a.mul(&sine, &cosine)?;
            let segment = a.sub(&angle, &product)?;
            let square = a.mul(&r, &r)?;
            let segment = a.mul(&square, &segment)?;
            let raised = a.add(&start, &lift)?;
            let offset = a.sub(&raised, &segment)?;
            bands.push(MarchBand {
                centre_x: band.centre_x.clone(),
                centre_y: band.centre_y.clone(),
                radius: band.radius.clone(),
                start_angle: band.start_angle.clone(),
                end_angle,
                start_depth: band.start_depth.clone(),
                end_depth,
                start_perimeter: band.start_perimeter.clone(),
                area_offset: [offset.lower().clone(), offset.upper().clone()],
            });
        }
        let mut least_radius = rows.bands[0].radius[0].clone();
        for band in rows.bands.iter().skip(1) {
            least_radius = c.minimum(&least_radius, &band.radius[0])?;
        }
        // The arch arc runs from its floor normal to pi.
        let top = &rows.bands[2].start_angle[1];
        let middle = c.mul_scalar(&c.add(top, &pi[0])?, 0.5)?;
        let harmless = c.where_cond(&rows.floored, &middle, &fill(like, 0.5)?)?;
        Ok(Self {
            arced: rows.arced.clone(),
            floored: rows.floored.clone(),
            bands,
            least_radius,
            harmless,
        })
    }

    fn map(
        &self,
        take: &dyn Fn(&Tensor<R>) -> numr::error::Result<Tensor<R>>,
    ) -> numr::error::Result<Self> {
        Ok(Self {
            arced: take(&self.arced)?,
            floored: take(&self.floored)?,
            bands: self
                .bands
                .iter()
                .map(|band| band.map(take))
                .collect::<numr::error::Result<_>>()?,
            least_radius: take(&self.least_radius)?,
            harmless: take(&self.harmless)?,
        })
    }

    /// Gathers the rows at the I64 indices `rows`. Indices may repeat.
    pub(crate) fn select(&self, c: &R::Client, rows: &Tensor<R>) -> numr::error::Result<Self>
    where
        R::Client: BarrelClient<R>,
    {
        self.map(&|value| c.index_select(value, 0, rows))
    }

    /// Rows `start .. start + len`.
    pub(crate) fn narrow(&self, start: usize, len: usize) -> numr::error::Result<Self> {
        self.map(&|value| value.narrow(0, start, len))
    }

    /// 1 on rows whose band `k` certainly holds every coordinate of `[low, high]`, for each of
    /// the three bands, with at most one 1 per row. Arch rows take the top band.
    pub(crate) fn singles(
        &self,
        c: &R::Client,
        low: &Tensor<R>,
        high: &Tensor<R>,
    ) -> numr::error::Result<[Tensor<R>; 3]>
    where
        R::Client: BarrelClient<R>,
    {
        let [bottom, corner, top] = self.three()?;
        let one = fill(low, 1.0)?;
        let first = c.le(high, &bottom.end_angle[0])?;
        let second = c.mul(
            &c.ge(low, &corner.start_angle[1])?,
            &c.le(high, &corner.end_angle[0])?,
        )?;
        let third = c.ge(low, &top.start_angle[1])?;
        let three = c.sub(&one, &self.floored)?;
        let first = c.mul(&first, &three)?;
        let second = c.mul(&c.mul(&second, &three)?, &c.sub(&one, &first)?)?;
        let taken = c.maximum(&first, &second)?;
        let third = c.mul(&c.maximum(&third, &self.floored)?, &c.sub(&one, &taken)?)?;
        Ok([first, second, third])
    }

    /// 1 on rows whose band `k` may hold a coordinate of `[low, high]`, for each band. Arch rows
    /// take the top band.
    pub(crate) fn members(
        &self,
        c: &R::Client,
        low: &Tensor<R>,
        high: &Tensor<R>,
    ) -> numr::error::Result<[Tensor<R>; 3]>
    where
        R::Client: BarrelClient<R>,
    {
        let [bottom, corner, top] = self.three()?;
        let one = fill(low, 1.0)?;
        let three = c.sub(&one, &self.floored)?;
        let first = c.mul(&c.le(low, &bottom.end_angle[1])?, &three)?;
        let second = c.mul(
            &c.mul(
                &c.ge(high, &corner.start_angle[0])?,
                &c.le(low, &corner.end_angle[1])?,
            )?,
            &three,
        )?;
        let third = c.maximum(&c.ge(high, &top.start_angle[0])?, &self.floored)?;
        Ok([first, second, third])
    }

    /// 1 on rows whose band `k` may hold the depth fraction `s`, for each band. Arch rows take
    /// the top band.
    pub(crate) fn depth_members(
        &self,
        c: &R::Client,
        s: &Tensor<R>,
    ) -> numr::error::Result<[Tensor<R>; 3]>
    where
        R::Client: BarrelClient<R>,
    {
        let [bottom, corner, top] = self.three()?;
        let one = fill(s, 1.0)?;
        let three = c.sub(&one, &self.floored)?;
        let first = c.mul(&c.le(s, &bottom.end_depth[1])?, &three)?;
        let second = c.mul(
            &c.mul(
                &c.ge(s, &corner.start_depth[0])?,
                &c.le(s, &corner.end_depth[1])?,
            )?,
            &three,
        )?;
        let third = c.maximum(&c.ge(s, &top.start_depth[0])?, &self.floored)?;
        Ok([first, second, third])
    }

    /// Each lane `[left, right]` with only the band that holds it, and a mask of the lanes inside
    /// a junction sliver, where no one band holds every coordinate.
    ///
    /// `self` holds the three bands of each lane's row. A sliver lane keeps the top band.
    pub(crate) fn lanes(
        &self,
        c: &R::Client,
        left: &Tensor<R>,
        right: &Tensor<R>,
    ) -> numr::error::Result<(Self, Tensor<R>)>
    where
        R::Client: BarrelClient<R>,
    {
        let [first, second, third] = self.singles(c, left, right)?;
        let [bottom, corner, top] = self.three()?;
        let upper = MarchBand::choose(c, &second, corner, top)?;
        let band = MarchBand::choose(c, &first, bottom, &upper)?;
        let held = c.maximum(&c.maximum(&first, &second)?, &third)?;
        let wide = c.gt(right, left)?;
        let open = c.sub(&fill(left, 1.0)?, &held)?;
        let sliver = c.mul(&c.mul(&open, &wide)?, &self.arced)?;
        Ok((
            Self {
                bands: vec![band],
                ..self.clone()
            },
            sliver,
        ))
    }

    fn three(&self) -> numr::error::Result<[&MarchBand<R>; 3]> {
        match self.bands.as_slice() {
            [bottom, corner, top] => Ok([bottom, corner, top]),
            _ => Err(numr::error::Error::InvalidArgument {
                arg: "culvert outlet arc bands",
                reason: format!(
                    "{} bands where three are needed: report this runtime",
                    self.bands.len()
                ),
            }),
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::structures::numerics::culvert::arcs::meet;
    use crate::structures::numerics::culvert::outlet::coordinates::{area, coordinate, depth};
    use crate::structures::numerics::culvert::outlet::integrand::shape;
    use crate::structures::numerics::culvert::outlet::jet;
    use crate::structures::numerics::culvert::outlet::prepare;
    use crate::structures::numerics::culvert::section;
    use crate::structures::tensor::culvert::TensorBarrelContext;
    use crate::structures::{CulvertBarrel, CulvertOutletRow, StructureError};
    use numr::dtype::DType;
    use numr::runtime::Runtime;
    use numr::runtime::cpu::{CpuDevice, CpuRuntime};
    use numr::tensor::Tensor;
    use solvr::common::arithmetic::with_interval_arithmetic;

    type Cpu = CpuRuntime;
    type Pairs = Vec<[Vec<f64>; 2]>;

    /// One evidence barrel per arc shape.
    fn barrels() -> Result<Vec<CulvertBarrel>, StructureError> {
        Ok(vec![
            CulvertBarrel::horizontal_ellipse(1.524, 0.9652, 0.419_475_92)?,
            CulvertBarrel::vertical_ellipse(0.9652, 1.524, 0.419_475_92)?,
            CulvertBarrel::pipe_arch(1.8542, 1.397, 0.4572, 1.938_528)?,
            CulvertBarrel::arch(3.048, 1.6002)?,
        ])
    }

    fn row(barrel: CulvertBarrel) -> CulvertOutletRow {
        CulvertOutletRow {
            barrel,
            length_m: 40.0,
            manning_n: 0.013,
            entrance_loss: 0.5,
            slope: 0.001,
            gravity_m_per_s2: 9.81,
        }
    }

    /// Frozen criteria: at depth fractions across every band and at the F64 neighbours of each
    /// junction depth, the march geometry at the coordinate of a depth meets the section
    /// geometry there, the depth of that coordinate holds the depth, and the area jet's first
    /// coefficient meets `t ds/dpsi`.
    #[test]
    fn march_bands_meet_the_section_geometry() -> Result<(), StructureError> {
        let fractions = [1e-3, 0.02, 0.07, 0.15, 0.3, 0.5, 0.7, 0.9, 0.97, 0.999];
        let shapes = barrels()?;
        let device = CpuDevice::new();
        let client = Cpu::default_client(&device);
        let first = TensorBarrelContext::<Cpu>::new(client.clone(), &shapes)?;
        let rows = first
            .arcs
            .as_ref()
            .ok_or_else(|| StructureError::definition("no arc rows"))?;
        let mut barrels = Vec::new();
        let mut depths = Vec::new();
        for (index, culvert) in shapes.iter().enumerate() {
            let rise = culvert.rise_m();
            let mut points: Vec<f64> = fractions.iter().map(|f| f * rise).collect();
            for band in &rows.bands[1..] {
                for bound in &band.start_depth {
                    let junction = bound.try_to_vec::<f64>()?[index];
                    if junction > 0.0 {
                        points.extend([junction.next_down(), junction.next_up()]);
                    }
                }
            }
            for point in points {
                barrels.push(*culvert);
                depths.push(point);
            }
        }
        let barrel = TensorBarrelContext::<Cpu>::new(client.clone(), &barrels)?;
        let outlets: Vec<CulvertOutletRow> = barrels.iter().copied().map(row).collect();
        let outlet = prepare(&barrel, &outlets)?;
        let at = Tensor::<Cpu>::from_slice(&depths, &[depths.len()], &device)?;
        let read =
            |i: &solvr::common::arithmetic::Interval<Cpu>| -> numr::error::Result<[Vec<f64>; 2]> {
                Ok([
                    i.lower().try_to_vec::<f64>()?,
                    i.upper().try_to_vec::<f64>()?,
                ])
            };
        let (march, exact, back, slope, held): (Pairs, Pairs, [Vec<f64>; 2], Pairs, Vec<f64>) =
            with_interval_arithmetic(&client, DType::F64, "march bands", |a| {
                let outline = outlet.outline(&barrel);
                let y = a.point(&at)?;
                let v = coordinate(a, &outline, &barrel.rise, &y)?;
                let jets = shape(a, &outline, &v, 1)?;
                let rise = a.point(&barrel.rise)?;
                let square = a.mul(&rise, &rise)?;
                let like = &barrel.rise;
                // The area at both ends of the coordinate, as the march reads it.
                let area = area(a, &outline, &barrel.rise, &v)?;
                let whole = jet::coefficient(a, &jets.area, 0, like)?;
                let whole = a.mul(&whole, &square)?;
                let area = meet(a, &area, &whole)?;
                let top = jet::coefficient(a, &jets.top, 0, like)?;
                let top = a.mul(&top, &rise)?;
                let wetted = jet::coefficient(a, &jets.perimeter, 0, like)?;
                let wetted = a.mul(&wetted, &rise)?;
                let geometry = section::at_depth(a, &barrel, &at)?.geometry;
                let back = depth(a, &outline, &barrel.rise, &v)?;
                // A narrow coordinate enclosure inside each band carries smooth jets.
                let wide = shape(a, &outline, &v, 5)?;
                let rate = jet::coefficient(a, &wide.area, 1, like)?;
                let t = jet::coefficient(a, &wide.top, 0, like)?;
                let step = jet::coefficient(a, &wide.step, 0, like)?;
                let product = a.mul(&t, &step)?;
                let held = match &wide.valid {
                    Some(valid) => valid.try_to_vec::<f64>()?,
                    None => vec![1.0; depths.len()],
                };
                Ok((
                    vec![read(&area)?, read(&top)?, read(&wetted)?],
                    vec![
                        read(&geometry.area)?,
                        read(&geometry.top_width)?,
                        read(&geometry.perimeter)?,
                    ],
                    read(&back)?,
                    vec![read(&rate)?, read(&product)?],
                    held,
                ))
            })?;
        // Every depth away from a junction holds its jets in one band.
        let smooth = held.iter().filter(|&&h| h == 1.0).count();
        assert!(smooth >= fractions.len() * shapes.len());
        for (row, &y) in depths.iter().enumerate() {
            for (quantity, (x, e)) in march.iter().zip(&exact).enumerate() {
                let (low, high) = (x[0][row], x[1][row]);
                let (exact_low, exact_high) = (e[0][row], e[1][row]);
                assert!(
                    low <= exact_high && exact_low <= high,
                    "row {row} y={y:e} quantity {quantity}: [{low:e}, {high:e}] misses [{exact_low:e}, {exact_high:e}]"
                );
                assert!(
                    high - low <= 1e-9 * high.abs().max(1e-300),
                    "row {row} y={y:e} quantity {quantity}: [{low:e}, {high:e}] too wide"
                );
            }
            assert!(back[0][row] <= y && y <= back[1][row], "row {row} y={y:e}");
            if held[row] != 1.0 {
                continue;
            }
            let [rate, product] = [&slope[0], &slope[1]];
            assert!(
                rate[0][row] <= product[1][row] && product[0][row] <= rate[1][row],
                "row {row} y={y:e}: da/dpsi {:?} misses t ds/dpsi {:?}",
                (rate[0][row], rate[1][row]),
                (product[0][row], product[1][row])
            );
        }
        Ok(())
    }
}
