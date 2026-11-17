use super::band_terms::{BandTerms, stack};
use super::boundary_area::knot;
use super::inputs::{Inputs, column};
use crate::channel::flow::surveyed::equation_ops::choose_expansion;
use crate::channel::flow::surveyed::error::Arithmetic;
use numr::dtype::DType;
use numr::ops::{
    BinaryOps, CompareOps, ConditionalOps, NextRepresentableOps, ScalarOps, ShapeOps, TensorOps,
    TwoSumOps, UnaryOps,
};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;
use solvr::common::arithmetic::{ArithmeticResult, CertifiedExpansion};

/// Largest cap area, width, depth, integral, or band slope that the model accepts, `2^1000`.
///
/// Model products up to three band thicknesses above a knot stay inside the F64 range at this scale.
/// Sections above it keep certified geometry and take no model value.
const MODEL_SCALE_LIMIT: f64 = f64::from_bits(0x7e70_0000_0000_0000);

/// Selection-only band polynomials, compiled once per section batch.
///
/// Tables have shape `[section, band]`. Each band starts at its bottom knot.
/// At depth `s` above the knot, area is `A + w*s + m*s²/2`.
/// The hydrostatic integral is `I + A*s + w*s²/2 + m*s³/6`.
/// Knot areas equal the boundary-area points. Knot integrals accumulate in two-part arithmetic.
/// The slope `m` makes both polynomials meet the next knot to two-part precision.
/// The model never certifies geometry. Certified queries use the interval coefficients.
#[derive(Debug)]
pub(super) struct BandModel<R: Runtime<DType = DType>> {
    /// Area at each band bottom, in square metres.
    pub area: CertifiedExpansion<R>,
    /// Six times the hydrostatic integral at each band bottom, in cubic metres.
    ///
    /// The factor six keeps knot sums free of division.
    pub six_integral: CertifiedExpansion<R>,
    /// F64 width at each band bottom, in metres.
    pub width: Tensor<R>,
    /// Two-part width slope with respect to depth.
    pub slope: CertifiedExpansion<R>,
    /// F64 mask `[section]`: one where the section scale admits model arithmetic.
    pub usable: Tensor<R>,
}

/// Cap bounds `[section]` from the certified compilation, used to admit model arithmetic.
pub(super) struct CapScale<'a, R: Runtime<DType = DType>> {
    pub area: &'a Tensor<R>,
    pub width: &'a Tensor<R>,
    pub depth: &'a Tensor<R>,
    pub integral: &'a Tensor<R>,
}

struct Columns<R: Runtime<DType = DType>> {
    area: Vec<CertifiedExpansion<R>>,
    six_integral: Vec<CertifiedExpansion<R>>,
    width: Vec<Tensor<R>>,
    slope: Vec<CertifiedExpansion<R>>,
}

impl<R: Runtime<DType = DType>> Columns<R> {
    fn new(bands: usize) -> Self {
        Self {
            area: Vec::with_capacity(bands),
            six_integral: Vec::with_capacity(bands),
            width: Vec::with_capacity(bands),
            slope: Vec::with_capacity(bands),
        }
    }
}

impl<R> BandModel<R>
where
    R: Runtime<DType = DType>,
    R::Client: TensorOps<R>
        + CompareOps<R>
        + NextRepresentableOps<R>
        + ScalarOps<R>
        + TwoSumOps<R>
        + RuntimeClient<R>,
{
    /// Compile the model from original coordinates. Cap bounds are certified interval upper endpoints.
    pub fn compile(
        a: &mut Arithmetic<'_, R>,
        inputs: &Inputs<R>,
        bands: usize,
        cap: &CapScale<'_, R>,
    ) -> ArithmeticResult<Self> {
        let c = a.client();
        let rows = inputs.bed.numel();
        let zeros = Tensor::zeros(&[rows], DType::F64, c.device())?;
        let ones = Tensor::ones(&[rows], DType::F64, c.device())?;
        let limit = Tensor::full_scalar(&[], DType::F64, MODEL_SCALE_LIMIT, c.device())?;
        let mut usable = ones.clone();
        for bound in [cap.area, cap.width, cap.depth, cap.integral] {
            usable = c.mul(&usable, &c.le(bound, &limit)?)?;
        }
        let half = a.expansion_point(&c.mul_scalar(&ones, 0.5)?)?;
        let zero = a.expansion_point(&zeros)?;
        let one = a.expansion_point(&ones)?;
        let mut area = a.expansion_point(&zeros)?;
        let mut six_integral = a.expansion_point(&zeros)?;
        let mut columns = Columns::new(bands);
        for band in 0..bands {
            let valid = column(&inputs.valid, band, rows)?;
            let knot = knot::band(a, inputs, band, &zeros, &half)?;
            let height = a.expansion_interval(&knot.height)?;
            let admitted = c.mul(&valid, &usable)?;
            let start = c.where_cond(
                &admitted,
                &c.maximum(knot.start_width.high(), &zeros)?,
                &zeros,
            )?;
            let start_point = a.expansion_point(&start)?;
            let mean = choose_expansion(a, &admitted, &knot.mean_width, &zero)?;
            // m = 2(mean - w)/t, so the area polynomial meets the band area at depth t.
            let excess = a.expansion_sub(&mean, &start_point)?;
            let twice = a.expansion_add(&excess, &excess)?;
            let twice_bounds = a.expansion_interval(&twice)?;
            let magnitude =
                c.maximum(&c.abs(twice_bounds.lower())?, &c.abs(twice_bounds.upper())?)?;
            let finite_slope = c.le(
                &magnitude,
                &c.mul_scalar(height.lower(), MODEL_SCALE_LIMIT)?,
            )?;
            usable = c.mul(&usable, &c.maximum(&finite_slope, &c.eq(&valid, &zeros)?)?)?;
            let active = c.mul(&c.mul(&valid, &usable)?, &c.gt(height.lower(), &zeros)?)?;
            let width = c.where_cond(&active, &start, &zeros)?;
            let width_point = a.expansion_point(&width)?;
            let twice = choose_expansion(a, &active, &twice, &zero)?;
            let divisor = choose_expansion(a, &active, &knot.height, &one)?;
            let slope = a.expansion_div_positive(&twice, &divisor)?;
            let slope = choose_expansion(a, &active, &slope, &zero)?;
            let thickness = choose_expansion(a, &active, &knot.height, &zero)?;
            let terms = BandTerms {
                area: a.expansion_bounds(area.high(), area.low(), area.abs_error())?,
                six_integral: a.expansion_bounds(
                    six_integral.high(),
                    six_integral.low(),
                    six_integral.abs_error(),
                )?,
                width: width_point,
                slope,
            };
            let next = terms.six_integral_at(a, &thickness)?;
            six_integral = choose_expansion(a, &active, &next, &terms.six_integral)?;
            area = a.expansion_add(&area, &knot.increment)?;
            columns.area.push(terms.area);
            columns.six_integral.push(terms.six_integral);
            columns.width.push(width);
            columns.slope.push(terms.slope);
        }
        let widths: Vec<_> = columns.width.iter().collect();
        Ok(Self {
            area: stack(a, &columns.area)?,
            six_integral: stack(a, &columns.six_integral)?,
            width: c.stack(&widths, 1)?,
            slope: stack(a, &columns.slope)?,
            usable,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::super::area_query::tests::test_support::{
        self as support, Client, Result, hydraulic,
    };
    use super::super::band_terms::BandTerms;
    use super::super::boundary_area::knot;
    use super::super::inputs::column;
    use super::CertifiedExpansion;
    use crate::channel::flow::surveyed::error::{self, Arithmetic, HydraulicResult};
    use crate::channel::section::HydraulicGeometry;
    use numr::{
        dtype::DType,
        ops::ScalarOps,
        runtime::{Runtime, RuntimeClient},
        tensor::Tensor,
    };

    fn band_column<R>(
        a: &mut Arithmetic<'_, R>,
        value: &CertifiedExpansion<R>,
        band: usize,
    ) -> HydraulicResult<CertifiedExpansion<R>>
    where
        R: Runtime<DType = DType>,
        R::Client: Client<R>,
    {
        let part = |tensor: &Tensor<R>| -> HydraulicResult<Tensor<R>> {
            Ok(tensor.narrow(1, band, 1)?.squeeze(Some(1)))
        };
        Ok(a.expansion_bounds(
            &part(value.high())?,
            &part(value.low())?,
            &part(value.abs_error())?,
        )?)
    }

    fn continuity<R>(client: &R::Client) -> Result
    where
        R: Runtime<DType = DType>,
        R::Client: Client<R>,
    {
        let section = support::batch::<R>(client)?;
        let context = &section.context;
        hydraulic(error::checked::<R, _>(
            client,
            "band model knot continuity",
            |a| {
                let model = &context.band_model;
                let rows = context.source.definitions.len();
                let c = a.client();
                let zeros = Tensor::zeros(&[rows], DType::F64, c.device())?;
                let ones = Tensor::ones(&[rows], DType::F64, c.device())?;
                let half = a.expansion_point(&c.mul_scalar(&ones, 0.5)?)?;
                assert_eq!(model.usable.try_to_vec::<f64>()?, vec![1.0; rows]);
                // Knot areas carry the boundary-area bits, so the model and the charts share every knot.
                let owners: Vec<i64> = (0..).take(rows).collect();
                let owners = Tensor::from_slice(&owners, &[rows], c.device())?;
                let geometry = HydraulicGeometry::from_batch(&section);
                let boundaries = geometry.boundary_areas(a)?.gather(a, &owners)?;
                let bands = context.maximum_bands;
                for (compiled, boundary) in [
                    (model.area.high(), boundaries.areas().high()),
                    (model.area.low(), boundaries.areas().low()),
                ] {
                    let boundary = boundary.narrow(1, 0, bands)?.contiguous()?;
                    let to_bits = |tensor: &Tensor<R>| -> HydraulicResult<Vec<u64>> {
                        Ok(tensor
                            .contiguous()?
                            .try_to_vec::<f64>()?
                            .into_iter()
                            .map(f64::to_bits)
                            .collect())
                    };
                    assert_eq!(to_bits(compiled)?, to_bits(&boundary)?);
                }
                let mut knots = 0;
                for band in 0..context.maximum_bands.saturating_sub(1) {
                    let width = model.width.narrow(1, band, 1)?.squeeze(Some(1));
                    let terms = BandTerms {
                        area: band_column(a, &model.area, band)?,
                        six_integral: band_column(a, &model.six_integral, band)?,
                        width: a.expansion_point(&width)?,
                        slope: band_column(a, &model.slope, band)?,
                    };
                    let height = knot::band(a, &context.raw_inputs, band, &zeros, &half)?.height;
                    let area = terms.area_at(a, &height)?;
                    let six = terms.six_integral_at(a, &height)?;
                    let next_area = band_column(a, &model.area, band + 1)?;
                    let next_six = band_column(a, &model.six_integral, band + 1)?;
                    let area_gap = a.expansion_sub(&area, &next_area)?;
                    let next_valid = column(&context.raw_inputs.valid, band + 1, rows)?;
                    let next_valid = next_valid.try_to_vec::<f64>()?;
                    let gap_high = area_gap.high().try_to_vec::<f64>()?;
                    let gap_low = area_gap.low().try_to_vec::<f64>()?;
                    let scale = next_area.high().try_to_vec::<f64>()?;
                    let six_parts = [
                        six.high().try_to_vec::<f64>()?,
                        six.low().try_to_vec::<f64>()?,
                    ];
                    let next_parts = [
                        next_six.high().try_to_vec::<f64>()?,
                        next_six.low().try_to_vec::<f64>()?,
                    ];
                    for row in 0..rows {
                        if next_valid[row] != 1.0 {
                            continue;
                        }
                        let gap = gap_high[row] + gap_low[row];
                        assert!(
                            gap.abs() <= 1e-28 * scale[row].abs().max(1.0),
                            "row={row}, band={band}, area gap={gap:e}"
                        );
                        for (part, next) in six_parts.iter().zip(&next_parts) {
                            assert_eq!(
                                part[row].to_bits(),
                                next[row].to_bits(),
                                "row={row}, band={band}"
                            );
                        }
                        knots += 1;
                    }
                }
                // The three-band section alone contributes two interior knots.
                assert!(knots >= 2);
                Ok(())
            },
        ))
    }

    fn three_band_knots<R>(client: &R::Client) -> Result
    where
        R: Runtime<DType = DType>,
        R::Client: Client<R>,
    {
        let section = support::batch::<R>(client)?;
        let model = &section.context.band_model;
        // Owner 2 has knots at depths 0, 1, 2 with widths 2, 4, 5 and cap width 8.
        let row = |tensor: &Tensor<R>| -> Result<Vec<f64>> {
            Ok(tensor.narrow(0, 2, 1)?.contiguous()?.try_to_vec::<f64>()?)
        };
        assert_eq!(row(model.area.high())?, [0.0, 3.0, 7.5]);
        assert_eq!(row(model.six_integral.high())?, [0.0, 8.0, 39.0]);
        assert_eq!(row(&model.width)?, [2.0, 4.0, 5.0]);
        assert_eq!(row(model.slope.high())?, [2.0, 1.0, 3.0]);
        for value in [&model.area, &model.six_integral, &model.slope] {
            assert_eq!(row(value.low())?, [0.0; 3]);
        }
        Ok(())
    }

    #[test]
    fn cpu_band_model_meets_every_knot() -> Result {
        use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime, ParallelismConfig};
        let client = CpuClient::new(CpuDevice::new())
            .with_parallelism(ParallelismConfig::new(Some(1), None));
        continuity::<CpuRuntime>(&client)?;
        three_band_knots::<CpuRuntime>(&client)
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_band_model_meets_every_knot() -> Result {
        use numr::runtime::cuda::{CudaClient, CudaDevice, CudaRuntime};
        let client = CudaClient::new(CudaDevice::new(0))?;
        continuity::<CudaRuntime>(&client)?;
        three_band_knots::<CudaRuntime>(&client)
    }
}
