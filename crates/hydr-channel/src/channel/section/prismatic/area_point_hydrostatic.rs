use super::{
    area_point_input::Inputs,
    area_point_query::PrismaticAreaPointBandCoordinates,
    band_terms::{BandTerms, constant},
    context::Context,
    query_input::{Arithmetic, QueryResult},
};
use crate::channel::flow::surveyed::equation_ops::choose_expansion;
use numr::dtype::DType;
use numr::ops::{
    BinaryOps, CompareOps, ConditionalOps, IndexingOps, NextRepresentableOps, ReduceOps, ScalarOps,
    TensorOps, TwoSumOps, TypeConversionOps, UnaryOps, UtilityOps,
};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;
use solvr::common::arithmetic::{CertifiedExpansion, Interval};

/// Newton steps from the F64 seed. Each step squares the relative depth error.
const NEWTON_STEPS: usize = 2;

/// Selection value of the hydrostatic integral at exact two-part area points, in cubic metres.
///
/// The band is the lowest candidate column. The seed is the F64 midpoint of its depth above the band start.
/// Newton steps solve the model area for that depth in two-part arithmetic.
/// The model integral at the solved depth is the value.
/// The radius covers model arithmetic at the solved depth only. It never certifies the surveyed integral.
/// Dry rows hold exact zero.
/// Rows without a provably positive model width, with a step above one band thickness,
/// or whose section exceeds the model scale lift the certified `integral` interval instead.
/// Equal input bits give equal output bits.
pub(super) fn point<R>(
    a: &mut Arithmetic<'_, R>,
    context: &Context<R>,
    inputs: &Inputs<R>,
    coordinates: &PrismaticAreaPointBandCoordinates<R>,
    integral: &Interval<R>,
) -> QueryResult<CertifiedExpansion<R>>
where
    R: Runtime<DType = DType>,
    R::Client: TensorOps<R>
        + CompareOps<R>
        + NextRepresentableOps<R>
        + ScalarOps<R>
        + TwoSumOps<R>
        + RuntimeClient<R>,
{
    let c = a.client();
    let device = c.device();
    let count = inputs.owners.numel();
    let bands = context.maximum_bands;
    let model = &context.band_model;
    let zeros = Tensor::zeros(&[count], DType::F64, device)?;
    let ones = Tensor::ones(&[count], DType::F64, device)?;

    // Rank candidate columns by index. Columns without a candidate rank after every band.
    let last_rank = bands.saturating_sub(1) as f64;
    let last = Tensor::full_scalar(&[], DType::F64, last_rank, device)?;
    let absent = Tensor::full_scalar(&[], DType::F64, last_rank + 1.0, device)?;
    let ranks = c
        .arange(0.0, bands as f64, 1.0, DType::F64)?
        .reshape(&[1, bands])?;
    let lowest = c.min(
        &c.where_cond(&coordinates.candidate, &ranks, &absent)?,
        &[1],
        false,
    )?;
    let found = c.lt(&lowest, &absent)?;
    let band = c
        .cast(&c.minimum(&lowest, &last)?, DType::I64)?
        .reshape(&[count, 1])?;
    let local = |table: &Tensor<R>| -> QueryResult<Tensor<R>> {
        Ok(c.gather(table, 1, &band)?.reshape(&[count])?)
    };
    let owned = |table: &Tensor<R>| -> QueryResult<Tensor<R>> {
        local(&c.index_select(table, 0, &inputs.owners)?)
    };

    let usable = c.index_select(&model.usable, 0, &inputs.owners)?;
    let active = c.mul(&c.mul(&inputs.wet, &found)?, &usable)?;
    // Inactive rows use zero area, unit width, and zero depth, so every quotient stays defined.
    let pick = |a: &mut Arithmetic<'_, R>,
                value: &CertifiedExpansion<R>|
     -> QueryResult<CertifiedExpansion<R>> {
        Ok(a.expansion_bounds(
            &c.where_cond(&active, &owned(value.high())?, &zeros)?,
            &c.where_cond(&active, &owned(value.low())?, &zeros)?,
            &c.where_cond(&active, &owned(value.abs_error())?, &zeros)?,
        )?)
    };
    let terms = BandTerms {
        area: pick(a, &model.area)?,
        six_integral: pick(a, &model.six_integral)?,
        width: a.expansion_point(&c.where_cond(&active, &owned(&model.width)?, &ones)?)?,
        slope: pick(a, &model.slope)?,
    };
    let target = a.expansion_bounds(
        &c.where_cond(&active, inputs.areas.high(), &zeros)?,
        &c.where_cond(&active, inputs.areas.low(), &zeros)?,
        &c.where_cond(&active, inputs.areas.abs_error(), &zeros)?,
    )?;
    let thickness = c.where_cond(&active, &owned(context.thickness.upper())?, &ones)?;
    let lower = local(coordinates.depth_above_start.lower())?;
    let upper = local(coordinates.depth_above_start.upper())?;
    // Halve each endpoint before adding, so the midpoint cannot overflow.
    let seed = c.add(&c.mul_scalar(&lower, 0.5)?, &c.mul_scalar(&upper, 0.5)?)?;
    let seed = c.where_cond(&active, &seed, &zeros)?;

    let zero = a.expansion_point(&zeros)?;
    let one = a.expansion_point(&ones)?;
    let mut depth = a.expansion_point(&seed)?;
    let mut resolved = active;
    for _ in 0..NEWTON_STEPS {
        let area = terms.area_at(a, &depth)?;
        let residual = a.expansion_sub(&area, &target)?;
        let width = terms.width_at(a, &depth)?;
        let width_bounds = a.expansion_interval(&width)?;
        let residual_bounds = a.expansion_interval(&residual)?;
        let magnitude = c.maximum(
            &c.abs(residual_bounds.lower())?,
            &c.abs(residual_bounds.upper())?,
        )?;
        // A step stays below one band thickness, so later model products stay in range.
        let bounded = c.mul(
            &c.gt(width_bounds.lower(), &zeros)?,
            &c.le(&magnitude, &c.mul(width_bounds.lower(), &thickness)?)?,
        )?;
        resolved = c.mul(&resolved, &bounded)?;
        let residual = choose_expansion(a, &resolved, &residual, &zero)?;
        let width = choose_expansion(a, &resolved, &width, &one)?;
        let step = a.expansion_div_positive(&residual, &width)?;
        depth = a.expansion_sub(&depth, &step)?;
    }
    let six_value = terms.six_integral_at(a, &depth)?;
    let six = constant(a, &six_value, 6.0)?;
    let value = a.expansion_div_positive(&six_value, &six)?;
    // Dry rows carry a zero certified interval, so the lift gives exact zero.
    let fallback = a.expansion_from_interval(integral)?;
    Ok(choose_expansion(a, &resolved, &value, &fallback)?)
}

#[cfg(test)]
mod tests {
    use super::super::area_point_query::PrismaticAreaPointRecovery;
    use super::super::area_query::test_support::{self as support, Client, Result};
    use super::super::config::PrismaticRecoveryConfig;
    use numr::{
        dtype::DType,
        runtime::{Runtime, RuntimeClient},
        tensor::Tensor,
    };

    /// Section owners in the shared test batch.
    const RECTANGLE: i64 = 1;
    const THREE_BAND: i64 = 2;
    const TRIANGLE: i64 = 4;

    fn recover<R>(
        client: &R::Client,
        owners: &[i64],
        areas: &[(f64, f64)],
    ) -> Result<PrismaticAreaPointRecovery<R>>
    where
        R: Runtime<DType = DType>,
        R::Client: Client<R>,
    {
        let section = support::batch::<R>(client)?;
        let owners = Tensor::from_slice(owners, &[owners.len()], client.device())?;
        let points: Vec<[u64; 3]> = areas
            .iter()
            .map(|(high, low)| [high.to_bits(), low.to_bits(), 0])
            .collect();
        let areas = support::expansion::<R>(client, &points, false)?;
        Ok(section.recover_area_points_tensor(
            &owners,
            &areas,
            &PrismaticRecoveryConfig::default(),
        )?)
    }

    /// Host rows `[high, low, radius]` of the selection value.
    fn rows<R>(recovery: &PrismaticAreaPointRecovery<R>) -> Result<Vec<[f64; 3]>>
    where
        R: Runtime<DType = DType>,
    {
        let point = recovery.hydrostatic_point_m3();
        let high = point.high().try_to_vec::<f64>()?;
        let low = point.low().try_to_vec::<f64>()?;
        let radius = point.abs_error().try_to_vec::<f64>()?;
        Ok(high
            .into_iter()
            .zip(low)
            .zip(radius)
            .map(|((high, low), radius)| [high, low, radius])
            .collect())
    }

    fn bits(row: &[f64; 3]) -> [u64; 3] {
        row.map(f64::to_bits)
    }

    /// Two-part reference `(high, low)` minus a computed row, rounded once.
    fn gap(row: &[f64; 3], reference: (f64, f64)) -> f64 {
        (row[0] - reference.0) + (row[1] - reference.1)
    }

    /// A model-arithmetic radius, far below the certified interval width.
    fn model_radius(row: &[f64; 3]) -> bool {
        row[2] <= 1e-24
    }

    fn wet_fan<R>(client: &R::Client) -> Result
    where
        R: Runtime<DType = DType>,
        R::Client: Client<R>,
    {
        let offsets = [2.5e-17, 8e-17];
        let recovery = recover::<R>(
            client,
            &[THREE_BAND; 4],
            &[
                (14.0, 0.0),
                (14.0, -offsets[0]),
                (14.0, -offsets[1]),
                (14.0, -offsets[0]),
            ],
        )?;
        let values = rows(&recovery)?;
        assert_eq!(values.len(), 4);
        assert_eq!(values[0][0].to_bits(), 17.0_f64.to_bits());
        assert_eq!(values[0][1], 0.0);
        for (row, offset) in values[1..3].iter().zip(offsets) {
            // The high part lies within one ulp of 17, so high - 17 is exact.
            let difference = (row[0] - 17.0) + row[1];
            // I1(A) near the cap: dI1/dA = A/T = 1.75 and d²I1/dA² = 11/256.
            let expected = (-1.75_f64).mul_add(offset, 11.0 / 512.0 * offset * offset);
            assert!(
                (difference - expected).abs() <= 1e-30,
                "offset={offset:e}, difference={difference:e}, expected={expected:e}"
            );
            assert!(model_radius(row), "radius={:e}", row[2]);
        }
        assert_eq!(bits(&values[1]), bits(&values[3]));
        Ok(())
    }

    fn repeated<R>(client: &R::Client) -> Result
    where
        R: Runtime<DType = DType>,
        R::Client: Client<R>,
    {
        let owners = [THREE_BAND, RECTANGLE, TRIANGLE, THREE_BAND];
        let areas = [(5.25, 0.0), (2.5, 0.0), (3.0, 0.0), (14.0, -3e-17)];
        let first = rows(&recover::<R>(client, &owners, &areas)?)?;
        let second = rows(&recover::<R>(client, &owners, &areas)?)?;
        assert_eq!(
            first.iter().map(bits).collect::<Vec<_>>(),
            second.iter().map(bits).collect::<Vec<_>>()
        );
        Ok(())
    }

    fn closed_forms<R>(client: &R::Client) -> Result
    where
        R: Runtime<DType = DType>,
        R::Client: Client<R>,
    {
        // Rectangle of width 2: I1 = A²/4. Triangle with width 3h: A = 1.5h² and I1 = h³/2.
        let recovery = recover::<R>(
            client,
            &[
                RECTANGLE, RECTANGLE, RECTANGLE, TRIANGLE, TRIANGLE, TRIANGLE,
            ],
            &[
                (0.0, 0.0),
                (2.0, 0.0),
                (3.0, 0.0),
                (1.5, 0.0),
                (6.0, 0.0),
                (3.0, 0.0),
            ],
        )?;
        let values = rows(&recovery)?;
        assert_eq!(values.len(), 6);
        assert_eq!(values[0], [0.0; 3]);
        // At A = 3 the triangle depth is √2, so I1 = √2. FMA rounds the square residual once.
        let root = std::f64::consts::SQRT_2;
        let root_low = -root.mul_add(root, -2.0) / (2.0 * root);
        let references = [
            (1.0, 0.0),
            (2.25, 0.0),
            (0.5, 0.0),
            (4.0, 0.0),
            (root, root_low),
        ];
        for (row, reference) in values[1..].iter().zip(references) {
            let error = gap(row, reference);
            assert!(
                error.abs() <= 1e-30,
                "reference={reference:?}, gap={error:e}"
            );
            assert!(model_radius(row), "radius={:e}", row[2]);
        }
        Ok(())
    }

    fn films<R>(client: &R::Client) -> Result
    where
        R: Runtime<DType = DType>,
        R::Client: Client<R>,
    {
        // Flat-bottom rectangle of width 2: A = 2^-100 gives I1 = A²/4 = 2^-202.
        // Pointed triangle with width 3h: A = 1.5·2^-100 gives h = 2^-50 and I1 = h³/2 = 2^-151.
        let tiny = 2.0_f64.powi(-100);
        let recovery = recover::<R>(
            client,
            &[RECTANGLE, TRIANGLE],
            &[(tiny, 0.0), (1.5 * tiny, 0.0)],
        )?;
        let values = rows(&recovery)?;
        assert_eq!(values.len(), 2);
        for (row, reference) in values.iter().zip([2.0_f64.powi(-202), 2.0_f64.powi(-151)]) {
            let relative = gap(row, (reference, 0.0)).abs() / reference;
            assert!(
                relative <= 1e-28,
                "reference={reference:e}, row={row:?}, relative={relative:e}"
            );
            assert!(row[2] <= reference * 1e-24, "radius={:e}", row[2]);
        }
        Ok(())
    }

    fn run<R>(client: &R::Client) -> Result
    where
        R: Runtime<DType = DType>,
        R::Client: Client<R>,
    {
        wet_fan::<R>(client)?;
        repeated::<R>(client)?;
        closed_forms::<R>(client)?;
        films::<R>(client)
    }

    #[test]
    fn cpu_hydrostatic_point_matches_model_closed_forms() -> Result {
        use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime, ParallelismConfig};
        let client = CpuClient::new(CpuDevice::new())
            .with_parallelism(ParallelismConfig::new(Some(1), None));
        run::<CpuRuntime>(&client)
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_hydrostatic_point_matches_model_closed_forms() -> Result {
        use numr::runtime::cuda::{CudaClient, CudaDevice, CudaRuntime};
        run::<CudaRuntime>(&CudaClient::new(CudaDevice::new(0))?)
    }
}
