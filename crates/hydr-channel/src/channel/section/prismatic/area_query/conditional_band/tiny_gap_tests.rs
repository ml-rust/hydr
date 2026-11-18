use super::super::{AreaDirection, AreaQuery, test_support as support};
use super::ConditionalBandSegments;
use crate::channel::{
    flow::{
        surveyed::{equation_ops::scalar, error},
        surveyed_points::potential_callbacks,
    },
    section::{HydraulicGeometry, PrismaticRecoveryConfig},
};
use numr::{
    dtype::DType,
    runtime::{Runtime, RuntimeClient},
    tensor::Tensor,
};
use support::{Client, Result, hydraulic};

pub(super) fn run<R>(client: &R::Client) -> Result
where
    R: Runtime<DType = DType>,
    R::Client: Client<R>,
{
    let section = support::batch::<R>(client)?;
    let geometry = HydraulicGeometry::from_batch(&section);
    let owners = Tensor::<R>::from_slice(&[1_i64; 2], &[2], client.device())?;
    let areas = support::expansion::<R>(client, &[[4.0_f64.to_bits(), 0, 0]; 2], false)?;
    let source =
        section.recover_area_points_tensor(&owners, &areas, &PrismaticRecoveryConfig::default())?;
    let distance = f64::from_bits(0x3c30000000000000);
    hydraulic(error::checked::<R, _>(
        client,
        "tiny unclipped conditional gaps",
        |a| {
            let bands = geometry.gather(a, &owners)?;
            let boundaries = geometry.boundary_areas(a)?.gather(a, &owners)?;
            let scale =
                a.expansion_point(&Tensor::from_slice(&[distance; 2], &[2], client.device())?)?;
            let parameter = a.point(&Tensor::from_slice(&[1.0, -1.0], &[2], client.device())?)?;
            let endpoint = AreaQuery::source_affine(
                a,
                &bands,
                &source,
                AreaDirection::Add,
                &scale,
                &parameter,
            )?
            .prove_support(a, &bands)?;
            let segments =
                ConditionalBandSegments::prepare(a, &bands, &source, &endpoint, &boundaries)?;
            let lanes = Tensor::from_slice(&[0_i64, 3], &[2], client.device())?;
            let selected = segments.select(a, &lanes)?;
            assert_eq!(
                selected.retained_gap().high().try_to_vec::<f64>()?,
                [distance, -distance]
            );
            assert_eq!(selected.retained_gap().low().try_to_vec::<f64>()?, [0.0; 2]);
            assert_eq!(
                selected.retained_gap().abs_error().try_to_vec::<f64>()?,
                [0.0; 2]
            );
            assert_eq!(
                selected.physical_gap().lower().try_to_vec::<f64>()?,
                [distance, -distance]
            );
            assert_eq!(
                selected.physical_gap().upper().try_to_vec::<f64>()?,
                [distance, -distance]
            );
            let middle = a.point(&Tensor::from_slice(&[0.5; 2], &[2], client.device())?)?;
            let panel = segments.panel(a, &lanes, &middle)?;
            assert_eq!(
                panel.gap.lower().try_to_vec::<f64>()?,
                [distance, -distance]
            );
            assert_eq!(
                panel.gap.upper().try_to_vec::<f64>()?,
                [distance, -distance]
            );
            let gravity = scalar(a, 1.0)?;
            let integrand =
                potential_callbacks::integrand(a, &segments, &lanes, &middle, &gravity)?;
            let lower = integrand.lower().try_to_vec::<f64>()?;
            let upper = integrand.upper().try_to_vec::<f64>()?;
            // Rectangle G=D/sqrt(2*A), with A=4+D/2 and D=+/-2^-60.
            let reference = [
                f64::from_bits(0x3c16a09e667f3bcc),
                f64::from_bits(0x3c16a09e667f3bcd),
            ];
            for (lane, expected) in [reference, [-reference[1], -reference[0]]]
                .into_iter()
                .enumerate()
            {
                assert!(lower[lane] <= expected[0] && expected[1] <= upper[lane]);
                let excess = (lower[lane] - expected[1])
                    .abs()
                    .max((upper[lane] - expected[0]).abs());
                assert!(excess <= 1e-10 * reference[0]);
            }
            let reordered = Tensor::from_slice(&[3_i64, 0, 3, 1, 4], &[5], client.device())?;
            let parameter = a.point(&Tensor::from_slice(&[0.5; 5], &[5], client.device())?)?;
            let value =
                potential_callbacks::integrand(a, &segments, &reordered, &parameter, &gravity)?;
            let bound =
                potential_callbacks::fourth_bound(a, &segments, &reordered, &parameter, &gravity)?;
            let value_lower = value.lower().try_to_vec::<f64>()?;
            let value_upper = value.upper().try_to_vec::<f64>()?;
            let fourth = bound.try_to_vec::<f64>()?;
            for (row, original) in [1_usize, 0, 1].into_iter().enumerate() {
                assert_eq!(
                    [value_lower[row], value_upper[row]],
                    [lower[original], upper[original]]
                );
                assert!(fourth[row].is_finite() && fourth[row] > 0.0);
            }
            for row in [3_usize, 4] {
                assert_eq!([value_lower[row], value_upper[row]], [0.0; 2]);
                assert_eq!(fourth[row], 0.0);
            }
            Ok(())
        },
    ))?;
    exact_rectangle_fourth::<R>(client, &section, &geometry)
}

fn exact_rectangle_fourth<R>(
    client: &R::Client,
    section: &crate::channel::section::PrismaticSectionBatch<R>,
    geometry: &HydraulicGeometry<R>,
) -> Result
where
    R: Runtime<DType = DType>,
    R::Client: Client<R>,
{
    let owners = Tensor::<R>::from_slice(&[1_i64; 2], &[2], client.device())?;
    let areas = support::expansion::<R>(client, &[[2.0_f64.to_bits(), 0, 0]; 2], false)?;
    let source =
        section.recover_area_points_tensor(&owners, &areas, &PrismaticRecoveryConfig::default())?;
    let distance = f64::from_bits(0x3c30000000000000);
    hydraulic(error::checked::<R, _>(
        client,
        "rectangular ordinary fourth derivative",
        |a| {
            let bands = geometry.gather(a, &owners)?;
            let boundaries = geometry.boundary_areas(a)?.gather(a, &owners)?;
            let scale =
                a.expansion_point(&Tensor::from_slice(&[distance; 2], &[2], client.device())?)?;
            let direction = a.point(&Tensor::from_slice(&[1.0, -1.0], &[2], client.device())?)?;
            let query = AreaQuery::source_affine(
                a,
                &bands,
                &source,
                AreaDirection::Add,
                &scale,
                &direction,
            )?
            .prove_support(a, &bands)?;
            let segments =
                ConditionalBandSegments::prepare(a, &bands, &source, &query, &boundaries)?;
            let lanes = Tensor::from_slice(&[0_i64, 3], &[2], client.device())?;
            let parameter = a.point(&Tensor::zeros(&[2], DType::F64, client.device())?)?;
            let gravity = scalar(a, 1.0)?;
            let bound =
                potential_callbacks::fourth_bound(a, &segments, &lanes, &parameter, &gravity)?;
            // For A=2, T=2, g=1, the exact magnitude at s=0 is 105*|D|^5/512.
            let independent = 105.0 * distance.powi(5) / 512.0;
            assert!(independent > 0.0);
            for value in bound.try_to_vec::<f64>()? {
                assert!(
                    value.is_finite() && value >= independent,
                    "M4={value}, exact={independent}"
                );
            }
            let first = Tensor::from_slice(&[0_i64], &[1], client.device())?;
            let mut widened = segments.select(a, &first)?;
            let at_start = a.point(&Tensor::zeros(&[1], DType::F64, client.device())?)?;
            let original = widened.panel(a, &first, &at_start)?;
            let radius = 2.0 * distance;
            let center = Tensor::zeros(&[1], DType::F64, client.device())?;
            let uncertainty = Tensor::from_slice(&[radius], &[1], client.device())?;
            // Only the gap certificates widen; the original A=2 to A=2+D endpoint geometry remains intact.
            widened.gap = a.expansion_bounds(&center, &center, &uncertainty)?;
            widened.physical_gap = a.bounds(
                &Tensor::from_slice(&[-radius], &[1], client.device())?,
                &uncertainty,
            )?;
            assert_eq!(widened.retained_gap().high().item::<f64>()?, 0.0);
            assert_eq!(widened.retained_gap().low().item::<f64>()?, 0.0);
            assert_eq!(widened.retained_gap().abs_error().item::<f64>()?, radius);
            assert!(-radius <= distance && distance <= radius);
            let retained_geometry = widened.panel(a, &first, &at_start)?;
            for (before, after) in [
                (&original.area, &retained_geometry.area),
                (
                    &original.hydraulic_depth,
                    &retained_geometry.hydraulic_depth,
                ),
            ] {
                assert_eq!(
                    before.lower().try_to_vec::<f64>()?,
                    after.lower().try_to_vec::<f64>()?
                );
                assert_eq!(
                    before.upper().try_to_vec::<f64>()?,
                    after.upper().try_to_vec::<f64>()?
                );
            }
            let value = potential_callbacks::integrand(a, &widened, &first, &at_start, &gravity)?;
            let lower = value.lower().item::<f64>()?;
            let upper = value.upper().item::<f64>()?;
            assert!(lower < 0.0 && upper > 0.0);
            assert!(lower <= distance / 2.0 && distance / 2.0 <= upper);
            let bound =
                potential_callbacks::fourth_bound(a, &widened, &first, &at_start, &gravity)?;
            let bound = bound.item::<f64>()?;
            assert!(
                bound.is_finite() && bound >= independent,
                "widened M4={bound}, exact={independent}"
            );
            Ok(())
        },
    ))
}
