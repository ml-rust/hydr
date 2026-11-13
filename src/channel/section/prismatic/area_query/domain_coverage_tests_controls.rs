use super::super::{cover_domain, domain_coverage::proof_candidates};
use super::{rectangles, support};
use crate::channel::{
    flow::surveyed::error::{self, HydraulicError},
    section::{CrossSectionError, HydraulicGeometry, PrismaticSectionBatch},
};
use numr::{
    dtype::DType,
    runtime::{Runtime, RuntimeClient},
    tensor::Tensor,
};
use support::{Client, Result, hydraulic};

pub(super) fn run<R>(client: &R::Client, section: &PrismaticSectionBatch<R>) -> Result
where
    R: Runtime<DType = DType>,
    R::Client: Client<R>,
{
    rejected_pressure::<R>(client)?;
    empty::<R>(client, section)?;
    admission::<R>(client, section)?;
    wide_arithmetic::<R>(client)
}

fn rejected_pressure<R>(client: &R::Client) -> Result
where
    R: Runtime<DType = DType>,
    R::Client: Client<R>,
{
    let error = rectangles::<R>(client, &[(2.0_f64.powi(-500), 2.0_f64.powi(-573))])
        .err()
        .ok_or("Pressure-underflow rectangle unexpectedly passes ordinary geometry admission")?;
    assert!(
        matches!(error.downcast_ref::<CrossSectionError>(), Some(CrossSectionError::Numerical { context, .. })
        if context.contains("hydrostatic pressure integral=0") && context.contains("rescale profile coordinates"))
    );
    Ok(())
}

fn empty<R>(client: &R::Client, section: &PrismaticSectionBatch<R>) -> Result
where
    R: Runtime<DType = DType>,
    R::Client: Client<R>,
{
    let geometry = HydraulicGeometry::from_batch(section);
    let rows = Tensor::<R>::empty(&[0], DType::I64, client.device())?;
    let coverage = hydraulic(error::checked::<R, _>(
        client,
        "empty geometry domain coverage",
        |a| {
            let bands = geometry.gather(a, &rows)?;
            let boundaries = geometry.boundary_areas(a)?.gather(a, &rows)?;
            cover_domain(a, &bands, &boundaries, &rows)
        },
    ))?;
    assert_eq!(coverage.proved.shape(), &[0]);
    assert_eq!(coverage.proved.dtype(), DType::F64);
    assert_eq!(coverage.owner.shape(), &[0]);
    assert_eq!(coverage.owner.dtype(), DType::I64);
    assert!(coverage.pair.is_none());
    Ok(())
}

fn invalid<T>(result: crate::channel::flow::surveyed::error::HydraulicResult<T>) -> Result {
    let error = result
        .err()
        .ok_or("Domain coverage unexpectedly accepts malformed geometry binding")?;
    assert!(
        matches!(
            error,
            HydraulicError::Geometry(CrossSectionError::InvalidInput { .. })
        ),
        "{error:?}"
    );
    Ok(())
}

fn admission<R>(client: &R::Client, section: &PrismaticSectionBatch<R>) -> Result
where
    R: Runtime<DType = DType>,
    R::Client: Client<R>,
{
    let geometry = HydraulicGeometry::from_batch(section);
    let foreign_section = support::batch::<R>(client)?;
    let foreign_geometry = HydraulicGeometry::from_batch(&foreign_section);
    let owners = Tensor::from_slice(&[1_i64], &[1], client.device())?;
    let diagnostics = Tensor::from_slice(&[8000000001_i64], &[1], client.device())?;
    let (bands, boundaries, foreign, wrong) = hydraulic(error::checked::<R, _>(
        client,
        "domain binding controls",
        |a| {
            let bands = geometry.gather(a, &owners)?;
            let boundaries = geometry.boundary_areas(a)?.gather(a, &owners)?;
            let foreign = foreign_geometry.boundary_areas(a)?.gather(a, &owners)?;
            let different = Tensor::from_slice(&[2_i64], &[1], client.device())?;
            let wrong = geometry.boundary_areas(a)?.gather(a, &different)?;
            Ok((bands, boundaries, foreign, wrong))
        },
    ))?;
    invalid(error::checked::<R, _>(
        client,
        "foreign coverage boundary model",
        |a| cover_domain(a, &bands, &foreign, &diagnostics),
    ))?;
    invalid(error::checked::<R, _>(
        client,
        "coverage boundary owner mismatch",
        |a| cover_domain(a, &bands, &wrong, &diagnostics),
    ))?;
    for malformed in [
        Tensor::from_slice(&[-1_i64], &[1], client.device())?,
        Tensor::from_slice(&[7_i64], &[], client.device())?,
        Tensor::from_slice(&[7_i64], &[1, 1], client.device())?,
        Tensor::from_slice(&[7_i64, 8], &[2], client.device())?,
        Tensor::from_slice(&[7.0_f64], &[1], client.device())?,
    ] {
        invalid(error::checked::<R, _>(
            client,
            "coverage diagnostic metadata",
            |a| cover_domain(a, &bands, &boundaries, &malformed),
        ))?;
    }
    assert_eq!(diagnostics.try_to_vec::<i64>()?, [8000000001]);
    Ok(())
}

fn wide_arithmetic<R>(client: &R::Client) -> Result
where
    R: Runtime<DType = DType>,
    R::Client: Client<R>,
{
    // These are honest arithmetic certificates, not substituted compiled-model boundaries.
    let capacity = support::expansion::<R>(
        client,
        &[
            [4.0_f64.to_bits(), 0, 0],
            [4.0_f64.to_bits(), 0, 2.0_f64.to_bits()],
        ],
        false,
    )?;
    hydraulic(error::checked::<R, _>(
        client,
        "wide capacity overlap proof",
        |a| {
            let candidates = proof_candidates(a, &capacity)?;
            assert_eq!(candidates.owner.try_to_vec::<i64>()?, [0, 1]);
            let span = candidates.span.high().try_to_vec::<f64>()?;
            assert_eq!(span[0], 3.0);
            assert!(span[1].is_finite() && span[1] > 0.0 && span[1] <= 1.5);
            let certified_lower = a
                .expansion_interval(&capacity)?
                .lower()
                .try_to_vec::<f64>()?;
            assert_eq!(
                span.into_iter().map(f64::to_bits).collect::<Vec<_>>(),
                certified_lower
                    .into_iter()
                    .map(|lower| (0.75 * lower).to_bits())
                    .collect::<Vec<_>>()
            );
            assert_eq!(candidates.span.low().try_to_vec::<f64>()?, [0.0, 0.0]);
            assert_eq!(candidates.span.abs_error().try_to_vec::<f64>()?, [0.0, 0.0]);
            assert_eq!(candidates.proved.try_to_vec::<f64>()?, [1.0, 0.0]);
            let low = a.expansion_interval(&candidates.cap_minus_lower)?;
            let cap = a.expansion_interval(&candidates.cap_minus_deficit)?;
            let gap = a.expansion_interval(&candidates.overlap_gap)?;
            assert_eq!(low.lower().try_to_vec::<f64>()?[0], 1.0);
            assert_eq!(low.upper().try_to_vec::<f64>()?[0], 1.0);
            assert_eq!(cap.lower().try_to_vec::<f64>()?[0], 1.0);
            assert_eq!(cap.upper().try_to_vec::<f64>()?[0], 1.0);
            assert_eq!(gap.lower().try_to_vec::<f64>()?[0], -2.0);
            assert_eq!(gap.upper().try_to_vec::<f64>()?[0], -2.0);
            assert!(
                low.lower().try_to_vec::<f64>()?[1] > 0.0
                    && cap.lower().try_to_vec::<f64>()?[1] > 0.0
            );
            assert!(
                gap.lower().try_to_vec::<f64>()?[1] < 0.0
                    && gap.upper().try_to_vec::<f64>()?[1] > 0.0
            );
            Ok(())
        },
    ))
}
