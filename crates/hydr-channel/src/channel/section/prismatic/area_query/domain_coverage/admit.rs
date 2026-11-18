use super::super::super::{boundary_area::BoundaryAreas, hydraulic_geometry::GatheredBands};
use super::super::admission;
use crate::channel::flow::surveyed::error::{Arithmetic, HydraulicResult};
use numr::{
    dtype::DType,
    ops::{
        BinaryOps, CompareOps, NextRepresentableOps, ScalarOps, TensorOps, TwoSumOps,
        TypeConversionOps,
    },
    runtime::{Runtime, RuntimeClient},
    tensor::Tensor,
};

pub(super) fn admit<R>(
    a: &Arithmetic<'_, R>,
    bands: &GatheredBands<R>,
    boundaries: &BoundaryAreas<R>,
    original_rows: &Tensor<R>,
) -> HydraulicResult<Tensor<R>>
where
    R: Runtime<DType = DType>,
    R::Client: TensorOps<R>
        + CompareOps<R>
        + NextRepresentableOps<R>
        + ScalarOps<R>
        + TwoSumOps<R>
        + RuntimeClient<R>,
{
    admission::matching_model(boundaries.same_context(&bands.model))?;
    let count = bands.section_owner.numel();
    admission::metadata(a, boundaries.owners(), &[count], DType::I64)?;
    admission::matching_owners(a, bands, boundaries.owners())?;
    admission::metadata(a, original_rows, &[count], DType::I64)?;
    let c = a.client();
    let captured = Tensor::empty(original_rows.shape(), DType::I64, c.device())?;
    c.copy_into(&captured, original_rows)?;
    let zero = Tensor::zeros(&[], DType::I64, c.device())?;
    admission::reject(
        a,
        &c.cast(&c.lt(&captured, &zero)?, a.dtype())?,
        "domain coverage diagnostic rows are negative. Supply nonnegative invocation identities",
    )?;
    Ok(captured)
}

#[cfg(test)]
pub(super) mod tests {
    use super::super::super::test_support as support;
    use super::super::cover_domain;
    use crate::channel::{
        flow::surveyed::error::{self, HydraulicError},
        section::{CrossSectionError, HydraulicGeometry, PrismaticSectionBatch},
    };
    use numr::runtime::RuntimeClient;
    use support::{Client, Result, hydraulic};

    use super::*;

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

    pub(in super::super) fn admission<R>(
        client: &R::Client,
        section: &PrismaticSectionBatch<R>,
    ) -> Result
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
}
