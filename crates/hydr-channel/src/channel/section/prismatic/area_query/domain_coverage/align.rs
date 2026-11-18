use super::super::{admission, chart::SupportedAreaQuery, support_selection::SupportSelection};
use crate::channel::flow::surveyed::error::{Arithmetic, HydraulicError, HydraulicResult};
use numr::{
    dtype::DType,
    ops::{
        BinaryOps, CompareOps, IndexingOps, NextRepresentableOps, ScalarOps, SortingOps, TensorOps,
        TwoSumOps, TypeConversionOps, UtilityOps,
    },
    runtime::{Runtime, RuntimeClient},
    tensor::Tensor,
};

pub(super) fn align<R>(
    a: &mut Arithmetic<'_, R>,
    support: SupportSelection<R>,
    common: &Tensor<R>,
    common_owner: &Tensor<R>,
    name: &str,
) -> HydraulicResult<SupportedAreaQuery<R>>
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
    let count = common.numel();
    admission::metadata(a, common, &[count], a.dtype())?;
    admission::metadata(a, &support.proved, &[count], a.dtype())?;
    admission::metadata(a, &support.owner, &[support.owner.numel()], DType::I64)?;
    admission::metadata(a, common_owner, &[common_owner.numel()], DType::I64)?;
    let invalid_indexing = || HydraulicError::Numerical {
        context: format!(
            "Domain coverage {name} count={count}, supported={} exceeds exact owner indexing. Split the coverage batch",
            support.owner.numel(),
        ),
    };
    let limit = i64::try_from(count).map_err(|_| invalid_indexing())?;
    if support.owner.numel() as u128 > (1_u128 << f64::MANTISSA_DIGITS) {
        return Err(invalid_indexing());
    }
    let zero = Tensor::from_slice(&[0_i64], &[], c.device())?;
    let upper = Tensor::from_slice(&[limit], &[], c.device())?;
    for owner in [&support.owner, common_owner] {
        let invalid = c.maximum(&c.lt(owner, &zero)?, &c.ge(owner, &upper)?)?;
        admission::reject(
            a,
            &c.cast(&invalid, a.dtype())?,
            "domain coverage owner lies outside its invocation. Supply admitted owner indices",
        )?;
    }
    let query = support.query.ok_or_else(|| HydraulicError::Numerical {
        context: format!(
            "Domain coverage has common rows but no supported {name} query. Discard the inconsistent cover",
        ),
    })?;
    admission::metadata(a, &query.query.owners, &[support.owner.numel()], DType::I64)?;
    let inverse = Tensor::from_slice(&[-1_i64], &[], c.device())?
        .broadcast_to(&[count])?
        .contiguous()?;
    let positions = c.arange(0.0, support.owner.numel() as f64, 1.0, DType::I64)?;
    // select_supported proves unique support owners. Their positions follow the supported query, not mask compaction order.
    let inverse = c.index_put(&inverse, 0, &support.owner, &positions)?;
    let positions = c.index_select(&inverse, 0, common_owner)?;
    if c.nonzero(&c.lt(&positions, &zero)?)?.numel() != 0 {
        return Err(HydraulicError::Numerical {
            context: format!(
                "Domain coverage {name} common owner lacks a supported query. Discard the inconsistent cover",
            ),
        });
    }
    query.select(a, &positions)
}

#[cfg(test)]
mod tests {
    use super::super::super::{
        chart::{AreaAnchor, AreaDirection, AreaQuery},
        test_support::{Client, Result, hydraulic},
    };
    use super::super::fixture::rectangles;
    use super::*;
    use crate::channel::{flow::surveyed::error, section::HydraulicGeometry};

    fn permute<R>(
        client: &R::Client,
        support: SupportSelection<R>,
        desired: &[i64],
    ) -> Result<SupportSelection<R>>
    where
        R: Runtime<DType = DType>,
        R::Client: Client<R>,
    {
        let owners = support.owner.try_to_vec::<i64>()?;
        let positions = desired
            .iter()
            .map(|owner| {
                let position = owners
                    .iter()
                    .position(|value| value == owner)
                    .ok_or("Requested permutation owner lacks a supported query")?;
                Ok(i64::try_from(position)?)
            })
            .collect::<Result<Vec<_>>>()?;
        let positions = Tensor::from_slice(&positions, &[positions.len()], client.device())?;
        let query = support.query.ok_or("Supported permutation has no query")?;
        let query = hydraulic(error::checked::<R, _>(
            client,
            "coverage permutation preserves supported certificates",
            |a| query.select(a, &positions),
        ))?;
        Ok(SupportSelection {
            proved: support.proved,
            owner: Tensor::from_slice(desired, &[desired.len()], client.device())?,
            query: Some(query),
        })
    }

    fn check<R>(query: &SupportedAreaQuery<R>, owners: &[i64], cap: bool) -> Result
    where
        R: Runtime<DType = DType>,
    {
        let tags = [901_i64, 702, 1103, 504, 1305];
        let capacity = [2.0, 4.0, 8.0, 16.0, 32.0];
        let span = [1.0, 2.0, 3.0, 4.0, 5.0];
        assert_eq!(query.query.owners.try_to_vec::<i64>()?, owners);
        assert_eq!(
            query.query.original_rows.try_to_vec::<i64>()?,
            owners
                .iter()
                .map(|owner| tags[*owner as usize])
                .collect::<Vec<_>>()
        );
        let parameter_lower = query.query.parameter.lower().try_to_vec::<f64>()?;
        let parameter_upper = query.query.parameter.upper().try_to_vec::<f64>()?;
        let lower = query.area.lower().try_to_vec::<f64>()?;
        let upper = query.area.upper().try_to_vec::<f64>()?;
        for (position, owner) in owners.iter().enumerate() {
            let owner = usize::try_from(*owner)?;
            assert_eq!(parameter_lower[position], 0.0);
            assert_eq!(parameter_upper[position], span[owner]);
            let truth = if cap {
                [capacity[owner] - span[owner], capacity[owner]]
            } else {
                [0.0, span[owner]]
            };
            assert!(
                lower[position] <= truth[0] && truth[1] <= upper[position],
                "cap={cap}, owner={owner}, bounds={:?}, independent={truth:?}",
                [lower[position], upper[position]]
            );
            assert!((lower[position] - truth[0]).abs() < 1e-10);
            assert!((upper[position] - truth[1]).abs() < 1e-10);
        }
        Ok(())
    }

    fn matrix<R>(client: R::Client) -> Result
    where
        R: Runtime<DType = DType>,
        R::Client: Client<R>,
    {
        let section = rectangles::<R>(
            &client,
            &[(1., 2.), (2., 2.), (4., 2.), (8., 2.), (16., 2.)],
        )?;
        let geometry = HydraulicGeometry::from_batch(&section);
        let original = [901_i64, 702, 1103, 504, 1305];
        let owners = Tensor::from_slice(&[0_i64, 1, 2, 3, 4], &[5], client.device())?;
        for (lower_order, cap_order, common_order) in [
            (
                &[2_i64, 0, 3, 1][..],
                &[4_i64, 3, 2, 0][..],
                &[0_i64, 2, 3][..],
            ),
            (
                &[1_i64, 3, 2, 0][..],
                &[2_i64, 0, 4, 3][..],
                &[3_i64, 0, 2][..],
            ),
        ] {
            let (lower, cap) = hydraulic(error::checked::<R, _>(
                &client,
                "distinct supported coverage subsets",
                |a| {
                    let bands = geometry.gather(a, &owners)?;
                    let zero = Tensor::zeros(&[5], a.dtype(), client.device())?;
                    let scale =
                        a.expansion_point(&Tensor::ones(&[5], a.dtype(), client.device())?)?;
                    let lower_parameter = a.bounds(
                        &zero,
                        &Tensor::from_slice(&[1., 2., 3., 4., 100.], &[5], client.device())?,
                    )?;
                    let cap_parameter = a.bounds(
                        &zero,
                        &Tensor::from_slice(&[1., 100., 3., 4., 5.], &[5], client.device())?,
                    )?;
                    let lower = AreaQuery::band_affine(
                        a,
                        &bands,
                        AreaAnchor::Zero,
                        AreaDirection::Add,
                        &scale,
                        &lower_parameter,
                    )?;
                    let cap = AreaQuery::cap_deficit(a, &bands, &scale, &cap_parameter)?;
                    let tags = Tensor::from_slice(&original, &[5], client.device())?;
                    client.copy_into(&lower.original_rows, &tags)?;
                    client.copy_into(&cap.original_rows, &tags)?;
                    Ok((
                        lower.select_supported(a, &bands)?,
                        cap.select_supported(a, &bands)?,
                    ))
                },
            ))?;
            assert_eq!(lower.proved.try_to_vec::<f64>()?, [1., 1., 1., 1., 0.]);
            assert_eq!(cap.proved.try_to_vec::<f64>()?, [1., 0., 1., 1., 1.]);
            let lower = permute(&client, lower, lower_order)?;
            let cap = permute(&client, cap, cap_order)?;
            let cap_query = cap.query.as_ref().ok_or("Cap support has no query")?;
            let cap_tags = cap_query.query.original_rows.try_to_vec::<i64>()?;
            let copy_positions = Tensor::from_slice(&[0_i64, 1, 2, 3], &[4], client.device())?;
            let missing_common = Tensor::from_slice(&[0., 1., 0., 0., 0.], &[5], client.device())?;
            let missing_owner = Tensor::from_slice(&[1_i64], &[1], client.device())?;
            let missing =
                error::checked::<R, _>(&client, "coverage missing common-owner rejection", |a| {
                    let copied = SupportSelection {
                        proved: cap.proved.clone(),
                        owner: cap.owner.clone(),
                        query: Some(cap_query.select(a, &copy_positions)?),
                    };
                    align(a, copied, &missing_common, &missing_owner, "cap")
                });
            assert!(matches!(
                missing,
                Err(error::HydraulicError::Numerical { .. })
            ));
            for invalid in [-1_i64, 5, 6_148_914_691_236_517_206] {
                for invalid_support in [false, true] {
                    let requested = Tensor::from_slice(
                        &[if invalid_support { 0 } else { invalid }],
                        &[1],
                        client.device(),
                    )?;
                    let support_owner = if invalid_support {
                        Tensor::from_slice(&[invalid, 3, 2, 0], &[4], client.device())?
                    } else {
                        cap.owner.clone()
                    };
                    let rejected = error::checked::<R, _>(
                        &client,
                        "coverage owner range rejection before inverse gather",
                        |a| {
                            let copied = SupportSelection {
                                proved: cap.proved.clone(),
                                owner: support_owner,
                                query: Some(cap_query.select(a, &copy_positions)?),
                            };
                            align(a, copied, &missing_common, &requested, "cap")
                        },
                    );
                    assert!(
                        matches!(rejected, Err(error::HydraulicError::Geometry(_))),
                        "invalid={invalid}, invalid_support={invalid_support}"
                    );
                }
            }
            assert_eq!(cap_tags, cap_query.query.original_rows.try_to_vec::<i64>()?);
            let common = Tensor::from_slice(&[1., 0., 1., 1., 0.], &[5], client.device())?;
            let common_owner = Tensor::<R>::from_slice(common_order, &[3], client.device())?;
            let (lower, cap) = hydraulic(error::checked::<R, _>(
                &client,
                "coverage authoritative common-owner ordering",
                |a| {
                    Ok((
                        align(a, lower, &common, &common_owner, "lower")?,
                        align(a, cap, &common, &common_owner, "cap")?,
                    ))
                },
            ))?;
            check(&lower, common_order, false)?;
            check(&cap, common_order, true)?;
        }
        Ok(())
    }

    #[test]
    fn configured_cpu_domain_coverage_preserves_common_owner_order() -> Result {
        use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime, ParallelismConfig};
        let original = CpuClient::new(CpuDevice::new());
        let before = original.parallelism();
        let config = ParallelismConfig::new(Some(2), Some(257));
        let client = original.with_parallelism(config);
        let retained = client.clone();
        matrix::<CpuRuntime>(client)?;
        assert_eq!(retained.parallelism(), config);
        assert_eq!(original.parallelism(), before);
        Ok(())
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn actual_cuda_domain_coverage_preserves_common_owner_order() -> Result {
        use numr::runtime::cuda::{CudaClient, CudaDevice, CudaRuntime};
        matrix::<CudaRuntime>(CudaClient::new(CudaDevice::new(0))?)
    }
}
