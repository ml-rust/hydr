use super::potential_increment::endpoint_increment;
use super::potential_range::{PotentialContext, PotentialEvaluation, select_expansion};
use super::shock_points;
#[cfg(test)]
use super::wave_curve_types::CurveRange;
use super::wave_curve_types::{
    CurveClassification, CurveValue, EndpointCurve, empty_offset, scatter_offset, scatter_value,
    unavailable,
};
use crate::channel::{
    flow::surveyed::{
        equation_ops::{gravity_depth_root, scalar, select_interval},
        error::{Arithmetic, HydraulicResult, checked},
    },
    section::{PhysicalEndpoint, pressure_endpoint, source_difference},
};
use numr::{
    dtype::DType,
    ops::{
        BinaryOps, CompareOps, IndexingOps, MidpointOps, NextRepresentableOps, ScalarOps,
        SortingOps, TensorOps, TwoSumOps,
    },
    runtime::{Runtime, RuntimeClient},
    tensor::Tensor,
};
use solvr::{common::arithmetic::CertifiedExpansion, integrate::CertifiedSimpsonAlgorithms};

pub(super) fn endpoint<R>(
    context: &PotentialContext<'_, R>,
    endpoint: PhysicalEndpoint,
) -> HydraulicResult<EndpointCurve<R>>
where
    R: Runtime<DType = DType>,
    R::Client: TensorOps<R>
        + CompareOps<R>
        + MidpointOps<R>
        + NextRepresentableOps<R>
        + ScalarOps<R>
        + TwoSumOps<R>
        + RuntimeClient<R>
        + CertifiedSimpsonAlgorithms<R>,
{
    let c = context.geometry.client();
    let n = context.bands.section_owner.numel();
    let (query, delta, mut result) = checked(c, "point side curve classification", |a| {
        let query = context.query.endpoint(a, context.bands, endpoint)?;
        let delta =
            source_difference(a, context.bands, &query, context.source, context.boundaries)?.lower;
        let classification = classify(a, context.source.dry_mask(), &delta)?;
        let result = EndpointCurve {
            curve: unavailable(a, n)?,
            classification,
            actual_shock: empty_offset(a, n)?,
            conditional_shock: empty_offset(a, n)?,
            equality_limit: unavailable(a, n)?,
        };
        Ok((query, delta, result))
    })?;
    let equal_rows = c.nonzero(&result.classification.equality)?.flatten()?;
    if equal_rows.numel() != 0 {
        let (curve, limit) = checked(c, "point equality curve and sound limit", |a| {
            let hydraulic_depth =
                select_interval(a, context.source.hydraulic_depth_m(), &equal_rows)?;
            let gravity = scalar(a, context.evaluation.config.gravity_m_per_s2)?;
            let sound = gravity_depth_root(a, &gravity, &hydraulic_depth, None)?;
            let retained = a.expansion_from_interval(&sound)?;
            let one = Tensor::ones(&[equal_rows.numel()], a.dtype(), c.device())?;
            let mut curve = unavailable(a, equal_rows.numel())?;
            curve.available = one.clone();
            Ok((
                curve,
                CurveValue {
                    retained,
                    physical: sound,
                    available: one,
                },
            ))
        })?;
        result.curve = checked(c, "point equality curve assembly", |a| {
            scatter_value(a, result.curve, &equal_rows, &curve)
        })?;
        result.equality_limit = checked(c, "point equality limit assembly", |a| {
            scatter_value(a, result.equality_limit, &equal_rows, &limit)
        })?;
    }
    let (rare_rows, shock_rows, conditional_rows) =
        checked(c, "point side curve branch rows", |a| {
            let zero = Tensor::zeros(&[n], a.dtype(), c.device())?;
            let bounds = a.expansion_interval(&delta)?;
            let possible_rare = c.mul(
                &result.classification.uncertain,
                &c.lt(bounds.lower(), &zero)?,
            )?;
            let possible_shock = c.mul(
                &result.classification.uncertain,
                &c.gt(bounds.upper(), &zero)?,
            )?;
            Ok((
                c.nonzero(&c.maximum(&result.classification.rarefaction, &possible_rare)?)?
                    .flatten()?,
                c.nonzero(&result.classification.shock)?.flatten()?,
                c.nonzero(&possible_shock)?.flatten()?,
            ))
        })?;
    let mut rare = checked(c, "point unavailable rare branch", |a| unavailable(a, n))?;
    if rare_rows.numel() != 0 {
        let (bands, source, query, boundaries) = checked(c, "point rare curve selection", |a| {
            Ok((
                context.bands.select(a, &rare_rows)?,
                context.source.select(a, &rare_rows)?,
                query.select(a, &rare_rows)?,
                context.boundaries.select(a, &rare_rows)?,
            ))
        })?;
        let parents = match &context.source_rows {
            Some(parents) => Some(parents.select(c, &rare_rows)?),
            None => None,
        };
        let selected = PotentialContext::new(
            context.geometry,
            &bands,
            &source,
            &query,
            &boundaries,
            PotentialEvaluation {
                config: context.evaluation.config,
                panels: context.evaluation.panels,
                work: context.evaluation.work,
            },
        )?
        .with_source_rows(parents);
        let increment = endpoint_increment(&selected, PhysicalEndpoint::Lower)?;
        rare = checked(c, "point rare branch capture", |a| {
            let value = CurveValue {
                retained: increment.retained,
                physical: increment.physical,
                available: Tensor::ones(&[rare_rows.numel()], a.dtype(), c.device())?,
            };
            scatter_value(a, rare, &rare_rows, &value)
        })?;
        let rows = c.nonzero(&result.classification.rarefaction)?.flatten()?;
        if rows.numel() != 0 {
            let value = checked(c, "point definite rare curve", |a| {
                let mut value = select_value(a, &rare, &rows)?;
                let zero = Tensor::zeros(value.physical.lower().shape(), a.dtype(), c.device())?;
                value.physical = a.bounds(
                    value.physical.lower(),
                    &c.minimum(value.physical.upper(), &zero)?,
                )?;
                Ok(value)
            })?;
            result.curve = checked(c, "point rare curve assembly", |a| {
                scatter_value(a, result.curve, &rows, &value)
            })?;
        }
    }
    let mut shock = checked(c, "point unavailable conditional shock", |a| {
        unavailable(a, n)
    })?;
    for (rows, conditional) in [(&shock_rows, false), (&conditional_rows, true)] {
        if rows.numel() == 0 {
            continue;
        }
        let (value, offset) = checked(c, "point shock curve evaluation", |a| {
            let bands = context.bands.select(a, rows)?;
            let source = context.source.select(a, rows)?;
            let query = query.select(a, rows)?;
            let boundaries = context.boundaries.select(a, rows)?;
            let target_area = query.retained_endpoint_area(a, &bands, &boundaries)?;
            let pressure = pressure_endpoint(
                a,
                context.geometry,
                &bands,
                &query,
                &source,
                &boundaries,
                context.evaluation.config.gravity_m_per_s2,
            )?;
            if conditional {
                shock_points::conditional(a, source.requested_areas_m2(), &target_area, &pressure)
            } else {
                shock_points::actual(a, source.requested_areas_m2(), &target_area, &pressure)
            }
        })?;
        if conditional {
            shock = checked(c, "point conditional shock curve capture", |a| {
                scatter_value(a, shock, rows, &value)
            })?;
            result.conditional_shock =
                checked(c, "point conditional shock domain assembly", |a| {
                    scatter_offset(a, result.conditional_shock, rows, &offset)
                })?;
        } else {
            result.curve = checked(c, "point actual shock curve assembly", |a| {
                scatter_value(a, result.curve, rows, &value)
            })?;
            result.actual_shock = checked(c, "point actual shock domain assembly", |a| {
                scatter_offset(a, result.actual_shock, rows, &offset)
            })?;
        }
    }
    let rows = c.nonzero(&result.classification.uncertain)?.flatten()?;
    if rows.numel() != 0 {
        let value = checked(c, "point conditional side curve envelope", |a| {
            let rare = select_interval(a, &rare.physical, &rows)?;
            let shock = select_interval(a, &shock.physical, &rows)?;
            let zero = Tensor::zeros(&[rows.numel()], a.dtype(), c.device())?;
            // Equality zero joins possible branches without asserting an equality identity.
            let lower = c.minimum(rare.lower(), &zero)?;
            let upper = c.maximum(shock.upper(), &zero)?;
            let physical = a.bounds(&lower, &upper)?;
            Ok(CurveValue {
                retained: a.expansion_from_interval(&physical)?,
                physical,
                available: Tensor::ones(&[rows.numel()], a.dtype(), c.device())?,
            })
        })?;
        result.curve = checked(c, "point conditional side curve assembly", |a| {
            scatter_value(a, result.curve, &rows, &value)
        })?;
    }
    Ok(result)
}

#[cfg(test)]
pub(super) fn range<R>(context: &PotentialContext<'_, R>) -> HydraulicResult<CurveRange<R>>
where
    R: Runtime<DType = DType>,
    R::Client: TensorOps<R>
        + CompareOps<R>
        + MidpointOps<R>
        + NextRepresentableOps<R>
        + ScalarOps<R>
        + TwoSumOps<R>
        + RuntimeClient<R>
        + CertifiedSimpsonAlgorithms<R>,
{
    let lower = endpoint(context, PhysicalEndpoint::Lower)?;
    let upper = endpoint(context, PhysicalEndpoint::Upper)?;
    let curve = checked::<R, _>(
        context.geometry.client(),
        "point monotone side curve range",
        |a| {
            let available = a
                .client()
                .mul(&lower.curve.available, &upper.curve.available)?;
            let physical = a.bounds(lower.curve.physical.lower(), upper.curve.physical.upper())?;
            Ok(CurveValue {
                retained: a.expansion_from_interval(&physical)?,
                physical,
                available,
            })
        },
    )?;
    Ok(CurveRange {
        lower,
        upper,
        curve,
    })
}

pub(super) fn classify<R>(
    a: &mut Arithmetic<'_, R>,
    dry: &Tensor<R>,
    delta: &CertifiedExpansion<R>,
) -> HydraulicResult<CurveClassification<R>>
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
    let zero = Tensor::zeros(dry.shape(), a.dtype(), c.device())?;
    let wet = c.eq(dry, &zero)?;
    let bounds = a.expansion_interval(delta)?;
    let shock = c.mul(&wet, &c.gt(bounds.lower(), &zero)?)?;
    let rarefaction = c.mul(&wet, &c.lt(bounds.upper(), &zero)?)?;
    let equality = c.mul(
        &wet,
        &c.mul(
            &c.eq(delta.high(), &zero)?,
            &c.mul(&c.eq(delta.low(), &zero)?, &c.eq(delta.abs_error(), &zero)?)?,
        )?,
    )?;
    let known = c.maximum(&equality, &c.maximum(&shock, &rarefaction)?)?;
    let uncertain = c.mul(&wet, &c.eq(&known, &zero)?)?;
    Ok(CurveClassification {
        shock,
        rarefaction,
        equality,
        uncertain,
        #[cfg(test)]
        dry_source: dry.clone(),
    })
}

fn select_value<R>(
    a: &mut Arithmetic<'_, R>,
    value: &CurveValue<R>,
    rows: &Tensor<R>,
) -> HydraulicResult<CurveValue<R>>
where
    R: Runtime<DType = DType>,
    R::Client: TensorOps<R>
        + CompareOps<R>
        + NextRepresentableOps<R>
        + ScalarOps<R>
        + TwoSumOps<R>
        + RuntimeClient<R>,
{
    Ok(CurveValue {
        retained: select_expansion(a, &value.retained, rows)?,
        physical: select_interval(a, &value.physical, rows)?,
        available: a.client().index_select(&value.available, 0, rows)?,
    })
}

#[cfg(test)]
mod tests {
    use super::super::source_fixture_support::{
        Client, Result, bounds as source_bounds, components, endpoints, expansion, hydraulic,
    };
    use super::super::wave_curve;
    use super::super::wave_curve_fixture_support::{self as support, check_endpoint};
    use super::super::wave_curve_fixtures as fixtures;
    use super::*;
    use crate::channel::{
        flow::surveyed::{
            PrismaticRiemannConfig, PrismaticRiemannRejection,
            error::{self, HydraulicError},
            work::WorkLedger,
        },
        section::HydraulicGeometry,
    };
    use solvr::common::arithmetic::ArithmeticError;

    mod conditional {
        use super::*;
        use crate::channel::section::{PrismaticSectionBatch, pressure_endpoint};

        pub(super) fn controls<R>(
            client: &R::Client,
            sections: &[PrismaticSectionBatch<R>],
        ) -> Result
        where
            R: Runtime<DType = DType>,
            R::Client: Client<R> + CertifiedSimpsonAlgorithms<R>,
        {
            let control: fixtures::UncertainSign = fixtures::UNCERTAIN_SIGN;
            let original = &fixtures::CASES[control.case_index];
            support::with_case::<R, _>(
                client,
                sections,
                original,
                true,
                1 << 24,
                |context, endpoint| {
                    let (classification, curve, offset, delta) =
                        hydraulic(error::checked::<R, _>(
                            client,
                            "frozen uncertain sign wave certificate",
                            |a| {
                                let query = context.query.endpoint(a, context.bands, endpoint)?;
                                let target = query.retained_endpoint_area(
                                    a,
                                    context.bands,
                                    context.boundaries,
                                )?;
                                let mut pressure = pressure_endpoint(
                                    a,
                                    context.geometry,
                                    context.bands,
                                    &query,
                                    context.source,
                                    context.boundaries,
                                    context.evaluation.config.gravity_m_per_s2,
                                )?;
                                let center = a.expansion_bounds(
                                    &Tensor::from_slice(
                                        &[f64::from_bits(control.delta_center[0]); 3],
                                        &[3],
                                        client.device(),
                                    )?,
                                    &Tensor::from_slice(
                                        &[f64::from_bits(control.delta_center[1]); 3],
                                        &[3],
                                        client.device(),
                                    )?,
                                    &Tensor::zeros(&[3], DType::F64, client.device())?,
                                )?;
                                let radius = a.expansion_bounds(
                                    &Tensor::from_slice(
                                        &[f64::from_bits(control.delta_radius[0]); 3],
                                        &[3],
                                        client.device(),
                                    )?,
                                    &Tensor::from_slice(
                                        &[f64::from_bits(control.delta_radius[1]); 3],
                                        &[3],
                                        client.device(),
                                    )?,
                                    &Tensor::zeros(&[3], DType::F64, client.device())?,
                                )?;
                                let radius = a.expansion_interval(&radius)?;
                                // Only this existing separation certificate widens. Physical source and target inputs stay unchanged.
                                let widened = a.expansion_bounds(
                                    center.high(),
                                    center.low(),
                                    radius.upper(),
                                )?;
                                let old = a.expansion_interval(&pressure.delta_area)?;
                                let widened_bounds = a.expansion_interval(&widened)?;
                                let old =
                                    endpoints(&old).map_err(|e| HydraulicError::Numerical {
                                        context: e.to_string(),
                                    })?;
                                let broad = endpoints(&widened_bounds).map_err(|e| {
                                    HydraulicError::Numerical {
                                        context: e.to_string(),
                                    }
                                })?;
                                for (old, broad) in old.into_iter().zip(broad) {
                                    assert!(broad[0] <= old[0] && old[1] <= broad[1]);
                                    assert!(broad[0] < 0.0 && broad[1] > 0.0);
                                }
                                pressure.delta_area = widened;
                                let classification = wave_curve::classify(
                                    a,
                                    context.source.dry_mask(),
                                    &pressure.delta_area,
                                )?;
                                let (curve, offset) = shock_points::conditional(
                                    a,
                                    context.source.requested_areas_m2(),
                                    &target,
                                    &pressure,
                                )?;
                                Ok((classification, curve, offset, pressure.delta_area))
                            },
                        ))?;
                    assert_eq!(classification.uncertain.try_to_vec::<f64>()?, [1.0; 3]);
                    if control.overlap_proves_neither_equality_nor_shock {
                        assert_eq!(classification.equality.try_to_vec::<f64>()?, [0.0; 3]);
                        assert_eq!(classification.shock.try_to_vec::<f64>()?, [0.0; 3]);
                    }
                    if control.actual_shock_offset_unavailable {
                        assert_eq!(classification.shock.try_to_vec::<f64>()?, [0.0; 3]);
                    }
                    assert_eq!(
                        components(&delta)?,
                        components(&offset.domain.retained_separation)?
                    );
                    let expected = match original.expected {
                        fixtures::Expected::Shock {
                            curve: fixtures::Reference::Local { bounds, .. },
                            ..
                        } => bounds.map(f64::from_bits),
                        _ => {
                            return Err(
                                "Frozen uncertain sign control lacks its physical shock reference"
                                    .into(),
                            );
                        }
                    };
                    for bounds in endpoints(&curve.physical)? {
                        assert!(bounds[0] <= 0.0 && bounds[1] >= expected[1]);
                        assert!(bounds[0] <= expected[0]);
                    }
                    for bounds in endpoints(&offset.domain.contraction)? {
                        assert!(bounds[0] >= 0.0 && bounds[1] <= 1.0);
                    }
                    for bounds in endpoints(&offset.domain.separation)? {
                        assert_eq!(bounds[0], 0.0);
                        assert!(bounds[1] > 0.0);
                    }
                    assert_eq!(context.evaluation.work.snapshot(), Default::default());
                    Ok(())
                },
            )?;
            let underflow: fixtures::ContractionUnderflow = fixtures::CONTRACTION_UNDERFLOW;
            support::with_case::<R, _>(
                client,
                sections,
                &fixtures::CASES[underflow.case_index],
                true,
                1 << 24,
                |context, endpoint| {
                    let result = hydraulic(wave_curve::endpoint(context, endpoint))?;
                    check_endpoint::<R>(
                        client,
                        &result,
                        fixtures::CASES[underflow.case_index].expected,
                        fixtures::CASES[underflow.case_index].id,
                    )?;
                    if underflow.nonzero_physical_truth {
                        for bounds in endpoints(&result.actual_shock.domain.separation)? {
                            assert!(bounds[0] > 0.0);
                        }
                        for bounds in endpoints(&result.curve.physical)? {
                            assert!(bounds[0] > 0.0);
                        }
                    }
                    let contraction = hydraulic(error::checked::<R, _>(
                        client,
                        "contraction underflow certificate",
                        |a| Ok(a.expansion_from_interval(&result.actual_shock.domain.contraction)?),
                    ))?;
                    if underflow.zero_center_requires_nonzero_radius {
                        for point in components(&contraction)? {
                            let [high, low, radius] = point.map(f64::from_bits);
                            if high == 0.0 && low == 0.0 {
                                assert!(radius > 0.0);
                            }
                        }
                    }
                    assert_eq!(context.evaluation.work.snapshot(), Default::default());
                    Ok(())
                },
            )
        }
    }

    mod mixed {
        use super::*;
        use crate::channel::flow::surveyed_points::wave_curve_types::{ShockDomain, ShockOffset};
        use crate::channel::section::{AreaDirection, AreaQuery};

        #[derive(Debug, PartialEq)]
        struct ValueRow {
            retained: [u64; 3],
            physical: [u64; 2],
            available: u64,
        }
        #[derive(Debug, PartialEq)]
        struct DomainRow {
            source: [u64; 2],
            target: [u64; 2],
            retained_gap: [u64; 3],
            gap: [u64; 2],
            contraction: [u64; 2],
        }
        #[derive(Debug, PartialEq)]
        struct OffsetRow {
            value: ValueRow,
            domain: DomainRow,
        }
        #[derive(Debug, PartialEq)]
        struct EndpointRow {
            curve: ValueRow,
            actual: OffsetRow,
            conditional: OffsetRow,
            equality_limit: ValueRow,
            classification: [u64; 5],
        }

        fn value_rows<R: Runtime<DType = DType>>(value: &CurveValue<R>) -> Result<Vec<ValueRow>> {
            let retained = components(&value.retained)?;
            let physical = endpoints(&value.physical)?;
            let available = value.available.try_to_vec::<f64>()?;
            Ok(retained
                .into_iter()
                .zip(physical)
                .zip(available)
                .map(|((retained, physical), available)| ValueRow {
                    retained,
                    physical: physical.map(f64::to_bits),
                    available: available.to_bits(),
                })
                .collect())
        }

        fn domain_rows<R: Runtime<DType = DType>>(
            value: &ShockDomain<R>,
        ) -> Result<Vec<DomainRow>> {
            let source = endpoints(&value.source_area)?;
            let target = endpoints(&value.target_area)?;
            let retained_gap = components(&value.retained_separation)?;
            let gap = endpoints(&value.separation)?;
            let contraction = endpoints(&value.contraction)?;
            Ok(source
                .into_iter()
                .zip(target)
                .zip(retained_gap)
                .zip(gap)
                .zip(contraction)
                .map(
                    |((((source, target), retained_gap), gap), contraction)| DomainRow {
                        source: source.map(f64::to_bits),
                        target: target.map(f64::to_bits),
                        retained_gap,
                        gap: gap.map(f64::to_bits),
                        contraction: contraction.map(f64::to_bits),
                    },
                )
                .collect())
        }

        fn offset_rows<R: Runtime<DType = DType>>(
            value: &ShockOffset<R>,
        ) -> Result<Vec<OffsetRow>> {
            Ok(value_rows(&value.value)?
                .into_iter()
                .zip(domain_rows(&value.domain)?)
                .map(|(value, domain)| OffsetRow { value, domain })
                .collect())
        }

        fn snapshot<R: Runtime<DType = DType>>(
            value: &EndpointCurve<R>,
        ) -> Result<Vec<EndpointRow>> {
            let curve = value_rows(&value.curve)?;
            let actual = offset_rows(&value.actual_shock)?;
            let conditional = offset_rows(&value.conditional_shock)?;
            let equality_limit = value_rows(&value.equality_limit)?;
            let masks = [
                &value.classification.shock,
                &value.classification.rarefaction,
                &value.classification.equality,
                &value.classification.uncertain,
                &value.classification.dry_source,
            ]
            .into_iter()
            .map(|v| v.try_to_vec::<f64>())
            .collect::<std::result::Result<Vec<_>, _>>()?;
            Ok(curve
                .into_iter()
                .zip(actual)
                .zip(conditional)
                .zip(equality_limit)
                .enumerate()
                .map(
                    |(row, (((curve, actual), conditional), equality_limit))| EndpointRow {
                        curve,
                        actual,
                        conditional,
                        equality_limit,
                        classification: std::array::from_fn(|mask| masks[mask][row].to_bits()),
                    },
                )
                .collect())
        }

        #[derive(Debug, PartialEq)]
        struct Inputs {
            source: Vec<[u64; 3]>,
            source_owners: Vec<i64>,
            target: Vec<[u64; 3]>,
            scale: Vec<[u64; 3]>,
            parameter: Vec<[u64; 2]>,
            area: Vec<[u64; 2]>,
            above_start: Vec<[u64; 2]>,
            below_end: Vec<[u64; 2]>,
            candidate: Vec<u64>,
            owners: Vec<i64>,
            original_rows: Vec<i64>,
            boundaries: Vec<[u64; 3]>,
        }

        fn input_snapshot<R>(context: &PotentialContext<'_, R>) -> Result<Inputs>
        where
            R: Runtime<DType = DType>,
            R::Client: Client<R>,
        {
            let target = hydraulic(error::checked::<R, _>(
                context.geometry.client(),
                "mixed query identity snapshot",
                |a| {
                    context
                        .query
                        .retained_endpoint_area(a, context.bands, context.boundaries)
                },
            ))?;
            let bits =
                |interval: &solvr::common::arithmetic::Interval<R>| -> Result<Vec<[u64; 2]>> {
                    Ok(endpoints(interval)?
                        .into_iter()
                        .map(|v| v.map(f64::to_bits))
                        .collect())
                };
            Ok(Inputs {
                source: components(context.source.requested_areas_m2())?,
                source_owners: context.source.owners().try_to_vec::<i64>()?,
                target: components(&target)?,
                scale: components(&context.query.query.scale)?,
                parameter: bits(&context.query.query.parameter)?,
                area: bits(&context.query.area)?,
                above_start: bits(&context.query.above_start)?,
                below_end: bits(&context.query.below_end)?,
                candidate: context
                    .query
                    .candidate
                    .try_to_vec::<f64>()?
                    .into_iter()
                    .map(f64::to_bits)
                    .collect(),
                owners: context.query.query.owners.try_to_vec::<i64>()?,
                original_rows: context.query.query.original_rows.try_to_vec::<i64>()?,
                boundaries: components(context.boundaries.areas())?,
            })
        }

        fn mixed<R>(client: R::Client) -> Result
        where
            R: Runtime<DType = DType>,
            R::Client: Client<R> + CertifiedSimpsonAlgorithms<R>,
        {
            let sections = support::sections::<R>(&client)?;
            let section = &sections[1];
            let geometry = HydraulicGeometry::from_batch(section);
            let owner_storage =
                Tensor::from_slice(&[-1_i64, 0, -1, 0, -1, 0, -1, 0], &[4, 2], client.device())?;
            let owners = owner_storage.narrow(1, 1, 1)?.squeeze(Some(1));
            let original_owners = owner_storage.try_to_vec::<i64>()?;
            let source_area = expansion::<R>(
                &client,
                &[
                    [1.0_f64.to_bits(), 0, 0],
                    [2.0_f64.to_bits(), 0, 0],
                    [1.0_f64.to_bits(), 0, 0],
                    [0, 0, 0],
                ],
                true,
            )?;
            let original_area = components(&source_area)?;
            let config = PrismaticRiemannConfig {
                gravity_m_per_s2: 3.0,
                ..PrismaticRiemannConfig::default()
            };
            let source =
                section.recover_area_points_tensor(&owners, &source_area, &config.recovery)?;
            let parameter_storage = Tensor::from_slice(
                &[f64::NAN, 1.0, f64::NAN, -1.0, f64::NAN, 0.0, f64::NAN, 1.0],
                &[4, 2],
                client.device(),
            )?;
            let parameters = parameter_storage.narrow(1, 1, 1)?.squeeze(Some(1));
            let original_parameters = parameter_storage
                .try_to_vec::<f64>()?
                .into_iter()
                .map(f64::to_bits)
                .collect::<Vec<_>>();
            let (bands, boundaries, query) = hydraulic(error::checked::<R, _>(
                &client,
                "mixed side curve graph",
                |a| {
                    let bands = geometry.gather(a, &owners)?;
                    let boundaries = geometry.boundary_areas(a)?.gather(a, &owners)?;
                    let scale =
                        a.expansion_point(&Tensor::ones(&[4], a.dtype(), client.device())?)?;
                    let parameter = a.point(&parameters)?;
                    let query = AreaQuery::source_affine(
                        a,
                        &bands,
                        &source,
                        AreaDirection::Add,
                        &scale,
                        &parameter,
                    )?
                    .prove_support(a, &bands)?;
                    Ok((bands, boundaries, query))
                },
            ))?;
            let work = WorkLedger::new(&config);
            let context = hydraulic(PotentialContext::new(
                &geometry,
                &bands,
                &source,
                &query,
                &boundaries,
                PotentialEvaluation {
                    config: &config,
                    panels: 4,
                    work: &work,
                },
            ))?;
            let before = input_snapshot(&context)?;
            let batch = hydraulic(wave_curve::endpoint(&context, PhysicalEndpoint::Lower))?;
            assert_eq!(
                batch.classification.shock.try_to_vec::<f64>()?,
                [1.0, 0.0, 0.0, 0.0]
            );
            assert_eq!(
                batch.classification.rarefaction.try_to_vec::<f64>()?,
                [0.0, 1.0, 0.0, 0.0]
            );
            assert_eq!(
                batch.classification.equality.try_to_vec::<f64>()?,
                [0.0, 0.0, 1.0, 0.0]
            );
            assert_eq!(
                batch.classification.uncertain.try_to_vec::<f64>()?,
                [0.0; 4]
            );
            assert_eq!(
                batch.classification.dry_source.try_to_vec::<f64>()?,
                [0.0, 0.0, 0.0, 1.0]
            );
            assert_eq!(
                batch.curve.available.try_to_vec::<f64>()?,
                [1.0, 1.0, 1.0, 0.0]
            );
            assert_eq!(
                batch.actual_shock.value.available.try_to_vec::<f64>()?,
                [1.0, 0.0, 0.0, 0.0]
            );
            assert_eq!(
                batch
                    .conditional_shock
                    .value
                    .available
                    .try_to_vec::<f64>()?,
                [0.0; 4]
            );
            assert_eq!(
                batch.equality_limit.available.try_to_vec::<f64>()?,
                [0.0, 0.0, 1.0, 0.0]
            );
            let batch_rows = snapshot(&batch)?;
            let mut singles = Vec::with_capacity(4);
            assert_eq!(batch_rows.len(), 4);
            for (row, batch_row) in batch_rows.iter().enumerate() {
                let original_row = i64::try_from(row)?;
                let rows = Tensor::from_slice(&[original_row], &[1], client.device())?;
                let (bands, source, query, boundaries) = hydraulic(error::checked::<R, _>(
                    &client,
                    "singleton side curve selection",
                    |a| {
                        Ok((
                            context.bands.select(a, &rows)?,
                            context.source.select(a, &rows)?,
                            context.query.select(a, &rows)?,
                            context.boundaries.select(a, &rows)?,
                        ))
                    },
                ))?;
                assert_eq!(
                    query.query.original_rows.try_to_vec::<i64>()?,
                    [original_row]
                );
                let singleton = hydraulic(PotentialContext::new(
                    &geometry,
                    &bands,
                    &source,
                    &query,
                    &boundaries,
                    PotentialEvaluation {
                        config: &config,
                        panels: 4,
                        work: &work,
                    },
                ))?;
                let value = hydraulic(wave_curve::endpoint(&singleton, PhysicalEndpoint::Lower))?;
                let rows = snapshot(&value)?;
                assert_eq!(rows.len(), 1);
                assert_eq!(batch_row, &rows[0], "mixed row={row}");
                singles.extend(rows);
            }
            let order = [2_i64, 0, 2, 3, 1];
            let rows = Tensor::from_slice(&order, &[order.len()], client.device())?;
            let (selected_bands, selected_source, selected_query, selected_boundaries) = hydraulic(
                error::checked::<R, _>(&client, "repeated mixed side curve selection", |a| {
                    Ok((
                        bands.select(a, &rows)?,
                        source.select(a, &rows)?,
                        query.select(a, &rows)?,
                        boundaries.select(a, &rows)?,
                    ))
                }),
            )?;
            assert_eq!(
                selected_query.query.original_rows.try_to_vec::<i64>()?,
                order
            );
            let selected = hydraulic(PotentialContext::new(
                &geometry,
                &selected_bands,
                &selected_source,
                &selected_query,
                &selected_boundaries,
                PotentialEvaluation {
                    config: &config,
                    panels: 4,
                    work: &work,
                },
            ))?;
            let selected_before = input_snapshot(&selected)?;
            let result = hydraulic(wave_curve::endpoint(&selected, PhysicalEndpoint::Lower))?;
            for (row, result) in snapshot(&result)?.into_iter().enumerate() {
                assert_eq!(
                    result,
                    singles[usize::try_from(order[row])?],
                    "selected row={row}"
                );
            }
            assert_eq!(selected_before, input_snapshot(&selected)?);
            assert_eq!(before, input_snapshot(&context)?);
            assert_eq!(original_area, components(&source_area)?);
            assert_eq!(original_owners, owner_storage.try_to_vec::<i64>()?);
            assert_eq!(
                original_parameters,
                parameter_storage
                    .try_to_vec::<f64>()?
                    .into_iter()
                    .map(f64::to_bits)
                    .collect::<Vec<_>>()
            );
            assert_eq!(work.snapshot(), Default::default());

            let physical_curve = endpoints(&batch.curve.physical)?;
            let physical_offset = endpoints(&batch.actual_shock.value.physical)?;
            let retained_curve = source_bounds::<R>(&client, &batch.curve.retained)?;
            let retained_curve = endpoints(&retained_curve)?;
            let retained_offset = source_bounds::<R>(&client, &batch.actual_shock.value.retained)?;
            let retained_offset = endpoints(&retained_offset)?;
            for (row, id) in [
                (0, "legacy_stationary_compression"),
                (1, "legacy_stationary_expansion"),
            ] {
                let reference = fixtures::CASES
                    .iter()
                    .find(|case| case.id == id)
                    .ok_or("Missing frozen stationary wave reference")?;
                let curve = match reference.expected {
                    fixtures::Expected::Shock { curve, offset } => {
                        support::check(physical_offset[row], offset, id);
                        support::check(retained_offset[row], offset, id);
                        curve
                    }
                    fixtures::Expected::Rare { curve } => curve,
                    _ => return Err("Stationary wave reference has an inconsistent branch".into()),
                };
                support::check(physical_curve[row], curve, id);
                support::check(retained_curve[row], curve, id);
            }
            Ok(())
        }

        #[test]
        fn configured_cpu_mixed_side_curves_preserve_selection_and_singleton_certificates() -> Result
        {
            use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime, ParallelismConfig};
            let original = CpuClient::new(CpuDevice::new());
            let before = original.parallelism();
            let config = ParallelismConfig::new(Some(2), Some(257));
            let client = original.with_parallelism(config);
            let retained = client.clone();
            mixed::<CpuRuntime>(client)?;
            assert_eq!(retained.parallelism(), config);
            assert_eq!(original.parallelism(), before);
            Ok(())
        }

        #[cfg(feature = "cuda")]
        #[test]
        fn cuda_mixed_side_curves_preserve_selection_and_singleton_certificates() -> Result {
            use numr::runtime::cuda::{CudaClient, CudaDevice, CudaRuntime};
            mixed::<CudaRuntime>(CudaClient::new(CudaDevice::new(0))?)
        }
    }

    mod underflow {
        use super::super::super::wave_curve_scaling_fixture as scaling;
        use super::*;
        use crate::channel::section::source_difference;

        fn contraction_underflow<R>(client: R::Client) -> Result
        where
            R: Runtime<DType = DType>,
            R::Client: Client<R> + CertifiedSimpsonAlgorithms<R>,
        {
            let sections = support::sections::<R>(&client)?;
            let case = &fixtures::CASES[fixtures::CONTRACTION_UNDERFLOW.case_index];
            support::with_case::<R, _>(
                &client,
                &sections,
                case,
                true,
                1 << 24,
                |context, endpoint| {
                    let fixtures::Input::SourceRelative { source, delta, .. } = case.input else {
                        return Err(
                            "Contraction control requires its original source-relative input"
                                .into(),
                        );
                    };
                    assert_eq!(delta, [1, 0], "{}: original minimum-subnormal gap", case.id);
                    let expected_delta = [[delta[0], delta[1], 0]; 3];
                    let source_before = components(context.source.requested_areas_m2())?;
                    let scale_before = components(&context.query.query.scale)?;
                    let parameter_before = endpoints(&context.query.query.parameter)?;
                    let owners_before = context.query.query.owners.try_to_vec::<i64>()?;
                    let source_owners_before = context.source.owners().try_to_vec::<i64>()?;
                    assert_eq!(
                        source_before,
                        [[source[0], source[1], 0]; 3],
                        "{}: original source",
                        case.id
                    );
                    assert_eq!(
                        scale_before, expected_delta,
                        "{}: exact query scale",
                        case.id
                    );
                    assert_eq!(
                        parameter_before, [[1.0; 2]; 3],
                        "{}: unit map parameter",
                        case.id
                    );
                    let (differences, endpoint_difference, product) =
                        hydraulic(error::checked::<R, _>(
                            &client,
                            "frozen contraction underflow stages",
                            |a| {
                                let query = context.query.endpoint(a, context.bands, endpoint)?;
                                let differences = source_difference(
                                    a,
                                    context.bands,
                                    context.query,
                                    context.source,
                                    context.boundaries,
                                )?;
                                let endpoint_difference = source_difference(
                                    a,
                                    context.bands,
                                    &query,
                                    context.source,
                                    context.boundaries,
                                )?;
                                let parameter =
                                    a.expansion_point(context.query.query.parameter.lower())?;
                                let product =
                                    a.expansion_mul(&context.query.query.scale, &parameter)?;
                                Ok((differences, endpoint_difference, product))
                            },
                        ))?;
                    for (label, value) in [
                        ("minimum-subnormal times one", &product),
                        ("stored map difference lower", &differences.lower),
                        ("stored map difference upper", &differences.upper),
                        ("endpoint difference lower", &endpoint_difference.lower),
                        ("endpoint difference upper", &endpoint_difference.upper),
                    ] {
                        assert_eq!(components(value)?, expected_delta, "{}: {label}", case.id);
                    }
                    let classification = hydraulic(error::checked::<R, _>(
                        &client,
                        "underflow shock classification",
                        |a| {
                            wave_curve::classify(
                                a,
                                context.source.dry_mask(),
                                &endpoint_difference.lower,
                            )
                        },
                    ))?;
                    assert_eq!(
                        classification.shock.try_to_vec::<f64>()?,
                        [1.0; 3],
                        "{}: definite shock",
                        case.id
                    );
                    for (label, mask) in [
                        ("rarefaction", &classification.rarefaction),
                        ("equality", &classification.equality),
                        ("uncertain", &classification.uncertain),
                        ("dry source", &classification.dry_source),
                    ] {
                        assert_eq!(mask.try_to_vec::<f64>()?, [0.0; 3], "{}: {label}", case.id);
                    }
                    let value = hydraulic(wave_curve::endpoint(context, endpoint))?;
                    check_endpoint::<R>(&client, &value, case.expected, case.id)?;
                    assert_eq!(
                        components(context.source.requested_areas_m2())?,
                        source_before,
                        "{}: source snapshot",
                        case.id
                    );
                    assert_eq!(
                        components(&context.query.query.scale)?,
                        scale_before,
                        "{}: scale snapshot",
                        case.id
                    );
                    assert_eq!(
                        endpoints(&context.query.query.parameter)?,
                        parameter_before,
                        "{}: parameter snapshot",
                        case.id
                    );
                    assert_eq!(
                        context.query.query.owners.try_to_vec::<i64>()?,
                        owners_before,
                        "{}: owner snapshot",
                        case.id
                    );
                    assert_eq!(
                        context.source.owners().try_to_vec::<i64>()?,
                        source_owners_before,
                        "{}: source owner snapshot",
                        case.id
                    );
                    let stored_after = hydraulic(error::checked::<R, _>(
                        &client,
                        "underflow stored map snapshot",
                        |a| {
                            source_difference(
                                a,
                                context.bands,
                                context.query,
                                context.source,
                                context.boundaries,
                            )
                        },
                    ))?;
                    for (label, value) in [
                        ("stored map lower snapshot", &stored_after.lower),
                        ("stored map upper snapshot", &stored_after.upper),
                    ] {
                        assert_eq!(components(value)?, expected_delta, "{}: {label}", case.id);
                    }
                    Ok(())
                },
            )
        }

        fn large_area<R>(client: R::Client) -> Result
        where
            R: Runtime<DType = DType>,
            R::Client: Client<R> + CertifiedSimpsonAlgorithms<R>,
        {
            use crate::channel::section::{
                CrossSectionBatch, CrossSectionDefinition, CrossSectionPoint,
            };
            let definition = CrossSectionDefinition {
                id: "large area and small gravity".into(),
                vertical_datum: "Independent scaling reference datum".into(),
                points: scaling::COORDINATES
                    .iter()
                    .map(|point| CrossSectionPoint {
                        station_m: f64::from_bits(point[0]),
                        elevation_m: f64::from_bits(point[1]),
                    })
                    .collect(),
                maximum_stage_m: f64::from_bits(scaling::CAP_STAGE),
                profile_provenance: "Frozen independent side-curve scaling addendum".into(),
            };
            let section = CrossSectionBatch::<R>::new(client.clone(), vec![definition])?
                .certify_prismatic()?;
            let case = fixtures::Case {
                id: "large area and small gravity compression",
                input: fixtures::Input::SourceRelative {
                    profile: fixtures::ProfileRef::Query(0),
                    gravity_bits: scaling::GRAVITY,
                    source: scaling::SOURCE,
                    delta: scaling::DELTA,
                },
                expected: fixtures::Expected::Shock {
                    curve: fixtures::Reference::Local {
                        bounds: scaling::CURVE,
                        allowance: scaling::CURVE_ALLOWANCE,
                    },
                    offset: fixtures::Reference::Local {
                        bounds: scaling::OFFSET,
                        allowance: scaling::OFFSET_ALLOWANCE,
                    },
                },
            };
            support::with_case::<R, _>(
                &client,
                &[section],
                &case,
                true,
                1 << 24,
                |context, endpoint| {
                    let value = hydraulic(wave_curve::endpoint(context, endpoint))?;
                    check_endpoint::<R>(&client, &value, case.expected, case.id)
                },
            )
        }

        #[test]
        fn configured_cpu_contraction_underflow_preserves_exact_map_and_shock_curve() -> Result {
            use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime, ParallelismConfig};
            let client = CpuClient::new(CpuDevice::new())
                .with_parallelism(ParallelismConfig::new(Some(2), Some(257)));
            contraction_underflow::<CpuRuntime>(client)
        }

        #[cfg(feature = "cuda")]
        #[test]
        fn cuda_contraction_underflow_preserves_exact_map_and_shock_curve() -> Result {
            use numr::runtime::cuda::{CudaClient, CudaDevice, CudaRuntime};
            contraction_underflow::<CudaRuntime>(CudaClient::new(CudaDevice::new(0))?)
        }

        #[test]
        fn configured_cpu_small_gravity_large_area_side_curve_preserves_ordering() -> Result {
            use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime, ParallelismConfig};
            let client = CpuClient::new(CpuDevice::new())
                .with_parallelism(ParallelismConfig::new(Some(2), Some(257)));
            large_area::<CpuRuntime>(client)
        }

        #[cfg(feature = "cuda")]
        #[test]
        fn cuda_small_gravity_large_area_side_curve_preserves_ordering() -> Result {
            use numr::runtime::cuda::{CudaClient, CudaDevice, CudaRuntime};
            large_area::<CudaRuntime>(CudaClient::new(CudaDevice::new(0))?)
        }
    }

    fn matrix<R>(client: R::Client) -> Result
    where
        R: Runtime<DType = DType>,
        R::Client: Client<R> + CertifiedSimpsonAlgorithms<R>,
    {
        let sections = support::sections::<R>(&client)?;
        for strided in [false, true] {
            for case in fixtures::CASES {
                support::with_case::<R, _>(
                    &client,
                    &sections,
                    case,
                    strided,
                    1 << 24,
                    |context, endpoint| {
                        let before = context.evaluation.work.snapshot();
                        let result = wave_curve::endpoint(context, endpoint);
                        if matches!(case.expected, fixtures::Expected::NumericalRange) {
                            let error = result
                                .err()
                                .ok_or("Numerical range case returned a wave certificate")?;
                            assert!(
                                matches!(
                                    error,
                                    HydraulicError::Arithmetic(ArithmeticError::Range { .. })
                                ),
                                "{}: expected arithmetic range, actual={error:?}",
                                case.id
                            );
                            return Ok(());
                        }
                        let value = hydraulic(result)?;
                        check_endpoint::<R>(&client, &value, case.expected, case.id)?;
                        if matches!(
                            case.expected,
                            fixtures::Expected::Dry
                                | fixtures::Expected::Equality { .. }
                                | fixtures::Expected::Shock { .. }
                        ) {
                            assert_eq!(
                                context.evaluation.work.snapshot(),
                                before,
                                "{}: unrequested quadrature",
                                case.id
                            );
                        }
                        if matches!(case.expected, fixtures::Expected::Dry) {
                            // Dry-source availability remains separate from a lazily requested target potential.
                            if let fixtures::Input::Baseline {
                                binding_index,
                                original_parameter_endpoint,
                            } = case.input
                            {
                                let reference =
                                    super::super::potential_increment_fixtures::query_profiles::CASES
                                        [binding_index]
                                        .endpoints[original_parameter_endpoint]
                                        .potential;
                                let target = hydraulic(
                                    super::super::potential_absolute::endpoint_potential(
                                        context, endpoint,
                                    ),
                                )?;
                                let expected = fixtures::Reference::Local {
                                    bounds: reference.bounds,
                                    allowance: reference.native_allowance,
                                };
                                for bounds in endpoints(&target.physical)? {
                                    support::check(bounds, expected, case.id);
                                }
                            }
                        }
                        Ok(())
                    },
                )
                .map_err(|error| format!("{} (strided={strided}): {error}", case.id))?;
            }
        }
        for index in fixtures::SOURCE_CROSSING {
            let case = &fixtures::CASES[index];
            support::with_case::<R, _>(&client, &sections, case, true, 1 << 24, |context, _| {
                let range = hydraulic(wave_curve::range(context))?;
                let lower_case = &fixtures::CASES[fixtures::SOURCE_CROSSING[0]];
                let upper_case = &fixtures::CASES[fixtures::SOURCE_CROSSING[1]];
                check_endpoint::<R>(&client, &range.lower, lower_case.expected, lower_case.id)?;
                check_endpoint::<R>(&client, &range.upper, upper_case.expected, upper_case.id)?;
                assert_eq!(range.curve.available.try_to_vec::<f64>()?, [1.0; 3]);
                let physical = endpoints(&range.curve.physical)?;
                let retained = source_bounds::<R>(&client, &range.curve.retained)?;
                let retained = endpoints(&retained)?;
                let lower = endpoints(&range.lower.curve.physical)?;
                let upper = endpoints(&range.upper.curve.physical)?;
                for row in 0..3 {
                    assert_eq!(physical[row], [lower[row][0], upper[row][1]]);
                    assert!(
                        retained[row][0] <= physical[row][0]
                            && physical[row][1] <= retained[row][1]
                    );
                }
                Ok(())
            })?;
        }
        let rare = fixtures::CASES
            .iter()
            .find(|case| {
                matches!(
                    case.input,
                    fixtures::Input::Baseline {
                        binding_index: 13,
                        ..
                    }
                )
            })
            .ok_or("Missing frozen multiband rarefaction")?;
        support::with_case::<R, _>(&client, &sections, rare, true, 0, |context, endpoint| {
            let result = wave_curve::endpoint(context, endpoint);
            assert!(matches!(
                result,
                Err(HydraulicError::Rejected {
                    reason: PrismaticRiemannRejection::WorkLimit,
                    ..
                })
            ));
            assert_eq!(context.evaluation.work.snapshot(), Default::default());
            Ok(())
        })?;
        conditional::controls::<R>(&client, &sections)
    }

    #[test]
    fn configured_cpu_side_curves_preserve_retained_offsets_and_physical_branches() -> Result {
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
    fn cuda_side_curves_preserve_retained_offsets_and_physical_branches() -> Result {
        use numr::runtime::cuda::{CudaClient, CudaDevice, CudaRuntime};
        matrix::<CudaRuntime>(CudaClient::new(CudaDevice::new(0))?)
    }
}
