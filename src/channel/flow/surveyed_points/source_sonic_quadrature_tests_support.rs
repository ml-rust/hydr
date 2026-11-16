use super::*;
use crate::channel::flow::surveyed::error::checked;

pub(super) fn owner<R>(
    session: &PointRootSession<'_, '_, R>,
    side: SonicSide,
    owner: i64,
) -> Result<Vec<Vec<u64>>>
where
    R: Runtime<DType = DType>,
    R::Client: support::DepthClient<R>,
{
    for request in &session.sonic.requests {
        if request.binding.side != side {
            continue;
        }
        for domain in &request.domains {
            let owners = domain.invocation_owner.try_to_vec::<i64>()?;
            let Some(position) = owners.iter().position(|value| *value == owner) else {
                continue;
            };
            let client = session.sources.geometry.client();
            let rows = support::rows::<R>(client, &[i64::try_from(position)?])?;
            let query = hydraulic(checked::<R, _>(
                client,
                "quadrature owner cached-query snapshot",
                |a| domain.query.select(a, &rows),
            ))?;
            let mut fields = Vec::new();
            for value in [
                query.area.lower(),
                query.area.upper(),
                query.query.parameter.lower(),
                query.query.parameter.upper(),
                query.query.scale.high(),
                query.query.scale.low(),
                query.query.scale.abs_error(),
                query.above_start.lower(),
                query.above_start.upper(),
                query.below_end.lower(),
                query.below_end.upper(),
                &query.candidate,
            ] {
                fields.push(
                    value
                        .try_to_vec::<f64>()?
                        .into_iter()
                        .map(f64::to_bits)
                        .collect(),
                );
            }
            for value in [
                domain.depth.lower(),
                domain.depth.upper(),
                domain.normalized_depth.lower(),
                domain.normalized_depth.upper(),
                domain.numerical.epsilon().high(),
                domain.numerical.epsilon().low(),
                domain.numerical.epsilon().abs_error(),
                domain.numerical.source_physical().lower(),
                domain.numerical.source_physical().upper(),
            ] {
                fields.push(
                    value
                        .narrow(0, position, 1)?
                        .try_to_vec::<f64>()?
                        .into_iter()
                        .map(f64::to_bits)
                        .collect(),
                );
            }
            for value in [
                &domain.invocation_owner,
                &domain.parent_rows,
                domain.numerical.levels(),
                &domain.status,
            ] {
                fields.push(
                    value
                        .narrow(0, position, 1)?
                        .try_to_vec::<i64>()?
                        .into_iter()
                        .map(|value| value as u64)
                        .collect(),
                );
            }
            fields.push(vec![u64::from(domain.refinement.is_some())]);
            if let Some(native) = &domain.refinement {
                assert_eq!(
                    domain.numerical.levels().try_to_vec::<i64>()?,
                    native.levels.try_to_vec::<i64>()?
                );
                fields.push(
                    native
                        .levels
                        .narrow(0, position, 1)?
                        .try_to_vec::<i64>()?
                        .into_iter()
                        .map(|value| value as u64)
                        .collect(),
                );
                let chart = hydraulic(checked::<R, _>(
                    client,
                    "quadrature owner native-chart snapshot",
                    |a| native.chart.select(a, &rows),
                ))?;
                for value in [
                    chart.query.parameter.lower(),
                    chart.query.parameter.upper(),
                    chart.query.scale.high(),
                    chart.query.scale.low(),
                    chart.query.scale.abs_error(),
                ] {
                    fields.push(
                        value
                            .try_to_vec::<f64>()?
                            .into_iter()
                            .map(f64::to_bits)
                            .collect(),
                    );
                }
            }
            return Ok(fields);
        }
    }
    Err("Quadrature owner snapshot lacks the requested live sonic domain".into())
}

pub(super) fn levels<R>(
    session: &PointRootSession<'_, '_, R>,
    side: SonicSide,
) -> Result<Vec<(i64, i64, i64)>>
where
    R: Runtime<DType = DType>,
{
    let mut result = Vec::new();
    for request in &session.sonic.requests {
        if request.binding.side != side {
            continue;
        }
        for domain in &request.domains {
            let owners = domain.invocation_owner.try_to_vec::<i64>()?;
            let parents = domain.parent_rows.try_to_vec::<i64>()?;
            let levels = domain.numerical.levels().try_to_vec::<i64>()?;
            for ((owner, parent), level) in owners.into_iter().zip(parents).zip(levels) {
                result.push((owner, parent, level));
            }
        }
    }
    result.sort_unstable();
    Ok(result)
}
