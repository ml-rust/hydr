use super::dry_vacuum_types::{DryVacuumEdges, DryVacuumPossibilities, DryVacuumRegime};
use crate::channel::flow::surveyed::error::{Arithmetic, HydraulicResult};
use numr::{
    dtype::DType,
    ops::{
        BinaryOps, CompareOps, ConditionalOps, NextRepresentableOps, ScalarOps, TensorOps,
        TwoSumOps,
    },
    runtime::{Runtime, RuntimeClient},
    tensor::Tensor,
};

pub(super) fn evaluate<R>(
    a: &mut Arithmetic<'_, R>,
    regime: &Tensor<R>,
    edges: &[DryVacuumEdges<R>; 2],
) -> HydraulicResult<DryVacuumPossibilities<R>>
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
    let zero = Tensor::zeros(regime.shape(), a.dtype(), c.device())?;
    let one = Tensor::full_scalar(regime.shape(), a.dtype(), 1., c.device())?;
    let is = |value: DryVacuumRegime| -> HydraulicResult<Tensor<R>> {
        Ok(c.where_cond(
            &c.eq(
                regime,
                &Tensor::full_scalar(&[], DType::I64, value as i64 as f64, c.device())?,
            )?,
            &one,
            &zero,
        )?)
    };
    let both = c.maximum(
        &is(DryVacuumRegime::Vacuum)?,
        &is(DryVacuumRegime::Touching)?,
    )?;
    let left = c.maximum(&is(DryVacuumRegime::LeftWet)?, &both)?;
    let right = c.maximum(&is(DryVacuumRegime::RightWet)?, &both)?;
    // Missing endpoints preserve alternatives and keep their required fan certificate pending.
    let possible = |value: &Tensor<R>, available: &Tensor<R>| -> HydraulicResult<Tensor<R>> {
        Ok(c.where_cond(available, value, &one)?)
    };
    let l = &edges[0];
    let r = &edges[1];
    let le = c.mul(
        &left,
        &possible(&c.ge(l.head.physical.upper(), &zero)?, &l.head.available)?,
    )?;
    let re = c.mul(
        &right,
        &possible(&c.le(r.head.physical.lower(), &zero)?, &r.head.available)?,
    )?;
    let lf = c.mul(
        &left,
        &c.mul(
            &possible(&c.le(l.head.physical.lower(), &zero)?, &l.head.available)?,
            &possible(&c.ge(l.front.physical.upper(), &zero)?, &l.front.available)?,
        )?,
    )?;
    let rf = c.mul(
        &right,
        &c.mul(
            &possible(&c.le(r.front.physical.lower(), &zero)?, &r.front.available)?,
            &possible(&c.ge(r.head.physical.upper(), &zero)?, &r.head.available)?,
        )?,
    )?;
    let ld = possible(&c.le(l.front.physical.lower(), &zero)?, &l.front.available)?;
    let rd = possible(&c.ge(r.front.physical.upper(), &zero)?, &r.front.available)?;
    let dry = c.maximum(
        &is(DryVacuumRegime::BothDry)?,
        &c.maximum(
            &c.mul(&is(DryVacuumRegime::LeftWet)?, &ld)?,
            &c.maximum(
                &c.mul(&is(DryVacuumRegime::RightWet)?, &rd)?,
                &c.mul(&both, &c.mul(&ld, &rd)?)?,
            )?,
        )?,
    )?;
    Ok(DryVacuumPossibilities {
        left_exterior: le,
        left_fan: lf,
        dry,
        right_fan: rf,
        right_exterior: re,
    })
}

#[cfg(test)]
mod tests {
    use crate::channel::flow::surveyed_points::{
        chart_fixture,
        dry_vacuum_types::{DryVacuumCandidate, DryVacuumRegime, DryVacuumStatus},
        point_root_session::PointRootSession,
        source_fixture_support as frozen,
        wet_depth::fixture as support,
    };
    use frozen::{Result, hydraulic};

    #[test]
    fn configured_cpu_dry_vacuum_positions_keep_moving_contacts_and_uncertain_fronts() -> Result {
        use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime, ParallelismConfig};
        let config = ParallelismConfig::new(Some(2), Some(257));
        let client = CpuClient::new(CpuDevice::new()).with_parallelism(config);
        controls::run::<CpuRuntime>(&client)
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn actual_cuda_dry_vacuum_positions_keep_moving_contacts_and_uncertain_fronts() -> Result {
        use numr::runtime::cuda::{CudaClient, CudaDevice, CudaRuntime};
        controls::run::<CudaRuntime>(&CudaClient::new(CudaDevice::new(0))?)
    }

    mod controls {
        use super::{
            DryVacuumCandidate, DryVacuumRegime, DryVacuumStatus, PointRootSession, Result,
            chart_fixture, frozen, hydraulic, support,
        };
        use crate::channel::flow::surveyed_points::source_sonic_types::{
            SonicSide, SourceSonicStatus,
        };
        use numr::{dtype::DType, runtime::Runtime};

        pub(super) fn run<R>(client: &R::Client) -> Result
        where
            R: Runtime<DType = DType>,
            R::Client: support::DepthClient<R>,
        {
            moving_contacts::<R>(client)?;
            uncertain_front::<R>(client)
        }

        fn moving_contacts<R>(client: &R::Client) -> Result
        where
            R: Runtime<DType = DType>,
            R::Client: support::DepthClient<R>,
        {
            let config = chart_fixture::config();
            let states = [
                [1., 2., 0., 0.],
                [0., 0., 1., -2.],
                [1., 1., 0., 0.],
                [0., 0., 1., -1.],
                [1., -2., 0., 0.],
                [0., 0., 1., 2.],
                [1., -1., 1., 3.],
                [1., -3., 1., 1.],
            ];
            support::with_rectangle::<R, _>(client, 1., 4., &states, &config, |pair, _| {
                let parents = support::rows::<R>(client, &[0, 1, 2, 3, 4, 5, 6, 7])?;
                let mut session = hydraulic(PointRootSession::new(
                    pair,
                    &parents,
                    config.root.max_points,
                ))?;
                for owner in 0..8 {
                    let owners = support::rows::<R>(client, &[owner])?;
                    let result = hydraulic(session.certify_dry_vacuum(&owners))?;
                    hydraulic(session.invocation.check(&result.invocation_witness))?;
                    assert_eq!(
                        result.status.try_to_vec::<i64>()?,
                        [DryVacuumStatus::Complete as i64],
                        "owner {owner}"
                    );
                    assert_eq!(result.callback_points, session.callback_points());
                    assert_eq!(result.point_limit, config.root.max_points);
                    let p = &result.possibilities;
                    let masks = [
                        &p.left_exterior,
                        &p.left_fan,
                        &p.dry,
                        &p.right_fan,
                        &p.right_exterior,
                    ]
                    .map(|m| m.try_to_vec::<f64>())
                    .into_iter()
                    .collect::<std::result::Result<Vec<_>, _>>()?;
                    let expected = match owner {
                        0 => [1., 0., 0., 0., 0.],
                        1 => [0., 0., 0., 0., 1.],
                        2 => [1., 1., 0., 0., 0.],
                        3 => [0., 0., 0., 1., 1.],
                        4 => [0., 1., 1., 0., 0.],
                        5 => [0., 0., 1., 1., 0.],
                        6 => [0., 1., 0., 0., 0.],
                        7 => [0., 0., 0., 1., 0.],
                        _ => return Err("Unexpected moving control owner".into()),
                    };
                    for (actual, expected) in masks.iter().zip(expected) {
                        assert_eq!(actual, &[expected], "owner {owner}");
                    }
                    for candidate in &result.candidates {
                        match candidate {
                            DryVacuumCandidate::Exterior(exterior) => {
                                let source = states[usize::try_from(owner)?];
                                let i = exterior.side.index() * 2;
                                assert_eq!(exterior.invocation_owner.try_to_vec::<i64>()?, [owner]);
                                assert_eq!(exterior.parent_rows.try_to_vec::<i64>()?, [owner]);
                                assert_eq!(
                                    frozen::components(&exterior.area)?,
                                    [[source[i].to_bits(), 0, 0]]
                                );
                                assert_eq!(
                                    frozen::components(&exterior.discharge)?,
                                    [[source[i + 1].to_bits(), 0, 0]]
                                );
                                assert_eq!(
                                    frozen::components(&exterior.velocity)?,
                                    [[source[i + 1].to_bits(), 0, 0]]
                                );
                                assert_eq!(frozen::endpoints(&exterior.query.area)?, [[1., 1.]]);
                                let depth = frozen::endpoints(exterior.recovery.depth_m())?[0];
                                assert!(depth[0] <= 1. && 1. <= depth[1]);
                            }
                            DryVacuumCandidate::Dry(dry) => {
                                assert_eq!(dry.invocation_owner.try_to_vec::<i64>()?, [owner]);
                                assert_eq!(dry.parent_rows.try_to_vec::<i64>()?, [owner]);
                                assert_eq!(frozen::endpoints(&dry.query.area)?, [[0., 0.]]);
                            }
                            DryVacuumCandidate::Fan(fan) => {
                                assert!(owner == 6 || owner == 7);
                                let expected =
                                    [0x3fbc71c71c71c71c, 0x3fbc71c71c71c71d].map(f64::from_bits);
                                let area = frozen::endpoints(&fan.query.area)?[0];
                                assert!(area[0] <= expected[0] && expected[1] <= area[1]);
                            }
                            DryVacuumCandidate::PendingFan(_) => {
                                return Err("Exact moving control leaves pending fan".into());
                            }
                        }
                    }
                    if owner >= 6 {
                        assert_eq!(
                            result.regime.try_to_vec::<i64>()?,
                            [DryVacuumRegime::Touching as i64]
                        );
                        assert_eq!(frozen::components(&result.invariant.retained)?, [[0; 3]]);
                        assert_eq!(result.invariant.available.try_to_vec::<f64>()?, [1.]);
                        let speed = if owner == 6 { 1. } else { -1. };
                        for edge in &result.edges {
                            let front = frozen::endpoints(&edge.front.physical)?[0];
                            assert!(front[0] <= speed && speed <= front[1]);
                            assert_eq!(edge.front_exact_zero.try_to_vec::<f64>()?, [0.]);
                        }
                    }
                    if owner == 2 || owner == 3 {
                        let side = if owner == 2 { 0 } else { 1 };
                        assert_eq!(
                            result.edges[side].head_exact_zero.try_to_vec::<f64>()?,
                            [1.]
                        );
                        assert_eq!(
                            result.sonic_status[side].try_to_vec::<i64>()?,
                            [SourceSonicStatus::SourceContact as i64]
                        );
                        assert!(
                            !result
                                .candidates
                                .iter()
                                .any(|v| matches!(v, DryVacuumCandidate::Fan(_)))
                        );
                    }
                    if owner == 4 || owner == 5 {
                        let side = if owner == 4 { 0 } else { 1 };
                        assert_eq!(
                            result.edges[side].front_exact_zero.try_to_vec::<f64>()?,
                            [1.]
                        );
                        assert_eq!(
                            result.sonic_status[side].try_to_vec::<i64>()?,
                            [SourceSonicStatus::DryContact as i64]
                        );
                    }
                }
                Ok(())
            })
        }
        fn uncertain_front<R>(client: &R::Client) -> Result
        where
            R: Runtime<DType = DType>,
            R::Client: support::DepthClient<R>,
        {
            let config = chart_fixture::config();
            let section = chart_fixture::rectangle::<R>(client, 3., 4.)?;
            let owners = support::rows::<R>(client, &[0])?;
            // Exact Q=-6+2^-1074 gives a positive, unrepresentable epsilon=2^-1074/3.
            let input = [
                [3.0_f64.to_bits(), 0, 0],
                [(-6.0_f64).to_bits(), 1, 0],
                [0, 0, 0],
                [0, 0, 0],
            ]
            .iter()
            .map(|v| frozen::expansion::<R>(client, &[*v], true))
            .collect::<Result<Vec<_>>>()?;
            chart_fixture::with_inputs(client, section, owners, input, &config, |pair, _| {
                let owners = support::rows::<R>(client, &[0])?;
                let mut session =
                    hydraulic(PointRootSession::new(pair, &owners, config.root.max_points))?;
                let result = hydraulic(session.certify_dry_vacuum(&owners))?;
                assert_eq!(
                    result.regime.try_to_vec::<i64>()?,
                    [DryVacuumRegime::LeftWet as i64]
                );
                assert_eq!(
                    result.status.try_to_vec::<i64>()?,
                    [DryVacuumStatus::Pending as i64]
                );
                let front = frozen::endpoints(&result.edges[0].front.physical)?[0];
                assert!(front[0] <= 0. && front[1] >= 0.);
                assert_eq!(result.edges[0].front_exact_zero.try_to_vec::<f64>()?, [0.]);
                assert_eq!(result.possibilities.left_fan.try_to_vec::<f64>()?, [1.]);
                assert_eq!(result.possibilities.dry.try_to_vec::<f64>()?, [1.]);
                let pending = result
                    .candidates
                    .iter()
                    .find_map(|v| {
                        if let DryVacuumCandidate::PendingFan(p) = v {
                            Some(p)
                        } else {
                            None
                        }
                    })
                    .ok_or("Uncertain front loses pending fan")?;
                assert_eq!(pending.side, SonicSide::Left);
                assert_eq!(pending.invocation_owner.try_to_vec::<i64>()?, [0]);
                assert_eq!(pending.parent_rows.try_to_vec::<i64>()?, [0]);
                assert_eq!(
                    pending.source_status.try_to_vec::<i64>()?,
                    [SourceSonicStatus::Unresolved as i64]
                );
                Ok(())
            })
        }
    }
}
