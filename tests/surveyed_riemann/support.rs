use super::data::{Case, PROFILES};
use hydr::channel::{
    CrossSectionBatch, CrossSectionDefinition, CrossSectionPoint, PrismaticRiemannConfig,
    PrismaticRiemannResult, PrismaticRiemannWork, PrismaticSectionBatch,
};
use numr::dtype::DType;
use numr::ops::{
    CompareOps, HypotOps, MidpointOps, NextRepresentableOps, ScalarOps, TensorOps, TwoSumOps,
};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;
use solvr::common::arithmetic::Interval;
use solvr::integrate::CertifiedSimpsonAlgorithms;
use solvr::optimize::roots::{CertifiedBisectionAlgorithms, CertifiedUpperBracketAlgorithms};

pub(super) type Result<T = ()> = std::result::Result<T, Box<dyn std::error::Error>>;

pub(super) trait Client<R: Runtime<DType = DType>>:
    TensorOps<R>
    + CompareOps<R>
    + NextRepresentableOps<R>
    + ScalarOps<R>
    + TwoSumOps<R>
    + HypotOps<R>
    + MidpointOps<R>
    + RuntimeClient<R>
    + CertifiedSimpsonAlgorithms<R>
    + CertifiedBisectionAlgorithms<R>
    + CertifiedUpperBracketAlgorithms<R>
{
}

impl<R, C> Client<R> for C
where
    R: Runtime<DType = DType>,
    C: TensorOps<R>
        + CompareOps<R>
        + NextRepresentableOps<R>
        + ScalarOps<R>
        + TwoSumOps<R>
        + HypotOps<R>
        + MidpointOps<R>
        + RuntimeClient<R>
        + CertifiedSimpsonAlgorithms<R>
        + CertifiedBisectionAlgorithms<R>
        + CertifiedUpperBracketAlgorithms<R>,
{
}

pub(super) fn batch<R>(client: &R::Client) -> Result<PrismaticSectionBatch<R>>
where
    R: Runtime<DType = DType>,
    R::Client: Client<R>,
{
    let definitions = PROFILES
        .iter()
        .map(|profile| CrossSectionDefinition {
            id: profile.id.into(),
            vertical_datum: "Frozen original survey datum".into(),
            points: profile
                .points
                .iter()
                .map(|p| CrossSectionPoint {
                    station_m: f64::from_bits(p[0]),
                    elevation_m: f64::from_bits(p[1]),
                })
                .collect(),
            maximum_stage_m: f64::from_bits(profile.cap),
            profile_provenance: "Independent original-bit Arb/acb reference".into(),
        })
        .collect();
    Ok(CrossSectionBatch::<R>::new(client.clone(), definitions)?.certify_prismatic()?)
}

pub(super) fn config(case: &Case) -> PrismaticRiemannConfig {
    let mut config = PrismaticRiemannConfig {
        gravity_m_per_s2: f64::from_bits(case.gravity),
        ..PrismaticRiemannConfig::default()
    };
    if case.loose_recovery {
        config.recovery.depth_atol_m = 1.0;
        config.recovery.depth_rtol = 1.0;
    }
    if case.zero_work {
        config.max_native_query_points = 0;
    }
    config
}

pub(super) fn values<R, T>(client: &R::Client, input: &[T], strided: bool) -> Result<Tensor<R>>
where
    R: Runtime<DType = DType>,
    T: numr::dtype::Element + Copy,
{
    let n = input.len();
    if strided && n != 0 {
        let sentinel = T::from_f64(if T::DTYPE == DType::I64 {
            -1.0
        } else {
            f64::NAN
        });
        let storage: Vec<_> = input.iter().flat_map(|&v| [sentinel, v]).collect();
        let tensor = Tensor::<R>::from_slice(&storage, &[n, 2], client.device())?
            .narrow(1, 1, 1)?
            .squeeze(Some(1));
        assert!(tensor.offset() > 0);
        if n > 1 {
            assert!(!tensor.is_contiguous());
        }
        Ok(tensor)
    } else {
        Ok(Tensor::<R>::from_slice(input, &[n], client.device())?)
    }
}

pub(super) fn inputs<R>(
    client: &R::Client,
    cases: &[&Case],
    strided: bool,
) -> Result<[Tensor<R>; 5]>
where
    R: Runtime<DType = DType>,
{
    let owners = cases
        .iter()
        .map(|c| i64::try_from(c.profile))
        .collect::<std::result::Result<Vec<_>, _>>()?;
    let mut result = vec![values::<R, _>(client, &owners, strided)?];
    for field in 0..4 {
        let field: Vec<_> = cases
            .iter()
            .map(|c| f64::from_bits(c.input[field]))
            .collect();
        result.push(values::<R, _>(client, &field, strided)?);
    }
    result
        .try_into()
        .map_err(|_| "Expected five input tensors".into())
}

pub(super) fn bounds<R: Runtime<DType = DType>>(
    interval: &Interval<R>,
    n: usize,
) -> Result<Vec<[f64; 2]>> {
    for endpoint in [interval.lower(), interval.upper()] {
        assert_eq!(endpoint.shape(), &[n]);
        assert_eq!(endpoint.dtype(), DType::F64);
    }
    let lower = interval.lower().try_to_vec::<f64>()?;
    let upper = interval.upper().try_to_vec::<f64>()?;
    Ok(lower
        .into_iter()
        .zip(upper)
        .map(|(lo, hi)| {
            assert!(
                lo.is_finite() && hi.is_finite() && lo <= hi,
                "Invalid enclosure [{lo}, {hi}]"
            );
            [lo, hi]
        })
        .collect())
}

pub(super) fn contains(actual: [f64; 2], expected: [u64; 2], label: &str) {
    let expected = expected.map(f64::from_bits);
    assert!(
        actual[0] <= expected[0] && actual[1] >= expected[1],
        "{label}: {actual:?} excludes {expected:?}"
    );
}

pub(super) fn work(work: PrismaticRiemannWork, config: &PrismaticRiemannConfig) -> Result {
    let total = work
        .residual_points
        .checked_add(work.simpson_value_points)
        .and_then(|n| n.checked_add(work.simpson_derivative_points))
        .ok_or("Work counter sum overflows usize. Check the returned callback counters")?;
    assert!(total <= config.max_native_query_points);
    assert!(work.refinements <= config.max_refinements);
    assert!(work.maximum_simpson_panels <= config.max_simpson_panels);
    assert!(work.maximum_simpson_panels == 0 || work.maximum_simpson_panels.is_power_of_two());
    if work.root_solves == 0 {
        assert_eq!(work.root_iterations, 0);
    }
    if work.root_solves == 0 && work.bracket_solves == 0 {
        assert_eq!(work.residual_points, 0);
    }
    assert!(work.root_iterations <= work.root_solves.saturating_mul(config.root.max_iterations));
    assert!(
        work.bracket_expansion_rounds
            <= work
                .bracket_solves
                .saturating_mul(config.max_bracket_expansions)
    );
    assert!(work.maximum_bracket_expansion_rounds <= config.max_bracket_expansions);
    assert!(work.maximum_bracket_expansion_rounds <= work.bracket_expansion_rounds);
    Ok(())
}

pub(super) fn snapshots<R: Runtime<DType = DType>>(
    result: &PrismaticRiemannResult<R>,
    cases: &[&Case],
) -> Result {
    assert_eq!(result.owners().shape(), &[cases.len()]);
    assert_eq!(result.owners().dtype(), DType::I64);
    let owners: Vec<_> = cases
        .iter()
        .map(|c| i64::try_from(c.profile))
        .collect::<std::result::Result<_, _>>()?;
    assert_eq!(result.owners().try_to_vec::<i64>()?, owners);
    for (field, tensor) in [
        result.left_area_m2(),
        result.left_discharge_m3_per_s(),
        result.right_area_m2(),
        result.right_discharge_m3_per_s(),
    ]
    .into_iter()
    .enumerate()
    {
        assert_eq!(tensor.shape(), &[cases.len()]);
        assert_eq!(tensor.dtype(), DType::F64);
        let bits: Vec<_> = tensor
            .try_to_vec::<f64>()?
            .into_iter()
            .map(f64::to_bits)
            .collect();
        assert_eq!(
            bits,
            cases.iter().map(|c| c.input[field]).collect::<Vec<_>>()
        );
    }
    Ok(())
}

pub(super) fn cpu() -> numr::runtime::cpu::CpuClient {
    use numr::runtime::cpu::{CpuClient, CpuDevice, ParallelismConfig};
    let original = CpuClient::new(CpuDevice::new());
    let before = original.parallelism();
    let configured = original.with_parallelism(ParallelismConfig::new(Some(2), Some(257)));
    assert_eq!(original.parallelism(), before);
    configured
}
