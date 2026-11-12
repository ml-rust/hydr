use super::PrismaticRiemannConfig;
use super::error::{
    HydraulicError, HydraulicResult, PrismaticRiemannError, PrismaticRiemannRejection, checked,
    into_public,
};
use super::lane_quadrature::numerical_contributions;
use super::potential_difference_derivatives::{integrand, panel_fourth_bound};
use super::potential_difference_lanes::DifferenceLanes;
use super::potential_difference_test_data::{CASES, GRAVITY, PROFILES};
use super::potential_difference_test_support::{batch, contains, values};
use super::work::{PrismaticRiemannWork, QueryWork, WorkLedger};
use crate::channel::section::HydraulicGeometry;
use numr::dtype::DType;
use numr::ops::{
    CompareOps, HypotOps, MidpointOps, NextRepresentableOps, ScalarOps, TensorOps, TwoSumOps,
};
use numr::runtime::{Device, Runtime, RuntimeClient};
use numr::tensor::Tensor;
use solvr::common::arithmetic::ArithmeticError;
use solvr::integrate::{CertifiedSimpsonAlgorithms, CertifiedSimpsonOptions};
use std::cell::Cell;

trait Client<R: Runtime<DType = DType>>:
    TensorOps<R>
    + CompareOps<R>
    + HypotOps<R>
    + MidpointOps<R>
    + NextRepresentableOps<R>
    + ScalarOps<R>
    + TwoSumOps<R>
    + RuntimeClient<R>
    + CertifiedSimpsonAlgorithms<R>
{
}

impl<R, C> Client<R> for C
where
    R: Runtime<DType = DType>,
    C: TensorOps<R>
        + CompareOps<R>
        + HypotOps<R>
        + MidpointOps<R>
        + NextRepresentableOps<R>
        + ScalarOps<R>
        + TwoSumOps<R>
        + RuntimeClient<R>
        + CertifiedSimpsonAlgorithms<R>,
{
}

fn fixture<R>(client: &R::Client, strided: bool) -> HydraulicResult<DifferenceLanes<R>>
where
    R: Runtime<DType = DType>,
    R::Client: Client<R>,
{
    let first = &CASES[0];
    let second = &CASES[1];
    assert_eq!(first.profile, second.profile);
    let section = batch::<R>(client, &PROFILES[first.profile])?;
    let geometry = HydraulicGeometry::from_batch(&section);
    let owners = values::<R, _>(client, &[0_i64; 3], strided)?;
    let original = values::<R, _>(
        client,
        &[
            f64::from_bits(first.original),
            f64::from_bits(second.original),
            f64::from_bits(first.original),
        ],
        strided,
    )?;
    let lower = values::<R, _>(
        client,
        &[
            f64::from_bits(first.target[0]),
            f64::from_bits(second.target[0]),
            f64::from_bits(first.original),
        ],
        strided,
    )?;
    let upper = values::<R, _>(
        client,
        &[
            f64::from_bits(first.target[1]),
            f64::from_bits(second.target[1]),
            f64::from_bits(first.original),
        ],
        strided,
    )?;
    checked(client, "lane contribution hydraulic fixtures", |a| {
        let bands = geometry.gather(a, &owners)?;
        let target = a.bounds(&lower, &upper)?;
        let lanes = DifferenceLanes::build(a, &bands, &original, &target, f64::from_bits(GRAVITY))?;
        assert_eq!(lanes.band_count, 1);
        assert_eq!(lanes.active.try_to_vec::<f64>()?, [1.0, 1.0, 0.0]);
        Ok(lanes)
    })
}

fn options(config: &PrismaticRiemannConfig) -> CertifiedSimpsonOptions {
    CertifiedSimpsonOptions {
        panels: 4,
        max_points: config.max_native_query_points,
    }
}

fn contributions<R>(client: &R::Client) -> HydraulicResult<()>
where
    R: Runtime<DType = DType>,
    R::Client: Client<R>,
{
    let config = PrismaticRiemannConfig::default();
    for strided in [false, true] {
        let lanes = fixture::<R>(client, strided)?;
        let left = values::<R, _>(client, &[0.0; 3], strided)?;
        let right = lanes.active.clone();
        let work = WorkLedger::new(&config);
        work.charge(QueryWork::Residual, 3)?;
        let value_entries = Cell::new(0_usize);
        let derivative_entries = Cell::new(0_usize);
        let result = numerical_contributions(
            client,
            &left,
            &right,
            options(&config),
            &work,
            |a, owners, parameter| {
                value_entries.set(value_entries.get() + owners.numel());
                assert!(
                    owners
                        .try_to_vec::<i64>()?
                        .iter()
                        .all(|&owner| owner == 0 || owner == 1)
                );
                integrand(a, &lanes, owners, parameter)
            },
            |a, owners, parameter| {
                derivative_entries.set(derivative_entries.get() + owners.numel());
                assert!(
                    owners
                        .try_to_vec::<i64>()?
                        .iter()
                        .all(|&owner| owner == 0 || owner == 1)
                );
                panel_fourth_bound(a, &lanes, owners, parameter)
            },
        )?;
        assert!(value_entries.get() > 0 && derivative_entries.get() > 0);
        assert_eq!(value_entries.get() % 2, 0);
        assert_eq!(
            work.snapshot(),
            PrismaticRiemannWork {
                residual_points: 3,
                simpson_value_points: value_entries.get() / 2,
                simpson_derivative_points: derivative_entries.get(),
                ..Default::default()
            }
        );
        for value in [
            result.value(),
            result.lower(),
            result.upper(),
            result.abs_error_bound(),
            result.quadrature_error_bound(),
        ] {
            assert_eq!(value.shape(), &[3]);
            assert_eq!(value.dtype(), DType::F64);
            assert!(value.device().is_same(client.device()));
            assert_eq!(value.narrow(0, 2, 1)?.item::<f64>()?, 0.0);
        }
        checked(client, "lane contribution independent references", |a| {
            for (row, case) in CASES[..2].iter().enumerate() {
                let value = a.bounds(
                    &result.lower().narrow(0, row, 1)?,
                    &result.upper().narrow(0, row, 1)?,
                )?;
                contains(&value, case.reference, case.id, true)?;
            }
            Ok(())
        })?;
    }
    let empty = Tensor::zeros(&[0], DType::F64, client.device())?;
    let work = WorkLedger::new(&config);
    let callbacks = Cell::new(0_usize);
    let result = numerical_contributions::<R, _, _>(
        client,
        &empty,
        &empty,
        options(&config),
        &work,
        |a, _, parameter| {
            callbacks.set(callbacks.get() + 1);
            Ok(a.bounds(parameter.lower(), parameter.upper())?)
        },
        |_, owners, _| {
            callbacks.set(callbacks.get() + 1);
            Ok(Tensor::zeros(owners.shape(), DType::F64, client.device())?)
        },
    )?;
    assert_eq!(result.lower().shape(), &[0]);
    assert_eq!(result.upper().shape(), &[0]);
    assert_eq!(callbacks.get(), 0);
    assert_eq!(work.snapshot(), PrismaticRiemannWork::default());
    Ok(())
}

fn typed_causes<R>(client: &R::Client) -> HydraulicResult<()>
where
    R: Runtime<DType = DType>,
    R::Client: Client<R>,
{
    let config = PrismaticRiemannConfig::default();
    let lanes = fixture::<R>(client, false)?;
    let left = Tensor::zeros(&[3], DType::F64, client.device())?;
    let right = &lanes.active;
    let work = WorkLedger::new(&config);
    let result = numerical_contributions(
        client,
        &left,
        right,
        options(&config),
        &work,
        |_, _, _| {
            Err(HydraulicError::Rejected {
                reason: PrismaticRiemannRejection::GeometryUnresolved,
                query: Some(1),
                section_owner: Some(0),
                context: "hydraulic lane value support".into(),
            })
        },
        |a, owners, parameter| panel_fourth_bound(a, &lanes, owners, parameter),
    );
    assert!(
        matches!(result, Err(HydraulicError::Rejected { reason: PrismaticRiemannRejection::GeometryUnresolved, query: Some(1), section_owner: Some(0), ref context }) if context == "hydraulic lane value support")
    );
    assert!(work.snapshot().simpson_value_points > 0);

    let work = WorkLedger::new(&config);
    let result = numerical_contributions(
        client,
        &left,
        right,
        options(&config),
        &work,
        |a, owners, parameter| integrand(a, &lanes, owners, parameter),
        |_, _, _| {
            Err(HydraulicError::Runtime(numr::error::Error::Backend(
                "hydraulic lane derivative runtime".into(),
            )))
        },
    );
    assert!(
        matches!(result, Err(HydraulicError::Runtime(numr::error::Error::Backend(ref context))) if context == "hydraulic lane derivative runtime")
    );
    assert!(work.snapshot().simpson_derivative_points > 0);

    let work = WorkLedger::new(&config);
    let result = numerical_contributions(
        client,
        &left,
        right,
        options(&config),
        &work,
        |a, owners, _| {
            let one = Tensor::ones(owners.shape(), DType::F64, client.device())?;
            let zero = Tensor::zeros(owners.shape(), DType::F64, client.device())?;
            let _invalid = a.bounds(&one, &zero)?;
            Err(HydraulicError::Rejected {
                reason: PrismaticRiemannRejection::GeometryUnresolved,
                query: None,
                section_owner: None,
                context: "hydraulic lane invalid bounds".into(),
            })
        },
        |a, owners, parameter| panel_fourth_bound(a, &lanes, owners, parameter),
    );
    let actual = result.as_ref().err();
    // Inverted bounds accumulate device status; checked completion returns the exact arithmetic stage.
    assert!(
        matches!(
            actual,
            Some(HydraulicError::Arithmetic(ArithmeticError::Range {
                stage: "surveyed Simpson values"
            }))
        ),
        "deferred invalid bounds must precede the retained geometry rejection: {actual:?}"
    );

    let work = WorkLedger::new(&config);
    work.charge(QueryWork::Residual, config.max_native_query_points)?;
    let before = work.snapshot();
    let callbacks = Cell::new(0_usize);
    let result = numerical_contributions(
        client,
        &left,
        right,
        options(&config),
        &work,
        |a, owners, parameter| {
            callbacks.set(callbacks.get() + 1);
            integrand(a, &lanes, owners, parameter)
        },
        |a, owners, parameter| {
            callbacks.set(callbacks.get() + 1);
            panel_fourth_bound(a, &lanes, owners, parameter)
        },
    );
    assert!(matches!(
        result,
        Err(HydraulicError::Rejected {
            reason: PrismaticRiemannRejection::WorkLimit,
            ..
        })
    ));
    assert_eq!(callbacks.get(), 0);
    assert_eq!(work.snapshot(), before);
    Ok(())
}

fn matrix<R>(client: R::Client) -> HydraulicResult<()>
where
    R: Runtime<DType = DType>,
    R::Client: Client<R>,
{
    contributions::<R>(&client)?;
    typed_causes::<R>(&client)
}

#[test]
fn configured_cpu_lane_quadrature_preserves_contributions_work_and_causes()
-> Result<(), PrismaticRiemannError> {
    use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime, ParallelismConfig};
    let original = CpuClient::new(CpuDevice::new());
    let before = original.parallelism();
    let config = ParallelismConfig::new(Some(2), Some(257));
    let client = original.with_parallelism(config);
    let retained = client.clone();
    matrix::<CpuRuntime>(client).map_err(|error| into_public(error, Default::default()))?;
    assert_eq!(retained.parallelism(), config);
    assert_eq!(original.parallelism(), before);
    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_lane_quadrature_preserves_contributions_work_and_causes()
-> Result<(), Box<dyn std::error::Error>> {
    use numr::runtime::cuda::{CudaClient, CudaDevice, CudaRuntime};
    matrix::<CudaRuntime>(CudaClient::new(CudaDevice::new(0))?)
        .map_err(|error| into_public(error, Default::default()))?;
    Ok(())
}
