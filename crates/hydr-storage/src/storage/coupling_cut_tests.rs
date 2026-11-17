use super::*;
use crate::storage::{
    LevelPoolConfig, LevelPoolCurve, LevelPoolElement, LevelPoolKnot, LevelPoolState,
};
use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime, ParallelismConfig};

type TestResult = Result<(), Box<dyn std::error::Error>>;

fn batch(
    volumes: &[f64],
    stages: &[f64],
    current: f64,
) -> Result<LevelPoolBatch<CpuRuntime>, LevelPoolError> {
    let client = CpuClient::new(CpuDevice::new())
        .with_parallelism(ParallelismConfig::new(Some(1), Some(64)));
    let curve = LevelPoolCurve::new(
        volumes
            .iter()
            .zip(stages)
            .map(|(&volume_m3, &stage_m)| LevelPoolKnot { volume_m3, stage_m })
            .collect(),
        "analytical datum".into(),
        "exact affine storage segments".into(),
    )?;
    LevelPoolBatch::new(
        client,
        vec![LevelPoolElement {
            id: "pool".into(),
            curve,
            connected_pool_assertion: "one connected level surface".into(),
            storage_support_identity: "finite original storage".into(),
        }],
        vec![LevelPoolState { volume_m3: current }],
        0.0,
        LevelPoolConfig::default(),
    )
}

#[test]
fn cuts_retain_knots_and_enclose_affine_inverse_in_requested_order() -> TestResult {
    let batch = batch(&[0.0, 3.0, 11.0], &[-2.0, 0.0, 4.0], 8.0)?;
    let plan = batch.bind_coupling_cuts(&[
        LevelPoolCut {
            pool_index: 0,
            elevation_m: 1.5,
        },
        LevelPoolCut {
            pool_index: 0,
            elevation_m: 0.0,
        },
        LevelPoolCut {
            pool_index: 0,
            elevation_m: -2.0,
        },
        LevelPoolCut {
            pool_index: 0,
            elevation_m: 4.0,
        },
    ])?;
    let nominal = plan.nominal_volume_m3().try_to_vec::<f64>()?;
    let lower = plan.lower_volume_m3().try_to_vec::<f64>()?;
    let upper = plan.upper_volume_m3().try_to_vec::<f64>()?;
    assert_eq!(nominal, vec![6.0, 3.0, 0.0, 11.0]);
    assert!(lower[0] <= 6.0 && upper[0] >= 6.0);
    for index in 1..4 {
        assert_eq!(lower[index].to_bits(), nominal[index].to_bits());
        assert_eq!(upper[index].to_bits(), nominal[index].to_bits());
    }
    let available = batch
        .coupling_cut_availability(&plan)?
        .available_lower_m3
        .try_to_vec::<f64>()?;
    assert!(available[0] > 0.0 && available[0] <= 2.0);
    assert_eq!(&available[1..], &[5.0, 8.0, 0.0]);
    Ok(())
}

#[test]
fn availability_rounds_subtraction_down_without_water_tolerance_credit() -> TestResult {
    let small = 2.0f64.powi(-54);
    let batch = batch(&[0.0, small, 2.0], &[0.0, 1.0, 2.0], 1.0)?;
    let plan = batch.bind_coupling_cuts(&[LevelPoolCut {
        pool_index: 0,
        elevation_m: 1.0,
    }])?;
    assert_eq!(
        plan.upper_volume_m3().try_to_vec::<f64>()?[0].to_bits(),
        small.to_bits()
    );
    let available = batch
        .coupling_cut_availability(&plan)?
        .available_lower_m3
        .try_to_vec::<f64>()?[0];
    assert!(available > 0.0 && available < 1.0);
    Ok(())
}

#[test]
fn cuts_reject_foreign_models_and_unsupported_inputs() -> TestResult {
    let original = batch(&[0.0, 4.0], &[0.0, 2.0], 2.0)?;
    let foreign = batch(&[0.0, 4.0], &[0.0, 2.0], 2.0)?;
    let plan = original.bind_coupling_cuts(&[LevelPoolCut {
        pool_index: 0,
        elevation_m: 1.0,
    }])?;
    assert!(matches!(
        foreign.coupling_cut_availability(&plan),
        Err(LevelPoolError::InvalidInput { .. })
    ));
    for elevation_m in [-1.0, 3.0, f64::NAN, f64::INFINITY] {
        assert!(matches!(
            original.bind_coupling_cuts(&[LevelPoolCut {
                pool_index: 0,
                elevation_m
            }]),
            Err(LevelPoolError::InvalidInput { .. })
        ));
    }
    assert!(matches!(
        original.bind_coupling_cuts(&[LevelPoolCut {
            pool_index: 1,
            elevation_m: 1.0
        }]),
        Err(LevelPoolError::InvalidInput { .. })
    ));
    assert!(original.clone().coupling_cut_availability(&plan).is_ok());
    Ok(())
}

#[test]
fn empty_cuts_preserve_empty_native_support() -> TestResult {
    let batch = batch(&[0.0, 4.0], &[0.0, 2.0], 2.0)?;
    let plan = batch.bind_coupling_cuts(&[])?;
    assert_eq!(plan.upper_volume_m3().shape(), &[0]);
    assert_eq!(
        batch
            .coupling_cut_availability(&plan)?
            .available_lower_m3
            .shape(),
        &[0]
    );
    Ok(())
}

#[test]
fn zero_certified_availability_grants_no_positive_water() -> TestResult {
    let batch = batch(&[0.0, 3.0, 11.0], &[-2.0, 0.0, 4.0], 6.0)?;
    let plan = batch.bind_coupling_cuts(&[LevelPoolCut {
        pool_index: 0,
        elevation_m: 1.5,
    }])?;
    assert_eq!(
        batch
            .coupling_cut_availability(&plan)?
            .available_lower_m3
            .try_to_vec::<f64>()?,
        vec![0.0]
    );
    Ok(())
}

#[test]
fn interleaved_cuts_preserve_pool_order_across_repeated_curve_groups() -> TestResult {
    let first = batch(&[0.0, 3.0, 11.0], &[-2.0, 0.0, 4.0], 8.0)?;
    let second = batch(&[0.0, 8.0], &[0.0, 4.0], 6.0)?;
    let mut elements = vec![
        first.elements()[0].clone(),
        second.elements()[0].clone(),
        first.elements()[0].clone(),
    ];
    for (index, element) in elements.iter_mut().enumerate() {
        element.id = format!("pool-{index}");
        element.storage_support_identity = format!("finite-storage-{index}");
    }
    let client = CpuClient::new(CpuDevice::new())
        .with_parallelism(ParallelismConfig::new(Some(1), Some(64)));
    let batch = LevelPoolBatch::<CpuRuntime>::new(
        client,
        elements,
        vec![
            LevelPoolState { volume_m3: 8.0 },
            LevelPoolState { volume_m3: 6.0 },
            LevelPoolState { volume_m3: 10.0 },
        ],
        0.0,
        LevelPoolConfig::default(),
    )?;
    let cuts = [
        LevelPoolCut {
            pool_index: 1,
            elevation_m: 1.5,
        },
        LevelPoolCut {
            pool_index: 2,
            elevation_m: 0.0,
        },
        LevelPoolCut {
            pool_index: 0,
            elevation_m: 1.5,
        },
        LevelPoolCut {
            pool_index: 1,
            elevation_m: 4.0,
        },
        LevelPoolCut {
            pool_index: 2,
            elevation_m: 2.5,
        },
        LevelPoolCut {
            pool_index: 0,
            elevation_m: -2.0,
        },
    ];
    let plan = batch.bind_coupling_cuts(&cuts)?;
    let expected = [3.0, 3.0, 6.0, 8.0, 8.0, 0.0];
    let lower = plan.lower_volume_m3().try_to_vec::<f64>()?;
    let upper = plan.upper_volume_m3().try_to_vec::<f64>()?;
    assert_eq!(plan.nominal_volume_m3().try_to_vec::<f64>()?, expected);
    for ((lower, upper), expected) in lower.iter().zip(&upper).zip(expected) {
        assert!(*lower <= expected && *upper >= expected);
    }
    let available = batch
        .coupling_cut_availability(&plan)?
        .available_lower_m3
        .try_to_vec::<f64>()?;
    for (value, ceiling) in available.iter().zip([3.0, 7.0, 2.0, 0.0, 2.0, 8.0]) {
        assert!(*value >= 0.0 && *value <= ceiling);
        if ceiling > 0.0 {
            assert!(*value > 0.0);
        }
    }
    assert_eq!(available[1], 7.0);
    assert_eq!(available[3], 0.0);
    assert_eq!(available[5], 8.0);
    Ok(())
}

#[test]
fn nonrepresentable_affine_inverse_contains_adjacent_binary64_bounds() -> TestResult {
    let batch = batch(&[0.0, 3.5], &[0.125, 0.875], 3.5)?;
    let plan = batch.bind_coupling_cuts(&[LevelPoolCut {
        pool_index: 0,
        elevation_m: 0.375,
    }])?;
    // Exact V = 7/6 lies strictly between these adjacent binary64 operands.
    let exact_lower = f64::from_bits(0x3ff2_aaaa_aaaa_aaaa);
    let exact_upper = f64::from_bits(0x3ff2_aaaa_aaaa_aaab);
    let lower = plan.lower_volume_m3().try_to_vec::<f64>()?[0];
    let upper = plan.upper_volume_m3().try_to_vec::<f64>()?[0];
    assert!(lower <= exact_lower);
    assert!(upper >= exact_upper);
    let available = batch
        .coupling_cut_availability(&plan)?
        .available_lower_m3
        .try_to_vec::<f64>()?[0];
    // Exact eligibility = 7/3 exceeds this operand and is below its successor.
    let eligibility_lower = f64::from_bits(0x4002_aaaa_aaaa_aaaa);
    assert!(available > 0.0 && available <= eligibility_lower);
    Ok(())
}

#[test]
fn subnormal_equal_scale_cut_encloses_exact_unit_slope_volume() -> TestResult {
    let top = 1.0e-308;
    let cut = 5.0e-309;
    let batch = batch(&[0.0, top], &[0.0, top], top)?;
    let plan = batch.bind_coupling_cuts(&[LevelPoolCut {
        pool_index: 0,
        elevation_m: cut,
    }])?;
    let lower = plan.lower_volume_m3().try_to_vec::<f64>()?[0];
    let upper = plan.upper_volume_m3().try_to_vec::<f64>()?[0];
    assert!(lower.is_finite() && lower >= 0.0 && lower <= cut);
    assert!(upper.is_finite() && upper >= cut && upper <= top);
    let available = batch
        .coupling_cut_availability(&plan)?
        .available_lower_m3
        .try_to_vec::<f64>()?[0];
    assert!(available > 0.0 && available <= top - cut);
    Ok(())
}

#[test]
fn near_overflow_equal_scale_cut_encloses_exact_unit_slope_volume() -> TestResult {
    let top = 1.0e308;
    let cut = 5.0e307;
    let batch = batch(&[0.0, top], &[0.0, top], top)?;
    let plan = batch.bind_coupling_cuts(&[LevelPoolCut {
        pool_index: 0,
        elevation_m: cut,
    }])?;
    let lower = plan.lower_volume_m3().try_to_vec::<f64>()?[0];
    let upper = plan.upper_volume_m3().try_to_vec::<f64>()?[0];
    assert!(lower.is_finite() && lower >= 0.0 && lower <= cut);
    assert!(upper.is_finite() && upper >= cut && upper <= top);
    let available = batch
        .coupling_cut_availability(&plan)?
        .available_lower_m3
        .try_to_vec::<f64>()?[0];
    assert!(available > 0.0 && available <= cut);
    Ok(())
}

#[test]
fn nonfinite_static_segment_difference_returns_typed_input_error() -> TestResult {
    let result = batch(&[0.0, 1.0e308], &[-1.0e308, 1.0e308], 0.0);
    assert!(matches!(result, Err(LevelPoolError::InvalidInput { .. })));
    Ok(())
}

#[test]
fn nonfinite_directed_cut_intermediate_returns_typed_numerical_error() -> TestResult {
    let batch = batch(&[0.0, f64::MAX], &[0.0, f64::MAX], 0.0)?;
    let cut = f64::from_bits(f64::MAX.to_bits() - 1);
    let result = batch.bind_coupling_cuts(&[LevelPoolCut {
        pool_index: 0,
        elevation_m: cut,
    }]);
    assert!(matches!(result, Err(LevelPoolError::Numerical { .. })));
    Ok(())
}
