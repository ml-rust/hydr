use super::*;
use numr::runtime::cpu::{CpuDevice, ParallelismConfig};

type TestResult = Result<(), Box<dyn std::error::Error>>;

const GRAVITY: f64 = 9.81;
const DEPTHS: [(f64, f64); 4] = [
    (3.209e-105, 2.370e-162),
    (2.370e-162, 3.209e-105),
    (1.0, 1e-120),
    (1e-120, 1.0),
];

fn setup() -> Result<(CpuClient, TensorFaceState, TensorFaceState), EngineError> {
    let device = CpuDevice::new();
    let tensor = |values: &[f64]| {
        CpuTensor::from_slice(values, &[DEPTHS.len()], &device).map_err(EngineError::from)
    };
    let left = TensorFaceState {
        h: tensor(&DEPTHS.map(|(left, _)| left))?,
        q_normal: tensor(&[0.0; DEPTHS.len()])?,
        q_tangent: tensor(&[0.0; DEPTHS.len()])?,
        bed: tensor(&[0.0; DEPTHS.len()])?,
    };
    let right = TensorFaceState {
        h: tensor(&DEPTHS.map(|(_, right)| right))?,
        q_normal: left.q_normal.clone(),
        q_tangent: left.q_tangent.clone(),
        bed: left.bed.clone(),
    };
    Ok((
        CpuClient::new(device).with_parallelism(ParallelismConfig::new(Some(1), None)),
        left,
        right,
    ))
}

// Normalize H by sqrt(hDeep*hShallow) and the residual by sqrt(g*hDeep).
fn normalized_residual(z: f64, deep: f64, shallow: f64) -> f64 {
    let depth_ratio = shallow.sqrt() / deep.sqrt();
    let rarefaction = 2.0 * ((z * depth_ratio).sqrt() - 1.0);
    let shock = (z - depth_ratio) * (0.5 * (1.0 + depth_ratio / z)).sqrt();
    rarefaction + shock
}

fn physical_upper(deep: f64, shallow: f64) -> f64 {
    let (mut lo, mut hi) = (2.0, 3.0);
    for _ in 0..80 {
        let middle = 0.5 * (lo + hi);
        if normalized_residual(middle, deep, shallow) >= 0.0 {
            hi = middle;
        } else {
            lo = middle;
        }
    }
    hi * deep.sqrt() * shallow.sqrt()
}

#[test]
fn unequal_positive_films_keep_a_tight_nonnegative_root_bracket() -> TestResult {
    let (client, left, right) = setup()?;
    let zero = constant(&left.h, 0.0)?;
    let solve = constant(&left.h, 1.0)?;
    let upper = root_upper(
        &client,
        &left.h,
        &right.h,
        &zero,
        &solve,
        GRAVITY,
        &SolverConfig::default(),
    )?;
    let residuals =
        residual(&client, &upper, &left.h, &right.h, &zero, GRAVITY)?.try_to_vec::<f64>()?;
    for ((&(hl, hr), upper), residual) in
        DEPTHS.iter().zip(upper.try_to_vec::<f64>()?).zip(residuals)
    {
        let deep = hl.max(hr);
        let shallow = hl.min(hr);
        let scale = deep.sqrt() * shallow.sqrt();
        let normalized_upper = upper / scale;
        assert!(upper.is_finite() && upper > shallow && upper < deep);
        assert!(
            (2.0..3.0).contains(&normalized_upper),
            "H/sqrt(hl*hr)={normalized_upper}"
        );
        assert!(
            residual.is_finite() && residual >= 0.0,
            "residual={residual}"
        );
        assert!((upper / physical_upper(deep, shallow) - 1.0).abs() <= 2e-12);
    }
    Ok(())
}

#[test]
fn unequal_positive_films_enclose_the_physical_shock_in_both_orientations() -> TestResult {
    let (client, left, right) = setup()?;
    let (sl, sr) = bounds(&client, &left, &right, GRAVITY, &SolverConfig::default())?;
    for ((&(hl, hr), sl), sr) in DEPTHS
        .iter()
        .zip(sl.try_to_vec::<f64>()?)
        .zip(sr.try_to_vec::<f64>()?)
    {
        let deep = hl.max(hr);
        let shallow = hl.min(hr);
        let star = physical_upper(deep, shallow);
        let head = GRAVITY.sqrt() * deep.sqrt();
        let shock =
            GRAVITY.sqrt() * (star / shallow.sqrt()) * (0.5 * (1.0 + shallow / star)).sqrt();
        let (physical_sl, physical_sr) = if hl > hr {
            (-head, shock)
        } else {
            (-shock, head)
        };
        let rounding = 32.0 * f64::EPSILON * shock;
        assert!(sl.is_finite() && sr.is_finite());
        assert!(
            sl <= physical_sl + rounding,
            "sl={sl}, physical_sl={physical_sl}"
        );
        assert!(
            sr >= physical_sr - rounding,
            "sr={sr}, physical_sr={physical_sr}"
        );
        assert!(sl < 0.0 && sr > 0.0);
    }
    Ok(())
}

#[test]
fn compressed_states_skip_a_nonfinite_one_sided_candidate() -> TestResult {
    let (hl_value, hr_value, gravity, jump_value) = (1e300_f64, 1e-200_f64, 1e-300_f64, -1e100_f64);
    let numerator_l = (2.0 * gravity.sqrt() * hr_value.sqrt() - jump_value).max(0.0);
    let factor_l = hl_value.sqrt() * (std::f64::consts::SQRT_2 / gravity.sqrt());
    let candidate_l = hl_value + numerator_l * factor_l;
    assert_eq!(candidate_l, f64::INFINITY);

    let device = CpuDevice::new();
    let client =
        CpuClient::new(device.clone()).with_parallelism(ParallelismConfig::new(Some(1), None));
    let hl = CpuTensor::from_slice(&[hl_value], &[1], &device)?;
    let hr = CpuTensor::from_slice(&[hr_value], &[1], &device)?;
    let jump = CpuTensor::from_slice(&[jump_value], &[1], &device)?;
    let solve = constant(&hl, 1.0)?;
    let upper = root_upper(
        &client,
        &hl,
        &hr,
        &jump,
        &solve,
        gravity,
        &SolverConfig::default(),
    )?;
    let value = upper.item::<f64>()?;
    let residual = residual(&client, &upper, &hl, &hr, &jump, gravity)?.item::<f64>()?;
    let expected = std::f64::consts::SQRT_2 * 1e150;
    assert!(value.is_finite() && value > 0.0);
    assert!(
        residual.is_finite() && residual >= 0.0,
        "residual={residual}"
    );
    assert!((value / expected - 1.0).abs() <= 2e-12, "upper={value}");
    Ok(())
}
