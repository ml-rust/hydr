use super::{
    ChannelConfig, ChannelError, ChannelGeometry, ChannelInitialState, ChannelReach,
    ChannelReachDefinition, ChannelSurveyedConfig, SurveyedChannelCell,
};
use crate::channel::{CrossSectionDefinition, CrossSectionPoint};
use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime};

type TestResult<T = ()> = Result<T, Box<dyn std::error::Error>>;

fn reach() -> TestResult<ChannelReach<CpuRuntime>> {
    Ok(ChannelReach::new(
        CpuClient::new(CpuDevice::new()),
        ChannelReachDefinition {
            vertical_datum: "Synthetic SI datum".into(),
            provenance: "Surveyed candidate-state admission".into(),
            geometry: ChannelGeometry::PrismaticSurveyed {
                profile: CrossSectionDefinition {
                    id: "section".into(),
                    vertical_datum: "Synthetic SI datum".into(),
                    points: [(-1.0, 3.0), (-1.0, 0.0), (1.0, 0.0), (1.0, 3.0)]
                        .into_iter()
                        .map(|(station_m, elevation_m)| CrossSectionPoint {
                            station_m,
                            elevation_m,
                        })
                        .collect(),
                    maximum_stage_m: 3.0,
                    profile_provenance: "Two-metre rectangular survey".into(),
                },
                cells: vec![SurveyedChannelCell {
                    id: "cell".into(),
                    length_m: 0.5,
                    manning_n: 0.0,
                }],
                numerical: ChannelSurveyedConfig::default(),
            },
            lateral_bindings: Vec::new(),
            config: ChannelConfig::default(),
        },
        ChannelInitialState {
            time_s: 0.0,
            volume_m3: vec![0.25],
            longitudinal_momentum_m4_per_s: vec![0.0],
        },
    )?)
}

fn check_rejected_candidate(volume: f64) -> TestResult {
    let reach = reach()?;
    let before = reach.snapshot_data()?;
    let candidate = reach.context.tensor(&[volume])?;
    let result = reach
        .context
        .fields(&candidate, &reach.accepted.momentum.high);
    let error = result
        .err()
        .ok_or("Candidate requires numerical rejection")?;
    assert!(
        matches!(error, ChannelError::Numerical { cell: Some(0), .. }),
        "{error}"
    );
    assert!(
        error.to_string().contains("before surveyed recovery"),
        "{error}"
    );
    assert_eq!(candidate.item::<f64>()?.to_bits(), volume.to_bits());
    assert_eq!(reach.snapshot_data()?, before);
    Ok(())
}

#[test]
fn surveyed_overflowing_candidate_area_rejects_numerically_without_state_change() -> TestResult {
    assert!((f64::MAX / 0.5).is_infinite());
    check_rejected_candidate(f64::MAX)
}

#[test]
fn surveyed_negative_candidate_volume_rejects_numerically_without_state_change() -> TestResult {
    check_rejected_candidate(-0.25)
}

#[test]
fn surveyed_candidate_shape_error_remains_invalid_input() -> TestResult {
    let reach = reach()?;
    let before = reach.snapshot_data()?;
    let candidate = reach.context.tensor(&[0.25, 0.25])?;
    let result = reach
        .context
        .fields(&candidate, &reach.accepted.momentum.high);
    let error = result
        .err()
        .ok_or("Candidate shape requires input rejection")?;
    assert!(
        matches!(error, ChannelError::InvalidInput { .. }),
        "{error}"
    );
    assert_eq!(reach.snapshot_data()?, before);
    Ok(())
}
