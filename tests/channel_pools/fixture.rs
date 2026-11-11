use super::reference::Case;
use hydr::channel::*;
use hydr::coupling::*;
use hydr::storage::*;

pub type Result<T = ()> = std::result::Result<T, Box<dyn std::error::Error>>;
pub type Batch = ChannelPoolBatch<crate::runtime::Runtime>;
pub const DATUM: &str = "Synthetic channel/pool datum";

pub fn config() -> ChannelPoolConfig {
    ChannelPoolConfig {
        storage: LevelPoolConfig::default(),
        head_absolute_m: 1e-12,
        head_relative: 1e-12,
        transfer_absolute_m3: 1e-12,
        transfer_relative: 1e-12,
        mass_absolute_m3: 1e-9,
        mass_relative: 1e-12,
        minimum_step_s: 1e-10,
        maximum_step_s: 1.0,
        iteration_limit: 64,
        retry_limit: 16,
    }
}

pub fn definition(
    triangle: bool,
    gravity: f64,
    coefficient: f64,
    count: usize,
    length: f64,
    bed: f64,
) -> ChannelReachDefinition {
    let geometry = if triangle {
        ChannelGeometry::PrismaticTriangular {
            left_bank_slope_h_per_v: coefficient,
            right_bank_slope_h_per_v: coefficient,
            bed_elevation_m: bed,
            cells: (0..count)
                .map(|i| TriangularChannelCell {
                    id: format!("cell-{i}"),
                    length_m: length,
                    maximum_depth_m: 4.0,
                    manning_n: 0.0,
                })
                .collect(),
        }
    } else {
        ChannelGeometry::Rectangular {
            cells: (0..count)
                .map(|i| RectangularChannelCell {
                    id: format!("cell-{i}"),
                    length_m: length,
                    width_m: coefficient,
                    bed_elevation_m: bed,
                    maximum_depth_m: 4.0,
                    manning_n: 0.0,
                })
                .collect(),
        }
    };
    ChannelReachDefinition {
        vertical_datum: DATUM.into(),
        provenance: "Explicit finite-pool analytical reach".into(),
        geometry,
        lateral_bindings: Vec::new(),
        config: ChannelConfig {
            gravity,
            max_step: 1.0,
            ..ChannelConfig::default()
        },
    }
}

pub fn pool(id: &str, area: f64, origin: f64) -> Result<LevelPoolElement> {
    Ok(LevelPoolElement {
        id: id.into(),
        curve: LevelPoolCurve::new(
            vec![
                LevelPoolKnot {
                    volume_m3: 0.0,
                    stage_m: origin,
                },
                LevelPoolKnot {
                    volume_m3: area * 4.0,
                    stage_m: origin + 4.0,
                },
            ],
            DATUM.into(),
            "Explicit constant-area analytical basin".into(),
        )?,
        connected_pool_assertion: "Connected approximately level synthetic basin".into(),
        storage_support_identity: format!("finite-basin-{id}"),
    })
}

pub fn connection(
    id: &str,
    reach_id: &str,
    endpoint: ChannelPoolEndpoint,
    pool_id: &str,
    loss: f64,
) -> ChannelPoolConnection {
    ChannelPoolConnection {
        id: id.into(),
        reach_id: reach_id.into(),
        endpoint,
        pool_id: pool_id.into(),
        entrance_loss_coefficient: loss,
        physical_connection_assertion: "Unobstructed named channel endpoint at its actual bed"
            .into(),
        quiescent_reservoir_assertion: "Negligible basin velocity with complete exit mixing".into(),
    }
}

pub fn single(
    case: Case,
    endpoint: ChannelPoolEndpoint,
) -> Result<(ChannelPoolSetup, ChannelPoolInitialState)> {
    Ok((
        ChannelPoolSetup {
            reaches: vec![ChannelPoolReach {
                id: "reach".into(),
                definition: definition(
                    case.triangle,
                    case.gravity,
                    case.coefficient,
                    1,
                    128.0,
                    0.0,
                ),
            }],
            pools: vec![pool("pool", case.pool_area, 0.0)?],
            connections: vec![connection(
                "connection",
                "reach",
                endpoint,
                "pool",
                case.loss,
            )],
            vertical_datum: DATUM.into(),
            nonoverlap_assertion: "Basin and reach occupy separate physical storage".into(),
            config: config(),
        },
        ChannelPoolInitialState {
            time_s: 0.0,
            reaches: vec![ChannelInitialState {
                time_s: 0.0,
                volume_m3: vec![128.0 * case.coefficient],
                longitudinal_momentum_m4_per_s: vec![0.0],
            }],
            pools: vec![LevelPoolState {
                volume_m3: case.pool_area * case.elevation,
            }],
        },
    ))
}

pub fn forcing(setup: &ChannelPoolSetup, start_s: f64, end_s: f64) -> ChannelPoolForcingInterval {
    let reaches = setup
        .reaches
        .iter()
        .map(|reach| {
            let external = |endpoint| {
                if setup
                    .connections
                    .iter()
                    .any(|c| c.reach_id == reach.id && c.endpoint == endpoint)
                {
                    None
                } else {
                    Some(ChannelBoundary::Exterior {
                        depth_m: 1.0,
                        velocity_m_per_s: 0.0,
                    })
                }
            };
            ChannelPoolReachForcing {
                reach_id: reach.id.clone(),
                left: external(ChannelPoolEndpoint::Left),
                right: external(ChannelPoolEndpoint::Right),
                lateral_inputs: Vec::new(),
            }
        })
        .collect();
    ChannelPoolForcingInterval {
        start_s,
        end_s,
        reaches,
    }
}

pub fn batch(setup: ChannelPoolSetup, initial: ChannelPoolInitialState) -> Result<Batch> {
    Ok(ChannelPoolBatch::new(
        crate::runtime::client(),
        setup,
        initial,
    )?)
}

pub fn near(actual: f64, expected: f64) {
    assert!(
        (actual - expected).abs() <= 1e-10 + 1e-10 * expected.abs(),
        "actual={actual:e}, independent reference={expected:e}"
    );
}

pub fn side_index(side: ChannelPoolEndpoint) -> usize {
    match side {
        ChannelPoolEndpoint::Left => 0,
        ChannelPoolEndpoint::Right => 1,
    }
}

pub fn check_water(batch: &Batch) -> Result {
    let ledger = batch.ledger()?;
    let scale = ledger
        .initial_storage_m3
        .abs()
        .max(ledger.storage_m3.abs())
        .max(ledger.external_boundary_in_m3)
        .max(ledger.external_boundary_out_m3)
        .max(ledger.lateral_in_m3)
        .max(ledger.lateral_out_m3);
    assert!(
        ledger.residual_m3.abs() + ledger.roundoff_bound_m3
            <= batch.setup().config.mass_absolute_m3 + batch.setup().config.mass_relative * scale
    );
    if let Some(step) = batch.last_step()? {
        assert!(
            step.mass_residual_m3.abs() + step.aggregation_roundoff_bound_m3
                <= batch.setup().config.mass_absolute_m3
                    + batch.setup().config.mass_relative
                        * step
                            .storage_before_m3
                            .abs()
                            .max(step.storage_after_m3.abs())
        );
        assert!(step.iterations <= batch.setup().config.iteration_limit);
        assert!(step.iteration.attachment_wet.iter().all(|wet| *wet));
        for cut in step.donor_cuts {
            assert!(cut.outgoing_upper_m3 <= cut.available_lower_m3);
        }
    }
    Ok(())
}
