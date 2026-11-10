//! Production observations for frozen changing-area atmospheric source cases.

#[allow(dead_code)]
#[path = "../tests/support/runtime.rs"]
mod runtime;

use std::io::{self, Write};

use hydr::storage::{
    LevelPoolBatch, LevelPoolConfig, LevelPoolCurve, LevelPoolElement, LevelPoolError,
    LevelPoolInput, LevelPoolKnot, LevelPoolSourcePolicy, LevelPoolSources, LevelPoolState,
};

struct Case {
    id: &'static str,
    initial_volume_m3: f64,
    rain_rate_m_per_s: f64,
    evaporation_rate_m_per_s: f64,
    bottom_loss_rate_m3_per_s: f64,
}

const CASES: [Case; 4] = [
    Case {
        id: "filling",
        initial_volume_m3: 9.1,
        rain_rate_m_per_s: 0.3,
        evaporation_rate_m_per_s: 0.1,
        bottom_loss_rate_m3_per_s: 0.5,
    },
    Case {
        id: "draining",
        initial_volume_m3: 12.1,
        rain_rate_m_per_s: 0.0,
        evaporation_rate_m_per_s: 0.1,
        bottom_loss_rate_m3_per_s: 0.5,
    },
    Case {
        id: "rain_only",
        initial_volume_m3: 9.1,
        rain_rate_m_per_s: 0.2,
        evaporation_rate_m_per_s: 0.0,
        bottom_loss_rate_m3_per_s: 0.0,
    },
    Case {
        id: "evaporation_only",
        initial_volume_m3: 12.1,
        rain_rate_m_per_s: 0.0,
        evaporation_rate_m_per_s: 0.1,
        bottom_loss_rate_m3_per_s: 0.0,
    },
];

const GRIDS: [(f64, u32); 7] = [
    (0.5, 4),
    (0.25, 8),
    (0.125, 16),
    (0.0625, 32),
    (0.03125, 64),
    (0.015625, 128),
    (0.0078125, 256),
];

fn observe(
    output: &mut impl Write,
    case: &Case,
    dt_s: f64,
    steps: u32,
) -> Result<(), Box<dyn std::error::Error>> {
    let curve = LevelPoolCurve::new(
        vec![
            LevelPoolKnot {
                volume_m3: 0.0,
                stage_m: 100.0,
            },
            LevelPoolKnot {
                volume_m3: 10.0,
                stage_m: 101.0,
            },
            LevelPoolKnot {
                volume_m3: 30.0,
                stage_m: 102.0,
            },
        ],
        "synthetic vertical datum".into(),
        "frozen piecewise-linear source reference fixture".into(),
    )?;
    let mut batch = LevelPoolBatch::<crate::runtime::Runtime>::new_with_sources(
        crate::runtime::client(),
        vec![LevelPoolElement {
            id: case.id.into(),
            curve,
            connected_pool_assertion: "synthetic connected level pool with two secant areas".into(),
            storage_support_identity: format!("synthetic-source-reference-{}", case.id),
        }],
        vec![LevelPoolState {
            volume_m3: case.initial_volume_m3,
        }],
        0.0,
        LevelPoolConfig::default(),
        vec![LevelPoolSourcePolicy {
            dry_basal_footprint_assertion:
                "synthetic first curve segment defines the rain-receiving basal footprint".into(),
            bottom_loss_provenance:
                "explicit constant potential bottom demand from the frozen reference fixture".into(),
        }],
    )?;
    let exchanges = [LevelPoolInput {
        incoming_volume_m3: 0.0,
        outgoing_volume_m3: 0.0,
    }];
    let sources = [LevelPoolSources {
        rain_rate_m_per_s: case.rain_rate_m_per_s,
        potential_evaporation_rate_m_per_s: case.evaporation_rate_m_per_s,
        bottom_loss_rate_m3_per_s: case.bottom_loss_rate_m3_per_s,
    }];
    for step in 1..=steps {
        batch = batch
            .trial_with_sources_to(f64::from(step) * dt_s, &exchanges, &sources)?
            .into_next();
    }
    let outputs = batch.outputs()?;
    let stage_m = outputs.first().and_then(|value| value.stage_m).ok_or_else(|| {
        LevelPoolError::Numerical {
            element: Some(0),
            context: format!(
                "reference case={},dt_s={dt_s}: missing wet stage; check fixture and source results",
                case.id
            ),
        }
    })?;
    let totals = batch.totals()?;
    let source_totals = totals.sources.ok_or_else(|| LevelPoolError::Numerical {
        element: Some(0),
        context: format!(
            "reference case={},dt_s={dt_s}: missing enabled source ledger; check source policy",
            case.id
        ),
    })?;
    writeln!(
        output,
        "{},{:.17e},{:.17e},{:.17e},{:.17e},{:.17e},{:.17e},{:.17e},{:.17e}",
        case.id,
        dt_s,
        totals.initial_volume_m3,
        totals.volume_m3,
        stage_m,
        source_totals.cumulative_rain_volume_m3,
        source_totals.cumulative_evaporation_volume_m3,
        source_totals.cumulative_bottom_loss_volume_m3,
        totals.residual_m3,
    )?;
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut output = io::BufWriter::new(io::stdout().lock());
    writeln!(
        output,
        "case,dt_s,initial_volume_m3,volume_m3,stage_m,rain_volume_m3,evaporation_volume_m3,bottom_loss_volume_m3,residual_m3"
    )?;
    for case in &CASES {
        for (dt_s, steps) in GRIDS {
            observe(&mut output, case, dt_s, steps)?;
        }
    }
    output.flush()?;
    Ok(())
}
