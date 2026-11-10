use super::fixture::{Result, assemble};
use super::laws::{passive, pump, running, weir};
use hydr::coupling::{
    CoupledLevelPoolBatch, CoupledLevelPoolInitialState, CoupledLevelPoolSetup,
    CoupledLevelPoolSourceInterval,
};
use hydr::storage::{LevelPoolSourcePolicy, LevelPoolSources};

pub type Batch = CoupledLevelPoolBatch<crate::runtime::Runtime>;

pub fn policies(count: usize) -> Vec<LevelPoolSourcePolicy> {
    vec![LevelPoolSourcePolicy {
        dry_basal_footprint_assertion:
            "Synthetic first segment is the rain-receiving basal footprint, including dry storage".into(),
        bottom_loss_provenance:
            "Explicit synthetic potential bottom demand without inferred soil state".into(),
    }; count]
}

pub fn rates(rain: f64, bottom: f64, evaporation: f64) -> LevelPoolSources {
    LevelPoolSources {
        rain_rate_m_per_s: rain,
        bottom_loss_rate_m3_per_s: bottom,
        potential_evaporation_rate_m_per_s: evaporation,
    }
}

pub fn interval(sources: Vec<LevelPoolSources>) -> CoupledLevelPoolSourceInterval {
    CoupledLevelPoolSourceInterval {
        start_s: 0.0,
        end_s: 2.0,
        sources,
    }
}

pub fn enabled(
    setup: CoupledLevelPoolSetup,
    initial: CoupledLevelPoolInitialState,
) -> Result<Batch> {
    let count = setup.pools.len();
    Ok(Batch::new_with_sources(
        crate::runtime::client(),
        setup,
        initial,
        policies(count),
    )?)
}

pub fn pump_case(
    volumes: [f64; 2],
) -> Result<(CoupledLevelPoolSetup, CoupledLevelPoolInitialState)> {
    assemble(
        &[("a", 0.0), ("b", 0.0)],
        &volumes,
        vec![pump("pump", "a", "b", &[(-16.0, 1.0), (16.0, 1.0)])?],
        vec![running("pump")],
    )
}

pub fn inactive_case(
    volumes: [f64; 2],
) -> Result<(CoupledLevelPoolSetup, CoupledLevelPoolInitialState)> {
    assemble(
        &[("a", 0.0), ("b", 0.0)],
        &volumes,
        vec![weir("inactive", "a", "b", 16.0, 1.0)?],
        vec![passive("inactive")],
    )
}
