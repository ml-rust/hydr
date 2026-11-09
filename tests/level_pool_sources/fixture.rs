use hydr::storage::{
    LevelPoolBatch, LevelPoolConfig, LevelPoolCurve, LevelPoolElement, LevelPoolError,
    LevelPoolInput, LevelPoolKnot, LevelPoolSourcePolicy, LevelPoolSources, LevelPoolState,
};

pub type Result<T = ()> = std::result::Result<T, Box<dyn std::error::Error>>;

pub fn curve(knots: &[(f64, f64)]) -> Result<LevelPoolCurve> {
    Ok(LevelPoolCurve::new(
        knots
            .iter()
            .map(|&(volume_m3, stage_m)| LevelPoolKnot { volume_m3, stage_m })
            .collect(),
        "synthetic vertical datum".into(),
        "Explicit synthetic pool-volume table".into(),
    )?)
}

pub fn changing() -> Result<LevelPoolCurve> {
    curve(&[(0.0, 100.0), (10.0, 101.0), (30.0, 102.0)])
}

pub fn element(id: &str, curve: LevelPoolCurve) -> LevelPoolElement {
    LevelPoolElement {
        id: id.into(),
        curve,
        connected_pool_assertion: "Explicit connected approximately level synthetic pool".into(),
        storage_support_identity: format!("support-{id}"),
    }
}

pub fn policy() -> LevelPoolSourcePolicy {
    LevelPoolSourcePolicy {
        dry_basal_footprint_assertion:
            "First segment represents the rain-receiving basal footprint, including dry storage"
                .into(),
        bottom_loss_provenance:
            "Explicit supplied potential bottom demand without inferred soil state".into(),
    }
}

pub fn enabled(curve: LevelPoolCurve, volume_m3: f64) -> Result<LevelPoolBatch> {
    Ok(LevelPoolBatch::new_with_sources(
        vec![element("pool", curve)],
        vec![LevelPoolState { volume_m3 }],
        0.0,
        LevelPoolConfig::default(),
        vec![policy()],
    )?)
}

pub fn sources(rain: f64, evaporation: f64, bottom: f64) -> LevelPoolSources {
    LevelPoolSources {
        rain_rate_m_per_s: rain,
        potential_evaporation_rate_m_per_s: evaporation,
        bottom_loss_rate_m3_per_s: bottom,
    }
}

pub fn exchange(incoming_volume_m3: f64, outgoing_volume_m3: f64) -> LevelPoolInput {
    LevelPoolInput {
        incoming_volume_m3,
        outgoing_volume_m3,
    }
}
pub fn tolerance(scale: f64) -> f64 {
    1e-9 + 1e-12 * scale
}
pub fn rejection<T>(result: std::result::Result<T, LevelPoolError>) -> Result<LevelPoolError> {
    match result {
        Ok(_) => Err("unsupported pool-source input requires rejection".into()),
        Err(error) => Ok(error),
    }
}
