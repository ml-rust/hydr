use hydr::storage::{
    LevelPoolBatch, LevelPoolConfig, LevelPoolCurve, LevelPoolElement, LevelPoolError,
    LevelPoolInput, LevelPoolKnot, LevelPoolState,
};

pub type Result<T = ()> = std::result::Result<T, Box<dyn std::error::Error>>;

pub fn curve(knots: &[(f64, f64)]) -> Result<LevelPoolCurve> {
    Ok(LevelPoolCurve::new(
        knots
            .iter()
            .map(|&(volume_m3, stage_m)| LevelPoolKnot { volume_m3, stage_m })
            .collect(),
        "synthetic vertical datum".into(),
        "Explicit synthetic connected-storage table".into(),
    )?)
}

pub fn constant() -> Result<LevelPoolCurve> {
    curve(&[(0.0, 10.0), (4.0, 12.0)])
}
pub fn multiple() -> Result<LevelPoolCurve> {
    curve(&[(0.0, 10.0), (2.0, 11.0), (8.0, 13.0)])
}
pub fn unit(maximum: f64) -> Result<LevelPoolCurve> {
    curve(&[(0.0, 0.0), (maximum, maximum)])
}

pub fn element(id: &str, curve: LevelPoolCurve) -> LevelPoolElement {
    LevelPoolElement {
        id: id.into(),
        curve,
        connected_pool_assertion: "Explicit synthetic connected approximately level pool".into(),
        storage_support_identity: format!("storage-support-{id}"),
    }
}

pub fn single(curve: LevelPoolCurve, volume_m3: f64) -> Result<LevelPoolBatch> {
    Ok(LevelPoolBatch::new(
        vec![element("pool", curve)],
        vec![LevelPoolState { volume_m3 }],
        0.0,
        LevelPoolConfig::default(),
    )?)
}

pub fn input(incoming_volume_m3: f64, outgoing_volume_m3: f64) -> LevelPoolInput {
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
        Ok(_) => Err("unsupported storage input requires rejection".into()),
        Err(error) => Ok(error),
    }
}
