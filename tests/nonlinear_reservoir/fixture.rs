use hydr::runoff::{
    ReservoirBatch, ReservoirConfig, ReservoirElement, ReservoirParameters, ReservoirState,
};

pub type Result<T = ()> = std::result::Result<T, Box<dyn std::error::Error>>;

pub fn parameters() -> Result<ReservoirParameters> {
    Ok(ReservoirParameters::from_native(
        1024.0,
        1024.0,
        32.0,
        1.0 / 64.0,
        1.0 / 32.0,
        1.0 / 8.0,
    )?)
}

pub fn single(
    parameters: ReservoirParameters,
    depth_m: f64,
    config: ReservoirConfig,
) -> Result<ReservoirBatch<crate::runtime::Runtime>> {
    Ok(ReservoirBatch::<crate::runtime::Runtime>::new(
        crate::runtime::client(),
        vec![ReservoirElement {
            id: "storage".into(),
            parameters,
        }],
        vec![ReservoirState { depth_m }],
        0.0,
        config,
    )?)
}
