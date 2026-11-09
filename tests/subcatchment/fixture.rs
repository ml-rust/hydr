use hydr::runoff::{
    SubcatchmentBatch, SubcatchmentConfig, SubcatchmentDefinition, SubcatchmentGeometry,
    SubcatchmentParameters, SubcatchmentRouting, SubcatchmentState,
};
use hydr::soil::ModifiedGreenAmptParameters;

pub type Result<T = ()> = std::result::Result<T, Box<dyn std::error::Error>>;

pub fn geometry(impervious_fraction: f64, zero_storage_fraction: f64) -> SubcatchmentGeometry {
    SubcatchmentGeometry {
        total_area_m2: 1024.0 * (0.3048 * 0.3048),
        width_m: 32.0 * 0.3048,
        slope: 1.0 / 64.0,
        impervious_fraction,
        zero_storage_impervious_fraction: zero_storage_fraction,
        impervious_manning: 1.0 / 32.0,
        pervious_manning: 1.0 / 32.0,
        impervious_depression_m: 0.125 * 0.3048,
        pervious_depression_m: 0.125 * 0.3048,
    }
}

pub fn parameters(
    impervious_fraction: f64,
    zero_storage_fraction: f64,
    routing: SubcatchmentRouting,
) -> Result<SubcatchmentParameters> {
    let soil = if impervious_fraction < 1.0 {
        Some(ModifiedGreenAmptParameters::from_native(
            0.125,
            1.0 / 1024.0,
            1.0,
        )?)
    } else {
        None
    };
    Ok(SubcatchmentParameters::new(
        geometry(impervious_fraction, zero_storage_fraction),
        soil,
        routing,
    )?)
}

pub fn single(
    parameters: SubcatchmentParameters,
    state: SubcatchmentState,
    config: SubcatchmentConfig,
) -> Result<SubcatchmentBatch> {
    Ok(SubcatchmentBatch::new(
        vec![SubcatchmentDefinition {
            id: "catchment".into(),
            parameters,
        }],
        vec![state],
        0.0,
        config,
    )?)
}
