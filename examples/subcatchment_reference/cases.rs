use crate::Result;
use hydr::runoff::{
    SubcatchmentGeometry, SubcatchmentInput, SubcatchmentParameters, SubcatchmentRouting,
    SubcatchmentState,
};
use hydr::soil::ModifiedGreenAmptParameters;

pub struct Case {
    pub name: &'static str,
    pub parameters: SubcatchmentParameters,
    pub initial: SubcatchmentState,
    pub input: SubcatchmentInput,
    pub horizon: f64,
    pub timesteps: &'static [f64],
}

fn geometry(impervious_fraction: f64) -> SubcatchmentGeometry {
    SubcatchmentGeometry {
        total_area_m2: 1024.0 * (0.3048 * 0.3048),
        width_m: 32.0 * 0.3048,
        slope: 1.0 / 64.0,
        impervious_fraction,
        zero_storage_impervious_fraction: 0.5,
        impervious_manning: 1.0 / 32.0,
        pervious_manning: 1.0 / 32.0,
        impervious_depression_m: 0.125 * 0.3048,
        pervious_depression_m: 0.125 * 0.3048,
    }
}

fn build(name: &'static str, routing: SubcatchmentRouting, smooth: bool) -> Result<Case> {
    let pervious_only = name == "pervious_sources";
    let geometry = geometry(if pervious_only { 0.0 } else { 0.5 });
    let soil = ModifiedGreenAmptParameters::from_native(
        0.125,
        if smooth { 1.0 / 2048.0 } else { 1.0 / 1024.0 },
        if smooth { 0.0 } else { 1.0 },
    )?;
    let parameters = SubcatchmentParameters::new(geometry, Some(soil), routing)?;
    let mut initial = SubcatchmentState::initialized(&parameters)?;
    initial.depths_m = if smooth {
        [0.3048 / 32.0, 5.0 * 0.3048 / 32.0, 5.0 * 0.3048 / 32.0]
    } else if pervious_only {
        [0.0, 0.0, 0.125 * 0.3048]
    } else if matches!(routing, SubcatchmentRouting::ImperviousToPervious { .. }) {
        [0.125 * 0.3048, 0.25 * 0.3048, 0.0]
    } else {
        [0.0, 0.125 * 0.3048, 0.25 * 0.3048]
    };
    Ok(Case {
        name,
        parameters,
        initial,
        input: SubcatchmentInput {
            rain_rate_m_per_s: if pervious_only {
                0.3048 / 512.0
            } else {
                0.3048 / 1024.0
            },
            potential_evaporation_m_per_s: if pervious_only { 0.3048 / 8192.0 } else { 0.0 },
            external_runon_volumes_m3: [0.0; 3],
        },
        horizon: if smooth { 64.0 } else { 256.0 },
        timesteps: if smooth {
            &[1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125]
        } else {
            &[64.0, 32.0, 16.0, 8.0]
        },
    })
}

pub fn catalog(mode: &str) -> Result<Vec<Case>> {
    let mut cases = Vec::new();
    if mode == "all" || mode == "smooth" {
        cases.push(build(
            "smooth_impervious_to_pervious",
            SubcatchmentRouting::ImperviousToPervious { fraction: 0.5 },
            true,
        )?);
        cases.push(build(
            "smooth_pervious_to_impervious",
            SubcatchmentRouting::PerviousToImpervious { fraction: 0.5 },
            true,
        )?);
    }
    if mode == "all" || mode == "histories" {
        cases.push(build(
            "pervious_sources",
            SubcatchmentRouting::Direct,
            false,
        )?);
        cases.push(build(
            "impervious_to_pervious",
            SubcatchmentRouting::ImperviousToPervious { fraction: 0.5 },
            false,
        )?);
        cases.push(build(
            "pervious_to_impervious",
            SubcatchmentRouting::PerviousToImpervious { fraction: 0.5 },
            false,
        )?);
    }
    if cases.is_empty() {
        return Err(format!("mode={mode}: require all, smooth, or histories").into());
    }
    Ok(cases)
}
