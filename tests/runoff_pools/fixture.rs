use super::laws::{passive, weir};
pub use super::pool_fixture::{DATUM, Result};
use hydr::coupling::{
    CoupledLevelPoolSetup, CoupledPoolInflowBinding, RunoffPoolBatch, RunoffPoolConfig,
    RunoffPoolForcingInterval, RunoffPoolInitialState, RunoffPoolMeteorology,
    RunoffPoolOutletBinding, RunoffPoolSetup,
};
use hydr::runoff::{
    Subarea, SubcatchmentConfig, SubcatchmentDefinition, SubcatchmentGeometry,
    SubcatchmentParameters, SubcatchmentRouting, SubcatchmentState,
};
use hydr::soil::ModifiedGreenAmptParameters;

pub type Batch = RunoffPoolBatch<crate::runtime::Runtime>;

pub fn impervious(area: f64, zero_fraction: f64) -> Result<SubcatchmentParameters> {
    Ok(SubcatchmentParameters::new(
        SubcatchmentGeometry {
            total_area_m2: area,
            width_m: 1.0,
            slope: 0.01,
            impervious_fraction: 1.0,
            zero_storage_impervious_fraction: zero_fraction,
            impervious_manning: 0.0,
            pervious_manning: 0.0,
            impervious_depression_m: 0.0,
            pervious_depression_m: 0.0,
        },
        None,
        SubcatchmentRouting::Direct,
    )?)
}

pub fn pervious(routing: SubcatchmentRouting) -> Result<SubcatchmentParameters> {
    Ok(SubcatchmentParameters::new(
        SubcatchmentGeometry {
            total_area_m2: 1024.0 * (0.3048 * 0.3048),
            width_m: 32.0 * 0.3048,
            slope: 1.0 / 64.0,
            impervious_fraction: 0.5,
            zero_storage_impervious_fraction: 0.5,
            impervious_manning: 1.0 / 32.0,
            pervious_manning: 1.0 / 32.0,
            impervious_depression_m: 0.125 * 0.3048,
            pervious_depression_m: 0.125 * 0.3048,
        },
        Some(ModifiedGreenAmptParameters::from_native(
            0.125,
            1.0 / 1024.0,
            1.0,
        )?),
        routing,
    )?)
}

pub fn inactive_pools(
    volumes: [f64; 2],
) -> Result<(CoupledLevelPoolSetup, Vec<hydr::storage::LevelPoolState>)> {
    let (pools, initial) = super::pool_fixture::assemble(
        &[("a", 0.0), ("b", 0.0)],
        &volumes,
        vec![weir("inactive", "a", "b", 16.0, 1.0)?],
        vec![passive("inactive")],
    )?;
    Ok((pools, initial.states))
}

pub fn setup(
    parameters: Vec<SubcatchmentParameters>,
    mut pools: CoupledLevelPoolSetup,
    pool_states: Vec<hydr::storage::LevelPoolState>,
) -> Result<(RunoffPoolSetup, RunoffPoolInitialState)> {
    let mut definitions = Vec::new();
    let mut states = Vec::new();
    let mut outlets = Vec::new();
    for (index, parameters) in parameters.into_iter().enumerate() {
        let id = format!("catchment-{index}");
        states.push(SubcatchmentState::initialized(&parameters)?);
        for subarea in Subarea::ALL {
            let fraction = match parameters.routing() {
                SubcatchmentRouting::ImperviousToPervious { fraction }
                    if subarea != Subarea::Pervious =>
                {
                    fraction
                }
                SubcatchmentRouting::PerviousToImpervious { fraction }
                    if subarea == Subarea::Pervious =>
                {
                    fraction
                }
                _ => 0.0,
            };
            if parameters.area_m2(subarea) > 0.0 && fraction < 1.0 {
                let boundary_id = format!("{id}-outlet-{}", subarea.index());
                pools.external_inflows.push(CoupledPoolInflowBinding {
                    boundary_id: boundary_id.clone(), pool_id: "a".into(),
                    input_id: format!("{id}-native-export-{}", subarea.index()),
                    physical_support_id: format!("{id}-subarea-{}", subarea.index()),
                    boundary_assertion: "Declared free native runoff outlet enters its named receiving pool".into(),
                    ownership_assertion: "Native runoff area excludes represented pool storage and local pool rainfall, without duplicated exports".into(),
                });
                outlets.push(RunoffPoolOutletBinding {
                    subcatchment_id: id.clone(), subarea, boundary_id,
                    free_outlet_assertion: "Free runoff export has no downstream backwater or hydraulic feedback".into(),
                    nonoverlap_assertion: "Separate runoff support excludes pool storage and local pool rain, with no external runon or disposal routes".into(),
                });
            }
        }
        definitions.push(SubcatchmentDefinition { id, parameters });
    }
    let minimum_step_s = pools.config.minimum_step_s;
    let maximum_step_s = pools.config.maximum_step_s;
    Ok((
        RunoffPoolSetup {
            subcatchments: definitions,
            runoff_config: SubcatchmentConfig::default(),
            pools,
            outlets,
            config: RunoffPoolConfig {
                minimum_step_s,
                maximum_step_s,
                retry_limit: 16,
                mass_absolute_m3: 1e-9,
                mass_relative: 1e-12,
            },
        },
        RunoffPoolInitialState {
            time_s: 0.0,
            subcatchments: states,
            pools: pool_states,
        },
    ))
}

pub fn exact_case() -> Result<(RunoffPoolSetup, RunoffPoolInitialState)> {
    let (mut pools, states) = inactive_pools([0.0, 1.0])?;
    pools.config.maximum_step_s = 2.0;
    setup(vec![impervious(2.0, 1.0)?], pools, states)
}

pub fn forcing(count: usize, rain: f64, evaporation: f64) -> RunoffPoolForcingInterval {
    RunoffPoolForcingInterval {
        start_s: 0.0,
        end_s: 2.0,
        subcatchments: vec![
            RunoffPoolMeteorology {
                rain_rate_m_per_s: rain,
                potential_evaporation_m_per_s: evaporation
            };
            count
        ],
        pool_sources: None,
    }
}

pub fn shortened_case() -> Result<(
    RunoffPoolSetup,
    RunoffPoolInitialState,
    RunoffPoolForcingInterval,
)> {
    let (mut pools, initial) = super::pool_fixture::orifice_case([4.0, 2.0])?;
    pools.config.iteration_limit = 8;
    let (setup, mut initial) = setup(
        vec![pervious(SubcatchmentRouting::ImperviousToPervious {
            fraction: 0.5,
        })?],
        pools,
        initial.states,
    )?;
    initial.subcatchments[0].depths_m = [0.125 * 0.3048, 0.25 * 0.3048, 0.0];
    Ok((setup, initial, forcing(1, 0.3048 / 1024.0, 0.0)))
}
