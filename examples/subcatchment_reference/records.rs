use super::cases::Case;
use crate::Result;
use hydr::runoff::{Subarea, SubcatchmentBatch, SubcatchmentRouting, SubcatchmentStepDiagnostics};
use hydr::soil::NativeSoilState;
use std::io::Write;

pub const HEADER: &str = "case,dt_s,step,time_s,area_m2,width_m,slope,impervious_fraction,zero_storage_fraction,impervious_manning,pervious_manning,impervious_depression_m,pervious_depression_m,soil_suction_ft,soil_conductivity_ft_per_s,soil_maximum_deficit,rain_m_per_s,evaporation_m_per_s,route_code,route_fraction,initial_depth0_m,initial_depth1_m,initial_depth2_m,area0_m2,area1_m2,area2_m2,alpha0_si,alpha1_si,alpha2_si,depth0_m,depth1_m,depth2_m,storage0_m3,storage1_m3,storage2_m3,gross0_m3,gross1_m3,gross2_m3,outlet0_m3,outlet1_m3,outlet2_m3,routed0_m3,routed1_m3,routed2_m3,infiltration_m3,evaporation_m3,rain_m3,total_storage_m3,total_gross_m3,total_outlet_m3,total_routed_m3,rate0_m_per_s,rate1_m_per_s,rate2_m_per_s,soil_imd,soil_f_ft,soil_fu_ft,soil_saturated,soil_t_s,soil_adjustment_m3,soil_adjustment_charge_m3,transfer_uncertainty_m3,physical_scale_lower_m3,water_residual_m3";
pub const SOIL_HEADER: &str = "case,dt_s,step,start_s,end_s,suction_ft,conductivity_ft_per_s,maximum_deficit,old_imd,old_f_ft,old_fu_ft,old_saturated,old_t_s,ponded_depth_m,constructed_rate_m_per_s,input_depth_m,available_depth_m,native_rate_ft_per_s,native_available_depth_ft,native_infiltration_depth_ft,infiltration_depth_m,adjustment_depth_m,imd,f_ft,fu_ft,saturated,t_s";

pub struct Endpoint<'a> {
    pub case: &'a Case,
    pub dt: f64,
    pub step: usize,
    pub batch: &'a SubcatchmentBatch,
    pub gross: [f64; 3],
    pub outlet: [f64; 3],
    pub routed: [f64; 3],
    pub rates: [Option<f64>; 3],
}

fn values(writer: &mut impl Write, values: impl IntoIterator<Item = Option<f64>>) -> Result {
    for value in values {
        write!(writer, ",")?;
        if let Some(value) = value {
            if !value.is_finite() {
                return Err("public observation contains a nonfinite value".into());
            }
            write!(writer, "{value:.17e}")?;
        }
    }
    writeln!(writer)?;
    Ok(())
}

fn native(state: Option<NativeSoilState>) -> [Option<f64>; 5] {
    match state {
        Some(state) => [
            Some(state.imd),
            Some(state.f_ft),
            Some(state.fu_ft),
            Some(if state.saturated { 1.0 } else { 0.0 }),
            Some(state.t_s),
        ],
        None => [None; 5],
    }
}

pub fn endpoint(writer: &mut impl Write, row: Endpoint<'_>) -> Result {
    let p = &row.case.parameters;
    let g = p.geometry();
    let soil = p
        .soil_parameters()
        .ok_or("observer case requires explicit soil parameters")?;
    let (code, fraction) = match p.routing() {
        SubcatchmentRouting::Direct => (0.0, 0.0),
        SubcatchmentRouting::PerviousToImpervious { fraction } => (1.0, fraction),
        SubcatchmentRouting::ImperviousToPervious { fraction } => (2.0, fraction),
    };
    let areas = Subarea::ALL.map(|subarea| p.area_m2(subarea));
    let alpha = Subarea::ALL.map(|subarea| {
        p.closure(subarea)
            .and_then(|closure| closure.coefficient_si())
    });
    let states = row.batch.states()?;
    let state = states.first().ok_or("missing observed catchment state")?;
    let storage = std::array::from_fn::<_, 3, _>(|index| areas[index] * state.depths_m[index]);
    let ledger = row.batch.ledger();
    let inputs = [
        g.total_area_m2,
        g.width_m,
        g.slope,
        g.impervious_fraction,
        g.zero_storage_impervious_fraction,
        g.impervious_manning,
        g.pervious_manning,
        g.impervious_depression_m,
        g.pervious_depression_m,
        soil.suction_ft(),
        soil.conductivity_ft_per_s(),
        soil.maximum_deficit(),
        row.case.input.rain_rate_m_per_s,
        row.case.input.potential_evaporation_m_per_s,
        code,
        fraction,
    ];
    let output = [
        ledger.infiltration_volume_m3,
        ledger.evaporation_volume_m3,
        ledger.applied_rain_volume_m3,
        ledger.storage_m3,
        row.gross.iter().sum(),
        ledger.outlet_volume_m3,
        ledger.internal_transfer_volume_m3,
    ];
    let auxiliary = [
        ledger.soil_water_adjustment_volume_m3,
        ledger.soil_water_adjustment_abs_bound_m3,
        ledger.transfer_uncertainty_charge_m3,
        ledger.physical_scale_lower_m3,
        ledger.residual_m3,
    ];
    write!(
        writer,
        "{},{:.17e},{},{:.17e}",
        row.case.name,
        row.dt,
        row.step,
        row.batch.time()
    )?;
    values(
        writer,
        inputs
            .into_iter()
            .map(Some)
            .chain(row.case.initial.depths_m.into_iter().map(Some))
            .chain(areas.into_iter().map(Some))
            .chain(alpha)
            .chain(state.depths_m.into_iter().map(Some))
            .chain(storage.into_iter().map(Some))
            .chain(row.gross.into_iter().map(Some))
            .chain(row.outlet.into_iter().map(Some))
            .chain(row.routed.into_iter().map(Some))
            .chain(output.into_iter().map(Some))
            .chain(row.rates)
            .chain(native(state.soil))
            .chain(auxiliary.into_iter().map(Some)),
    )
}

pub struct SoilCall<'a> {
    pub case: &'a Case,
    pub dt: f64,
    pub step: usize,
    pub diagnostics: &'a SubcatchmentStepDiagnostics,
    pub old: Option<NativeSoilState>,
    pub accepted: Option<NativeSoilState>,
}

pub fn soil_call(writer: &mut impl Write, row: SoilCall<'_>) -> Result {
    let catchment = row
        .diagnostics
        .receipts
        .first()
        .ok_or("missing catchment receipt")?;
    let Some(source) = &catchment.subareas[Subarea::Pervious.index()] else {
        return Ok(());
    };
    let receipt = source
        .soil
        .as_ref()
        .ok_or("missing pervious soil receipt")?;
    let parameters = row
        .case
        .parameters
        .soil_parameters()
        .ok_or("missing soil parameters")?;
    let old = row.old.ok_or("missing interval-start native soil")?;
    let accepted = row.accepted.ok_or("missing accepted native soil")?;
    write!(
        writer,
        "{},{:.17e},{},{:.17e},{:.17e}",
        row.case.name, row.dt, row.step, row.diagnostics.start, row.diagnostics.end
    )?;
    let input = [
        parameters.suction_ft(),
        parameters.conductivity_ft_per_s(),
        parameters.maximum_deficit(),
        old.imd,
        old.f_ft,
        old.fu_ft,
        if old.saturated { 1.0 } else { 0.0 },
        old.t_s,
        source.old_depth_m,
        source.bridge.total_rate_m_per_s,
        source.bridge.input_depth_m,
        source.bridge.available_depth_m,
        receipt.native_rate_ft_per_s,
        receipt.native_available_depth_ft,
        receipt.native_infiltration_depth_ft,
        source.infiltration_depth_m,
        source.adjustment_depth_m,
        accepted.imd,
        accepted.f_ft,
        accepted.fu_ft,
        if accepted.saturated { 1.0 } else { 0.0 },
        accepted.t_s,
    ];
    values(writer, input.into_iter().map(Some))
}
