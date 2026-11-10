use hydr::coupling::{
    CoupledLevelPoolForcingInterval, CoupledPoolInflow, CoupledPoolInflowBinding,
    CoupledPoolInflowValue,
};

pub fn binding(boundary: &str, pool: &str) -> CoupledPoolInflowBinding {
    CoupledPoolInflowBinding {
        boundary_id: boundary.into(),
        pool_id: pool.into(),
        input_id: format!("selected-input-{boundary}"),
        physical_support_id: format!("external-support-{boundary}"),
        boundary_assertion: "Prescribed synthetic inflow enters the named receiving pool".into(),
        ownership_assertion:
            "External origin excludes represented pool storage and local rainfall, with no duplicated supplied input".into(),
    }
}

pub fn rate(boundary: &str, rate_m3_per_s: f64) -> CoupledPoolInflow {
    CoupledPoolInflow {
        boundary_id: boundary.into(),
        input_id: format!("selected-input-{boundary}"),
        value: CoupledPoolInflowValue::Rate { rate_m3_per_s },
    }
}

pub fn volume(boundary: &str, volume_m3: f64) -> CoupledPoolInflow {
    CoupledPoolInflow {
        boundary_id: boundary.into(),
        input_id: format!("selected-input-{boundary}"),
        value: CoupledPoolInflowValue::IntervalVolume { volume_m3 },
    }
}

pub fn forcing(inflows: Vec<CoupledPoolInflow>) -> CoupledLevelPoolForcingInterval {
    span(0.0, 2.0, inflows)
}

pub fn span(
    start_s: f64,
    end_s: f64,
    inflows: Vec<CoupledPoolInflow>,
) -> CoupledLevelPoolForcingInterval {
    CoupledLevelPoolForcingInterval {
        start_s,
        end_s,
        sources: None,
        inflows,
    }
}
