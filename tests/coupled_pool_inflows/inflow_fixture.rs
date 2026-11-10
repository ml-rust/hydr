use hydr::coupling::{
    CoupledLevelPoolForcingInterval, CoupledPoolInflowBinding, CoupledPoolInflowRate,
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

pub fn rate(boundary: &str, rate_m3_per_s: f64) -> CoupledPoolInflowRate {
    CoupledPoolInflowRate {
        boundary_id: boundary.into(),
        input_id: format!("selected-input-{boundary}"),
        rate_m3_per_s,
    }
}

pub fn forcing(inflows: Vec<CoupledPoolInflowRate>) -> CoupledLevelPoolForcingInterval {
    CoupledLevelPoolForcingInterval {
        start_s: 0.0,
        end_s: 2.0,
        sources: None,
        inflows,
    }
}
