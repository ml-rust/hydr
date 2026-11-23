//! Text transforms mapping current coordinator package Debug text back through earlier formats.

pub fn before_ports(text: &str) -> String {
    text.replace(
        "hydr-surface-channel-checkpoint-4",
        "hydr-surface-channel-checkpoint-3",
    )
    .replace(
        "hydr-surface-channel-numr-cpu-6",
        "hydr-surface-channel-numr-cpu-4",
    )
    .replace("hydr-channel-1d-numr-cpu-3", "hydr-channel-1d-numr-cpu-2")
    .replace("hydr-swe-numr-cpu-2", "hydr-swe-numr-cpu-1")
    .replace(
        ", ports: SurfaceChannelPortCounterData { ledgers: [], last_signed_volume_m3: [], surface_in_m3: 0.0, surface_out_m3: 0.0, paired_bound_m3: 0.0 }",
        "",
    )
    .replace(", ports: [], port_paired_bound_m3: 0.0", "")
    .replace(
        "], ports: [], storage_exclusion_evidence",
        "], storage_exclusion_evidence",
    )
    .replace("left: Some(Wall)", "left: Wall")
    .replace("right: Some(Wall)", "right: Wall")
}

pub fn before_pools(text: &str) -> String {
    text.replace(
        "hydr-surface-channel-checkpoint-5",
        "hydr-surface-channel-checkpoint-4",
    )
    .replace(
        "hydr-surface-channel-numr-cpu-7",
        "hydr-surface-channel-numr-cpu-6",
    )
    .replace(
        ", pools: [], pool_support: [], pool_sources: None, pool_inflows: [], storage: LevelPoolConfig { volume_absolute: 1e-9, volume_relative: 1e-12 }, structures: [], pool_ends: [], operating: None",
        "",
    )
    .replace(")], pools: [] }, surface: ", ")] }, surface: ")
    .replace(", pools: None, operating: None, counters: ", ", counters: ")
    .replace(", structures: [], pool_ends: [], pool_inflows: []", "")
    .replace(", pool_sources: None, pool_inflows: []", "")
    .replace(", structures: [], pool_ends: [], pools: []", "")
}

pub fn before_root_probe(text: &str) -> String {
    text.replace(
        "hydr-surface-channel-checkpoint-6",
        "hydr-surface-channel-checkpoint-5",
    )
    .replace(
        "hydr-surface-channel-numr-cpu-8",
        "hydr-surface-channel-numr-cpu-7",
    )
    .replace("hydr-channel-checkpoint-3", "hydr-channel-checkpoint-2")
    .replace("hydr-channel-1d-numr-cpu-4", "hydr-channel-1d-numr-cpu-3")
    .replace(", probe: Midpoint", "")
}

pub fn before_culverts(text: &str) -> String {
    text.replace(
        "hydr-surface-channel-checkpoint-7",
        "hydr-surface-channel-checkpoint-6",
    )
}

pub fn before_culvert_roots(text: &str) -> String {
    text.replace(
        "hydr-surface-channel-numr-cpu-9",
        "hydr-surface-channel-numr-cpu-8",
    )
}

pub fn before_runoff(text: &str) -> String {
    text.replace(
        "hydr-surface-channel-checkpoint-8",
        "hydr-surface-channel-checkpoint-7",
    )
    .replace(
        ", runoff: None, storage_exclusion_evidence",
        ", storage_exclusion_evidence",
    )
    .replace(
        ", pools: [], runoff: [] }, surface: ",
        ", pools: [] }, surface: ",
    )
    .replace(", pool_inflows: [], runoff: [] }", ", pool_inflows: [] }")
}

pub fn before_runoff_reports(text: &str) -> String {
    text.replace(
        "hydr-surface-channel-numr-cpu-10",
        "hydr-surface-channel-numr-cpu-9",
    )
    .replace(
        ", operating: None, runoff: None, counters: ",
        ", operating: None, counters: ",
    )
    .replace(
        ", runoff: SurfaceChannelRunoffCounterData { ledgers: [], split_bound_m3: 0.0 }",
        "",
    )
    .replace(
        ", pool_ends: [], pools: [], runoff: [] }",
        ", pool_ends: [], pools: [] }",
    )
    .replace(", runoff_storage_m3: 0.0, runoff_delivered_m3: 0.0", "")
}

pub fn before_surface_runoff(text: &str) -> String {
    text.replace(
        "hydr-surface-channel-numr-cpu-11",
        "hydr-surface-channel-numr-cpu-10",
    )
}

pub fn before_blockage(text: &str) -> String {
    text.replace(
        "hydr-surface-channel-checkpoint-9",
        "hydr-surface-channel-checkpoint-8",
    )
    .replace(", condition: AsBuilt }", " }")
}

pub fn before_geometry_schedules(text: &str) -> String {
    text.replace(
        "hydr-surface-channel-checkpoint-10",
        "hydr-surface-channel-checkpoint-9",
    )
    .replace(
        "hydr-surface-channel-numr-cpu-12",
        "hydr-surface-channel-numr-cpu-11",
    )
    .replace(", geometry_schedules: []", "")
    .replace(", geometry_epochs: []", "")
    .replace(", geometry: []", "")
}

pub fn before_pool_curve_epochs(text: &str) -> String {
    text.replace(
        "hydr-surface-channel-checkpoint-11",
        "hydr-surface-channel-checkpoint-10",
    )
    .replace(
        "hydr-surface-channel-numr-cpu-13",
        "hydr-surface-channel-numr-cpu-12",
    )
    .replace(", pool_curve_schedules: []", "")
    .replace(", pool_curve_epochs: []", "")
}

pub fn before_surface_bed_epochs(text: &str) -> String {
    text.replace(
        "hydr-surface-channel-checkpoint-12",
        "hydr-surface-channel-checkpoint-11",
    )
    .replace(
        "hydr-surface-channel-numr-cpu-14",
        "hydr-surface-channel-numr-cpu-13",
    )
    .replace(", surface_bed_schedules: []", "")
    .replace(", surface_bed_epochs: []", "")
}

pub fn before_channel_section_epochs(text: &str) -> String {
    text.replace(
        "hydr-surface-channel-checkpoint-13",
        "hydr-surface-channel-checkpoint-12",
    )
    .replace(
        "hydr-surface-channel-numr-cpu-15",
        "hydr-surface-channel-numr-cpu-14",
    )
    .replace(", channel_section_schedules: []", "")
    .replace(", channel_section_epochs: []", "")
}
