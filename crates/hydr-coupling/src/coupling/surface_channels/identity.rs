//! Bitwise checkpoint comparisons. Each rejection returns the path of the first differing field.

use super::snapshot::{
    SurfaceChannelCounterData, SurfaceChannelPortCounterData, SurfaceChannelSetupData,
    SurfaceChannelSnapshotData,
};
use super::{
    SurfaceChannelConfig, SurfaceChannelDefinition, SurfaceChannelInitialState, SurfaceChannelLink,
    SurfaceChannelPortTransfer, SurfaceChannelStepDiagnostics,
};
use crate::channel::{ChannelInitialState, ChannelSnapshotData};
use crate::solver::{InitialState, SnapshotData, SolverSetup};
use hydr_structures::internal::bits;
use std::fmt::Debug;

/// Path of the first differing field.
pub(super) type Check = Result<(), String>;

fn require(same: bool, path: impl FnOnce() -> String) -> Check {
    if same { Ok(()) } else { Err(path()) }
}

/// Equal values with equal debug text.
///
/// Debug text prints the shortest round-trip form of every float, so it separates every
/// finite bit pattern and signed zero. Equality rejects every NaN.
fn same_value<T: PartialEq + Debug>(a: &T, b: &T) -> bool {
    a == b && format!("{a:?}") == format!("{b:?}")
}

/// Compares two coordinator setups without their surface definitions.
pub(super) fn setup(a: &SurfaceChannelSetupData, b: &SurfaceChannelSetupData) -> Check {
    require(
        a.storage_exclusion_evidence == b.storage_exclusion_evidence,
        || "setup.storage_exclusion_evidence".into(),
    )?;
    require(same_config(&a.config, &b.config), || "setup.config".into())?;
    require(a.reaches.len() == b.reaches.len(), || {
        "setup.reaches".into()
    })?;
    for (index, (a, b)) in a.reaches.iter().zip(&b.reaches).enumerate() {
        require(
            a.id == b.id && a.definition.same_bits(&b.definition),
            || format!("setup.reaches[{index}] ({:?})", a.id),
        )?;
    }
    require(a.links.len() == b.links.len(), || "setup.links".into())?;
    for (index, (a, b)) in a.links.iter().zip(&b.links).enumerate() {
        require(same_link(a, b), || {
            format!("setup.links[{index}] ({:?})", a.id)
        })?;
    }
    require(a.ports.len() == b.ports.len(), || "setup.ports".into())?;
    for (index, (a, b)) in a.ports.iter().zip(&b.ports).enumerate() {
        require(a == b, || format!("setup.ports[{index}] ({:?})", a.id))?;
    }
    require(same_value(&a.pools, &b.pools), || "setup.pools".into())?;
    require(same_value(&a.pool_support, &b.pool_support), || {
        "setup.pool_support".into()
    })?;
    require(same_value(&a.pool_sources, &b.pool_sources), || {
        "setup.pool_sources".into()
    })?;
    require(same_value(&a.pool_inflows, &b.pool_inflows), || {
        "setup.pool_inflows".into()
    })?;
    require(same_value(&a.storage, &b.storage), || {
        "setup.storage".into()
    })?;
    require(a.structures.len() == b.structures.len(), || {
        "setup.structures".into()
    })?;
    for (index, (a, b)) in a.structures.iter().zip(&b.structures).enumerate() {
        require(same_value(a, b), || {
            format!("setup.structures[{index}] ({:?})", a.element.id)
        })?;
    }
    require(same_value(&a.pool_ends, &b.pool_ends), || {
        "setup.pool_ends".into()
    })?;
    require(same_value(&a.operating, &b.operating), || {
        "setup.operating".into()
    })?;
    require(
        same_value(&a.geometry_schedules, &b.geometry_schedules),
        || "setup.geometry_schedules".into(),
    )?;
    require(
        same_value(&a.pool_curve_schedules, &b.pool_curve_schedules),
        || "setup.pool_curve_schedules".into(),
    )?;
    require(
        same_value(&a.surface_bed_schedules, &b.surface_bed_schedules),
        || "setup.surface_bed_schedules".into(),
    )?;
    require(same_value(&a.runoff, &b.runoff), || "setup.runoff".into())
}

fn same_config(a: &SurfaceChannelConfig, b: &SurfaceChannelConfig) -> bool {
    let values = |c: &SurfaceChannelConfig| {
        [
            c.head_absolute_m,
            c.head_relative,
            c.transfer_absolute_m3,
            c.transfer_relative,
            c.mass_absolute_m3,
            c.mass_relative,
            c.minimum_step_s,
            c.maximum_step_s,
            c.contraction_target,
        ]
    };
    a.iteration_limit == b.iteration_limit
        && a.retry_limit == b.retry_limit
        && bits(&values(a), &values(b))
}

fn same_link(a: &SurfaceChannelLink, b: &SurfaceChannelLink) -> bool {
    let values = |l: &SurfaceChannelLink| {
        [
            l.crest_elevation_m,
            l.crest_length_m,
            l.weir_coefficient,
            l.reach_tangent[0],
            l.reach_tangent[1],
        ]
    };
    a.id == b.id
        && a.surface_cell == b.surface_cell
        && a.reach_id == b.reach_id
        && a.channel_cell == b.channel_cell
        && a.applicability == b.applicability
        && a.mapping_evidence == b.mapping_evidence
        && bits(&values(a), &values(b))
}

/// Compares two original epochs and child states.
pub(super) fn initial(a: &SurfaceChannelInitialState, b: &SurfaceChannelInitialState) -> Check {
    require(a.time_s.to_bits() == b.time_s.to_bits(), || {
        "initial.time_s".into()
    })?;
    require(same_value(&a.surface, &b.surface), || {
        "initial.surface".into()
    })?;
    require(a.reaches.len() == b.reaches.len(), || {
        "initial.reaches".into()
    })?;
    for (index, ((a_id, a), (b_id, b))) in a.reaches.iter().zip(&b.reaches).enumerate() {
        require(a_id == b_id && same_reach_initial(a, b), || {
            format!("initial.reaches[{index}] ({a_id:?})")
        })?;
    }
    require(same_value(&a.pools, &b.pools), || "initial.pools".into())?;
    require(same_value(&a.runoff, &b.runoff), || "initial.runoff".into())
}

/// Compares an original surface state with a zero-history accepted surface state.
pub(super) fn surface_epoch(path: &str, a: &InitialState, b: &InitialState) -> Check {
    require(same_value(a, b), || path.to_string())
}

fn same_reach_initial(a: &ChannelInitialState, b: &ChannelInitialState) -> bool {
    a.time_s.to_bits() == b.time_s.to_bits()
        && bits(&a.volume_m3, &b.volume_m3)
        && bits(
            &a.longitudinal_momentum_m4_per_s,
            &b.longitudinal_momentum_m4_per_s,
        )
}

/// Checks each reach package of one frame against the synthesized coordinator definitions.
///
/// Each linked reach must carry its `surface-exchange:` bindings exactly as the definition
/// synthesizes them from the links.
pub(super) fn reach_definitions(
    prefix: &str,
    definition: &SurfaceChannelDefinition,
    reaches: &[(String, ChannelSnapshotData)],
) -> Check {
    let setup = definition.setup();
    require(reaches.len() == setup.reaches.len(), || {
        format!("{prefix}reaches")
    })?;
    for (index, ((id, data), (reach, coordinated))) in reaches
        .iter()
        .zip(setup.reaches.iter().zip(&definition.coordinated))
        .enumerate()
    {
        require(*id == reach.id, || {
            format!("{prefix}reaches[{index}] ({id:?})")
        })?;
        require(data.definition.same_bits(coordinated), || {
            format!("{prefix}reaches[{index}] ({id:?}).definition")
        })?;
    }
    Ok(())
}

/// Compares the static surface definition of a frame package with the coordinator's.
///
/// Forcing stays out: the surface engine compares forcing windows itself.
pub(super) fn surface_definition(
    prefix: &str,
    a: &SolverSetup,
    b: &SolverSetup,
    check_bed: bool,
) -> Check {
    let same_domain = if check_bed {
        same_value(&a.domain, &b.domain)
    } else {
        same_value(&a.domain.grid(), &b.domain.grid())
            && bits(a.domain.manning(), b.domain.manning())
            && a.domain.active() == b.domain.active()
            && same_value(a.domain.reference(), b.domain.reference())
    };
    require(
        same_domain
            && same_value(&a.land, &b.land)
            && same_value(&a.boundaries, &b.boundaries)
            && same_value(&a.config, &b.config),
        || format!("{prefix}surface.setup"),
    )
}

/// Compares a replayed package with the supplied package, children and receipt included.
pub(super) fn replayed(
    found: &SurfaceChannelSnapshotData,
    given: &SurfaceChannelSnapshotData,
) -> Check {
    surface("surface", &found.surface, &given.surface)?;
    require(found.reaches.len() == given.reaches.len(), || {
        "reaches".into()
    })?;
    for (index, ((id, a), (_, b))) in found.reaches.iter().zip(&given.reaches).enumerate() {
        require(a.same_bits(b), || format!("reaches[{index}] ({id:?})"))?;
    }
    require(same_value(&found.pools, &given.pools), || "pools".into())?;
    require(same_value(&found.operating, &given.operating), || {
        "operating".into()
    })?;
    require(found.geometry_epochs == given.geometry_epochs, || {
        "geometry_epochs".into()
    })?;
    require(found.pool_curve_epochs == given.pool_curve_epochs, || {
        "pool_curve_epochs".into()
    })?;
    require(found.surface_bed_epochs == given.surface_bed_epochs, || {
        "surface_bed_epochs".into()
    })?;
    require(same_value(&found.runoff, &given.runoff), || "runoff".into())?;
    counters("counters", &found.counters, &given.counters)?;
    match (&found.last_interval, &given.last_interval) {
        (Some(a), Some(b)) => {
            diagnostics("last_interval.diagnostics", &a.diagnostics, &b.diagnostics)
        }
        (None, None) => Ok(()),
        _ => Err("last_interval".into()),
    }
}

fn surface(path: &str, a: &SnapshotData, b: &SnapshotData) -> Check {
    require(same_value(&a.setup, &b.setup), || format!("{path}.setup"))?;
    require(same_value(&a.forcing_prefix, &b.forcing_prefix), || {
        format!("{path}.forcing_prefix")
    })?;
    require(same_value(&a.accepted, &b.accepted), || {
        format!("{path}.accepted")
    })?;
    require(same_value(&a.ledger, &b.ledger), || {
        format!("{path}.ledger")
    })?;
    require(a.accepted_steps == b.accepted_steps, || {
        format!("{path}.accepted_steps")
    })?;
    require(same_value(&a.last_step, &b.last_step), || {
        format!("{path}.last_step")
    })
}

fn counters(path: &str, a: &SurfaceChannelCounterData, b: &SurfaceChannelCounterData) -> Check {
    require(a.time_s.to_bits() == b.time_s.to_bits(), || {
        format!("{path}.time_s")
    })?;
    require(a.accepted_steps == b.accepted_steps, || {
        format!("{path}.accepted_steps")
    })?;
    require(a.links.len() == b.links.len(), || format!("{path}.links"))?;
    for (index, (a, b)) in a.links.iter().zip(&b.links).enumerate() {
        let link = || format!("{path}.links[{index}] ({:?})", a.link_id);
        require(a.link_id == b.link_id, link)?;
        for (name, x, y) in [
            (
                "surface_to_channel_m3",
                a.surface_to_channel_m3,
                b.surface_to_channel_m3,
            ),
            (
                "channel_to_surface_m3",
                a.channel_to_surface_m3,
                b.channel_to_surface_m3,
            ),
            (
                "roundoff_bound_m3",
                a.roundoff_bound_m3,
                b.roundoff_bound_m3,
            ),
        ] {
            require(x.to_bits() == y.to_bits(), || format!("{}.{name}", link()))?;
        }
    }
    port_counters(&format!("{path}.ports"), &a.ports, &b.ports)?;
    require(same_value(&a.structures, &b.structures), || {
        format!("{path}.structures")
    })?;
    require(same_value(&a.pool_ends, &b.pool_ends), || {
        format!("{path}.pool_ends")
    })?;
    require(same_value(&a.pool_inflows, &b.pool_inflows), || {
        format!("{path}.pool_inflows")
    })?;
    require(same_value(&a.runoff, &b.runoff), || {
        format!("{path}.runoff")
    })
}

fn port_counters(
    path: &str,
    a: &SurfaceChannelPortCounterData,
    b: &SurfaceChannelPortCounterData,
) -> Check {
    require(a.ledgers.len() == b.ledgers.len(), || {
        format!("{path}.ledgers")
    })?;
    for (index, (a, b)) in a.ledgers.iter().zip(&b.ledgers).enumerate() {
        let port = || format!("{path}.ledgers[{index}] ({:?})", a.port_id);
        require(a.port_id == b.port_id, port)?;
        for (name, x, y) in [
            (
                "surface_to_channel_m3",
                a.surface_to_channel_m3,
                b.surface_to_channel_m3,
            ),
            (
                "channel_to_surface_m3",
                a.channel_to_surface_m3,
                b.channel_to_surface_m3,
            ),
            (
                "roundoff_bound_m3",
                a.roundoff_bound_m3,
                b.roundoff_bound_m3,
            ),
        ] {
            require(x.to_bits() == y.to_bits(), || format!("{}.{name}", port()))?;
        }
    }
    require(
        bits(&a.last_signed_volume_m3, &b.last_signed_volume_m3),
        || format!("{path}.last_signed_volume_m3"),
    )?;
    for (name, x, y) in [
        ("surface_in_m3", a.surface_in_m3, b.surface_in_m3),
        ("surface_out_m3", a.surface_out_m3, b.surface_out_m3),
        ("paired_bound_m3", a.paired_bound_m3, b.paired_bound_m3),
    ] {
        require(x.to_bits() == y.to_bits(), || format!("{path}.{name}"))?;
    }
    Ok(())
}

fn diagnostics(
    path: &str,
    a: &SurfaceChannelStepDiagnostics,
    b: &SurfaceChannelStepDiagnostics,
) -> Check {
    require(a.retries == b.retries, || format!("{path}.retries"))?;
    require(a.iterations == b.iterations, || {
        format!("{path}.iterations")
    })?;
    for (name, x, y) in [
        ("start_s", a.start_s, b.start_s),
        ("end_s", a.end_s, b.end_s),
        ("requested_end_s", a.requested_end_s, b.requested_end_s),
        ("contraction", a.contraction, b.contraction),
        ("head_residual_m", a.head_residual_m, b.head_residual_m),
        (
            "transfer_residual_m3",
            a.transfer_residual_m3,
            b.transfer_residual_m3,
        ),
        (
            "storage_before_m3",
            a.storage_before_m3,
            b.storage_before_m3,
        ),
        ("storage_after_m3", a.storage_after_m3, b.storage_after_m3),
        (
            "surface_exchange_in_m3",
            a.surface_exchange_in_m3,
            b.surface_exchange_in_m3,
        ),
        (
            "surface_exchange_out_m3",
            a.surface_exchange_out_m3,
            b.surface_exchange_out_m3,
        ),
        (
            "channel_exchange_in_m3",
            a.channel_exchange_in_m3,
            b.channel_exchange_in_m3,
        ),
        (
            "channel_exchange_out_m3",
            a.channel_exchange_out_m3,
            b.channel_exchange_out_m3,
        ),
        ("paired_bound_m3", a.paired_bound_m3, b.paired_bound_m3),
        ("mass_residual_m3", a.mass_residual_m3, b.mass_residual_m3),
        ("mass_bound_m3", a.mass_bound_m3, b.mass_bound_m3),
    ] {
        require(x.to_bits() == y.to_bits(), || format!("{path}.{name}"))?;
    }
    require(a.links.len() == b.links.len(), || format!("{path}.links"))?;
    for (index, (a, b)) in a.links.iter().zip(&b.links).enumerate() {
        let link = || format!("{path}.links[{index}] ({:?})", a.link_id);
        require(a.link_id == b.link_id && a.regime == b.regime, link)?;
        for (name, x, y) in [
            ("signed_volume_m3", a.signed_volume_m3, b.signed_volume_m3),
            ("weir_volume_m3", a.weir_volume_m3, b.weir_volume_m3),
            ("level_factor", a.level_factor, b.level_factor),
            ("donor_factor", a.donor_factor, b.donor_factor),
            (
                "midpoint_surface_stage_m",
                a.midpoint_surface_stage_m,
                b.midpoint_surface_stage_m,
            ),
            (
                "midpoint_channel_stage_m",
                a.midpoint_channel_stage_m,
                b.midpoint_channel_stage_m,
            ),
        ] {
            require(x.to_bits() == y.to_bits(), || format!("{}.{name}", link()))?;
        }
    }
    require(
        a.port_paired_bound_m3.to_bits() == b.port_paired_bound_m3.to_bits(),
        || format!("{path}.port_paired_bound_m3"),
    )?;
    require(a.ports.len() == b.ports.len(), || format!("{path}.ports"))?;
    for (index, (a, b)) in a.ports.iter().zip(&b.ports).enumerate() {
        require(same_port_transfer(a, b), || {
            format!("{path}.ports[{index}] ({:?})", a.port_id)
        })?;
    }
    require(a.structures.len() == b.structures.len(), || {
        format!("{path}.structures")
    })?;
    for (index, (a, b)) in a.structures.iter().zip(&b.structures).enumerate() {
        require(same_value(a, b), || {
            format!("{path}.structures[{index}] ({:?})", a.structure_id)
        })?;
    }
    require(same_value(&a.pool_ends, &b.pool_ends), || {
        format!("{path}.pool_ends")
    })?;
    require(same_value(&a.pools, &b.pools), || format!("{path}.pools"))?;
    require(same_value(&a.runoff, &b.runoff), || {
        format!("{path}.runoff")
    })?;
    require(same_value(&a.geometry, &b.geometry), || {
        format!("{path}.geometry")
    })
}

fn same_port_transfer(a: &SurfaceChannelPortTransfer, b: &SurfaceChannelPortTransfer) -> bool {
    let values = |t: &SurfaceChannelPortTransfer| {
        [
            t.signed_volume_m3,
            t.reach_momentum_impulse_m4_per_s,
            t.surface_momentum_impulse_m4_per_s[0],
            t.surface_momentum_impulse_m4_per_s[1],
            t.junction_reaction_m4_per_s[0],
            t.junction_reaction_m4_per_s[1],
            t.signal_speed_m_per_s,
            t.paired_gap_m3,
            t.paired_bound_m3,
        ]
    };
    a.port_id == b.port_id
        && a.regime == b.regime
        && a.reversed == b.reversed
        && bits(&values(a), &values(b))
}
