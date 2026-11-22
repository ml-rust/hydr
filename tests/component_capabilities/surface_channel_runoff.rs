//! Surface/channel runoff declarations checked against runs and admission.

use crate::surface_channel_support::pools_fixture::{Batch, Result, inflow_binding};
use crate::surface_channel_support::runoff_fixture::{
    OUTLET, POND, Receiver, SOURCE, forcing, outlet, parts,
};
use crate::surface_channels_fixture::{Outcome, construction_outcome, trial_outcome};
use crate::surface_channels_variants::Variant;
use hydr::capability::{BoundaryKind, ComponentId, EndpointId, UnsupportedReason};
use hydr::coupling::{
    SurfaceChannelRunoffDestination, SurfaceChannelRunoffShare, SurfaceChannelSetup,
    surface_channel_capabilities,
};
use hydr::solver::{ForcingInterval, ForcingSchedule};

/// Requested duration of every one-interval case, in s.
const DT: f64 = 0.25;

/// One rejection case: outcome, expected variant and message fragment.
type Rejection = (Outcome, Variant, &'static str);

/// The runoff case into `receiver` after `edit`, built once.
fn edited(receiver: Receiver, edit: impl FnOnce(&mut SurfaceChannelSetup)) -> Outcome {
    let (mut setup, initial) = parts(receiver)?;
    edit(&mut setup);
    construction_outcome(setup, initial)
}

/// One trial of the runoff case into `receiver` after `edit`.
fn trial(receiver: Receiver, edit: impl FnOnce(&mut SurfaceChannelSetup)) -> Outcome {
    let (mut setup, initial) = parts(receiver)?;
    edit(&mut setup);
    let batch = Batch::new(crate::runtime::client(), setup, initial)?;
    trial_outcome(&batch, DT, &forcing(0.0, DT, 0.0))
}

/// Edits the runoff declaration of `setup`.
fn runoff(
    setup: &mut SurfaceChannelSetup,
    edit: impl FnOnce(&mut hydr::coupling::SurfaceChannelRunoff),
) {
    if let Some(runoff) = setup.runoff.as_mut() {
        edit(runoff);
    }
}

/// Rain of `rate` m/s on the suppressed surface cell over the whole surface window.
fn rain_on_the_owned_cell(setup: &mut SurfaceChannelSetup, rate: f64) -> Result {
    let surface = &setup.surface;
    setup.surface.forcing = ForcingSchedule::new(
        &surface.domain,
        &surface.boundaries,
        vec![ForcingInterval {
            start: 0.0,
            end: 1000.0,
            rain: vec![rate],
            potential_evaporation: vec![0.0],
            withdrawal: vec![0.0],
            boundaries: Vec::new(),
        }],
    )?;
    Ok(())
}

/// Cases for each runoff `RejectedByAdmission` entry, or `None` for an unknown id.
pub fn rejection_cases(id: &str) -> Option<Vec<Rejection>> {
    use Variant::Input;
    Some(match id {
        "unmapped-runoff-outlet" => vec![
            (
                edited(Receiver::Pool, |s| runoff(s, |r| r.outlets.clear())),
                Input,
                "unmapped-runoff-outlet",
            ),
            (
                edited(Receiver::Pool, |s| {
                    runoff(s, |r| r.outlets[0].outlet_id = "unknown".into());
                }),
                Input,
                "unmapped-runoff-outlet",
            ),
        ],
        "duplicate-runoff-outlet" => vec![(
            edited(Receiver::Pool, |s| {
                runoff(s, |r| r.outlets.push(outlet(Receiver::Reach, 3.0)));
            }),
            Input,
            "duplicate-runoff-outlet",
        )],
        // The reach starts at 0.5 m, above an invert at 0.25 m.
        "runoff-outlet-submerged" => vec![(
            trial(Receiver::Reach, |s| {
                runoff(s, |r| r.outlets = vec![outlet(Receiver::Reach, 0.25)]);
            }),
            Input,
            "runoff-outlet-submerged",
        )],
        "runoff-duplicate-writer" => vec![(
            edited(Receiver::Pool, |s| {
                let mut binding = inflow_binding("feed", POND);
                binding.physical_support_id = SOURCE.into();
                s.pool_inflows.push(binding);
            }),
            Input,
            "runoff-duplicate-writer",
        )],
        "suppressed-rain" => {
            let (mut setup, initial) = match parts(Receiver::Pool) {
                Ok(parts) => parts,
                Err(error) => return Some(vec![(Err(error), Input, "suppressed-rain")]),
            };
            let outcome = rain_on_the_owned_cell(&mut setup, 1e-6)
                .and_then(|()| construction_outcome(setup, initial));
            vec![(outcome, Input, "suppressed-rain")]
        }
        _ => return None,
    })
}

/// The runoff endpoint delivers water into both admitted destinations.
///
/// Frozen criteria, fixed before the first run: the outlet endpoint admits only the no-feedback
/// runoff outlet, one interval into the pond and one into the reach cell each deliver positive
/// runoff, and the subcatchment nests and attaches here.
#[test]
fn surface_channels_deliver_runoff_through_the_declared_outlet() -> Result {
    let declared = surface_channel_capabilities();
    let endpoint = declared
        .endpoints
        .iter()
        .find(|endpoint| endpoint.id == EndpointId::Outlet)
        .ok_or("no outlet endpoint")?;
    assert_eq!(endpoint.admitted, [BoundaryKind::RunoffOutletNoFeedback]);
    for receiver in [Receiver::Pool, Receiver::Reach] {
        let (setup, initial) = parts(receiver)?;
        let batch = Batch::new(crate::runtime::client(), setup, initial)?;
        let next = batch.trial_to(DT, &forcing(0.0, DT, 0.0))?.into_next();
        let step = next.last_step()?.ok_or("an accepted interval")?;
        let [transfer] = step.runoff.as_slice() else {
            return Err("one delivery lane".into());
        };
        assert_eq!(transfer.outlet_id, OUTLET);
        assert!(transfer.volume_m3 > 0.0, "{receiver:?}: {transfer:?}");
    }
    assert!(declared.nested.contains(&ComponentId::Subcatchment));
    let child = hydr::runoff::subcatchment_capabilities();
    assert!(child.attachable_to.contains(&ComponentId::SurfaceChannels));
    Ok(())
}

/// Surface cells stay outside runoff delivery.
///
/// Frozen criteria, fixed before the first run: `runoff-attachment` is `NotAttached`, a
/// surface-cell destination builds, and its first trial rejects as input naming the entry.
#[test]
fn surface_cell_runoff_stays_unattached() -> Result {
    let entry = surface_channel_capabilities()
        .unsupported_by_id("runoff-attachment")
        .ok_or("`runoff-attachment` is undeclared")?;
    assert_eq!(entry.reason, UnsupportedReason::NotAttached);
    let cells = SurfaceChannelRunoffDestination::SurfaceCells {
        cells: vec![SurfaceChannelRunoffShare {
            index: 0,
            weight: 1.0,
        }],
    };
    let (variant, reason) = trial(Receiver::Pool, |s| {
        runoff(s, |r| r.outlets[0].destination = cells);
    })?
    .ok_or("a surface-cell destination stepped")?;
    assert_eq!(variant, Variant::Input, "{reason}");
    assert!(reason.starts_with("runoff-attachment"), "{reason}");
    Ok(())
}
