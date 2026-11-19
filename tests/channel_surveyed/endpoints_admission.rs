//! Rejections of surveyed scalar endpoints outside the wet subcritical branch.
//!
//! Frozen criterion, `resource/surveyed-endpoint-attachments-plan.md` §6, "Rejections":
//! dry, critical, supercritical and cap cases each reject with the entry's text. The reach
//! state stays unchanged bit for bit.
//!
//! The entries are `stage-outside-wet-subcritical`, `discharge-outside-wet-subcritical` and
//! `quiescent-reservoir-outside-wet-subcritical`. Each rejection is a `ChannelError::Numerical`
//! whose text names the failed quantity, as for rectangular and triangular reaches.

use super::support::{TestResult, forcing, profile, reach};
use super::surveyed_profiles::THREE_BANDS;
use hydr::channel::{ChannelBoundary, ChannelConfig, ChannelError, ChannelSurveyedConfig};

/// Three-band area at depth 1.5 m: `3 + 4 (0.5) + 0.25 (0.5)`.
const AREA: f64 = 5.125;
/// Three-band top width at depth 1.5 m.
const WIDTH: f64 = 4.5;
const GRAVITY: f64 = 9.81;
/// Cap depth of the three-band profile, in metres.
const CAP: f64 = 3.0;

#[derive(Clone, Copy)]
enum End {
    Left,
    Right,
}

impl End {
    fn outward(self) -> f64 {
        match self {
            Self::Left => -1.0,
            Self::Right => 1.0,
        }
    }
}

/// `c = sqrt(g A / T)` of the interior at depth 1.5 m.
fn sound() -> f64 {
    (GRAVITY * AREA / WIDTH).sqrt()
}

/// One cell of area `area` and global discharge `discharge`, with `policy` at `end`.
///
/// Returns the rejection text after checking the reach is unchanged.
fn rejection(area: f64, discharge: f64, end: End, policy: ChannelBoundary) -> TestResult<String> {
    let config = ChannelConfig {
        gravity: GRAVITY,
        ..ChannelConfig::default()
    };
    let section = profile(THREE_BANDS.name, THREE_BANDS.points, THREE_BANDS.cap);
    let origin = reach::<crate::runtime::Runtime>(
        crate::runtime::client(),
        section,
        &[1.0],
        &[0.0],
        &[area],
        &[discharge],
        config,
        ChannelSurveyedConfig::default(),
    )?;
    let before = origin.snapshot_data()?;
    let boundaries = match end {
        End::Left => [policy, ChannelBoundary::Wall],
        End::Right => [ChannelBoundary::Wall, policy],
    };
    let duration = 1.0 / 1024.0;
    let outcome = origin.step(duration, &forcing(0.0, duration, boundaries));
    assert_eq!(
        origin.snapshot_data()?,
        before,
        "a rejection changed the reach"
    );
    match outcome {
        Ok(_) => Err("the endpoint was admitted".into()),
        Err(ChannelError::Numerical { context, .. }) => Ok(context),
        Err(other) => Err(format!("expected a numerical rejection, got {other}").into()),
    }
}

fn policies(stage: f64, discharge: f64, head: f64) -> [ChannelBoundary; 3] {
    [
        ChannelBoundary::Stage { elevation_m: stage },
        ChannelBoundary::Discharge {
            discharge_m3_per_s: discharge,
        },
        ChannelBoundary::QuiescentReservoir {
            elevation_m: head,
            entrance_loss_coefficient: 0.5,
        },
    ]
}

fn expect(text: &str, fragment: &str) {
    assert!(text.contains(fragment), "`{text}` lacks `{fragment}`");
    assert!(
        text.contains("Supply a resolved wet subcritical target"),
        "`{text}` lacks the next action"
    );
}

#[test]
fn dry_interiors_and_dry_traces_reject() -> TestResult {
    for end in [End::Left, End::Right] {
        for policy in policies(1.5, 0.0, 1.5) {
            expect(&rejection(0.0, 0.0, end, policy)?, "interior depth is dry");
        }
        // A stage or reservoir head below the bed leaves the trace dry.
        let [stage, _, reservoir] = policies(-0.5, 0.0, -0.5);
        for policy in [stage, reservoir] {
            expect(&rejection(AREA, 0.0, end, policy)?, "trace depth is dry");
        }
    }
    Ok(())
}

#[test]
fn critical_and_supercritical_interiors_reject() -> TestResult {
    for end in [End::Left, End::Right] {
        for velocity in [sound(), 2.0 * sound()] {
            // Global discharge whose outward velocity equals or exceeds c.
            let discharge = end.outward() * velocity * AREA;
            for policy in policies(1.5, discharge, 1.5) {
                expect(
                    &rejection(AREA, discharge, end, policy)?,
                    "incoming characteristic is unresolved",
                );
            }
        }
    }
    Ok(())
}

#[test]
fn supercritical_traces_reject() -> TestResult {
    for end in [End::Left, End::Right] {
        // A deep drawdown from rest accelerates the leaving trace past its sound speed.
        let [stage, _, reservoir] = policies(0.05, 0.0, 0.05);
        for policy in [stage, reservoir] {
            expect(
                &rejection(AREA, 0.0, end, policy)?,
                "incoming characteristic is unresolved",
            );
        }
        // More outflow than the sonic trace carries has no subcritical trace.
        let leaving = ChannelBoundary::Discharge {
            discharge_m3_per_s: end.outward() * 1000.0,
        };
        expect(
            &rejection(AREA, 0.0, end, leaving)?,
            "requested discharge reaches the lower sonic uncertainty band",
        );
    }
    Ok(())
}

#[test]
fn traces_above_the_cap_reject() -> TestResult {
    for end in [End::Left, End::Right] {
        let entering = -end.outward() * 1000.0;
        for policy in policies(CAP + 0.5, entering, 10.0 * CAP) {
            expect(
                &rejection(AREA, 0.0, end, policy)?,
                "trace depth exceeds the geometric cap",
            );
        }
    }
    Ok(())
}
