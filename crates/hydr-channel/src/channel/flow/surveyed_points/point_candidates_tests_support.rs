use super::super::super::{
    point_candidate_types::{PointCandidate, PointCandidateCertificates},
    point_root_session::PointRootSession,
    source_sonic_types::SonicSide,
};
use super::{Result, frozen, hydraulic, support};
use crate::channel::flow::surveyed::error::checked;
use numr::{dtype::DType, runtime::Runtime};

#[derive(Debug)]
pub(super) struct State {
    pub(super) owner: i64,
    pub(super) label: &'static str,
    pub(super) area: [f64; 2],
    pub(super) depth: [f64; 2],
    pub(super) velocity: Option<[f64; 2]>,
    pub(super) discharge: [f64; 2],
}
pub(super) fn actual<R>(
    session: &PointRootSession<'_, '_, R>,
    result: &PointCandidateCertificates<R>,
) -> Result<Vec<State>>
where
    R: Runtime<DType = DType>,
    R::Client: support::DepthClient<R>,
{
    let mut states = Vec::new();
    for candidate in &result.candidates {
        match candidate {
            PointCandidate::Exterior(e) => {
                let c = session.sources.geometry.client();
                let (area, q, u) = hydraulic(checked::<R, _>(
                    c,
                    "candidate source diagnostic bounds",
                    |a| {
                        Ok((
                            a.expansion_interval(&e.area)?,
                            a.expansion_interval(&e.discharge)?,
                            a.expansion_interval(&e.velocity)?,
                        ))
                    },
                ))?;
                let area = frozen::endpoints(&area)?;
                let q = frozen::endpoints(&q)?;
                let u = frozen::endpoints(&u)?;
                let h = frozen::endpoints(e.recovery.depth_m())?;
                for (i, owner) in e
                    .invocation_owner
                    .try_to_vec::<i64>()?
                    .into_iter()
                    .enumerate()
                {
                    states.push(State {
                        owner,
                        label: if e.side == SonicSide::Left {
                            "left-exterior"
                        } else {
                            "right-exterior"
                        },
                        area: area[i],
                        depth: h[i],
                        velocity: Some(u[i]),
                        discharge: q[i],
                    });
                }
            }
            PointCandidate::Dry(d) => {
                let area = frozen::endpoints(&d.query.area)?;
                let h = frozen::endpoints(&d.depth)?;
                for (i, owner) in d
                    .invocation_owner
                    .try_to_vec::<i64>()?
                    .into_iter()
                    .enumerate()
                {
                    states.push(State {
                        owner,
                        label: "dry",
                        area: area[i],
                        depth: h[i],
                        velocity: None,
                        discharge: [0., 0.],
                    });
                }
            }
            PointCandidate::Star(star) => {
                let w = &result.waves[star.wave];
                let depth = hydraulic(checked::<R, _>(
                    session.sources.geometry.client(),
                    "candidate star diagnostic depth",
                    |a| {
                        let selected = session
                            .sources
                            .select_rows(&w.star.parent_rows, &w.star.query)?;
                        Ok(session
                            .sources
                            .geometry
                            .recover_query(a, &selected.bands, &w.star.query)?
                            .state
                            .depth)
                    },
                ))?;
                let area = frozen::endpoints(&w.star.query.area)?;
                let q = frozen::endpoints(&w.star.discharge)?;
                let u = frozen::endpoints(&w.star.velocity)?;
                let h = frozen::endpoints(&depth)?;
                for (owner, row) in star
                    .invocation_owner
                    .try_to_vec::<i64>()?
                    .into_iter()
                    .zip(star.rows.try_to_vec::<i64>()?)
                {
                    let i = usize::try_from(row)?;
                    states.push(State {
                        owner,
                        label: "star",
                        area: area[i],
                        depth: h[i],
                        velocity: Some(u[i]),
                        discharge: q[i],
                    });
                }
            }
            PointCandidate::Fan(f) => {
                let cert = &f.certificate;
                let active = f.active.try_to_vec::<f64>()?;
                let area = frozen::endpoints(&cert.query.area)?;
                let q = frozen::endpoints(&cert.discharge)?;
                let u = frozen::endpoints(&cert.velocity)?;
                let h = frozen::endpoints(&cert.depth)?;
                for (i, owner) in cert
                    .invocation_owner
                    .try_to_vec::<i64>()?
                    .into_iter()
                    .enumerate()
                {
                    if active[i] == 1. {
                        states.push(State {
                            owner,
                            label: if cert.side == SonicSide::Left {
                                "left-sonic"
                            } else {
                                "right-sonic"
                            },
                            area: area[i],
                            depth: h[i],
                            velocity: Some(u[i]),
                            discharge: q[i],
                        });
                    }
                }
            }
            PointCandidate::Pending(_) => {}
        }
    }
    Ok(states)
}
pub(super) fn contains(actual: [f64; 2], expected: [u64; 2], context: &str) {
    let expected = expected.map(f64::from_bits);
    assert!(
        actual[0] <= expected[0] && expected[1] <= actual[1],
        "{context}: {actual:?} excludes {expected:?}"
    );
}
pub(super) fn label(label: &str) -> &str {
    match label {
        "vacuum" | "sonic-dry-limit" => "dry",
        x => x,
    }
}
