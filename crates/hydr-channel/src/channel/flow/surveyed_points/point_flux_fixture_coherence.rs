use super::super::super::{
    point_flux_types::{PointFluxStatus, ResolvedPointFluxes},
    point_root_session::PointRootSession,
};
use super::{diagnostics, frozen, support};
use frozen::Result;
use numr::{dtype::DType, runtime::Runtime};
use std::collections::BTreeMap;

/// Checks every selection, and that each face holds its owning selection's values.
pub(in crate::channel::flow::surveyed_points) fn coherent<R>(
    session: &PointRootSession<'_, '_, R>,
    result: &ResolvedPointFluxes<R>,
) -> Result
where
    R: Runtime<DType = DType>,
    R::Client: support::DepthClient<R>,
{
    assert_eq!(result.callback_points, session.callback_points());
    assert_eq!(result.point_limit, session.point_limit());
    for selection in &result.selections {
        diagnostics::selection(session, selection)?;
    }
    let owners = result.invocation_owner.try_to_vec::<i64>()?;
    let status = result.status.try_to_vec::<i64>()?;
    let index = result.selection.try_to_vec::<i64>()?;
    let face = [
        frozen::endpoints(&result.physical_water)?,
        frozen::endpoints(&result.physical_momentum)?,
        frozen::endpoints(&result.joint_water)?,
        frozen::endpoints(&result.joint_momentum)?,
    ];
    let valid = [
        result.physical_valid.try_to_vec::<f64>()?,
        result.joint_valid.try_to_vec::<f64>()?,
    ];
    let held = result
        .selections
        .iter()
        .map(|selection| {
            Ok((
                selection.candidates.invocation_owner.try_to_vec::<i64>()?,
                [
                    frozen::endpoints(&selection.physical_water)?,
                    frozen::endpoints(&selection.physical_momentum)?,
                    frozen::endpoints(&selection.joint_water)?,
                    frozen::endpoints(&selection.joint_momentum)?,
                ],
                [
                    selection.physical_valid.try_to_vec::<f64>()?,
                    selection.joint_valid.try_to_vec::<f64>()?,
                ],
                selection.status.try_to_vec::<i64>()?,
            ))
        })
        .collect::<Result<Vec<_>>>()?;
    for (i, owner) in owners.iter().enumerate() {
        let (rows, endpoints, flags, statuses) = held
            .get(usize::try_from(index[i])?)
            .ok_or("Face names an absent selection")?;
        let p = rows
            .iter()
            .position(|row| row == owner)
            .ok_or("Owning selection lacks its face")?;
        for (value, expected) in face.iter().zip(endpoints) {
            assert_eq!(value[i], expected[p], "face {owner} mixes selections");
        }
        assert_eq!(valid[0][i], flags[0][p]);
        assert_eq!(valid[1][i], flags[1][p]);
        let original = statuses[p];
        if original == PointFluxStatus::FluxBudgetExceeded as i64 {
            assert!(
                [
                    PointFluxStatus::FluxBudgetExceeded as i64,
                    PointFluxStatus::WorkLimit as i64,
                    PointFluxStatus::RootBracketUnresolved as i64,
                ]
                .contains(&status[i])
            );
        } else if status[i] == PointFluxStatus::WaveEnvelopeUnresolved as i64 {
            assert_eq!(original, PointFluxStatus::Accepted as i64);
            assert_eq!(result.wave_valid.try_to_vec::<f64>()?[i], 0.);
        } else {
            assert_eq!(status[i], original);
        }
    }
    Ok(())
}

/// Per-face status from the initial selection and its positive-area reselection.
pub(in crate::channel::flow::surveyed_points) fn baseline<R: Runtime<DType = DType>>(
    result: &ResolvedPointFluxes<R>,
) -> Result<Vec<i64>> {
    let owners = result.invocation_owner.try_to_vec::<i64>()?;
    let position = owners
        .iter()
        .enumerate()
        .map(|(i, owner)| (*owner, i))
        .collect::<BTreeMap<_, _>>();
    let count = if result.positive_support.is_some() {
        2
    } else {
        1
    };
    let mut status = vec![-1; owners.len()];
    for selection in result.selections.iter().take(count) {
        let rows = selection.candidates.invocation_owner.try_to_vec::<i64>()?;
        for (owner, value) in rows.iter().zip(selection.status.try_to_vec::<i64>()?) {
            let face = position
                .get(owner)
                .ok_or("Selection owner lies outside the faces")?;
            status[*face] = value;
        }
    }
    Ok(status)
}
