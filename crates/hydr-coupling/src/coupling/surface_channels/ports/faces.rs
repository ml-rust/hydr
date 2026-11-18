//! Surface face ownership of one port: active sides, open faces, axis, side, and face line.

use super::super::SurfaceChannelError;
use crate::solver::{Axis, BoundaryFace};
use hydr_core::model::Domain;
use std::collections::BTreeSet;

/// Capability entry of every face ownership rejection.
pub(super) const FACE_OWNERSHIP: &str = "port-face-ownership";

/// Checked faces of one port, sorted along their face line.
#[derive(Debug)]
pub(super) struct LineFaces {
    pub faces: Vec<BoundaryFace>,
    /// The ghost lies on the face's negative side.
    pub ghost_left: bool,
    /// Sum of face widths along the line, in m.
    pub width_m: f64,
    /// Bed of each face's active surface cell, in face order, in m.
    pub beds_m: Vec<f64>,
}

/// `(line, position)` of `face`: X faces lie on a column line, Y faces on a row line.
fn line(face: BoundaryFace) -> (usize, usize) {
    match face.axis {
        Axis::X => (face.column, face.row),
        Axis::Y => (face.row, face.column),
    }
}

/// Active cells on the negative and positive side of `face`, or `None` off the grid.
fn sides(domain: &Domain, face: BoundaryFace) -> Option<[Option<usize>; 2]> {
    let (nx, ny) = domain.grid().dimensions();
    let active = |cell: usize| domain.active()[cell].then_some(cell);
    let BoundaryFace { axis, column, row } = face;
    match axis {
        Axis::X if row < ny && column <= nx => Some([
            column.checked_sub(1).and_then(|c| active(row * nx + c)),
            (column < nx).then(|| row * nx + column).and_then(active),
        ]),
        Axis::Y if column < nx && row <= ny => Some([
            row.checked_sub(1).and_then(|r| active(r * nx + column)),
            (row < ny).then(|| row * nx + column).and_then(active),
        ]),
        _ => None,
    }
}

/// Checks the faces of port `id` and claims them in `claimed`.
///
/// Every face has one active surface cell, is no caller open face, and belongs to no other
/// port. The faces share one axis, one face line, and one surface side, with no gap.
pub(super) fn check_faces(
    domain: &Domain,
    open: &BTreeSet<BoundaryFace>,
    claimed: &mut BTreeSet<BoundaryFace>,
    id: &str,
    declared: &[BoundaryFace],
) -> Result<LineFaces, SurfaceChannelError> {
    let reject = |detail: String| {
        SurfaceChannelError::input(format!("{FACE_OWNERSHIP}: port={id:?} {detail}"))
    };
    let Some(&first) = declared.first() else {
        return Err(reject(
            "has no faces: bind at least one surface face".into(),
        ));
    };
    let (dx, dy) = domain.grid().spacing();
    let mut faces = declared.to_vec();
    faces.sort_by_key(|face| line(*face).1);
    let mut ghost_left = None;
    let mut width_m = 0.0;
    let mut beds_m = Vec::with_capacity(faces.len());
    for (slot, face) in faces.iter().enumerate() {
        if !claimed.insert(*face) {
            return Err(reject(format!(
                "face={face:?} is already bound to a port: bind each face to one port once"
            )));
        }
        let Some([left, right]) = sides(domain, *face) else {
            return Err(reject(format!(
                "face={face:?} lies outside the grid: bind faces of the surface grid"
            )));
        };
        if left.is_none() && right.is_none() {
            return Err(reject(format!(
                "face={face:?} has no active surface cell: bind faces of active cells"
            )));
        }
        if left.is_some() && right.is_some() {
            return Err(reject(format!(
                "face={face:?} has two active cells: bind faces whose other side is inactive or off the grid"
            )));
        }
        if open.contains(face) {
            return Err(reject(format!(
                "face={face:?} is a caller open face: remove it from the open faces or from the port"
            )));
        }
        if face.axis != first.axis || line(*face).0 != line(first).0 {
            return Err(reject(format!(
                "face={face:?}, first face={first:?}: bind faces of one axis on one face line"
            )));
        }
        let left_ghost = left.is_none();
        if *ghost_left.get_or_insert(left_ghost) != left_ghost {
            return Err(reject(format!(
                "face={face:?}: its active cell lies on the other side from {:?}; bind faces with the surface on one side",
                faces[0]
            )));
        }
        if slot > 0 && line(*face).1 != line(faces[slot - 1]).1 + 1 {
            return Err(reject(format!(
                "face={face:?} follows {:?} with a gap: bind one contiguous face line",
                faces[slot - 1]
            )));
        }
        width_m += if face.axis == Axis::X { dy } else { dx };
        if let Some(cell) = left.or(right) {
            beds_m.push(domain.bed()[cell]);
        }
    }
    Ok(LineFaces {
        faces,
        ghost_left: ghost_left.unwrap_or(true),
        width_m,
        beds_m,
    })
}

#[cfg(test)]
mod tests {
    use super::super::fixture::{TestResult, case, y_face};
    use super::*;

    #[test]
    fn faces_sort_along_their_line_and_sum_their_widths() -> TestResult {
        let (setup, _) = case(&[])?;
        let mut claimed = BTreeSet::new();
        let line = check_faces(
            &setup.surface.domain,
            &BTreeSet::new(),
            &mut claimed,
            "p",
            &[y_face(2, 2), y_face(0, 2), y_face(1, 2)],
        )?;
        assert_eq!(line.faces, [y_face(0, 2), y_face(1, 2), y_face(2, 2)]);
        assert!(!line.ghost_left);
        assert_eq!(line.width_m.to_bits(), 6.0f64.to_bits());
        // The fixture surface bed is 1 m everywhere.
        assert_eq!(line.beds_m, [1.0; 3]);
        assert_eq!(claimed.len(), 3);
        Ok(())
    }
}
