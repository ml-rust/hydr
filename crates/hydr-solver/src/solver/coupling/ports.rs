//! Coordinator port faces: static face topology and admission.
//!
//! A port binds one contiguous line of surface faces to one reach end. Each face has one
//! active surface cell. Its other side is an inactive cell or lies off the grid. Transport
//! replaces that side with the reach ghost state before the HLL flux.

use super::super::Solver;
use super::super::tensor::port_faces::{PortFaceLane, PortFaces};
use crate::contract::{Axis, BoundaryConditions, BoundaryFace, EngineError};
use crate::solver::FaceTransfer;
use numr::dtype::DType;
use numr::ops::IndexingOps;
use numr::runtime::{Runtime, RuntimeClient};
use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

/// Reach end that a port attaches to.
#[doc(hidden)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PortReachEnd {
    /// The first reach cell. Reach-positive discharge leaves the surface.
    Left,
    /// The last reach cell. Reach-positive discharge enters the surface.
    Right,
}

/// One port declaration: its faces, the reach end bed, and the reach end it attaches to.
#[doc(hidden)]
#[derive(Clone, Debug, PartialEq)]
pub struct SurfacePortSpec {
    /// Surface faces in any order. Admission sorts them along their face line.
    pub faces: Vec<BoundaryFace>,
    /// Reach end cell bed elevation in the surface datum, in m.
    pub ghost_bed_m: f64,
    /// Reach end that supplies the ghost state.
    pub reach_end: PortReachEnd,
}

#[derive(Debug)]
struct PortEntry {
    faces: Vec<BoundaryFace>,
    face_indices: Vec<usize>,
    ghost_left: bool,
}

/// Host description of an admitted port set. Ghosts share it to prove their port set.
#[derive(Debug)]
pub(super) struct PortLayout {
    ports: Vec<PortEntry>,
    lanes: usize,
    cell_count: usize,
    face_count: usize,
    boundaries: BoundaryConditions,
}

/// Admitted port faces of one surface grid, built once like [`super::SurfaceLinks`].
///
/// Holds at least one port. Port order follows the coordinator. Face lanes run in port order,
/// then along each face line in increasing row or column.
#[doc(hidden)]
#[derive(Clone)]
pub struct SurfacePorts<R: Runtime<DType = DType>> {
    layout: Arc<PortLayout>,
    pub(in crate::solver) faces: Arc<PortFaces<R>>,
}

impl<R: Runtime<DType = DType>> SurfacePorts<R> {
    pub(super) fn layout(&self) -> &Arc<PortLayout> {
        &self.layout
    }

    /// Returns the number of ports.
    pub fn len(&self) -> usize {
        self.layout.ports.len()
    }

    /// Returns false: a port set always holds at least one port.
    pub fn is_empty(&self) -> bool {
        self.layout.ports.is_empty()
    }

    /// Returns the number of port faces over every port.
    pub fn face_lanes(&self) -> usize {
        self.layout.lanes
    }

    /// Returns the faces of `port` in lane order, or `None` past the last port.
    pub fn faces(&self, port: usize) -> Option<&[BoundaryFace]> {
        self.layout
            .ports
            .get(port)
            .map(|entry| entry.faces.as_slice())
    }

    /// Returns whether the ghost of `port` is the left side of its faces.
    ///
    /// Water entering the surface then moves along the face's positive axis.
    pub fn ghost_left(&self, port: usize) -> Option<bool> {
        self.layout.ports.get(port).map(|entry| entry.ghost_left)
    }

    /// Rejects ports built for a different grid, face list, or boundary declaration.
    pub(in crate::solver) fn check(&self, solver: &Solver<R>) -> Result<(), EngineError> {
        let layout = &self.layout;
        let topology = &solver.context.topology.faces;
        let cells = solver.definition.domain.grid().cell_count();
        let reject = |reason: String| EngineError::rejected("surface ports", reason);
        if layout.cell_count != cells || layout.face_count != topology.len() {
            return Err(reject(format!(
                "ports cover {} cells and {} faces, engine has {cells} cells and {} faces: build the ports from this engine",
                layout.cell_count,
                layout.face_count,
                topology.len()
            )));
        }
        if layout.boundaries != solver.definition.boundaries {
            return Err(reject(
                "ports were built under different open faces: build the ports from this engine"
                    .into(),
            ));
        }
        for entry in &layout.ports {
            for (face, &index) in entry.faces.iter().zip(&entry.face_indices) {
                if topology.get(index).map(|support| support.face) != Some(*face) {
                    return Err(reject(format!(
                        "face={face:?} differs from engine face {index}: build the ports from this engine"
                    )));
                }
            }
        }
        Ok(())
    }

    /// Sums port face volumes of a step receipt into the surface and out of it, in m³.
    pub(in crate::solver) fn face_volumes(
        &self,
        faces: &[FaceTransfer],
    ) -> Result<[f64; 2], EngineError> {
        let mut inflow = 0.0_f64;
        let mut outflow = 0.0_f64;
        for entry in &self.layout.ports {
            let into = if entry.ghost_left { 1.0 } else { -1.0 };
            for (face, &index) in entry.faces.iter().zip(&entry.face_indices) {
                let transfer = faces
                    .get(index)
                    .filter(|transfer| transfer.face == *face)
                    .ok_or_else(|| {
                        EngineError::rejected(
                            "checkpoint import",
                            format!(
                                "port face={face:?} at index {index} is missing from the step faces: supply an unchanged same-build checkpoint"
                            ),
                        )
                    })?;
                let volume = into * transfer.volume;
                inflow += volume.max(0.0);
                outflow += (-volume).max(0.0);
            }
        }
        Ok([inflow, outflow])
    }
}

fn line(face: BoundaryFace) -> (usize, usize) {
    match face.axis {
        Axis::X => (face.column, face.row),
        Axis::Y => (face.row, face.column),
    }
}

impl<R: Runtime<DType = DType>> Solver<R>
where
    R::Client: IndexingOps<R>,
{
    /// Builds a port set over `specs` for this engine's grid and boundary declaration.
    ///
    /// # Errors
    /// Rejects an empty set, a port without faces, a nonfinite ghost bed, a face with no
    /// active cell or two, a caller open face, a face in two ports or twice in one, faces off
    /// one axis, one ghost side, or one face line, and a gap in the face line.
    #[doc(hidden)]
    pub fn coupling_ports(
        &self,
        specs: &[SurfacePortSpec],
    ) -> Result<SurfacePorts<R>, EngineError> {
        let reject = |reason: String| EngineError::rejected("surface ports", reason);
        if specs.is_empty() {
            return Err(reject(
                "no ports: pass no ports instead of an empty port set".into(),
            ));
        }
        let domain = &self.definition.domain;
        let topology = &self.context.topology.faces;
        let supports: BTreeMap<_, _> = topology
            .iter()
            .enumerate()
            .map(|(index, support)| (support.face, (index, support)))
            .collect();
        let open: BTreeSet<_> = self
            .definition
            .boundaries
            .open_faces()
            .iter()
            .copied()
            .collect();
        let (dx, dy) = domain.grid().spacing();
        let mut seen = BTreeSet::new();
        let mut ports = Vec::with_capacity(specs.len());
        let mut lanes = Vec::new();
        for (port, spec) in specs.iter().enumerate() {
            if spec.faces.is_empty() {
                return Err(reject(format!(
                    "port={port} has no faces: bind at least one surface face"
                )));
            }
            if !spec.ghost_bed_m.is_finite() {
                return Err(reject(format!(
                    "port={port}, ghost_bed_m={}: supply a finite reach end bed",
                    spec.ghost_bed_m
                )));
            }
            let mut faces = spec.faces.clone();
            faces.sort_by_key(|face| line(*face).1);
            let first = faces[0];
            let mut face_indices = Vec::with_capacity(faces.len());
            let mut ghost_left = None;
            for (slot, face) in faces.iter().enumerate() {
                if !seen.insert(*face) {
                    return Err(reject(format!(
                        "port={port}, face={face:?} repeats: bind each face to one port once"
                    )));
                }
                let Some(&(index, support)) = supports.get(face) else {
                    return Err(reject(format!(
                        "port={port}, face={face:?} has no active surface cell: bind faces of active cells"
                    )));
                };
                if support.left.is_some() && support.right.is_some() {
                    return Err(reject(format!(
                        "port={port}, face={face:?} has two active cells: bind faces whose other side is inactive or off the grid"
                    )));
                }
                if open.contains(face) {
                    return Err(reject(format!(
                        "port={port}, face={face:?} is a caller open face: remove it from the open faces or from the port"
                    )));
                }
                if face.axis != first.axis || line(*face).0 != line(first).0 {
                    return Err(reject(format!(
                        "port={port}, face={face:?}, first face={first:?}: bind faces of one axis on one face line"
                    )));
                }
                let left = support.left.is_none();
                if *ghost_left.get_or_insert(left) != left {
                    return Err(reject(format!(
                        "port={port}, face={face:?}: the active cell lies on the other side from the first face; bind faces with the surface on one side"
                    )));
                }
                if slot > 0 && line(*face).1 != line(faces[slot - 1]).1 + 1 {
                    return Err(reject(format!(
                        "port={port}, face={face:?} follows {:?} with a gap: bind one contiguous face line",
                        faces[slot - 1]
                    )));
                }
                face_indices.push(index);
            }
            let ghost_left = ghost_left.unwrap_or(true);
            let into = if ghost_left { 1.0 } else { -1.0 };
            let reach = match spec.reach_end {
                PortReachEnd::Left => -1.0,
                PortReachEnd::Right => 1.0,
            };
            for (face, &index) in faces.iter().zip(&face_indices) {
                lanes.push(PortFaceLane {
                    port,
                    face_index: index,
                    ghost_left,
                    reach_sign: into * reach,
                    bed: spec.ghost_bed_m,
                    width: if face.axis == Axis::X { dy } else { dx },
                    x_axis: face.axis == Axis::X,
                });
            }
            ports.push(PortEntry {
                faces,
                face_indices,
                ghost_left,
            });
        }
        let faces = PortFaces::new(&lanes, ports.len(), self.context.client.device())?;
        Ok(SurfacePorts {
            layout: Arc::new(PortLayout {
                ports,
                lanes: lanes.len(),
                cell_count: domain.grid().cell_count(),
                face_count: topology.len(),
                boundaries: self.definition.boundaries.clone(),
            }),
            faces: Arc::new(faces),
        })
    }
}
