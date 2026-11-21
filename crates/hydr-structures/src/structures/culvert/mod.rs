//! Closed-conduit culvert barrel geometry for circular and box barrels.
//!
//! [`CulvertBarrelBatch`] encloses area, top width, wetted perimeter, hydraulic radius and
//! hydrostatic moment at represented depths and over depth ranges. Each row is dry at `y = 0`,
//! free for `0 < y < D` and full for `y >= D`. It also encloses the critical depth of each
//! barrel for a given discharge, and the Manning normal depth for a given discharge, roughness
//! and slope.

mod barrel;
mod batch;
mod capabilities;
mod depth;
mod input;
mod output;

pub use barrel::{CulvertBarrel, CulvertBarrelShape};
pub use batch::CulvertBarrelBatch;
pub use capabilities::culvert_barrel_capabilities;
pub use depth::{
    BarrelDepthStatus, CriticalDepthStatus, CulvertBarrelDepth, CulvertBarrelDepthRow,
    NormalDepthStatus,
};
pub use output::{BarrelState, CulvertBarrelGeometry, CulvertBarrelGeometryRow};
