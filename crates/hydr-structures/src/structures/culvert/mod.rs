//! Closed-conduit culvert barrels: circular, box, elliptical, pipe-arch and flat-floor arch.
//!
//! [`CulvertBarrelBatch`] encloses area, top width, wetted perimeter, hydraulic radius and
//! hydrostatic moment at represented depths and over depth ranges. Each row is dry at `y = 0`,
//! free for `0 < y < D` and full for `y >= D`. It also encloses the critical depth of each
//! barrel for a given discharge, and the Manning normal depth for a given discharge, roughness
//! and slope.
//!
//! [`CulvertInletBatch`] encloses the HDS-5 inlet-control headwater of a barrel and inlet.
//!
//! [`CulvertOutletBatch`] encloses the outlet-control headwater of a barrel, its length,
//! roughness and entrance loss, and the position of a hydraulic jump in a steep barrel.
//!
//! [`CulvertParameters`] defines the culvert structure law. Its discharge inverts the larger
//! of the two control headwaters.

mod barrel;
mod batch;
mod capabilities;
mod depth;
mod inlet;
mod input;
mod law_capabilities;
mod outlet;
mod output;
mod parameters;

pub use barrel::{CulvertBarrel, CulvertBarrelShape};
pub use batch::CulvertBarrelBatch;
pub use capabilities::culvert_barrel_capabilities;
pub use depth::{
    BarrelDepthStatus, CriticalDepthStatus, CulvertBarrelDepth, CulvertBarrelDepthRow,
    NormalDepthStatus,
};
pub use inlet::{
    CulvertInlet, CulvertInletBatch, CulvertInletForm, CulvertInletHeadwater,
    CulvertInletHeadwaterRow, CulvertInletRow, InletRegime, culvert_inlet_capabilities,
};
pub use law_capabilities::culvert_capabilities;
pub use outlet::{
    CulvertOutletBatch, CulvertOutletHeadwater, CulvertOutletHeadwaterRow, CulvertOutletJump,
    CulvertOutletJumpRow, CulvertOutletRow, JumpStatus, OutletRegime, culvert_outlet_capabilities,
};
pub use output::{BarrelState, CulvertBarrelGeometry, CulvertBarrelGeometryRow};
pub(crate) use parameters::CulvertSide;
pub use parameters::{CulvertAlignment, CulvertEnd, CulvertEnds, CulvertParameters};
