//! Outlet control for circular and box culvert barrels.
//!
//! [`CulvertOutletRow`] holds a barrel, its length, Manning `n`, entrance loss, slope and gravity.
//! [`CulvertOutletBatch`] encloses the outlet-control headwater `HW_o(Q, TW)` of each row from a
//! march upstream from the outlet, and the position of a hydraulic jump in a steep barrel.

mod batch;
mod capabilities;
mod output;
mod row;

pub use batch::CulvertOutletBatch;
pub use capabilities::culvert_outlet_capabilities;
pub use output::{
    CulvertOutletHeadwater, CulvertOutletHeadwaterRow, CulvertOutletJump, CulvertOutletJumpRow,
    JumpStatus, OutletRegime,
};
pub use row::CulvertOutletRow;
