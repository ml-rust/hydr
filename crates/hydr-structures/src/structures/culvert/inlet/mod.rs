//! HDS-5 inlet control for circular and box culvert barrels.
//!
//! [`CulvertInlet`] holds an HDS-5 code or explicit coefficients and the mitered flag.
//! [`CulvertInletBatch`] encloses the inlet-control headwater `HW_i(Q)` of each row: unsubmerged
//! up to `X = 3.5`, submerged from `X = 4`, and linear in `X` between them.

mod batch;
mod capabilities;
mod codes;
mod definition;
mod output;
mod row;

pub use batch::CulvertInletBatch;
pub use capabilities::culvert_inlet_capabilities;
pub use definition::{CulvertInlet, CulvertInletForm};
pub use output::{CulvertInletHeadwater, CulvertInletHeadwaterRow, InletRegime};
pub use row::CulvertInletRow;
