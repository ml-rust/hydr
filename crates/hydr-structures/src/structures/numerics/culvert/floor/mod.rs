//! Open sections above a flat bed at depth `h`, with depths measured up from the bed.
//!
//! The bed cuts the barrel at `y = h`. The open section is the barrel above that plane, of
//! rise `D' = D - h`, with a flat bed of width `b = T(h)`. A box or a flat-floor arch with
//! `h = 0` keeps its own floor as the bed.
//!
//! | Quantity | Form |
//! | --- | --- |
//! | `A`, `T`, `M` | band chain from the bed, or box closed forms on `D'` |
//! | `P_w` | wall arcs from the bed, plus the box crown `B` when full |
//! | `P` | `P_w + b` |
//! | `P_n` | `P_w + rho^(3/2) b`, the weighted perimeter of HDS-5 eq. 3.8 with exponent `2/3` |
//! | `R` | `A / P` |
//!
//! Each arc band starts at the bed normal `psi_s` with zero area, wall and moment. The band
//! formulas take each wall's chord and circular segment, so every area term is nonnegative and
//! no term cancels near the bed or under the crown. The full state closes the chain at the crown
//! and never subtracts the buried part from the barrel's full state.
//!
//! Every branch runs on every row with a harmless operand on rows it does not serve, and
//! `where_cond` selects per row.

mod bed;
mod chain;
mod frame;
mod layout;
mod point;
mod range;

pub(crate) use layout::{FloorInputs, FloorLayout, layout};
pub(crate) use point::{FloorSection, at_depth};
pub(crate) use range::over_depth;
