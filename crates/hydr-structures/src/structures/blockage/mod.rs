//! Typed blockage and closure of a structure's declared geometry.
//!
//! The law holds the unblocked geometry. The element's condition scales it by the open share
//! `fl(1 - f)` of each blocked share `f`, or closes the structure to exact zero discharge.

mod condition;
pub(crate) mod effective;
mod fraction;
mod report;

pub use condition::{
    BlockageGeometry, CulvertBlockage, CulvertSediment, StructureBlockage, StructureCondition,
    StructureConditionKind,
};
pub use fraction::BlockageFraction;
pub(crate) use report::{CulvertGeometrySummary, reports};
pub use report::{StructureGeometry, StructureGeometryReport};
