//! Certified outlet headwater and jump tensors, their status per row and their host projection.

use crate::structures::StructureError;
use crate::structures::numerics::culvert::outlet::OutletHeadwater;
use crate::structures::numerics::culvert::outlet::jump::OutletJump;
use crate::structures::tensor::codes;
use numr::dtype::DType;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

/// Outlet-control flow state of one row.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OutletRegime {
    /// Zero discharge. The headwater is the still level `max(TW - S L, 0)`.
    NoFlow,
    /// The barrel flows full from the outlet to the inlet.
    Full,
    /// A free-surface profile occupies part or all of the barrel.
    PartlyFull,
    /// A steep barrel runs at critical depth at the inlet: `HW_o = yc + (1 + ke) Vc^2 / (2 g)`.
    /// Critical depth is capped at the crown. A crown tie row holds the hull of this value and
    /// the full-barrel value.
    InletCritical,
    /// A certified comparison of the march did not resolve. Bounds are NaN.
    Unresolved,
}

impl OutletRegime {
    fn from_code(code: f64) -> Option<Self> {
        [
            (codes::OUTLET_NO_FLOW, Self::NoFlow),
            (codes::OUTLET_FULL, Self::Full),
            (codes::OUTLET_PARTLY_FULL, Self::PartlyFull),
            (codes::OUTLET_INLET_CRITICAL, Self::InletCritical),
            (codes::OUTLET_UNRESOLVED, Self::Unresolved),
        ]
        .into_iter()
        .find_map(|(value, regime)| (value == code).then_some(regime))
    }
}

/// Certified outlet-control headwater per row, as regime codes and lower and upper F64 `[m]`
/// tensors in metres above the inlet invert.
pub struct CulvertOutletHeadwater<R: Runtime<DType = DType>> {
    status: Tensor<R>,
    inlet: [Tensor<R>; 2],
    headwater: [Tensor<R>; 2],
}

/// One projected row of a [`CulvertOutletHeadwater`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CulvertOutletHeadwaterRow {
    /// Flow state of the row.
    pub regime: OutletRegime,
    /// `(lower, upper)` inlet head `h_in` in metres above the inlet invert: the depth, or the
    /// pressure head of a full inlet.
    pub inlet_head_m: (f64, f64),
    /// `(lower, upper)` headwater `HW_o = h_in + (1 + ke) V_in^2 / (2 g)` in metres above the
    /// inlet invert.
    pub headwater_m: (f64, f64),
}

fn regime_error(code: f64, row: usize) -> StructureError {
    StructureError::numerical(format!(
        "outlet regime code={code}: report this row, the code is not an outlet regime"
    ))
    .at(row)
}

impl<R: Runtime<DType = DType>> CulvertOutletHeadwater<R> {
    pub(crate) fn from_headwater(headwater: OutletHeadwater<R>) -> Self {
        Self {
            status: headwater.status,
            inlet: headwater.inlet,
            headwater: headwater.headwater,
        }
    }

    /// Lower and upper headwater in metres above the inlet invert.
    pub fn headwater_m(&self) -> (&Tensor<R>, &Tensor<R>) {
        (&self.headwater[0], &self.headwater[1])
    }

    /// Lower and upper inlet head in metres above the inlet invert.
    pub fn inlet_head_m(&self) -> (&Tensor<R>, &Tensor<R>) {
        (&self.inlet[0], &self.inlet[1])
    }

    /// Copies every row to the host.
    pub fn project(&self) -> Result<Vec<CulvertOutletHeadwaterRow>, StructureError> {
        let status = self.status.try_to_vec::<f64>()?;
        let inlet = [
            self.inlet[0].try_to_vec::<f64>()?,
            self.inlet[1].try_to_vec::<f64>()?,
        ];
        let headwater = [
            self.headwater[0].try_to_vec::<f64>()?,
            self.headwater[1].try_to_vec::<f64>()?,
        ];
        status
            .iter()
            .enumerate()
            .map(|(row, &code)| {
                let regime =
                    OutletRegime::from_code(code).ok_or_else(|| regime_error(code, row))?;
                let at = |values: &[Vec<f64>; 2]| -> (f64, f64) {
                    (
                        values[0].get(row).copied().unwrap_or(f64::NAN),
                        values[1].get(row).copied().unwrap_or(f64::NAN),
                    )
                };
                Ok(CulvertOutletHeadwaterRow {
                    regime,
                    inlet_head_m: at(&inlet),
                    headwater_m: at(&headwater),
                })
            })
            .collect()
    }
}

/// Hydraulic jump state of one row.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum JumpStatus {
    /// The row is not inlet-critical, so no supercritical profile leaves the inlet.
    NotSteep,
    /// Supercritical flow leaves the outlet. Any jump lies downstream of the barrel.
    Supercritical,
    /// A jump sits inside the barrel.
    InBarrel,
    /// A certified comparison did not resolve, the subcritical profile includes a full outlet
    /// segment, or the row is a crown tie.
    Unresolved,
}

impl JumpStatus {
    fn from_code(code: f64) -> Option<Self> {
        [
            (codes::JUMP_NOT_STEEP, Self::NotSteep),
            (codes::JUMP_SUPERCRITICAL, Self::Supercritical),
            (codes::JUMP_IN_BARREL, Self::InBarrel),
            (codes::JUMP_UNRESOLVED, Self::Unresolved),
        ]
        .into_iter()
        .find_map(|(value, status)| (value == code).then_some(status))
    }
}

/// Certified jump position and depths per row, as status codes and F64 `[m]` bound tensors.
///
/// It also carries the headwater of the march it ran on.
pub struct CulvertOutletJump<R: Runtime<DType = DType>> {
    headwater: CulvertOutletHeadwater<R>,
    status: Tensor<R>,
    position: [Tensor<R>; 2],
    upstream: [Tensor<R>; 2],
    downstream: [Tensor<R>; 2],
    outlet: [Tensor<R>; 2],
}

/// One projected row of a [`CulvertOutletJump`]. Fields that do not apply hold NaN.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CulvertOutletJumpRow {
    /// Jump state of the row.
    pub status: JumpStatus,
    /// `(lower, upper)` jump position in metres downstream of the inlet, for `InBarrel`.
    pub position_m: (f64, f64),
    /// `(lower, upper)` supercritical depth just upstream of the jump, for `InBarrel`.
    pub upstream_depth_m: (f64, f64),
    /// `(lower, upper)` subcritical depth just downstream of the jump, for `InBarrel`.
    pub downstream_depth_m: (f64, f64),
    /// `(lower, upper)` supercritical depth at the outlet, for `Supercritical`.
    pub outlet_depth_m: (f64, f64),
}

impl<R: Runtime<DType = DType>> CulvertOutletJump<R> {
    pub(crate) fn from_jump(jump: OutletJump<R>, headwater: CulvertOutletHeadwater<R>) -> Self {
        Self {
            headwater,
            status: jump.status,
            position: jump.position,
            upstream: jump.upstream,
            downstream: jump.downstream,
            outlet: jump.outlet,
        }
    }

    /// Outlet-control headwater of the same rows and inputs.
    pub fn headwater(&self) -> &CulvertOutletHeadwater<R> {
        &self.headwater
    }

    /// Lower and upper jump position in metres downstream of the inlet.
    pub fn position_m(&self) -> (&Tensor<R>, &Tensor<R>) {
        (&self.position[0], &self.position[1])
    }

    /// Copies every row to the host.
    pub fn project(&self) -> Result<Vec<CulvertOutletJumpRow>, StructureError> {
        let status = self.status.try_to_vec::<f64>()?;
        let read = |pair: &[Tensor<R>; 2]| -> Result<[Vec<f64>; 2], StructureError> {
            Ok([pair[0].try_to_vec::<f64>()?, pair[1].try_to_vec::<f64>()?])
        };
        let position = read(&self.position)?;
        let upstream = read(&self.upstream)?;
        let downstream = read(&self.downstream)?;
        let outlet = read(&self.outlet)?;
        status
            .iter()
            .enumerate()
            .map(|(row, &code)| {
                let status = JumpStatus::from_code(code).ok_or_else(|| {
                    StructureError::numerical(format!(
                        "jump status code={code}: report this row, the code is not a jump status"
                    ))
                    .at(row)
                })?;
                let at = |values: &[Vec<f64>; 2]| -> (f64, f64) {
                    (
                        values[0].get(row).copied().unwrap_or(f64::NAN),
                        values[1].get(row).copied().unwrap_or(f64::NAN),
                    )
                };
                Ok(CulvertOutletJumpRow {
                    status,
                    position_m: at(&position),
                    upstream_depth_m: at(&upstream),
                    downstream_depth_m: at(&downstream),
                    outlet_depth_m: at(&outlet),
                })
            })
            .collect()
    }
}
