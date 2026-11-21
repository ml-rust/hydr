//! Certified barrel geometry tensors and their explicit host projection.

use crate::structures::StructureError;
use crate::structures::numerics::culvert::section::SectionIntervals;
use crate::structures::tensor::codes;
use numr::dtype::DType;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

/// Flow state of one barrel row.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BarrelState {
    /// Depth zero. Every quantity is an exact zero, except `T = P = B` for a box.
    Dry,
    /// Free surface below the crown, `0 < y < D`.
    Free,
    /// Full from the crown upward, `y >= D`. Depth above the crown is pressure head.
    Full,
}

/// Certified barrel geometry per row, as lower and upper F64 tensors of shape `[m]`.
///
/// Every exact value lies inside its `[lower, upper]` pair.
pub struct CulvertBarrelGeometry<R: Runtime<DType = DType>> {
    state: Tensor<R>,
    area: [Tensor<R>; 2],
    top_width: [Tensor<R>; 2],
    perimeter: [Tensor<R>; 2],
    radius: [Tensor<R>; 2],
    moment: [Tensor<R>; 2],
}

/// One projected row of [`CulvertBarrelGeometry`], each quantity as `(lower, upper)`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CulvertBarrelGeometryRow {
    /// Flow state of the row.
    pub state: BarrelState,
    /// Flow area `A`, in square metres.
    pub area_m2: (f64, f64),
    /// Free-surface width `T`, in metres. Zero in the full state.
    pub top_width_m: (f64, f64),
    /// Wetted perimeter `P`, in metres.
    pub wetted_perimeter_m: (f64, f64),
    /// Hydraulic radius `A / P`, in metres.
    pub hydraulic_radius_m: (f64, f64),
    /// Hydrostatic moment `integral from 0 to y of (y - eta) T(eta) d eta`, in cubic metres.
    /// In the full state it is `A_full (y - D/2)`.
    pub hydrostatic_moment_m3: (f64, f64),
}

impl<R: Runtime<DType = DType>> CulvertBarrelGeometry<R> {
    pub(crate) fn from_section(section: &SectionIntervals<R>) -> Self {
        let g = &section.geometry;
        let ends =
            |i: &solvr::common::arithmetic::Interval<R>| [i.lower().clone(), i.upper().clone()];
        Self {
            state: section.state.clone(),
            area: ends(&g.area),
            top_width: ends(&g.top_width),
            perimeter: ends(&g.perimeter),
            radius: ends(&g.radius),
            moment: ends(&g.moment),
        }
    }

    /// Lower and upper flow area `A`, in square metres.
    pub fn area(&self) -> (&Tensor<R>, &Tensor<R>) {
        (&self.area[0], &self.area[1])
    }

    /// Lower and upper free-surface width `T`, in metres.
    pub fn top_width(&self) -> (&Tensor<R>, &Tensor<R>) {
        (&self.top_width[0], &self.top_width[1])
    }

    /// Lower and upper wetted perimeter `P`, in metres.
    pub fn wetted_perimeter(&self) -> (&Tensor<R>, &Tensor<R>) {
        (&self.perimeter[0], &self.perimeter[1])
    }

    /// Lower and upper hydraulic radius `A / P`, in metres.
    pub fn hydraulic_radius(&self) -> (&Tensor<R>, &Tensor<R>) {
        (&self.radius[0], &self.radius[1])
    }

    /// Lower and upper hydrostatic moment, in cubic metres.
    pub fn hydrostatic_moment(&self) -> (&Tensor<R>, &Tensor<R>) {
        (&self.moment[0], &self.moment[1])
    }

    /// Copies every row to the host.
    pub fn project(&self) -> Result<Vec<CulvertBarrelGeometryRow>, StructureError> {
        let read =
            |t: &Tensor<R>| -> Result<Vec<f64>, StructureError> { Ok(t.try_to_vec::<f64>()?) };
        let pair = |[lower, upper]: &[Tensor<R>; 2]| -> Result<Vec<(f64, f64)>, StructureError> {
            Ok(read(lower)?.into_iter().zip(read(upper)?).collect())
        };
        let states = read(&self.state)?;
        let area = pair(&self.area)?;
        let top_width = pair(&self.top_width)?;
        let perimeter = pair(&self.perimeter)?;
        let radius = pair(&self.radius)?;
        let moment = pair(&self.moment)?;
        states
            .iter()
            .enumerate()
            .map(|(row, &code)| {
                let state = if code == codes::BARREL_DRY {
                    BarrelState::Dry
                } else if code == codes::BARREL_FREE {
                    BarrelState::Free
                } else if code == codes::BARREL_FULL {
                    BarrelState::Full
                } else {
                    return Err(StructureError::numerical(format!(
                        "barrel state code={code}: report this row, the code is not a barrel state"
                    ))
                    .at(row));
                };
                Ok(CulvertBarrelGeometryRow {
                    state,
                    area_m2: area[row],
                    top_width_m: top_width[row],
                    wetted_perimeter_m: perimeter[row],
                    hydraulic_radius_m: radius[row],
                    hydrostatic_moment_m3: moment[row],
                })
            })
            .collect()
    }
}
