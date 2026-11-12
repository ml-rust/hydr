use super::super::area_point_query::{
    PrismaticAreaPointRecoveredProperties, PrismaticAreaPointRecovery,
};
use super::{contact::Fields, projection::Projection};
use crate::channel::section::CrossSectionError;
use numr::dtype::DType;
use numr::runtime::{Runtime, RuntimeClient};
use solvr::common::arithmetic::Interval;

/// Host geometry and bank contact at one exact two-component area point.
#[derive(Clone, Debug, PartialEq)]
pub struct PrismaticPerimeterAreaPointRecoveredProperties {
    /// Section recovery preserving both requested area components and local band coordinates.
    pub geometry: PrismaticAreaPointRecoveredProperties,
    /// Wetted perimeter bounds `[lower, upper]`, in metres. Dry rows contain zero.
    pub wetted_perimeter_m: [f64; 2],
    /// Original requested area divided by perimeter, in metres. Absent when dry.
    pub hydraulic_radius_m: Option<[f64; 2]>,
}

/// Owned geometry, perimeter, and radius conditioned on exact high-plus-low area points.
///
/// Contact tensors have shape `[N]`. Keep tensors, endpoints, and storage aliases immutable.
/// Dry contact fields contain zero placeholders. Geometry retains compatible dry endpoint coordinates.
#[derive(Debug)]
pub struct PrismaticPerimeterAreaPointRecovery<R: Runtime<DType = DType>> {
    pub(super) geometry: PrismaticAreaPointRecovery<R>,
    pub(super) fields: Fields<R>,
}

impl<R: Runtime<DType = DType>> PrismaticPerimeterAreaPointRecovery<R> {
    /// Section recovery with original owners, area components, and retained local coordinates.
    pub fn geometry(&self) -> &PrismaticAreaPointRecovery<R> {
        &self.geometry
    }

    /// Wetted perimeter bounds, in metres. Keep endpoints immutable.
    pub fn wetted_perimeter_m(&self) -> &Interval<R> {
        &self.fields.perimeter
    }

    /// Original requested area divided by perimeter, in metres. Dry rows contain zero placeholders.
    pub fn hydraulic_radius_m(&self) -> &Interval<R> {
        &self.fields.radius
    }

    /// Copy component-preserving geometry and contact bounds to host memory in original query order.
    pub fn properties(
        &self,
    ) -> Result<Vec<PrismaticPerimeterAreaPointRecoveredProperties>, CrossSectionError> {
        let geometry = self.geometry.properties()?;
        let projection = Projection::read(
            &self.fields,
            geometry.len(),
            self.geometry.context.source.client.device(),
        )?;
        geometry
            .into_iter()
            .enumerate()
            .map(|(index, geometry)| {
                let (wetted_perimeter_m, hydraulic_radius_m) =
                    projection.row(index, geometry.dry)?;
                Ok(PrismaticPerimeterAreaPointRecoveredProperties {
                    geometry,
                    wetted_perimeter_m,
                    hydraulic_radius_m,
                })
            })
            .collect()
    }
}
