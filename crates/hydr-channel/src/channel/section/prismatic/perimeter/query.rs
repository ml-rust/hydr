use super::super::{
    output::{PrismaticRecoveredProperties, PrismaticSectionProperties},
    query::{PrismaticSectionQuery, PrismaticSectionRecovery},
};
use super::{contact::Fields, projection::Projection};
use crate::channel::section::CrossSectionError;
use numr::dtype::DType;
use numr::runtime::{Runtime, RuntimeClient};
use solvr::common::arithmetic::Interval;

/// Host geometry and contact bounds for one original depth operand.
#[derive(Clone, Debug, PartialEq)]
pub struct PrismaticPerimeterProperties {
    /// Existing certified geometry, including its original requested depth.
    pub geometry: PrismaticSectionProperties,
    /// Wetted perimeter bounds `[lower, upper]`, in metres. Dry rows contain `[0, 0]`.
    pub wetted_perimeter_m: [f64; 2],
    /// Area divided by wetted perimeter, in metres. Absent when dry.
    pub hydraulic_radius_m: Option<[f64; 2]>,
}

/// Host geometry and contact bounds conditioned on one original conserved area.
#[derive(Clone, Debug, PartialEq)]
pub struct PrismaticPerimeterRecoveredProperties {
    /// Existing recovered geometry, including its original conserved area.
    pub geometry: PrismaticRecoveredProperties,
    /// Wetted perimeter bounds `[lower, upper]`, in metres. Dry rows contain `[0, 0]`.
    pub wetted_perimeter_m: [f64; 2],
    /// Original conserved area divided by wetted perimeter, in metres. Absent when dry.
    pub hydraulic_radius_m: Option<[f64; 2]>,
}

/// Native certified geometry, wetted perimeter, and radius at original represented depths.
///
/// Every tensor has shape `[N]`. Keep result storage and endpoint aliases immutable.
/// Results own their native storage. Dry perimeter and radius contain zero placeholders.
#[derive(Debug)]
pub struct PrismaticPerimeterQuery<R: Runtime<DType = DType>> {
    pub(super) geometry: PrismaticSectionQuery<R>,
    pub(super) fields: Fields<R>,
}

impl<R: Runtime<DType = DType>> PrismaticPerimeterQuery<R> {
    /// Existing certified query with original owners and depth operands.
    pub fn geometry(&self) -> &PrismaticSectionQuery<R> {
        &self.geometry
    }

    /// Wetted perimeter bounds, in metres. Keep endpoints immutable.
    pub fn wetted_perimeter_m(&self) -> &Interval<R> {
        &self.fields.perimeter
    }

    /// Area divided by wetted perimeter, in metres. Dry rows contain zero placeholders.
    pub fn hydraulic_radius_m(&self) -> &Interval<R> {
        &self.fields.radius
    }

    /// Copy geometry and contact bounds to host memory in original query order.
    pub fn properties(&self) -> Result<Vec<PrismaticPerimeterProperties>, CrossSectionError> {
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
                Ok(PrismaticPerimeterProperties {
                    geometry,
                    wetted_perimeter_m,
                    hydraulic_radius_m,
                })
            })
            .collect()
    }
}

/// Native recovered geometry and contact conditioned on original represented conserved areas.
///
/// Every tensor has shape `[N]`. Keep result storage and endpoint aliases immutable.
/// Radius uses the original conserved area. Dry radius contains a zero placeholder.
#[derive(Debug)]
pub struct PrismaticPerimeterRecovery<R: Runtime<DType = DType>> {
    pub(super) geometry: PrismaticSectionRecovery<R>,
    pub(super) fields: Fields<R>,
}

impl<R: Runtime<DType = DType>> PrismaticPerimeterRecovery<R> {
    /// Existing certified recovery with original owners and conserved area operands.
    pub fn geometry(&self) -> &PrismaticSectionRecovery<R> {
        &self.geometry
    }

    /// Wetted perimeter bounds, in metres. Keep endpoints immutable.
    pub fn wetted_perimeter_m(&self) -> &Interval<R> {
        &self.fields.perimeter
    }

    /// Original area divided by wetted perimeter, in metres. Dry rows contain zero placeholders.
    pub fn hydraulic_radius_m(&self) -> &Interval<R> {
        &self.fields.radius
    }

    /// Copy original conserved areas, geometry, and contact bounds to host memory.
    pub fn properties(
        &self,
    ) -> Result<Vec<PrismaticPerimeterRecoveredProperties>, CrossSectionError> {
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
                Ok(PrismaticPerimeterRecoveredProperties {
                    geometry,
                    wetted_perimeter_m,
                    hydraulic_radius_m,
                })
            })
            .collect()
    }
}
