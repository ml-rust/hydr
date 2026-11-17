//! Checked represented support, class-area closures, and immutable topology.

use super::{Subarea, SubcatchmentError};
use crate::runoff::ReservoirParameters;
use crate::soil::ModifiedGreenAmptParameters;

/// Explicit SI physical support without GIS ownership inference.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SubcatchmentGeometry {
    /// Total physical area in m².
    pub total_area_m2: f64,
    /// Runoff width in metres.
    pub width_m: f64,
    /// Dimensionless slope.
    pub slope: f64,
    /// Fraction of total area in the combined impervious class.
    pub impervious_fraction: f64,
    /// Fraction within the impervious class having zero depression storage.
    pub zero_storage_impervious_fraction: f64,
    /// Manning coefficient for both impervious subareas.
    pub impervious_manning: f64,
    /// Manning coefficient for pervious support.
    pub pervious_manning: f64,
    /// Depression depth of impervious-storage support in metres.
    pub impervious_depression_m: f64,
    /// Depression depth of pervious support in metres.
    pub pervious_depression_m: f64,
}

/// Same-interval conservative internal topology. Accepted boundaries retain no queue.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum SubcatchmentRouting {
    /// Every represented subarea exports directly to the component outlet.
    Direct,
    /// Route this fraction of each impervious export to pervious support.
    ImperviousToPervious {
        /// Finite fraction in `[0,1]`.
        fraction: f64,
    },
    /// Route this fraction of pervious export to impervious-storage support only.
    PerviousToImpervious {
        /// Finite fraction in `[0,1]`.
        fraction: f64,
    },
}
impl SubcatchmentRouting {
    pub(crate) fn fraction(self) -> f64 {
        match self {
            Self::Direct => 0.0,
            Self::ImperviousToPervious { fraction } | Self::PerviousToImpervious { fraction } => {
                fraction
            }
        }
    }
}

/// Checked physical definitions with cached positive-area drainage closures.
#[derive(Clone, Debug, PartialEq)]
pub struct SubcatchmentParameters {
    geometry: SubcatchmentGeometry,
    soil: Option<ModifiedGreenAmptParameters>,
    routing: SubcatchmentRouting,
    areas: [f64; 3],
    area_bounds: [f64; 3],
    closures: [Option<ReservoirParameters>; 3],
}
impl SubcatchmentParameters {
    /// Check explicit geometry, exactly supported soil, and internal routing destinations.
    pub fn new(
        geometry: SubcatchmentGeometry,
        soil: Option<ModifiedGreenAmptParameters>,
        routing: SubcatchmentRouting,
    ) -> Result<Self, SubcatchmentError> {
        let g = geometry;
        let values = [
            g.total_area_m2,
            g.width_m,
            g.slope,
            g.impervious_fraction,
            g.zero_storage_impervious_fraction,
            g.impervious_manning,
            g.pervious_manning,
            g.impervious_depression_m,
            g.pervious_depression_m,
        ];
        if values.iter().any(|v| !v.is_finite() || *v < 0.0)
            || g.total_area_m2 <= 0.0
            || g.impervious_fraction > 1.0
            || g.zero_storage_impervious_fraction > 1.0
        {
            return Err(SubcatchmentError::input(format!(
                "geometry={g:?}: require finite nonnegative geometry, positive total area, and fractions within [0,1]"
            )));
        }
        let fraction = routing.fraction();
        if !fraction.is_finite() || !(0.0..=1.0).contains(&fraction) {
            return Err(SubcatchmentError::input(
                "routing fraction requires a finite value within [0,1]",
            ));
        }
        let impervious = g.total_area_m2 * g.impervious_fraction;
        let areas = [
            impervious * g.zero_storage_impervious_fraction,
            impervious * (1.0 - g.zero_storage_impervious_fraction),
            g.total_area_m2 * (1.0 - g.impervious_fraction),
        ];
        let positive = [
            g.impervious_fraction > 0.0 && g.zero_storage_impervious_fraction > 0.0,
            g.impervious_fraction > 0.0 && g.zero_storage_impervious_fraction < 1.0,
            g.impervious_fraction < 1.0,
        ];
        let mut closures = [None; 3];
        for area in Subarea::ALL {
            let i = area.index();
            if !areas[i].is_finite() || (positive[i] && areas[i] <= 0.0) {
                return Err(SubcatchmentError::input("mathematically positive subarea underflows or overflows. Rescale physical support").at_subarea(area));
            }
            if areas[i] > 0.0 {
                let (class, n, ds) = if area == Subarea::Pervious {
                    (areas[2], g.pervious_manning, g.pervious_depression_m)
                } else {
                    (
                        impervious,
                        g.impervious_manning,
                        if area == Subarea::ImperviousZeroStorage {
                            0.0
                        } else {
                            g.impervious_depression_m
                        },
                    )
                };
                closures[i] = Some(
                    ReservoirParameters::from_si(areas[i], class, g.width_m, g.slope, n, ds)
                        .map_err(|e| SubcatchmentError::from(e).at_subarea(area))?,
                );
            }
        }
        if soil.is_some() != (areas[2] > 0.0) {
            return Err(SubcatchmentError::input(
                "soil parameters must be present exactly when pervious area is positive",
            )
            .at_subarea(Subarea::Pervious));
        }
        let target = match routing {
            SubcatchmentRouting::Direct => None,
            SubcatchmentRouting::ImperviousToPervious { .. } => Some(Subarea::Pervious),
            SubcatchmentRouting::PerviousToImpervious { .. } => Some(Subarea::ImperviousStorage),
        };
        if let Some(target) = target
            && fraction > 0.0
            && areas[target.index()] == 0.0
        {
            return Err(SubcatchmentError::input(
                "positive internal routing requires a positive-area destination",
            )
            .at_subarea(target));
        }
        let area_bounds = super::support_bounds::area_bounds(g, areas)?;
        Ok(Self {
            geometry,
            soil,
            routing,
            areas,
            area_bounds,
            closures,
        })
    }
    /// Immutable supplied SI geometry.
    pub fn geometry(&self) -> &SubcatchmentGeometry {
        &self.geometry
    }
    /// Immutable same-interval internal topology.
    pub fn routing(&self) -> SubcatchmentRouting {
        self.routing
    }
    /// Checked native soil parameters for positive pervious support only.
    pub fn soil_parameters(&self) -> Option<ModifiedGreenAmptParameters> {
        self.soil
    }
    /// Actual represented storage area of one physical subarea in m².
    pub fn area_m2(&self, subarea: Subarea) -> f64 {
        self.areas[subarea.index()]
    }
    /// Existing reservoir closure for positive support. None denotes an absent subarea.
    pub fn closure(&self, subarea: Subarea) -> Option<ReservoirParameters> {
        self.closures[subarea.index()]
    }
    pub(crate) fn area_bound(&self, subarea: Subarea) -> f64 {
        self.area_bounds[subarea.index()]
    }
    pub(crate) fn receiver(&self) -> Option<Subarea> {
        if self.routing.fraction() == 0.0 {
            return None;
        }
        match self.routing {
            SubcatchmentRouting::ImperviousToPervious { .. }
                if self.areas[0] > 0.0 || self.areas[1] > 0.0 =>
            {
                Some(Subarea::Pervious)
            }
            SubcatchmentRouting::PerviousToImpervious { .. } if self.areas[2] > 0.0 => {
                Some(Subarea::ImperviousStorage)
            }
            _ => None,
        }
    }
    #[doc(hidden)]
    pub fn donor_fraction(&self, subarea: Subarea) -> f64 {
        match self.routing {
            SubcatchmentRouting::ImperviousToPervious { fraction }
                if subarea != Subarea::Pervious =>
            {
                fraction
            }
            SubcatchmentRouting::PerviousToImpervious { fraction }
                if subarea == Subarea::Pervious =>
            {
                fraction
            }
            _ => 0.0,
        }
    }
    #[doc(hidden)]
    pub fn same_bits(&self, other: &Self) -> bool {
        let a = self.geometry;
        let b = other.geometry;
        let geometry = |g: SubcatchmentGeometry| {
            [
                g.total_area_m2,
                g.width_m,
                g.slope,
                g.impervious_fraction,
                g.zero_storage_impervious_fraction,
                g.impervious_manning,
                g.pervious_manning,
                g.impervious_depression_m,
                g.pervious_depression_m,
            ]
        };
        let bits = |a: &[f64], b: &[f64]| a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits());
        let soil = match (self.soil, other.soil) {
            (None, None) => true,
            (Some(a), Some(b)) => bits(
                &[
                    a.suction_ft(),
                    a.conductivity_ft_per_s(),
                    a.maximum_deficit(),
                    a.upper_zone_depth_ft(),
                    a.capacity_ft(),
                    a.recovered_deficit_ceiling(),
                ],
                &[
                    b.suction_ft(),
                    b.conductivity_ft_per_s(),
                    b.maximum_deficit(),
                    b.upper_zone_depth_ft(),
                    b.capacity_ft(),
                    b.recovered_deficit_ceiling(),
                ],
            ),
            _ => false,
        };
        let routing = matches!(
            (self.routing, other.routing),
            (SubcatchmentRouting::Direct, SubcatchmentRouting::Direct)
                | (
                    SubcatchmentRouting::ImperviousToPervious { .. },
                    SubcatchmentRouting::ImperviousToPervious { .. }
                )
                | (
                    SubcatchmentRouting::PerviousToImpervious { .. },
                    SubcatchmentRouting::PerviousToImpervious { .. }
                )
        ) && self.routing.fraction().to_bits() == other.routing.fraction().to_bits();
        bits(&geometry(a), &geometry(b))
            && bits(&self.areas, &other.areas)
            && bits(&self.area_bounds, &other.area_bounds)
            && soil
            && routing
            && self
                .closures
                .iter()
                .zip(&other.closures)
                .all(|(a, b)| match (a, b) {
                    (None, None) => true,
                    (Some(a), Some(b)) => a.same_bits(b),
                    _ => false,
                })
    }
}
